# vLLM + LMCache + GeminiFS UB 128K 性能汇报

日期：2026-07-07

## 结论

在 clean benchmark state 下，UB 后端已经达到本轮目标：128K prefix reuse
warm-load 延迟接近 DRAM，并明显快于 GeminiFS 和 SSD。

当前推荐汇报口径：

- UB clean-state repeat4 warm mean：`0.988s`
- DRAM repeat4 warm mean：`0.959s`
- GeminiFS repeat4 warm mean：`2.510s`
- SSD repeat4 warm mean：`4.873s`
- UB 只比 DRAM 慢约 `3%`，比 GeminiFS 快约 `2.54x`
- 早先同日 `1.8-1.9s` 的 UB 结果是 dirty SNVMe / machine state 结果，不是当前代码路径上限

## 当前有效 128K 结果

benchmark 形态：

- 模型：`/data/models/Llama-3.1-8B-Instruct`
- prompt tokens：`129162`
- warm hit tokens：`129024 / 129162`
- 生成 token：4
- vLLM：V1 engine，eager mode，chunked prefill
- `BENCH_MAX_MODEL_LEN=131072`
- `BENCH_MAX_NUM_BATCHED_TOKENS=8192`
- UB：`TARDIS_UB_BULK_LOAD=1`，`TARDIS_UB_ASYNC_READ=0`
- UB batch size：`TARDIS_UB_MAX_XFERS_PER_LAUNCH=1024`
- vLLM/Tardis block size：`512`

| Backend | Run1 cold/store (s) | Warm runs >=2 (s) | Warm mean (s) | vs DRAM | vs GeminiFS | Log |
|---|---:|---:|---:|---:|---:|---|
| DRAM | 16.060 | 0.970 / 0.942 / 0.964 | 0.959 | ref | 0.38x | `/tmp/lmcache_full_workload_20260707/dram_full_b512_128k_retry.log` |
| GeminiFS | 15.829 | 2.670 / 2.469 / 2.390 | 2.510 | 2.62x | ref | `/tmp/lmcache_full_workload_20260707/geminifs_full_b512_128k.log` |
| SSD | 16.223 | 5.708 / 4.608 / 4.303 | 4.873 | 5.08x | 1.94x | `/tmp/lmcache_full_workload_20260707/ssd_full_b512_128k.log` |
| UB batch clean state | 17.392 | 0.995 / 0.969 / 0.999 | 0.988 | 1.03x | 0.39x | `/tmp/lmcache_ub_trace_compare_20260707/vllm_ub_batch_repeat4_clean_b512_128k.log` |

这些 warm runs 基本是 KV load / IO 测试：warm run 复用 `129024 / 129162`
tokens，只生成 4 个新 token。

## UB 为什么现在快了

当前保留的稳定优化路径是：

1. UB batch transfer：把大量 per-block K/V transfer 合成 batch xfer。
2. block-pointer API：传整层 KV tensor + block ids，在 C++ 里算指针，避免 Python 构造大量 per-block tensor view。
3. bulk-load：warm run 前一次性读完所有层，走 `layerwise_batch_get_all()`。
4. block512：每个 K/V block 变成 1 MiB，减少 descriptor 数量。

128K warm load 的实际 UB 形态：

| Item | Value |
|---|---:|
| Layers | 32 |
| Blocks | 252 |
| Unique pool files | 252 |
| K/V bytes per block | 1 MiB |
| Per-layer read | 504 MiB |
| Total read | 15.75 GiB |
| Block id range | `[23, 274]` |
| Pool file id range | `[260, 511]` |

## 关键 profile / trace 证据

`TARDIS_UB_TRACE_BULK_TIMING=1` 是轻量 timing trace，用来定位 UB bulk read
内部耗时；正式性能结论仍使用 non-profiled repeat4。

| Run | Warm/load wall | UB bulk timing | Per-layer mean | Log |
|---|---:|---:|---:|---|
| standalone full-shape microbench | 0.5176s | 517.454ms total, 501.983ms layer total | 15.687ms | `/tmp/lmcache_ub_trace_compare_20260707/standalone_blocks32_trace.log` |
| vLLM 128K UB batch trace | 1.0259s | 500.058ms total, 499.367ms layer total | 15.605ms | `/tmp/lmcache_ub_trace_compare_20260707/vllm_ub_batch_trace_b512_128k.log` |

vLLM trace 中，`layerwise_batch_get_all()` 内部非 xfer 开销很小：

| Component | Time |
|---|---:|
| prepare pool ids | 0.037ms |
| current stream wait | 0.036ms |
| daemon launch | 0.244ms |
| daemon stop | 0.157ms |

结论：clean state 下，standalone microbench 和真实 vLLM 路径都能走到同一条
`~0.50s / 15.75GiB` 的 UB fast path，约 `31GiB/s`。瓶颈不在 Python metadata
或 daemon control。

## 为什么之前同一天有 1.8-1.9s 的 UB 结果

早先同日结果：

| Backend | Warm mean (s) | Log |
|---|---:|---|
| UB batch | 1.856 | `/tmp/lmcache_full_workload_20260707/ub_batch_full_b512_128k.log` |
| UB defer-sync 4-stream | 1.860 | `/tmp/lmcache_full_workload_20260707/ub_defer_sync4_full_b512_128k_retry.log` |

后续排查发现，当时机器/SNVMe 状态不干净：

- `/dev/snvme*n1` ext4 filesystem 残留挂载在 `/mnt/gpu*`
- `snvme` module `use_count` 卡在 4
- full-shape retry 一度在 GeminiFS init 阶段报：
  `ioctl_chrdev_helper err is -1` / `Failed to open device descriptor`

清理 stale mount、reload lmcache-dev 的 SNVMe module，并通过 hardened gate
之后，同一代码路径恢复到 `~0.50s` UB bulk time 和 `0.988s` vLLM warm mean。

因此，`1.8-1.9s` 行应该标记为 dirty-state diagnostic，不作为当前 UB 后端性能。

## 复现 checklist

正式跑 UB benchmark 前先在 lmcache-dev 里做 dry-run：

```bash
cd /home/zwh/lmcache-dev
benchmarks/geminifs_ub/recover_ub_bench_env.sh
```

如果 dry-run 发现 stale mount/node 或 unbound BDF，再执行：

```bash
benchmarks/geminifs_ub/recover_ub_bench_env.sh --apply --smoke
```

或者直接跑 gate：

```bash
CUDA_MODULE_LOADING=EAGER \
UB_BENCH_CUDA_SMOKE=1 \
UB_BENCH_CUDA_SMOKE_TIMEOUT_SECS=30 \
benchmarks/geminifs_ub/check_ub_bench_env.sh
```

clean state 必须满足：

- GPU0/GPU1 空闲
- 四个目标 NVMe BDF 都绑定 native `nvme`
- `/dev/snvm_control` 存在
- `snvme use_count=0`
- 没有 `/dev/snvme*` 或 `/dev/ssnvme*` stale node
- 没有 `/dev/snvme*n1` mount under `/mnt/gpu*`
- CUDA allocation smoke 通过

## 复现实验命令入口

最新机器可读结果：

```bash
cd /home/zwh/lmcache-dev
python benchmarks/geminifs_ub/summarize_results.py
```

详细 artifacts：

- 当前结果 JSON：`/home/zwh/lmcache-dev/benchmarks/geminifs_ub/results_20260707.json`
- 汇报版 report：`/home/zwh/lmcache-dev/benchmarks/geminifs_ub/ub_128k_report_20260707.md`
- 长状态文档：`/home/zwh/lmcache-dev/docs/ub_lmcache_perf_status_20260626.md`
- runbook：`/home/zwh/lmcache-dev/benchmarks/geminifs_ub/next_ub_profile_plan.md`
- environment gate：`/home/zwh/lmcache-dev/benchmarks/geminifs_ub/check_ub_bench_env.sh`
- recovery helper：`/home/zwh/lmcache-dev/benchmarks/geminifs_ub/recover_ub_bench_env.sh`

## 当前下一步

UB transfer fast path 已经达到接近 DRAM 的目标。继续优化时优先做：

1. 保证 clean-state recovery 可复现，避免把 dirty SNVMe 状态误算成 UB 性能。
2. 降低 non-UB warm wall：vLLM warm wall 约 `0.99-1.03s`，UB bulk read 本身约 `0.50s`。
3. 保持 `TARDIS_UB_ASYNC_READ=0` 默认；persistent async 之前会 hang / starve GPU。
4. deferred-sync 当前只保留为诊断实验；repeat4 没有稳定优于 batch。

