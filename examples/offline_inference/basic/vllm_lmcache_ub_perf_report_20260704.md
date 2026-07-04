# vLLM + LMCache + GeminiFS UB 性能进展汇报

日期：2026-07-04

## 结论

当前 UB 后端已经跑通真实 vLLM prefix reuse 推理链路，并且在 128K 复用前缀 benchmark 上达到 DRAM-level warm-load 延迟，稳定快于当前 GeminiFS 和 SSD。

保守汇报口径：

- UB 已经从最初明显慢于 GeminiFS，优化到 128K 场景下 DRAM-level。
- 当前最佳 repeat4 结果是 UB bulk-load block512 warm mean `0.962s`。
- 同组 DRAM warm mean 是 `1.029s`，GeminiFS warm mean 是 `1.076s`。
- UB 对 DRAM 的差距只有几十毫秒量级，不能宣传为大幅超过 DRAM；更准确说法是“达到 DRAM-level，并快于 GeminiFS/SSD”。
- UB 仍有明确优化空间：profile 显示剩余主要时间在 UB xfer/sync，不在 Python 元数据准备。

## 当前有效 128K 结果

benchmark 形态：

- 模型：`/data/models/Llama-3.1-8B-Instruct`
- prompt tokens：`129162`
- warm hit tokens：`129024 / 129162`
- vLLM：V1 engine，`enforce_eager=True`
- UB 运行：`CUDA_MODULE_LOADING=EAGER`，`CUDA_VISIBLE_DEVICES=0,1`
- 复用场景：同一进程内 run1 cold/store，run2-run4 warm/load

| Backend | Run1 cold/store (s) | Warm runs >=2 (s) | Warm mean (s) | vs DRAM mean | vs GeminiFS mean | Log |
|---|---:|---:|---:|---:|---:|---|
| UB bulk block512 | 17.697 | 0.982 / 0.949 / 0.956 | 0.962 | 0.94x | 0.89x | `/tmp/lmcache_block512_20260704/UB-bulk-block512-repeat4-128K.log` |
| UB bulk block256 | 17.561 | 1.006 / 0.957 / 0.961 | 0.974 | 0.95x | 0.91x | `/tmp/lmcache_bulk_load_20260704/UB-bulk-load-repeat4-128K.log` |
| UB blocks | 17.422 | 1.022 / 0.983 / 0.983 | 0.996 | 0.97x | 0.93x | `/tmp/lmcache_repeat_20260704/UB-blocks-repeat4-128K.log` |
| DRAM | 16.655 | 1.056 / 1.025 / 1.006 | 1.029 | ref | 0.96x | `/tmp/lmcache_repeat_20260704/DRAM-repeat4-128K.log` |
| GeminiFS | 16.276 | 1.095 / 1.068 / 1.064 | 1.076 | 1.05x | ref | `/tmp/lmcache_repeat_20260704/GeminiFS-repeat4-128K.log` |

单次 128K 对比里，当前 UB bulk-load warm 是 `1.008s`，DRAM 是 `0.953s`，GeminiFS 是 `1.086s`，SSD 是 `4.580s`。repeat4 更适合作为汇报依据，因为 DRAM/UB/GeminiFS 都在 1 秒附近，单次差异容易受噪声影响。

## 优化路径

1. UB batch xfer

最初 UB default 128K warm/load 是 `12.643s`。UB pybind 以前对每个 K/V block 单独发 xfer，128K warm path 约等于：

`32 layers * 504 blocks * 2 K/V = 32256` 次小 transfer。

加入 batch transfer 后，128K warm/load 降到 `1.749s`，后续把 `TARDIS_UB_MAX_XFERS_PER_LAUNCH` 调到 `1024`，降到约 `1.095s`。

2. block-pointer API

原始 batch path 仍然在 Python 侧构造大量 per-block tensor view。改成传整层 KV tensor + block ids，让 C++ 根据 stride 算 K/V 指针后，128K warm/load 从约 `1.095s` 继续降到 repeat mean `0.996s`。

profile 对比：

| Item | 旧路径 | block-pointer |
|---|---:|---:|
| `GeminiFSUBBackend.layerwise_batch_get` | 0.666s | 0.544s |
| `prepare ub kv cache read` total | 0.134s | 0.000635s |
| `deamon_pool_io_batch_xfer_kernel` | 0.493s | 0.493s |

结论：Python per-block view 构造已经不是瓶颈，剩余主要是 UB xfer 本身。

3. bulk-load

普通 UB blocks warm run 每层各读一次并启动/停止 daemon。bulk-load 在 forward 前一次性把所有层读完，避免每层调度路径重复。

repeat4 结果：

- UB blocks warm mean：`0.996s`
- UB bulk-load warm mean：`0.974s`
- 改善约 `22ms`

profile：

- `GeminiFSUBBackend.layerwise_batch_get_all`：`500.302ms`
- `TardisEngine.retrieve_all_layers`：`500.333ms`
- `prepare ub bulk kv cache read`：`0.027ms`
- `deamon_pool_io_batch_xfer_kernel` overlap：`492.299ms`

结论：bulk-load 已经把 Python/metadata 热路径压到很低，剩余主要是 UB transfer + 同步等待。

4. block512 tuning

默认 `block_size=256`、`chunk_size=256` 时，每个 K 或 V transfer 是 `512KiB`。block512 把 vLLM block 和 Tardis chunk 都设成 `512`，每个 K/V transfer 变成 `1MiB`，descriptor 数量减半。

repeat4 结果：

- UB bulk block256 warm mean：`0.974s`
- UB bulk block512 warm mean：`0.962s`
- 改善约 `12ms / 1.2%`

结论：block512 有轻微收益，可以作为 tuning knob；目前不建议把它宣传成主要优化。

## 失败或不保留的实验

| 实验 | 结果 | 结论 |
|---|---|---|
| async1 / persistent daemon | warm run 卡住或长时间无结果，GPU 100% | 当前 daemon 常驻会和 vLLM compute 抢 GPU 资源，不作为稳定路径 |
| 低资源 async1 daemon | 5K warm run 仍挂住 | 简单减少 daemon worker blocks/warps 不够 |
| metadata host-pack | 128K 单次 `1.054s`，变慢 | 不保留 |
| metadata device-cache | repeat mean `1.006s`，未优于 UB blocks `0.996s` | 不保留 |
| final-sync variant | repeat mean `0.981s`，略慢于 bulk-load `0.974s` | 不保留 |
| block1024 | 5K cold/store 提交 6 个任务后卡住，无 `BENCH_RESULT` | 不安全，不跑 128K |
| block512 Nsight retry | 推理跑出 run2 `1.002s`，但 Nsight/CUPTI 退出阶段留下 D-state zombie | 不能作为 profile 归因，需重启/驱动恢复后再 profile |

## 当前瓶颈判断

128K warm hit 约读 `15.75GiB` KV。bulk-load profile 里 UB xfer kernel overlap 约 `492ms`，等效读带宽约 `32GiB/s`。这说明：

- 当前慢点不是 Python 调度，也不是 metadata copy。
- 当前主要瓶颈是 UB transfer/daemon scheduling/sync。
- 想进一步稳定超过 DRAM，需要提高 UB read bandwidth，或者安全地把下一层 IO 和当前层 compute overlap。

## 下一步建议

1. 先恢复机器状态再继续跑实验。当前 Nsight/CUPTI 留下 PID `1836242`，`nvidia-smi -q` 仍显示占用 GPU0 约 `40510MiB`、GPU1 约 `2378MiB`。用户态 kill 和 `nvidia-smi --gpu-reset -i 0,1` 都清不掉，建议重启或做驱动级恢复。
2. 重启后优先补一个干净的 block512 Nsight profile，确认 `1MiB` transfer 是否真的缩短 `deamon_pool_io_batch_xfer_kernel`，还是只是 benchmark 噪声。
3. 后续优化重点放在 UB xfer/daemon scheduling，不再优先做 metadata 微优化。
4. async/persistent 方向需要重新设计 daemon residency，不能沿用当前 async1 直接让 daemon kernel 和模型 compute 常驻重叠的方案。

## 复现辅助

- 机器可读结果表：
  `/home/zwh/lmcache-dev/benchmarks/geminifs_ub/results_20260704.json`
- Markdown 摘要生成：
  `/home/zwh/lmcache-dev/benchmarks/geminifs_ub/summarize_results.py`
- 跑新 benchmark 前的环境检查：
  `/home/zwh/lmcache-dev/benchmarks/geminifs_ub/check_ub_bench_env.sh`

当前这台机器运行环境检查会失败，因为 GPU 仍有 Nsight/CUPTI 残留。重启或驱动级恢复后，先确认该检查通过，再采集新的 UB/DRAM/GeminiFS 对比数字。

## 关键提交

- `ac15c6c perf: batch GeminiFS UB pool transfers`
- `615fb41 perf: make UB batch launch size tunable`
- `e6d0fb4 perf: avoid per-block UB tensor views`
- `52eb954 perf: add opt-in UB bulk load`
- `aa6f522 bench: add UB block size tuning result`
- `c3ac1ce docs: record UB block1024 smoke hang`
