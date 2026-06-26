# vLLM + LMCache/Tardis Prefix Reuse Benchmark

Date: 2026-06-26

## Setup

- Model: `/data/models/Llama-3.1-8B-Instruct`
- GPU: NVIDIA H100 PCIe
- vLLM: V1 engine, `enforce_eager=True`, `enable_prefix_caching=False`
- Scenario: one long shared prefix plus different suffix questions
- Prompt length: 5094 tokens
- Shared prefix: 5078 tokens
- Chunk size: 256
- Max model length: 8192
- Max batched tokens: 8192
- Output: 4 tokens
- Runs:
  - Run1: cold miss, prefill and store KV
  - Run2: warm prefix reuse, external KV hit/load

The measured wall time is only the `llm.generate()` call. It excludes model
loading, vLLM engine initialization, and GeminiFS reset/setup.

## 5K Results

| Backend | Connector | GPUs visible | Run1 cold/store (s) | Run2 warm/load (s) | Hit tokens | Notes |
|---|---|---:|---:|---:|---:|---|
| DRAM | `LMCacheConnectorV1` | 1 | 0.324 | 0.114 | 4864 / 5094 | LMCache local CPU backend |
| SSD | `LMCacheConnectorV1` | 1 | 0.318 | 0.245 | 4864 / 5094 | LMCache `local_disk`, with 4GB CPU staging |
| GeminiFS | `TardisConnectorV1` | 2 | 0.282 | 0.176 | 4864 / 5094 | Tardis legacy GeminiFS backend |
| UB | `TardisConnectorV1` | 2 | 0.740 | 0.562 | 4864 / 5094 | `geminifs_ub`, default non-persistent daemon |
| UB persistent | `TardisConnectorV1` | 2 | 0.610 | 0.594 | 4864 / 5094 | `TARDIS_UB_PERSIST_DAEMON=1` |
| UB persistent async | `TardisConnectorV1` | 2 | 2.433 | 2.404 | 4864 / 5094 | `TARDIS_UB_PERSIST_DAEMON=1`, `TARDIS_UB_ASYNC_XFER=1` |
| GDS/NDS | `LMCacheConnectorV1` | - | - | - | - | Blocked by driver/backend issue; no valid data |

The persistent rows are historical experiments from this run. The cleaned
minimal UB backend keeps only the default non-persistent path because those
variants did not improve the real inference benchmark and added extra
green-context/submodule API dependencies.

## 128K Results

This is the larger prefix-reuse benchmark requested after the 5K smoke table.
Llama-3.1-8B-Instruct supports a 131072-token context window, so the benchmark
uses a 129162-token prompt with a 129146-token shared prefix. A direct
`max_num_batched_tokens=131072` run OOMed on H100 during prefill, so the valid
128K runs use vLLM chunked prefill with `max_num_batched_tokens=8192`.

| Backend | Connector | GPUs visible | Run1 cold/store (s) | Run2 warm/load (s) | Hit tokens | Notes |
|---|---|---:|---:|---:|---:|---|
| DRAM | `LMCacheConnectorV1` | 1 | 16.203 | 0.980 | 129024 / 129162 | `local_cpu`, 32GB limit |
| SSD | `LMCacheConnectorV1` | 1 | 16.549 | 5.593 | 129024 / 129162 | `local_disk`, 32GB CPU staging |
| GeminiFS | `TardisConnectorV1` | 2 | 15.989 | 2.198 | 129024 / 129162 | legacy GeminiFS backend, 512-file capacity |
| UB | `TardisConnectorV1` | 2 | 22.522 | 12.643 | 129024 / 129162 | `geminifs_ub`, 512-file capacity |
| UB default capacity | `TardisConnectorV1` | 2 | 17.560 | 16.959 | 32768 / 129162 | default 128 files only holds 32K tokens, not a valid full-prefix comparison |
| UB direct 128K batch | `TardisConnectorV1` | 2 | - | - | - | `max_num_batched_tokens=131072` OOMed before a valid result |

## Interpretation

UB is functionally running end to end:

`vLLM -> TardisConnectorV1 -> TardisEngine -> geminifs_ub -> GeminiFS UB pool-file -> multi-GPU daemon IO`

However, in this prefix-reuse inference benchmark, the current UB path is not
yet performance competitive. For the 5K warm hit path, default UB is:

- 4.95x slower than DRAM
- 2.30x slower than SSD
- 3.20x slower than legacy GeminiFS

For the 128K warm hit path, UB with enough file capacity is:

- 12.90x slower than DRAM
- 2.26x slower than SSD
- 5.75x slower than legacy GeminiFS

The main observed bottleneck is control overhead in the current UB integration:

- The Tardis UB backend does not expose a true batch interface equivalent to
  the legacy GeminiFS batched read/write path.
- The Python backend loops over smaller transfer groups.
- The default UB mode repeatedly launches and stops IO daemon kernels around
  transfers.
- The bounded persistent mode still logs daemon drain/relaunch events, so it is
  not acting like one stable long-lived daemon across the whole request.
- The async variant is currently worse in this integration.

## UB Profile Findings

The current profile is based on timestamped Python/C++ logs from the completed
5K and 128K benchmark runs, plus the UB/legacy backend code paths.

For 128K UB cold/store:

- Wall time: 22.522s.
- UB daemon launch count: 512.
- UB daemon stop count: 512.
- `launch_start -> launch_done` summed to about 12.991s.
- `launch_done -> stop_start`, which includes UB xfer and stream sync, summed
  to about 8.305s.
- `stop_start -> next_launch`, which includes daemon stop and Python/backend
  overhead, summed to about 0.768s.
- These three buckets account for about 98.0% of the measured cold/store wall
  time.

For 128K UB warm/load:

- Wall time: 12.643s.
- UB daemon launch count: 32.
- UB daemon stop count: 32.
- `launch_done -> stop_start`, which includes the per-layer UB read and stream
  sync, summed to about 12.021s, or about 95.1% of wall time.
- Average per-layer read+sync time was about 0.376s.

The key code-path differences are:

- UB calls `_ensure_io_daemon_started()` and `_finish_io_xfer()` inside each
  `layerwise_batch_put/get`. In default mode `_finish_io_xfer()` synchronizes
  the stream and then stops daemon kernels after every transfer.
- The UB pybind wrapper accepts lists, but internally loops over every cached
  block and launches one `deamon_pool_io_xfer_kernel` for K and one for V.
  For the 128K warm hit, that is:

  `32 layers * 504 blocks * 2 K/V = 32256 small UB xfer kernels`

- The legacy GeminiFS path uses `geminifs_batched_read/write`, which builds
  batched IO contexts and launches batched transfer kernels. Its layerwise read
  path can prefetch the next layer on a load stream and rely on stream
  dependencies instead of synchronizing every layer in the backend.

This explains why UB becomes much worse as the reused prefix grows: the warm
path scales with the number of blocks times layers as many small synchronous
UB transfers, while the legacy path has a real batched transfer path and better
overlap with model execution.

The highest-priority fixes are:

1. Add a true UB batch read/write kernel/API that submits all K/V block
   transfers for one layer in one or a small number of kernels instead of one
   kernel per block per K/V tensor.
2. Stop launching/stopping daemon kernels per layer/chunk. Use a safe
   request-scoped or process-scoped daemon lifetime that does not conflict with
   PyTorch synchronization.
3. Remove the unconditional per-layer backend `stream.synchronize()` from the
   read path; use CUDA stream dependencies so the model waits only when it
   reaches the layer whose KV is needed.
4. After the above, rerun Nsight Systems with the existing NVTX ranges
   (`prepare ub kv cache read/write`) to verify kernel-launch count, daemon
   lifetime, and IO/compute overlap.

## Current Status

- DRAM, SSD, GeminiFS, and UB all completed the same real vLLM inference prefix
  reuse benchmark at both 5K and 128K prompt sizes.
- UB backend is confirmed runnable, but current performance is worse than DRAM,
  SSD, and legacy GeminiFS in both tested single-request workloads.
- For 128K, UB must use a larger file capacity (`max_num_local_file: 512`) to
  hold the full prefix. The default 128-file config only hit 32768 tokens.
- GDS/NDS is marked blocked. Before reboot it caused driver/NVML calls to hang
  after the GDS path failed during backend initialization; this needs separate
  driver/backend debugging before a valid benchmark number can be reported.

## Logs

- `/tmp/vllm_lmcache_bench_cpu.log`
- `/tmp/vllm_lmcache_bench_ssd.log`
- `/tmp/vllm_lmcache_bench_geminifs.log`
- `/tmp/vllm_lmcache_bench_ub.log`
- `/tmp/vllm_lmcache_bench_ub_persist.log`
- `/tmp/vllm_lmcache_bench_ub_persist_async.log`
- `/tmp/vllm_lmcache_bench_dram_128k_chunk8192.log`
- `/tmp/vllm_lmcache_bench_ssd_128k_chunk8192_staging32.log`
- `/tmp/vllm_lmcache_bench_geminifs_128k_chunk8192_files512.log`
- `/tmp/vllm_lmcache_bench_ub_128k_chunk8192_files512.log`
- `/tmp/vllm_lmcache_bench_ub_128k_chunk8192.log`
- `/tmp/vllm_lmcache_bench_ub_128k.log`

## Reproduction Notes

After a reboot, UB/GeminiFS may fail with `Failed to open control descriptor`
if the GeminiFS `snvme` module and `/dev/snvm_control` are not restored. In this
run, the fix was to load `snvme-core.ko` and `snvme.ko`, then create
`/dev/snvm_control` from the major/minor exposed at:

`/sys/class/libsnvm helper/snvm_control/dev`

UB requires `CUDA_MODULE_LOADING=EAGER` before Python starts.
