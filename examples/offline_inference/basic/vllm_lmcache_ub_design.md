# vLLM-LMCache-UB 后端接入设计

## 背景

当前 offline 推理测试链路在
`/home/zwh/vllm/examples/offline_inference/basic/test_bash.md`：

```bash
conda activate zwhtest

sudo -E env \
  CUDA_MODULE_LOADING=EAGER \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/home/zwh/vllm:/home/zwh/lmcache-dev \
  LD_LIBRARY_PATH=/home/zwh/lmcache-dev/csrc/GeminiFS/build/lib:/home/zwh/.conda/envs/zwhtest/lib/python3.12/site-packages/torch/lib:/home/zwh/.conda/envs/zwhtest/lib:/usr/local/cuda/lib64 \
  /home/zwh/.conda/envs/zwhtest/bin/python \
  examples/offline_inference/basic/basic.py
```

`basic.py` 使用 vLLM 的 `TardisConnectorV1`：

```python
KVTransferConfig(
    kv_connector="TardisConnectorV1",
    kv_role="kv_both",
)
```

Tardis 侧代码在 `/home/zwh/lmcache-dev`。当前链路大致是：

1. vLLM scheduler 通过 `TardisConnectorV1Impl.get_num_new_matched_tokens()` 查询外部 KV 命中。
2. worker 在 `start_load_kv()` 中准备 load/store 元信息。
3. load 走 `TardisEngine.prepare_for_load()`，得到 `GPUFileMetadata` 和 vLLM block id。
4. store 走 `TardisEngine.prepare_for_store()`，分配 `GPUFileMetadata`。
5. layerwise IO 走：
   - `retrieve_layer()` -> `GeminiFSBackend.layerwise_batch_get()`
   - `store_layer()` -> `GeminiFSBackend.layerwise_batch_put()`
6. `GeminiFSBackend` 调用 `tardis._custom_ops`，最终进入 `tardis.c_ops` pybind：
   - `geminifs_batched_read(k_caches, v_caches, gpu_file_ids, layer, controller, stream)`
   - `geminifs_batched_write(k_caches, v_caches, gpu_file_ids, layer, controller, stream)`

当前 Tardis 后端是单 GPU local GeminiFS file 语义：一个 cache chunk 对应一个 `GPUFileId`，每次按 layer 对多个 chunk 做 batched read/write。

Geminifs UB 分支在 `/home/zwh/Geminifs`，当前分支是 `zwh-lmcache-dev`。UB 相关测试主要在：

- `/home/zwh/Geminifs/test/io_deamon/io_deamon_pool_bandwidth_test.cu`
- `/home/zwh/Geminifs/test/io_deamon/io_deamon_pool_iops_test.cu`
- `/home/zwh/Geminifs/test/io_deamon/io_deamon_aggregate_iops_test.cu`

其中 vLLM-LMCache 链路应该优先封装 pool-file 聚合路径，而不是直接暴露 aggregate IOPS 测试里的窗口模型。

## 目标

第一阶段目标是跑通 `vllm -> TardisConnectorV1 -> TardisEngine -> UB storage backend -> GeminiFS UB pool-file -> multi-GPU aggregate IO`。

成功标准：

1. offline `basic.py` 可以用 `CUDA_VISIBLE_DEVICES=0,1` 或更多 GPU 启动。
2. warm run 能从 UB 后端读取 KV，输出结果正常。
3. store/load 的 KV 数据校验能通过单元测试或专门 smoke test。
4. UB 后端在日志/NVTX 中能看到 pool-file 聚合 IO 路径，而不是旧的 `geminifs_batched_*` local file 路径。

非目标：

1. 第一阶段不重写 vLLM connector 调度逻辑。
2. 第一阶段不做跨进程、跨节点的 lookup/persistence 强一致性。
3. 第一阶段不追求最优 batch/concurrency，只先保证接口正确、链路可跑、性能可测。

## 当前验证状态

当前实现已经跑通过以下链路：

1. GeminiFS UB pool-file integrity 测试。
2. pybind UB K/V round trip，`k_equal=True`、`v_equal=True`。
3. vLLM + LMCache + Tardis + `storage_backend: geminifs_ub` offline smoke，`BASIC_NUM_RUNS=2` 时第二轮能读回 UB KV 并完成推理。
4. 2026-06-11 使用 `CUDA_MODULE_LOADING=EAGER`、`CUDA_VISIBLE_DEVICES=0,1`、Llama-3.1-8B-Instruct 跑通两轮 smoke；命令未设置旧的 `TARDIS_UB_STOP_DAEMON_AFTER_XFER`，日志见 `/tmp/tardis_ub_final_smoke.log`。
5. 2026-06-11 曾验证过 bounded persistent daemon smoke；但该路径性能没有优于 default UB，且依赖额外 green-context API。本轮代码精简后不把它放进主 UB backend。

本轮调试确认的关键运行约束：

- 必须在 Python 进程启动前设置 `CUDA_MODULE_LOADING=EAGER`。UB daemon kernel 不能用 CUDA lazy module loading 作为运行方案。
- 不要设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，否则 torch tensor 指针可能不能被 GeminiFS DMA map。
- 默认 in-process UB 模式仍可以在每次 UB transfer 前启动 daemon kernel，并在 transfer 完成后停止 daemon kernel。
- 旧的无限 resident persistent daemon 不能直接放在推理进程里。profile 和最小 repro 显示，daemon server kernels 原来在普通 CUDA stream 上 launch；PyTorch 的 device/current-stream synchronization 会等待 resident kernel，因此 torch allocation、`torch.cuda.synchronize()`、部分 vLLM/PyTorch 同步路径可能永久卡住。
- 当前最小实现不保留 persistent/green-context 分支。后续如果继续优化 daemon 生命周期，应作为独立设计重新引入，并用实际 benchmark 证明收益。

## UB 抽象与现有 Tardis 模型的映射

UB pool-file 的关键结构是 `deamon_pool_file_full_view_t`：

```cpp
typedef struct {
    int64_t gpu_id;
    uint64_t gpu_file_size;
    uint64_t gpu_file_granularity;
    nvl_queue_client_ctrl_t **client_ctrls;
    uint64_t gpu_num;
    deamon_gpu_file_full_view_t *local_gpu_file_view;
} deamon_pool_file_full_view_t;
```

核心 device primitive 是：

```cpp
deamon_pool_io_xfer(
    pool_file_view,
    buffer,
    size,
    pool_file_id,
    file_offset,
    is_read
)
```

它按 `gpu_file_granularity` 切分一个 pool file 的逻辑地址：

```text
chunk_index      = file_offset / gpu_file_granularity
target_gpu       = chunk_index % gpu_num
gpu_file_offset  = (chunk_index / gpu_num) * gpu_file_granularity + offset_in_chunk
```

因此一个 `pool_file_id` 是跨 GPU 聚合的逻辑文件 id。同一个逻辑 file 的不同 stripe 会分布到多个 GPU 的同名 GPU file 上。

这和 Tardis 当前的模型可以保持一致：

```text
CacheEngineKey -> GPUFileMetadata.id
              -> pool_file_id
```

也就是说，第一阶段不需要把 Tardis 的 key/index 改成 slot/offset 模型。`GPUFileMetadata.id` 继续作为后端内部 id，只是在 UB backend 中解释为 `pool_file_id`。

需要注意的容量语义变化：

- 旧 GeminiFS：每个 GPU file 只在当前 GPU 的本地 NVMe 上。
- UB pool-file：每个 `pool_file_id` 在所有可见 GPU 上都有对应 local GPU file，逻辑文件容量近似为 `gpu_num * local_gpu_file_size`，数据按 `DEAMON_GPU_FILE_GRANULARITY` stripe。
- 为了简化第一阶段，一个 Tardis chunk 仍然完整写入一个 `pool_file_id` 的 offset 0。

## 建议实现方案

### 1. 在 GeminiFS C++ 层增加 UB host wrapper

在 `/home/zwh/Geminifs` 已有 `deamon_pool_io_xfer_kernel`，但现在没有面向 Python/Tardis 的 batched K/V wrapper。需要在 `GeminiFS` 类上增加 host API：

```cpp
bool geminifs_ub_pool_read(
    const std::vector<torch::Tensor>& k_caches,
    const std::vector<torch::Tensor>& v_caches,
    const std::vector<GPUFileId>& pool_file_ids,
    int layer_idx,
    int device_id,
    cudaStream_t stream);

bool geminifs_ub_pool_write(
    const std::vector<torch::Tensor>& k_caches,
    const std::vector<torch::Tensor>& v_caches,
    const std::vector<GPUFileId>& pool_file_ids,
    int layer_idx,
    int device_id,
    cudaStream_t stream);
```

内部逻辑：

1. 确保 `launch_io_deamon_kernels()` 已经启动，daemon server kernel 已运行。
2. 通过 `get_deamon_manager()->get_deamon(device_id)->get_pool_file_view()` 获取当前 worker GPU 的 pool view。
3. 对每个 `(k_cache, v_cache, pool_file_id)` 发起 K/V 两次 pool IO：
   - K offset: `layer_idx * 2 * kv_bytes_per_layer`
   - V offset: `layer_idx * 2 * kv_bytes_per_layer + kv_bytes_per_layer`
4. `size` 使用 `k_cache.nbytes()` / `v_cache.nbytes()`。
5. 第一阶段可以每个 K/V launch 一个 `deamon_pool_io_xfer_kernel<<<1, PIPELINE_CTRL_THREADS + PIPELINE_PCIE_THREADS, 0, stream>>>`。

这个实现最小化风险，但 batch 性能有限。后续再实现真正 batched UB kernel。

### 2. 处理 UB 没有 batch 接口的问题

Tardis 当前上层天然按 layer 传入多 chunk：

```python
k_caches: list[Tensor]
v_caches: list[Tensor]
gpu_file_ids: list[int]
```

UB 当前只有单 buffer 的 `deamon_pool_io_xfer_kernel`。建议分两阶段：

第一阶段，host-side loop：

```text
for each file in batch:
  launch pool write K
  launch pool write V
```

优点：

- 最快跑通 vLLM-LMCache-UB。
- 复用已有 pool-file routing 和 daemon server kernel。
- 易于校验 correctness。

缺点：

- launch 数量是 `num_files * 2 * num_layers`。
- 每个 `deamon_pool_io_xfer()` 内部目前主要由 thread 0 submit/harvest pending remote IO，batch 内并行度不足。

第二阶段，新增真正 batch kernel：

```cpp
__global__ void ub_pool_batched_kv_xfer_kernel(
    deamon_pool_file_full_view_t* pool_file_view,
    void** k_ptrs,
    void** v_ptrs,
    uint32_t* pool_file_ids,
    uint32_t num_files,
    uint64_t kv_bytes,
    uint64_t layer_base_offset,
    bool is_read);
```

可以先采用 one thread/warp per file 的模型，参考
`io_deamon_pool_iops_test.cu` 的 `pool_iops_window_kernel`，把 pool-file routing 内联到 batch kernel 中。这样可以让 batch 内多个 file 并发 submit NVLink queue request。

### 3. 在 pybind 和 `_custom_ops.py` 暴露 UB 接口

修改 `/home/zwh/lmcache-dev/csrc/geminifs_pybind.cu`：

- 新增 `GeminiFS.launch_io_deamon_kernels()`
- 新增 `GeminiFS.stop_io_deamon_kernels()`
- 新增 `GeminiFS.is_io_deamon_kernel_launched()`
- 新增 `GeminiFS.geminifs_ub_pool_read(...)`
- 新增 `GeminiFS.geminifs_ub_pool_write(...)`

修改 `/home/zwh/lmcache-dev/tardis/_custom_ops.py`：

- `initialize_geminifs(..., backend_mode="tardis" | "ub_pool")`
- `launch_io_deamon_kernels()`
- `stop_io_deamon_kernels()`
- `geminifs_ub_pool_read(...)`
- `geminifs_ub_pool_write(...)`

daemon 生命周期由 UB backend 在 transfer 前按需管理：

```python
if need_ub_xfer:
    backend.ensure_io_daemon_started()
    geminifs_ub_pool_read_or_write(...)
```

当前 default 路径在每次 transfer 完成后显式 stop。persistent daemon 不放进本轮最小实现，后续需要结合 UB batch API 和 PyTorch/vLLM 同步行为重新设计。

### 4. 新增或改造 Tardis storage backend

建议新增文件：

```text
/home/zwh/lmcache-dev/tardis/v1/storage_backend/geminifs_ub_backend.py
```

并保留旧 `GeminiFSBackend`。原因：

- 旧后端使用 `GPUController`、tensor register、`geminifs_batched_xfer`，语义是 local file。
- UB 后端依赖 daemon manager、pool view、visible multi-GPU set，初始化和生命周期不同。
- 分开后更容易 A/B 测试 `tardis` vs `ub_pool`。

UB backend 可复用大部分 Python 层索引逻辑：

```python
self.dict: dict[CacheEngineKey, GPUFileMetadata]
self.allocate(key) -> GPUFileMetadata(pool_file_id, key)
self.contains/get/remove(...)
self.layerwise_batch_get(...)
self.layerwise_batch_put(...)
```

差异点：

- 初始化时调用 `initialize_geminifs(..., backend_mode="ub_pool")`。
- `allocate()` 仍可调用 `gpu_open_file(self.device_id)`，但返回 id 解释为 `pool_file_id`。
- `layerwise_batch_get/put()` 调用 `geminifs_ub_pool_read/write()`。
- 不需要 `register_tensor_with_gpu()`，除非 UB host wrapper 仍依赖旧 GPUController PRP 注册。pool IO path 使用 daemon staging/NVLink/NVMe 路径，应该避免复用旧 batched xfer 的 tensor DMA 注册。

### 5. 配置项

在 `TardisEngineConfig` 增加：

```yaml
storage_backend: "geminifs"      # "geminifs" | "geminifs_ub"
ub_enable_io_deamon: true
ub_client_gpu_id: 0              # 默认 torch.cuda.current_device()
ub_pool_file_size_multiplier: 1  # 预留
```

第一阶段最小配置：

```yaml
chunk_size: 256
max_num_local_file: 32780
max_local_disk_size: 100
sys_config_path: "/home/zwh/Geminifs/test/io_deamon/sys_config.ini"
storage_backend: "geminifs_ub"
ub_enable_io_deamon: true
```

在 `TardisEngine.__init__()` 中根据 `config.storage_backend` 选择 backend：

```python
if config.storage_backend == "geminifs_ub":
    self.storage_backend = GeminiFSUBBackend(...)
else:
    self.storage_backend = GeminiFSBackend(...)
```

### 6. vLLM 启动方式

UB 需要一个进程可见多张 GPU，因为 `GeminiFS::init()` 会遍历 `cudaGetDeviceCount()` 并为所有可见 GPU 初始化 daemon/pool view。

测试脚本建议新增：

```text
/home/zwh/vllm/examples/offline_inference/basic/test_ub_bash.md
```

内容基于现有 `test_bash.md`，关键变化：

```bash
sudo -E env \
  CUDA_MODULE_LOADING=EAGER \
  CUDA_VISIBLE_DEVICES=0,1 \
  PYTHONPATH=/home/zwh/vllm:/home/zwh/lmcache-dev \
  TARDIS_CONFIG_FILE=/home/zwh/open_sources/scripts/tardis_ub_config.yaml \
  LD_LIBRARY_PATH=/home/zwh/lmcache-dev/csrc/GeminiFS/build/lib:/home/zwh/.conda/envs/zwhtest/lib/python3.12/site-packages/torch/lib:/home/zwh/.conda/envs/zwhtest/lib:/usr/local/cuda/lib64 \
  /home/zwh/.conda/envs/zwhtest/bin/python \
  examples/offline_inference/basic/basic.py
```

如果 vLLM 使用 tensor parallel，多 worker 进程会改变每个 worker 的 visible device/rank 关系。第一阶段建议先使用单 vLLM worker、`CUDA_VISIBLE_DEVICES=0,1`，让 GPU0 作为 client GPU，GPU0+GPU1 作为 UB 聚合 IO 设备。后续再扩展到 TP worker 每个 rank 各自拥有一组 UB 聚合卡。

## 数据布局

当前 `GeminiFSBackend` 初始化：

```python
kv_size = chunk_size * num_kv_head * head_size * dtype.itemsize
gpu_file_shape = [kv_shape[1], num_layers, kv_size]
per_file_size = 2 * num_layers * kv_size
```

这个布局可以直接复用到 UB pool file：

```text
pool_file_id:
  layer 0 K: offset 0
  layer 0 V: offset kv_size
  layer 1 K: offset 2 * kv_size
  layer 1 V: offset 3 * kv_size
  ...
```

对于 MLA，`kv_shape[1] == 1`，需要在 wrapper 中按 `kv_shape[1]` 泛化，不能硬编码 K/V 两路。第一阶段如果只测 Llama-3.1-8B-Instruct，先支持 `kv_shape[1] == 2` 即可，但代码里应显式 assert 并报错。

## Correctness 验证计划

### 1. GeminiFS UB 单元测试

先在 `/home/zwh/Geminifs` 跑现有测试确认底层可用：

```bash
cd /home/zwh/Geminifs
# 具体 build 目录按当前机器已有构建为准
./build/test/io_deamon/io_deamon_pool_bandwidth_test
./build/test/io_deamon/io_deamon_pool_iops_test --gtest_filter='*Integrity*'
```

需要确认：

- daemon kernel 能启动。
- pool-file read/write 能通过数据校验。
- `sys_config.ini` 中 GPU/NVMe 拓扑和当前机器一致。

### 2. pybind smoke test

在 `/home/zwh/lmcache-dev/tests/v1` 增加 `test_geminifs_ub.py`：

1. 初始化 `GeminiFS(..., gpu_file_shape=[2, num_layers, kv_size])`。
2. 启动 daemon。
3. 构造一层或两层 K/V tensor。
4. 写入 `pool_file_id=0`。
5. 读回到另一个 tensor。
6. `torch.allclose()` 校验。
7. stop daemon。

### 3. Tardis backend smoke test

构造 mock `CacheEngineKey` 和 mock layer KV cache，直接调用：

```python
backend.allocate(key)
backend.layerwise_batch_put(...)
backend.layerwise_batch_get(...)
```

校验读回 KV 和写入 KV 一致。这个测试不依赖完整 vLLM。

### 4. vLLM offline smoke test

用 `basic.py` 跑两次相同 prompt：

- cold run：cache miss，然后 store 到 UB。
- warm run：cache hit，然后 load from UB。

日志中需要看到：

- `storage_backend=geminifs_ub`
- `launch_io_deamon_kernels`
- `geminifs_ub_pool_write/read`
- `Tardis hit tokens > 0`

## 风险与处理

1. **daemon persistent kernel 生命周期**
   - 风险：无限 resident daemon 常驻在同一 Python/vLLM 进程里，会被 PyTorch allocation、stream sync、device sync 等路径等待，导致推理挂住。
   - 处理：当前最小实现默认每次 UB transfer 后 stop daemon；persistent/green-context 分支不放进主路径，避免依赖未验证收益的实验代码。

2. **CUDA_VISIBLE_DEVICES 和 device id 映射**
   - 风险：Geminifs `sys_config.ini` 使用物理 `cudaDevice`，vLLM 进程使用 remapped visible id。
   - 处理：第一阶段固定 `CUDA_VISIBLE_DEVICES=0,1`，配置文件也用 0/1；后续增加 physical/logical mapping 配置。

3. **UB pool-file 容量和 Tardis max file 数不一致**
   - 风险：每个 pool file 需要每张 GPU 上同 id file 都存在。
   - 处理：初始化时所有可见 GPU 都创建相同数量 file；`max_num_local_file` 仍表示 pool file 数，不是所有 GPU file 总数。

4. **batch 性能不足**
   - 风险：host-side loop launch 太多 kernel，性能不代表 UB 能力。
   - 处理：先跑通 correctness；第二阶段实现 `ub_pool_batched_kv_xfer_kernel`。

5. **KV tensor contiguous/shape 假设**
   - 风险：vLLM KV cache layout 变化会导致 offset 错误。
   - 处理：复用当前 `layerwise_batch_get/put()` 中的 `kvcache[0, block_id]` / `kvcache[1, block_id]`，并 assert contiguous、CUDA、nbytes 等于 `kv_size`。

6. **local GPU path 也通过 NVLink queue**
   - 当前 `init_deamon_pool_file_view()` 对所有 GPU 都填 `client_ctrls[i]`，包括本 GPU。第一阶段按现有实现走；若本地路径性能异常，再单独优化 local chunk 直接走 local GPU file view。

## 建议改动清单

1. `/home/zwh/Geminifs`
   - 在 `GeminiFS` 增加 UB pool read/write host API。
   - 如需要，新增 batch kernel 文件或放在 `io_deamon.cu` 附近。
   - 确认 CMake 安装/导出新增符号。

2. `/home/zwh/lmcache-dev/csrc/GeminiFS`
   - 同步 UB 分支代码，确保包含 `io_deamon*`、`nvl_queue*`、pool-file 相关文件。
   - `setup.py` 链接仍使用 `libgeminifs.so` 和 `libnvm.so`。

3. `/home/zwh/lmcache-dev/csrc/geminifs_pybind.cu`
   - 绑定 daemon lifecycle 和 UB pool read/write。

4. `/home/zwh/lmcache-dev/tardis/_custom_ops.py`
   - 暴露 Python wrapper。

5. `/home/zwh/lmcache-dev/tardis/config.py`
   - 增加 `storage_backend` 和 UB 相关配置。

6. `/home/zwh/lmcache-dev/tardis/v1/tardis_engine.py`
   - 根据配置选择 backend。

7. `/home/zwh/lmcache-dev/tardis/v1/storage_backend/geminifs_ub_backend.py`
   - 新增 UB backend。

8. `/home/zwh/vllm/examples/offline_inference/basic`
   - 新增 `test_ub_bash.md`。
   - 可选新增 `tardis_ub_config.yaml` 示例或在文档中指向实际配置路径。

## 第一阶段实施顺序

1. 先在 `/home/zwh/Geminifs` 跑通 pool-file integrity/bandwidth 测试。
2. 将 UB 分支 GeminiFS 同步到 `/home/zwh/lmcache-dev/csrc/GeminiFS`，重新构建 `libgeminifs.so`。
3. 增加 pybind daemon lifecycle 和 `geminifs_ub_pool_read/write`。
4. 写 `test_geminifs_ub.py`，直接验证 pybind K/V round trip。
5. 新增 `GeminiFSUBBackend`，复用 Tardis key/index 逻辑。
6. 跑 Tardis backend smoke test。
7. 新增 `test_ub_bash.md`，用 `basic.py` 跑 vLLM offline cold/warm 链路。
8. 再做 UB batched kernel 性能优化。

## 结论

最稳妥的路线是把 UB pool-file 作为 Tardis 的一个新 storage backend，而不是改 vLLM connector。Tardis 现有 layerwise load/store 已经提供了合适的接入点；第一版只需要把 `GPUFileMetadata.id` 解释为 `pool_file_id`，并把旧的 `geminifs_batched_read/write` 替换成 UB pool-file read/write wrapper。

UB 缺少 batch 接口不是阻塞项。第一阶段用 host-side loop 封装单 file pool IO，先跑通 vLLM-LMCache-UB；第二阶段再参考 `io_deamon_pool_iops_test.cu` 把 pool routing 内联进真正的 batched kernel，提高并发和减少 kernel launch。
