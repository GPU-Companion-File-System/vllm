# vLLM + LMCache + Tutti UB

这条 runbook 使用新 `tardis.tutti_ops` 和 Tutti `Coordinator(SERVICE_CLIENT)`，不加载 Legacy `GeminiFS`/`tardis.c_ops`。

运行前必须确认两张 GPU 空闲、四个盘已由 `snvme` 绑定，并且 Tutti daemon 使用四盘配置启动：

```bash
export CUDA_MODULE_LOADING=EAGER
export SNVME_KERNEL_VERSION=5.15.0-public

cd /home/zwh/lmcache-tutti-ub
./scripts/build_tutti_lmcache.sh

sudo -E csrc/GeminiFS/build/bin/nvmeservice_daemon \
  --config csrc/GeminiFS/benchmarks/tutti_ub_pool/sys_config_4nvme.yaml
```

另一个终端启动 vLLM smoke：

```bash
conda activate zwhtest
cd /home/zwh/vllm-tutti-ub

CUDA_MODULE_LOADING=EAGER \
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONUNBUFFERED=1 \
BASIC_MODEL_PATH=/data/models/Llama-3.1-8B-Instruct \
BASIC_CONTEXT_CHARS=512 \
BASIC_NUM_PROMPTS=1 \
BASIC_NUM_RUNS=2 \
BASIC_TENSOR_PARALLEL_SIZE=2 \
BASIC_PREFIX_MODE=full \
BASIC_MAX_MODEL_LEN=2048 \
BASIC_MAX_NUM_BATCHED_TOKENS=2048 \
BASIC_MAX_TOKENS=4 \
BASIC_GPU_MEMORY_UTILIZATION=0.45 \
TARDIS_CONFIG_FILE=/home/zwh/vllm-tutti-ub/examples/offline_inference/basic/tardis_tutti_ub_smoke_config.yaml \
PYTHONPATH=/home/zwh/vllm-tutti-ub:/home/zwh/lmcache-tutti-ub \
LD_LIBRARY_PATH=/home/zwh/lmcache-tutti-ub/csrc/GeminiFS/build/lib:$LD_LIBRARY_PATH \
/home/zwh/.conda/envs/zwhtest/bin/python \
examples/offline_inference/basic/basic.py
```

`BASIC_TENSOR_PARALLEL_SIZE=2` 会显式传给 vLLM 的 `LLM(..., tensor_parallel_size=2)`，不能仅凭 `CUDA_VISIBLE_DEVICES=0,1` 判断这是 TP=2。将 `BASIC_PREFIX_MODE=partial` 可让第一轮写入完整前缀、后续轮次只复用前缀的一部分，用于验证 layerwise 路径。

验证顺序应为：先运行四盘 `pool_file_ub_bench` 的 subrange correctness，再运行 TP=1；确认 rank-local namespace 和文件恢复后，再并发运行 TP=2。任何只启动单个 rank 的结果都不能作为 TP=2 结论。

`TARDIS_TUTTI_DEFER_SYNC` 可以通过 YAML 的 `tutti_defer_sync` 打开。部分 prefix 命中保留 layerwise 路径；只有高复用/全命中并且调用方支持 bulk wait 时启用 bulk read。

也可以使用 `run_tutti_ub_benchmark.sh` 统一采集环境、版本、拓扑和日志：

```bash
export CUDA_MODULE_LOADING=EAGER
export SNVME_KERNEL_VERSION=5.15.0-public
export BASIC_MODEL_PATH=/data/models/Llama-3.1-8B-Instruct
export BASIC_TENSOR_PARALLEL_SIZE=2
export BASIC_PREFIX_MODE=full
export BASIC_CONTEXT_CHARS=512
export BASIC_NUM_PROMPTS=1
export BASIC_NUM_RUNS=2
export BASIC_MAX_MODEL_LEN=2048
export BASIC_MAX_NUM_BATCHED_TOKENS=2048
export BASIC_MAX_TOKENS=4
export BASIC_GPU_MEMORY_UTILIZATION=0.45

CUDA_VISIBLE_DEVICES=0,1 \
  ./run_tutti_ub_benchmark.sh
```

输出目录包含 `environment.json`、完整 `basic.log`、`metrics.json`、按 rank
写出的 `tutti_stats_rank*.json` 和退出状态。
其中 `metrics.json` 记录每轮 wall time、每请求 TTFT/E2E 和
`num_cached_tokens`。正式 TP=2
实验必须在两张 GPU 空闲、daemon 已启动后运行；脚本不会终止其他进程。
