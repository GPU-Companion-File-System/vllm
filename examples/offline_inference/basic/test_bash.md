# vLLM + LMCache + Tardis UB smoke

这个命令用于验证 `storage_backend: geminifs_ub` 的完整推理链路。

运行前确认：

- GPU0/GPU1 没有其他 vLLM 或 GeminiFS UB 任务。
- 没有其他任务占用 `/dev/snvm_control`、`/dev/ssnvme*` 或 `/mnt/gpu*/nvme-*`。
- 不要设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，否则 torch tensor 指针可能不能被 GeminiFS DMA map。
- UB daemon kernel 必须在 Python 启动前设置 `CUDA_MODULE_LOADING=EAGER`；不要用 lazy module loading 跑这条链路。
- 当前 UB default 模式会在每次 UB transfer 前启动 daemon kernel，并在 transfer 完成后停止 daemon kernel。

```bash
conda activate zwhtest

cd /home/zwh/vllm

sudo -E env \
  CUDA_MODULE_LOADING=EAGER \
  CUDA_VISIBLE_DEVICES=0,1 \
  PYTHONUNBUFFERED=1 \
  BASIC_MODEL_PATH=/data/models/Llama-3.1-8B-Instruct \
  BASIC_CONTEXT_CHARS=512 \
  BASIC_NUM_PROMPTS=1 \
  BASIC_NUM_RUNS=2 \
  BASIC_MAX_MODEL_LEN=2048 \
  BASIC_MAX_NUM_BATCHED_TOKENS=2048 \
  BASIC_MAX_TOKENS=4 \
  BASIC_GPU_MEMORY_UTILIZATION=0.45 \
  TARDIS_CONFIG_FILE=/home/zwh/vllm/examples/offline_inference/basic/tardis_ub_smoke_config.yaml \
  PYTHONPATH=/home/zwh/vllm:/home/zwh/lmcache-dev \
  LD_LIBRARY_PATH=/home/zwh/lmcache-dev/csrc/GeminiFS/build/lib:/home/zwh/.conda/envs/zwhtest/lib/python3.12/site-packages/torch/lib:/home/zwh/.conda/envs/zwhtest/lib:/usr/local/cuda/lib64 \
  /home/zwh/.conda/envs/zwhtest/bin/python \
  examples/offline_inference/basic/basic.py
```

如果上一次 UB 运行 hang 住，先只清理自己的残留进程，再 reset snvme：

```bash
sudo bash -lc '
set -e
for m in /mnt/gpu0/nvme-0000:50:00.0 /mnt/gpu0/nvme-0000:41:00.0 /mnt/gpu1/nvme-0000:51:00.0 /mnt/gpu1/nvme-0000:44:00.0; do
  if mountpoint -q "$m"; then
    umount "$m"
  fi
done
cd /home/zwh/lmcache-dev/csrc/GeminiFS/scripts
./reset_snvme.sh
'
```
