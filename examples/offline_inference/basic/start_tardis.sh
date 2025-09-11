export TARDIS_CONFIG_FILE="/home/wxt/open_sources/scripts/tardis_config.yaml"
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_FLASH_ATTN_VERSION=3
# export VLLM_PROFILE_START_STOP="0-20"

# sudo -E /usr/local/cuda/bin/nsys profile \
#     --capture-range cudaProfilerApi \
#     --capture-range-end=stop \
#     --backtrace none \
#     --sample none \
#     --cpuctxsw none \
#     --force-overwrite true \
#     --trace cuda,nvtx,osrt \
#     --gpu-metrics-devices=0 \
#     --trace-fork-before-exec=true \
#     -o tardis.nsys-rep \
sudo -E /home/wxt/miniconda3/envs/wxttest/bin/python basic.py 