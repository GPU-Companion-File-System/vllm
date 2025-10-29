sudo -E \
/usr/local/cuda/bin/nsys profile \
    --trace-fork-before-exec=true \
    --gpu-metrics-devices=0 \
    --capture-range cudaProfilerApi \
/home/wxt/miniconda3/envs/wxttest/bin/python basic.py