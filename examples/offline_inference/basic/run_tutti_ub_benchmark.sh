#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LMCACHE_ROOT="${LMCACHE_ROOT:-${VLLM_ROOT}/../lmcache-tutti-ub}"
PYTHON="${PYTHON:-python3}"
MODEL_PATH="${BASIC_MODEL_PATH:-/data/models/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${TUTTI_VLLM_OUTPUT_DIR:-/tmp/tutti-vllm-benchmark}"
RUN_ID="${TUTTI_VLLM_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${OUTPUT_DIR}/${RUN_ID}"

if [[ "${CUDA_MODULE_LOADING:-}" != "EAGER" ]]; then
    echo "Set CUDA_MODULE_LOADING=EAGER before starting vLLM." >&2
    exit 2
fi
if [[ ! -d "${LMCACHE_ROOT}" ]]; then
    echo "LMCache root does not exist: ${LMCACHE_ROOT}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model config was not found: ${MODEL_PATH}/config.json" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
export TARDIS_CONFIG_FILE="${TARDIS_CONFIG_FILE:-${SCRIPT_DIR}/tardis_tutti_ub_smoke_config.yaml}"
export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/dev/null}"
export LMCACHE_USE_EXPERIMENTAL=True
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_FLASH_ATTN_VERSION="${VLLM_FLASH_ATTN_VERSION:-2}"
export BASIC_MODEL_PATH="${MODEL_PATH}"
export BASIC_METRICS_OUTPUT="${BASIC_METRICS_OUTPUT:-${LOG_DIR}/metrics.json}"
if [[ -z "${TARDIS_TUTTI_STATS_PATH:-}" ]]; then
    export TARDIS_TUTTI_STATS_PATH="${LOG_DIR}/tutti_stats_rank{rank}.json"
fi
export BASIC_NUM_RUNS="${BASIC_NUM_RUNS:-2}"

export PYTHONPATH="${VLLM_ROOT}:${LMCACHE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${LMCACHE_ROOT}/csrc/GeminiFS/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python_metadata() {
    "${PYTHON}" - <<'PY'
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


root = Path(os.environ["LMCACHE_ROOT"])
vllm_root = Path(os.environ["VLLM_ROOT"])
output = {
    "python": sys.executable,
    "python_version": platform.python_version(),
    "torch": command([sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda)"]),
    "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "tardis_config": os.environ.get("TARDIS_CONFIG_FILE"),
    "tutti_endpoint": os.environ.get("TARDIS_TUTTI_ENDPOINT"),
    "tutti_device_ids": os.environ.get("TARDIS_TUTTI_DEVICE_IDS"),
    "tutti_owners": os.environ.get("TARDIS_TUTTI_OWNERS"),
    "tutti_rank_local_only": os.environ.get("TARDIS_TUTTI_RANK_LOCAL_ONLY"),
    "tutti_rank_remote_only": os.environ.get("TARDIS_TUTTI_RANK_REMOTE_ONLY"),
    "tutti_descriptor": os.environ.get("TARDIS_TUTTI_DESCRIPTOR"),
    "tutti_defer_sync": os.environ.get("TARDIS_TUTTI_DEFER_SYNC"),
    "tutti_bulk_load": os.environ.get("TARDIS_TUTTI_BULK_LOAD"),
    "tutti_stats_path": os.environ.get("TARDIS_TUTTI_STATS_PATH"),
    "model": os.environ.get("BASIC_MODEL_PATH"),
    "workload": {key: value for key, value in os.environ.items() if key.startswith("BASIC_")},
    "nvidia_smi": command(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv"]),
    "topology": command(["nvidia-smi", "topo", "-m"]),
    "lmcache_git_commit": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
    "lmcache_git_branch": command(["git", "-C", str(root), "branch", "--show-current"]),
    "geminifs_git_commit": command(["git", "-C", str(root / "csrc/GeminiFS"), "rev-parse", "HEAD"]),
    "geminifs_git_branch": command(["git", "-C", str(root / "csrc/GeminiFS"), "branch", "--show-current"]),
    "vllm_git_commit": command(["git", "-C", str(vllm_root), "rev-parse", "HEAD"]),
    "vllm_git_branch": command(["git", "-C", str(vllm_root), "branch", "--show-current"]),
}
print(json.dumps(output, indent=2))
PY
}

export LMCACHE_ROOT VLLM_ROOT
python_metadata > "${LOG_DIR}/environment.json"

set +e
"${PYTHON}" "${SCRIPT_DIR}/basic.py" > "${LOG_DIR}/basic.log" 2>&1
status=$?
set -e
echo "status=${status}" > "${LOG_DIR}/status.txt"
if [[ -f "${LOG_DIR}/metrics.json" ]]; then
    "${PYTHON}" "${LMCACHE_ROOT}/benchmarks/tutti_lmcache/summarize_vllm_run.py" \
        "${LOG_DIR}" \
        --output "${LOG_DIR}/report_zh.md" \
        > "${LOG_DIR}/summary.log" 2>&1 || true
fi
echo "output=${LOG_DIR}" >&2
exit "${status}"
