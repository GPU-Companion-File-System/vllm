# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.config import KVTransferConfig
ktc = KVTransferConfig(
    kv_connector="TardisConnectorV1",
    kv_role="kv_both",
)

import torch

import os
import json
from pathlib import Path
import time

MODEL_PATH = os.environ.get(
    "BASIC_MODEL_PATH", "/data/models/Llama-3.1-8B-Instruct")
TARDIS_CONFIG_FILE = os.environ.get(
    "TARDIS_CONFIG_FILE", "/home/zwh/open_sources/scripts/tardis_config.yaml")
LMCACHE_CONFIG_FILE = os.environ.get(
    "LMCACHE_CONFIG_FILE", "/home/zwh/open_sources/scripts/lmcache_config.yaml")
MAN_BASH_PATH = os.environ.get(
    "MAN_BASH_PATH",
    "/home/zwh/vllm/examples/offline_inference/basic/man-bash.txt")
CONTEXT_CHARS = int(os.environ.get("BASIC_CONTEXT_CHARS", "155000"))
NUM_PROMPTS = int(os.environ.get("BASIC_NUM_PROMPTS", "4"))
MAX_MODEL_LEN = int(os.environ.get("BASIC_MAX_MODEL_LEN", "131072"))
MAX_NUM_BATCHED_TOKENS = int(
    os.environ.get("BASIC_MAX_NUM_BATCHED_TOKENS", str(MAX_MODEL_LEN)))
MAX_TOKENS = int(os.environ.get("BASIC_MAX_TOKENS", "16"))
BLOCK_SIZE = int(os.environ.get("BASIC_BLOCK_SIZE", "256"))
GPU_MEMORY_UTILIZATION = float(
    os.environ.get("BASIC_GPU_MEMORY_UTILIZATION", "0.9"))
NUM_RUNS = int(os.environ.get("BASIC_NUM_RUNS", "2"))
TENSOR_PARALLEL_SIZE = int(
    os.environ.get("BASIC_TENSOR_PARALLEL_SIZE", "1"))
PREFIX_MODE = os.environ.get("BASIC_PREFIX_MODE", "full").strip().lower()
PARTIAL_PREFIX_CHARS = int(
    os.environ.get("BASIC_PARTIAL_PREFIX_CHARS", str(CONTEXT_CHARS // 2)))
METRICS_OUTPUT = os.environ.get("BASIC_METRICS_OUTPUT", "")
ENABLE_CUDA_PROFILER = os.environ.get(
    "BASIC_CUDA_PROFILER", "0").strip().lower() in {"1", "true", "yes", "on"}

os.environ["TARDIS_CONFIG_FILE"] = TARDIS_CONFIG_FILE
os.environ["LMCACHE_CONFIG_FILE"] = LMCACHE_CONFIG_FILE
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_FLASH_ATTN_VERSION"] = "3"

long_context = ""
with open(MAN_BASH_PATH, "r") as f:
    long_context = f.read()

# a truncation of the long context for the --max-model-len 16384
# if you increase the --max-model-len, you can decrease the truncation i.e.
# use more of the long context
long_context = long_context[:CONTEXT_CHARS]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
question = "Summarize bash in 2 sentences."

if PREFIX_MODE not in {"full", "partial"}:
    raise ValueError("BASIC_PREFIX_MODE must be 'full' or 'partial'")
if not 0 < PARTIAL_PREFIX_CHARS < len(long_context):
    PARTIAL_PREFIX_CHARS = len(long_context) // 2

prompt = f"{long_context}\n\n{question}"
partial_prefix = long_context[:PARTIAL_PREFIX_CHARS]
partial_suffix = long_context[PARTIAL_PREFIX_CHARS:]
prompt_partial = (
    f"{partial_prefix}\n\nThis is a new suffix. {partial_suffix}\n\n"
    f"{question}"
)

print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")

def prompts_for_run(run_idx: int) -> list[str]:
    # Populate the cache with the full prompt, then change only the suffix.
    base = prompt_partial if PREFIX_MODE == "partial" and run_idx > 0 else prompt
    return [base] * NUM_PROMPTS


def output_metrics(output) -> dict:
    metrics = getattr(output, "metrics", None)
    values = {}
    for name in (
        "arrival_time", "first_token_time", "finished_time",
        "last_token_time",
    ):
        value = getattr(metrics, name, None) if metrics is not None else None
        values[name] = value
    arrival = values["arrival_time"]
    first = values["first_token_time"]
    finished = values["finished_time"]
    values["ttft_seconds"] = (
        first - arrival if arrival is not None and first is not None else None
    )
    values["e2e_seconds"] = (
        finished - arrival
        if arrival is not None and finished is not None else None
    )
    values["num_cached_tokens"] = getattr(output, "num_cached_tokens", None)
    values["request_id"] = getattr(output, "request_id", None)
    return values


if (TENSOR_PARALLEL_SIZE <= 0 or NUM_PROMPTS <= 0 or NUM_RUNS <= 0 or
        BLOCK_SIZE <= 0):
    raise ValueError("TP size, block size, prompt count, and run count must be positive")

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS,
    # max_tokens=1024,
    # ignore_eos=True,
    # stop=None 
    )


def main():
    # Create an LLM.
    llm = LLM(model=MODEL_PATH,
            enforce_eager=True, enable_prefix_caching=False,
            kv_transfer_config=ktc, block_size=BLOCK_SIZE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION)

    # llm = LLM(model="/home/wxt/models/Llama-3.1-8B-Instruct",
    #         enforce_eager=True, 
    #         # tensor_parallel_size=2,
    #         max_num_batched_tokens=131072, max_model_len=131072)
    
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    if ENABLE_CUDA_PROFILER:
        torch.cuda.cudart().cudaProfilerStart()
    run_metrics = []
    for run_idx in range(NUM_RUNS):
        prompts = prompts_for_run(run_idx)
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        wall_seconds = time.perf_counter() - started
        run_metrics.append({
            "run_idx": run_idx,
            "prefix_mode": PREFIX_MODE,
            "prompt_tokens": [len(tokenizer.encode(prompt)) for prompt in prompts],
            "wall_seconds": wall_seconds,
            "requests": [output_metrics(output) for output in outputs],
        })
        # Print the outputs.
        print(f"\nGenerated Outputs Run {run_idx + 1}:\n" + "-" * 60)
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Output:    {generated_text!r}")
            print("-" * 60)
    
    # outputs = llm.generate(prompts, sampling_params)
    # # Print the outputs.
    # print("\nGenerated Outputs:\n" + "-" * 60)
    # for output in outputs:
    #     generated_text = output.outputs[0].text
    #     print(f"Output:    {generated_text!r}")
    #     print("-" * 60)
    if ENABLE_CUDA_PROFILER:
        torch.cuda.cudart().cudaProfilerStop()
    if METRICS_OUTPUT:
        output_path = Path(METRICS_OUTPUT)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "model": MODEL_PATH,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "num_prompts": NUM_PROMPTS,
            "num_runs": NUM_RUNS,
            "context_chars": CONTEXT_CHARS,
            "max_model_len": MAX_MODEL_LEN,
            "max_tokens": MAX_TOKENS,
            "block_size": BLOCK_SIZE,
            "cuda_profiler": ENABLE_CUDA_PROFILER,
            "runs": run_metrics,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Metrics: {output_path}")


if __name__ == "__main__":
    main()
