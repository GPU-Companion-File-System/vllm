# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

# Keep standalone invocation consistent with the reproducible shell runner.
# FA3 can still be selected explicitly after its extension is built.
os.environ.setdefault("VLLM_FLASH_ATTN_VERSION", "2")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.config import KVTransferConfig

import torch

import json
from pathlib import Path
import time

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATH = os.environ.get(
    "BASIC_MODEL_PATH", "/data/models/Llama-3.1-8B-Instruct")
TARDIS_CONFIG_FILE = os.environ.get(
    "TARDIS_CONFIG_FILE", str(SCRIPT_DIR / "tardis_tutti_ub_smoke_config.yaml"))
LMCACHE_CONFIG_FILE = os.environ.get(
    "LMCACHE_CONFIG_FILE", "/dev/null")
MAN_BASH_PATH = os.environ.get(
    "MAN_BASH_PATH", str(SCRIPT_DIR / "man-bash.txt"))
CONTEXT_CHARS = int(os.environ.get("BASIC_CONTEXT_CHARS", "155000"))
CONTEXT_TOKENS = int(os.environ.get("BASIC_CONTEXT_TOKENS", "0"))
PARTIAL_PREFIX_TOKENS = int(
    os.environ.get("BASIC_PARTIAL_PREFIX_TOKENS", "0"))
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
KV_CONNECTOR = os.environ.get(
    "BASIC_KV_CONNECTOR", "TardisConnectorV1").strip()
if KV_CONNECTOR not in {"TardisConnectorV1", "LMCacheConnectorV1"}:
    raise ValueError(
        "BASIC_KV_CONNECTOR must be TardisConnectorV1 or "
        "LMCacheConnectorV1"
    )
ktc = KVTransferConfig(kv_connector=KV_CONNECTOR, kv_role="kv_both")

os.environ["TARDIS_CONFIG_FILE"] = TARDIS_CONFIG_FILE
os.environ["LMCACHE_CONFIG_FILE"] = LMCACHE_CONFIG_FILE
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_FLASH_ATTN_VERSION", "2")

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

token_prompt_full = None
token_prompt_partial = None
if CONTEXT_TOKENS > 0:
    seed_tokens = tokenizer.encode(long_context, add_special_tokens=False)
    if not seed_tokens:
        raise ValueError("BASIC_CONTEXT_TOKENS requires a non-empty context")
    repeats = (CONTEXT_TOKENS + len(seed_tokens) - 1) // len(seed_tokens)
    context_tokens = (seed_tokens * repeats)[:CONTEXT_TOKENS]
    question_tokens = tokenizer.encode(
        f"\n\n{question}", add_special_tokens=False)
    token_prompt_full = context_tokens + question_tokens
    partial_length = PARTIAL_PREFIX_TOKENS or CONTEXT_TOKENS // 2
    partial_length = min(max(partial_length, 1), CONTEXT_TOKENS - 1)
    new_suffix = tokenizer.encode(
        "\n\nThis is a new suffix.", add_special_tokens=False)
    token_prompt_partial = (
        context_tokens[:partial_length]
        + new_suffix
        + context_tokens[partial_length:]
        + question_tokens
    )


def prompt_length(prompt_input) -> int:
    if isinstance(prompt_input, dict):
        return len(prompt_input["prompt_token_ids"])
    if isinstance(prompt_input, (list, tuple)):
        return len(prompt_input)
    return len(tokenizer.encode(prompt_input))


print(
    f"Prompt length: {prompt_length(token_prompt_full or prompt)} tokens; "
    f"context_tokens={CONTEXT_TOKENS or 'char-limited'}"
)

def prompts_for_run(run_idx: int) -> list:
    if token_prompt_full is not None and token_prompt_partial is not None:
        token_ids = (
            token_prompt_partial if PREFIX_MODE == "partial" and run_idx > 0
            else token_prompt_full
        )
        return [
            {"prompt_token_ids": token_ids.copy()}
            for _ in range(NUM_PROMPTS)
        ]
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
            "prompt_tokens": [prompt_length(prompt) for prompt in prompts],
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
            "context_tokens": CONTEXT_TOKENS,
            "partial_prefix_tokens": PARTIAL_PREFIX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "max_tokens": MAX_TOKENS,
            "block_size": BLOCK_SIZE,
            "kv_connector": KV_CONNECTOR,
            "lmcache_config_file": LMCACHE_CONFIG_FILE,
            "cuda_profiler": ENABLE_CUDA_PROFILER,
            "runs": run_metrics,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Metrics: {output_path}")


if __name__ == "__main__":
    main()
