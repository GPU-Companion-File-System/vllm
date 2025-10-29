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

# export TARDIS_CONFIG_FILE="/home/wxt/open_sources/scripts/tardis_config.yaml"
# export VLLM_USE_V1=1
# export VLLM_ENABLE_V1_MULTIPROCESSING=1
# export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export VLLM_FLASH_ATTN_VERSION=3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TARDIS_CONFIG_FILE"] = "/home/wxt/open_sources/scripts/tardis_config.yaml"
# os.environ["LMCACHE_CONFIG_FILE"] = "/home/wxt/open_sources/scripts/lmcache_config.yaml"
# os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_FLASH_ATTN_VERSION"] = "3"

long_context = ""
with open("/home/wxt/open_sources/vllm/examples/offline_inference/basic/man-bash.txt", "r") as f:
    long_context = f.read()

# a truncation of the long context for the --max-model-len 16384
# if you increase the --max-model-len, you can decrease the truncation i.e.
# use more of the long context
long_context = long_context[:70000]

tokenizer = AutoTokenizer.from_pretrained("/home/wxt/models/Llama-3.1-8B-Instruct")
question = "Summarize bash in 2 sentences."

prompt = f"{long_context}\n\n{question}"

# Sample prompts.
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    prompt
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(model="/home/wxt/models/Llama-3.1-8B-Instruct",
            enforce_eager=True, block_size=256, enable_prefix_caching=False,
            kv_transfer_config=ktc,
            max_num_batched_tokens=65536, max_model_len=131072)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    torch.cuda.cudart().cudaProfilerStart()
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
    
    outputs = llm.generate(prompt, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
