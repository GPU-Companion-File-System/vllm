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

os.environ["TARDIS_CONFIG_FILE"] = "/home/wxt/open_sources/scripts/tardis_config.yaml"
os.environ["LMCACHE_CONFIG_FILE"] = "/home/wxt/open_sources/scripts/lmcache_config.yaml"
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
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
long_context = long_context[:155000]

tokenizer = AutoTokenizer.from_pretrained("/home/wxt/models/Llama-3.1-8B-Instruct")
question = "Summarize bash in 2 sentences."

prompt = f"{long_context}\n\n{question}"
prompt1 = f"{long_context[1:]}\n\n{question}"
prompt2 = f"{long_context[2:]}\n\n{question}"
prompt3 = f"{long_context[3:]}\n\n{question}"

prompt4 = f"{long_context[:75000]} {long_context[:75000]}\n\n{question}"

prompt5 = f"{long_context[:75000]} {long_context[1:75000]}\n\n{question}"
prompt6 = f"{long_context[:75000]} {long_context[2:75000]}\n\n{question}"

print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")

# Sample prompts.
prompts = [
    prompt,
    prompt,
    prompt,
    prompt3,
    # prompt4,
    # prompt5,
    # prompt6,
]
# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.8, top_p=0.95,
    # max_tokens=1024,
    # ignore_eos=True,
    # stop=None 
    )


def main():
    # Create an LLM.
    llm = LLM(model="/home/wxt/models/Llama-3.1-8B-Instruct",
            enforce_eager=True, enable_prefix_caching=False,
            kv_transfer_config=ktc, block_size=256,
            max_num_batched_tokens=131072, max_model_len=131072)

    # llm = LLM(model="/home/wxt/models/Llama-3.1-8B-Instruct",
    #         enforce_eager=True, 
    #         # tensor_parallel_size=2,
    #         max_num_batched_tokens=131072, max_model_len=131072)
    
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
    
    # outputs = llm.generate(prompts, sampling_params)
    # # Print the outputs.
    # print("\nGenerated Outputs:\n" + "-" * 60)
    # for output in outputs:
    #     generated_text = output.outputs[0].text
    #     print(f"Output:    {generated_text!r}")
    #     print("-" * 60)
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
