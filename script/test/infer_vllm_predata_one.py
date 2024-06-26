# -*- coding:utf-8 -*-
# @FileName : infer_vllm.py
# @Time : 2023/12/29 18:38
# @Author :fiv
import argparse
import os
import random
import time
from typing import List

import torch
from vllm import SamplingParams, LLMEngine, EngineArgs

from pre_data import PreDataset
from util import get_log


def run_vllm(
        requests: {int, List[str]},
        vllm_params: dict,
):
    llm = LLMEngine.from_engine_args(EngineArgs(**vllm_params))

    input_num_tokens = []
    output_num_tokens = []
    items = requests.items()
    start = time.perf_counter()

    for output_len, prompts in items:
        para = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=output_len, skip_special_tokens=True)
        for prompt in prompts:
            llm.add_request("", prompt, para)

    while llm.has_unfinished_requests():
        step_outputs = llm.step()
        for output in step_outputs:
            if output.finished:
                i = len(output.prompt_token_ids)
                o = len(output.outputs[0].token_ids)
                input_num_tokens.append(i)
                output_num_tokens.append(o)

    end = time.perf_counter()
    return end - start, input_num_tokens, output_num_tokens + input_num_tokens


def main_vllm(args: argparse.Namespace):
    log = get_log(os.path.basename(__file__).split(".")[0])
    log.info(f"\n{os.popen('nvidia-smi').read()}")
    random.seed(args.seed)
    requests = PreDataset(args.dataset, args.num_samples)
    params = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": torch.cuda.device_count(),
        "seed": args.seed,
        "swap_space": 4,
        "block_size": 16,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": False,
        # "max_context_len_to_capture": 0,
        # "pipeline_parallel_size": torch.cuda.device_count(), NotImplementedError: Pipeline parallelism is not supported yet.
    }
    log.info(f"\n"
             f"args: {args} \n"
             f"params: {params} \n"
             f"request_length: {len(requests)} \n")

    elapsed_time, input_num_tokens, output_num_tokens = run_vllm(requests, params)
    prompt_num_tokens = sum(input_num_tokens)
    total_num_tokens = sum(output_num_tokens)
    log.info(f"\n"
             f"Elapsed time: {elapsed_time:.2f} s \n"
             f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
             f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
             f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
             f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")
