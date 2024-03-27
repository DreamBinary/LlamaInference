# -*- coding:utf-8 -*-
# @FileName : infer_vllm.py
# @Time : 2023/12/29 18:38
# @Author :fiv
import argparse
import json
import os
import random
import time
from typing import List, Tuple

from tqdm import tqdm
from util.logutil import get_log
from vllm import SamplingParams, LLM


def run_vllm(
        requests: List[Tuple[str, int, int]],
        model: str,
        tokenizer_path: str,
        trust_remote_code: bool,
):
    llm = LLM(model, tokenizer_path, dtype="float16", trust_remote_code=trust_remote_code)

    input_num_tokens = []
    output_num_tokens = []
    start = time.perf_counter()

    for i in tqdm(range(len(requests))):
        prompt, prompt_len, output_len = requests[i]
        para = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=output_len)
        llm_outputs = llm.generate(prompt, sampling_params=para, use_tqdm=False)
        # llm_outputs = llm.generate(prompt, sampling_params=para)
        # llm_outputs = llm.generate(prompt, sampling_params=para, use_tqdm=False)
        input_num_tokens.append(len(llm_outputs[0].prompt_token_ids))
        output_num_tokens.append(len(llm_outputs[0].outputs[0].token_ids) + input_num_tokens[-1])

    end = time.perf_counter()
    return end - start, input_num_tokens, output_num_tokens


def main_vllm(args: argparse.Namespace):
    log = get_log(os.path.basename(__file__).split(".")[0])
    log.info(args)
    random.seed(args.seed)
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_samples)]
    else:
        with open(args.dataset) as f:
            requests = json.load(f)

    if args.num_samples is not None:
        requests = requests[0:args.num_samples]

    elapsed_time, input_num_tokens, output_num_tokens = run_vllm(requests, args.model, args.tokenizer,
                                                                 args.trust_remote_code)
    prompt_num_tokens = sum(prompt_len for prompt_len in input_num_tokens)
    total_num_tokens = sum(output_len for output_len in output_num_tokens)
    log.info(f"\n"
             f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
             f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
             f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
             f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")
