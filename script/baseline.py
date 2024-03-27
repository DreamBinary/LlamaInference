# -*- coding: utf-8 -*-
# @FileName : baseline.py
# @Time : 2023/12/29 23:01
# @Author :fiv

import argparse
import json
import os
import random
import time
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from util import get_log


def run_hf(
        requests: List[Tuple[str, int, int]],
        model: str,
        tokenizer: PreTrainedTokenizerBase,
        trust_remote_code: bool,
):
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    input_num_tokens = []
    output_num_tokens = []
    start = time.perf_counter()
    for i in tqdm(range(len(requests))):
        prompt, prompt_len, output_len = requests[i]
        # Generate the sequences.
        input_ids = tokenizer(prompt, return_tensors="pt",
                              padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=False,
            num_return_sequences=1,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=output_len,
        )
        # Include the decoding time.
        tokenizer.decode(llm_outputs[0], skip_special_tokens=True)
        input_num_tokens.append(len(input_ids[0]))
        output_num_tokens.append(len(llm_outputs[0]))
    end = time.perf_counter()
    return end - start, input_num_tokens, output_num_tokens


def main_baseline(args: argparse.Namespace):
    log = get_log(os.path.basename(__file__).split(".")[0])
    log.info(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
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

    elapsed_time, input_num_tokens, output_num_tokens = run_hf(requests, args.model, tokenizer, args.trust_remote_code)
    prompt_num_tokens = sum(prompt_len for prompt_len in input_num_tokens)
    total_num_tokens = sum(output_len for output_len in output_num_tokens)
    log.info(f"\n"
             f"Throughput: {len(requests) / elapsed_time:.4f} requests/s \n"
             f"Tokens/s: {total_num_tokens / elapsed_time:.4f} tokens/s \n"
             f"Prompt_num_tokens:{prompt_num_tokens:.4f} tokens \n"
             f"Total_num_tokens:{total_num_tokens:.4f} tokens \n")
