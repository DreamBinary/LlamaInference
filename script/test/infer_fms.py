# -*- coding:utf-8 -*-
# @FileName : infer_fms.py
# @Time : 2024/1/4 13:42
# @Author :fiv

import os
import random
import time
from typing import List

from fms.models.llama import load_fms_llama
from fms.utils import tokenizers
from tqdm import tqdm
from vllm import SamplingParams
from fms.utils.generation import generate
from pre_data import PreDataset
from util import get_log
import torch

def run_fms(
        requests: {int, List[str]},
        model_path: str,
        tokenizer: str,
):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)

    torch.set_default_device(device)
    torch.set_default_dtype(torch.half)

    llm = load_fms_llama(model_path)
    tokenizer = tokenizers.get_tokenizer(tokenizer)

    llm.eval()
    torch.set_grad_enabled(False)
    input_num_tokens = []
    output_num_tokens = []
    items = requests.items()
    start = time.perf_counter()

    for output_len, prompts in tqdm(items, desc="Inference"):


        generate()


    end = time.perf_counter()
    return end - start, input_num_tokens, output_num_tokens


def main_fms(args):
    log = get_log(os.path.basename(__file__).split(".")[0])
    random.seed(args.seed)
    requests = PreDataset(args.dataset, args.num_samples)
    params = {
        "model_path": args.model,
    }
    log.info(f"\n"
             f"args: {args} \n"
             f"params: {params} \n"
             f"request_length: {len(requests)} \n")

    elapsed_time, input_num_tokens, output_num_tokens = run_fms(requests, params)
    prompt_num_tokens = sum(input_num_tokens)
    total_num_tokens = sum(output_num_tokens)
    log.info(f"\n"
             f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
             f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
             f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
             f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")
