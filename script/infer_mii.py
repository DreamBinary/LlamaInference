# -*- coding:utf-8 -*-
# @FileName : infer_mii_predata.py
# @Time : 2024/1/1 19:25
# @Author :fiv
import argparse
import os
import random
from typing import List
from typing import Tuple

import torch
from mii.config import ModelConfig
from mii.modeling.models import load_model
from mii.modeling.tokenizers import load_tokenizer

from my_mii_pipe import MyMIIPipeline
from pre_data import get_full_data
from util import get_log


def run_mii(
        requests: List[Tuple[int, str, int, int]],
        mii_params: dict,
):
    model_config = ModelConfig(**mii_params)
    inference_engine = load_model(model_config)
    tokenizer = load_tokenizer(model_config)
    pipe = MyMIIPipeline(
        inference_engine=inference_engine,
        tokenizer=tokenizer,
        model_config=model_config,
    )

    return pipe(requests)


def main_mii(args: argparse.Namespace):
    torch.cuda.empty_cache()

    os.environ['RANK'] = os.environ['SLURM_PROCID']
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['MASTER_PORT'] = "29500"
    os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
    os.environ['MASTER_ADDR'] = args.master_addr

    print("RANK", os.environ['RANK'], "WORLD_SIZE", os.environ['WORLD_SIZE'], "MASTER_PORT", os.environ['MASTER_PORT'],
          "LOCAL_RANK", os.environ['LOCAL_RANK'])

    log = get_log(os.path.basename(__file__).split(".")[0])
    log.info(f"\n{os.popen('nvidia-smi').read()}")
    random.seed(args.seed)

    requests = get_full_data(args.dataset, args.num_samples)

    params = {
        "model_name_or_path": args.model,
        "tokenizer": args.tokenizer,
        "tensor_parallel": os.environ['WORLD_SIZE']
    }

    rank = int(os.environ['RANK'])
    if rank == 0:
        log.info(f"\n"
                 f"args: {args} \n"
                 f"params: {params} \n"
                 f"request_length: {len(requests)} \n")

    elapsed_time, input_num_tokens, output_num_tokens, running_log = run_mii(requests, params)
    prompt_num_tokens = sum(input_num_tokens)
    total_num_tokens = sum(output_num_tokens)

    if rank == 0:
        log.info(f"\n"
                 f"Elapsed time: {elapsed_time:.2f} s \n"
                 f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
                 f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
                 f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
                 f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")

        log.info(f"\n"
                 f"Detail of each iteration:")
        start_time = running_log[0][0]

        for timestamp, num_generated_tokens in running_log:
            log.info(f"Time: {timestamp - start_time:8.2f} sec,\t"
                     f"Step_num_generated_tokens: {num_generated_tokens:6d},\t")
