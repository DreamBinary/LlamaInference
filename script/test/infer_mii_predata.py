# -*- coding:utf-8 -*-
# @FileName : infer_mii_predata.py
# @Time : 2024/1/1 19:25
# @Author :fiv
import argparse
import os
import random
import time
from typing import List

import torch
from mii import pipeline
from mii.batching.data_classes import Response
from tqdm import tqdm

from pre_data import PreDataset
from util import get_log


def run_mii(
        requests: {int, List[str]},
        mii_params: dict,
):
    pipe = pipeline(**mii_params)

    input_num_tokens = []
    output_num_tokens = []
    items = requests.items()
    start = time.perf_counter()

    for output_len, prompts in tqdm(items, desc="Inference"):
        llm_outputs: List[Response] = pipe(
            prompts,
            max_new_tokens=output_len,
            top_p=1.0,
            temperature=1.0,
            do_sample=False,
        )
        i = sum(output.prompt_length for output in llm_outputs)
        o = sum(output.generated_length for output in llm_outputs) + i
        input_num_tokens.append(i)
        output_num_tokens.append(o)

    end = time.perf_counter()
    return end - start, input_num_tokens, output_num_tokens


def main_mii(args: argparse.Namespace):
    log = get_log(os.path.basename(__file__).split(".")[0])
    random.seed(args.seed)
    requests = PreDataset(args.dataset, args.num_samples)

    # # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = str(local_rank)
    # # os.environ["WORLD_SIZE"] = str(cnt)
    # # os.environ["MASTER_ADDR"] = "localhost"
    # # os.environ["MASTER_PORT"] = torch_dist_port
    # # inference_engine_config: RaggedInferenceEngineConfig = RaggedInferenceEngineConfig(
    # #     tensor_parallel=DeepSpeedTPConfig(tp_size=cnt))
    # tensor_parallel = 2
    # tp_config = DeepSpeedTPConfig(tp_size=tensor_parallel)
    # inference_engine_config = RaggedInferenceEngineConfig(tensor_parallel=tp_config)

    params = {
        "model_name_or_path": args.model,
        "tokenizer": args.tokenizer,
        "tensor_parallel": torch.cuda.device_count(),
        "replica_num": torch.cuda.device_count(),
        # "max_length"
    }

    log.info(f"\n"
             f"args: {args} \n"
             f"params: {params} \n"
             f"request_length: {len(requests)} \n")

    elapsed_time, input_num_tokens, output_num_tokens = run_mii(requests, params)
    prompt_num_tokens = sum(input_num_tokens)
    total_num_tokens = sum(output_num_tokens)
    log.info(f"\n"
             f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
             f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
             f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
             f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")
