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

import torch
import vllm.engine.llm_engine as llm_engine
from vllm import SamplingParams, LLMEngine, EngineArgs, RequestOutput

from util import get_log


def run_vllm(
        requests: List[Tuple[str, int, int]],
        vllm_params: dict,
):
    llm = LLMEngine.from_engine_args(EngineArgs(**vllm_params))

    iter_times = []
    outputs: List[List[RequestOutput]] = []
    start = time.perf_counter()

    for i, (prompt, prompt_len, output_len) in enumerate(requests):
        para = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=output_len, skip_special_tokens=True)
        llm.add_request(str(i), prompt, para)

    s = time.perf_counter()
    while llm.has_unfinished_requests():
        step_outputs = llm.step()
        e = time.perf_counter()
        step_outputs = list(filter(lambda x: x.finished, step_outputs))
        if step_outputs:  # valid step
            outputs.append(step_outputs)
            iter_times.append(e - s)

    end = time.perf_counter()

    input_num_tokens = [sum([len(o.prompt_token_ids) for o in o_list]) for o_list in outputs]
    output_num_tokens = [sum([len(o.outputs[0].token_ids) + len(o.prompt_token_ids) for o in o_list])
                         for o_list in outputs]

    return end - start, iter_times, input_num_tokens, output_num_tokens


def main_vllm(args: argparse.Namespace):
    filename = os.path.basename(__file__).split(".")[0]
    log = get_log(filename)
    log.info(f"\n{os.popen('nvidia-smi').read()}")
    llm_engine.logger = log
    random.seed(args.seed)

    assert args.dataset is not None, "dataset is None"

    with open(args.dataset) as f:
        requests = json.load(f)

    if args.num_samples is not None:
        requests = requests[0:args.num_samples]

    num_gpu = torch.cuda.device_count()

    params = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "seed": args.seed,

        # "worker_use_ray": True,
        "pipeline_parallel_size": num_gpu,
        "tensor_parallel_size": num_gpu,

        "max_parallel_loading_workers": None,
        "block_size": 16,
        "swap_space": 4,

        "disable_log_stats": True,

        "gpu_memory_utilization": 0.9,
        "enforce_eager": False,

        # "max_context_len_to_capture": 0,
        # "pipeline_parallel_size": torch.cuda.device_count(), NotImplementedError: Pipeline parallelism is not supported yet.

    }
    log.info(f"\n"
             f"args: {args} \n"
             f"params: {params} \n"
             f"request_length: {len(requests)} \n")

    elapsed_time, iter_times, input_num_tokens, output_num_tokens = run_vllm(requests, params)

    print(sum(iter_times))

    prompt_num_tokens = sum(input_num_tokens)
    total_num_tokens = sum(output_num_tokens)
    log.info(f"\n"
             f"Elapsed time: {elapsed_time:.2f} s \n"
             f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
             f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
             f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
             f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")

    log.info(f"\n"
             f"Detail of each iteration:")
    for i, (iter_time, input_num_token, output_num_token) in enumerate(
            zip(iter_times, input_num_tokens, output_num_tokens)):
        log.info(
            f"iter {i:4d} :\ttime: {iter_time:.2f} s\tinput_num_token: {input_num_token:4d}\toutput_num_token: {output_num_token:4d}")

    from util.board_util import draw_tensorboard
    draw_tensorboard(filename=filename, data={
        "time": iter_times,
        "input_num_tokens": input_num_tokens,
        "output_num_tokens": output_num_tokens,
    })
