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
from vllm import SamplingParams, LLMEngine, EngineArgs, RequestOutput

from util import get_log


def run_vllm(
        requests: List[Tuple[str, int, int]],
        vllm_params: dict,
):
    llm = LLMEngine.from_engine_args(EngineArgs(**vllm_params))

    start_times = []
    iter_times = []
    outputs: List[RequestOutput] = []
    start = time.perf_counter()

    for i, (prompt, prompt_len, output_len) in enumerate(requests):
        para = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=output_len, skip_special_tokens=True)
        t = time.perf_counter()
        start_times.append(t)
        llm.add_request(str(i), prompt, para, arrival_time=t)

    while llm.has_unfinished_requests():
        step_outputs = llm.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                iter_times.append(time.perf_counter() - start_times[int(output.request_id)])

    end = time.perf_counter()

    result = sorted(zip(outputs, iter_times), key=lambda x: x[0].request_id)
    outputs, iter_times = zip(*result)
    input_num_tokens = [len(output.prompt_token_ids) for output in outputs]
    output_num_tokens = [len(output.outputs[0].token_ids) + len(output.prompt_token_ids) for output in outputs]

    return end - start, iter_times, input_num_tokens, output_num_tokens


def main_vllm(args: argparse.Namespace):
    filename = os.path.basename(__file__).split(".")[0]
    log = get_log(filename)
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

        "worker_use_ray": True,
        "pipeline_parallel_size": num_gpu,
        "tensor_parallel_size": num_gpu,

        "max_parallel_loading_workers": None,
        "block_size": 16,
        "swap_space": 4,

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
