# -*- coding:utf-8 -*-
# @FileName : infer_vllm.py
# @Time : 2023/12/29 18:38
# @Author :fiv
import argparse
import asyncio
import os
import random
import time
from typing import List, Tuple

import torch
from tqdm import tqdm
from vllm import SamplingParams, AsyncEngineArgs

from my_llm_engine import MyAsyncLLMEngine
from pre_data import get_data
from util import get_log

async def run_vllm(
        requests: List[Tuple[int, str, int]],
        vllm_params: dict,
):
    print("start load")
    llm = MyAsyncLLMEngine.from_engine_args(AsyncEngineArgs(**vllm_params), start_engine_loop=True)
    print("end load")
    pbar = tqdm(total=len(requests), desc="OUTPUT")
    outputs = []
    tasks: List[asyncio.Task] = []

    async def add(request_id: str, prompt, output_len):
        para = SamplingParams(
            n=1,
            # best_of=None,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            min_p=0.0,
            use_beam_search=False,
            length_penalty=1.0,
            early_stopping=False,
            stop=None,
            stop_token_ids=None,
            include_stop_str_in_output=False,
            ignore_eos=False,
            max_tokens=output_len,
            logprobs=None,
            prompt_logprobs=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            logits_processors=None,
        )
        stream = await llm.add_request(request_id, prompt, para)
        step_output = [output async for output in stream if output.finished]
        outputs.extend(step_output)
        pbar.update(len(step_output))

    start = time.perf_counter()

    for i, p, o in requests:
        tasks.append(asyncio.create_task(add(str(i), p, o)))
    await asyncio.gather(*tasks)

    end = time.perf_counter()
    pbar.close()

    input_num_tokens = [len(o.prompt_token_ids) for o in outputs]
    output_num_tokens = [len(o.outputs[0].token_ids) + len(o.prompt_token_ids) for o in outputs]

    return end - start, input_num_tokens, output_num_tokens, llm.engine.log


def main_vllm(args: argparse.Namespace):
    torch.cuda.empty_cache()

    # import ray
    # ray.init(address="auto", _redis_password=args.redis_password)
    # print(ray.available_resources())
    # print(ray.cluster_resources())
    # print(ray.nodes())

    os.environ[
        "RAY_DEDUP_LOGS"] = "0"  # Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication

    filename = os.path.basename(__file__).split(".")[0]
    log = get_log(filename)
    log.info(f"\n{os.popen('nvidia-smi').read()}")
    random.seed(args.seed)

    requests = get_data(args.dataset, args.num_samples)

    if args.num_gpus is not None:
        num_gpu = args.num_gpus
    else:
        num_gpu = torch.cuda.device_count()

    params = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "seed": args.seed,

        "worker_use_ray": True,
        # "pipeline_parallel_size": num_gpu, NotImplementedError: Pipeline parallelism is not supported yet.
        "tensor_parallel_size": num_gpu,

        # "max_parallel_loading_workers": None,
        "block_size": 16,
        "swap_space": 4,

        "disable_log_stats": False,

        "gpu_memory_utilization": 0.7,
        "enforce_eager": False,

        "disable_log_requests": True,

        # "max_context_len_to_capture": 0,
        # "pipeline_parallel_size": torch.cuda.device_count(), NotImplementedError: Pipeline parallelism is not supported yet.
    }
    log.info(f"\n"
             f"args: {args} \n"
             f"params: {params} \n"
             f"request_length: {len(requests)} \n")

    elapsed_time, input_num_tokens, output_num_tokens, running_log = asyncio.run(run_vllm(requests, params))

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
    start_time = running_log[0][0]

    for timestamp, prompt_num_batched_tokens, generation_num_batched_tokens, num_running, num_swapped, num_pending in running_log:
        log.info(f"Time: {timestamp - start_time:8.2f} sec,\t"
                 f"Prompt_num_batched_tokens: {prompt_num_batched_tokens:4d} tokens,\t"
                 f"Generation_num_batched_tokens: {generation_num_batched_tokens:4d} tokens,\t"
                 f"Running: {num_running:4d} reqs,\t"
                 f"Swapped: {num_swapped:4d} reqs,\t"
                 f"Pending: {num_pending:4d} reqs")

    # from util.board_util import draw_tensorboard
    # prompt_num_batched_tokens = [l[1] for l in running_log]
    # generation_num_batched_tokens = [l[2] for l in running_log]
    # draw_tensorboard(filename=filename, data={
    #     "input_num_tokens": input_num_tokens,
    #     "output_num_tokens": output_num_tokens,
    #     "prompt_num_batched_tokens": prompt_num_batched_tokens,
    #     "generation_num_batched_tokens": generation_num_batched_tokens,
    # })
