# -*- coding:utf-8 -*-
# @FileName : infer_lightllm.py
# @Time : 2024/1/9 18:33
# @Author :fiv

import argparse
import asyncio
import os
import random
import time
from typing import Tuple, List

import torch
import uvloop
from lightllm.server.detokenization.manager import start_detokenization_process
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.router.manager import start_router_process
from lightllm.server.sampling_params import SamplingParams
from lightllm.utils.net_utils import alloc_can_use_network_port
from lightllm.utils.start_utils import start_submodule_processes

from async_data import AsyncData
from my_manager import MyHttpServerManager
from pre_data import get_full_data
from script import get_log

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# isFirst = True


# async def get_request(
#         input_requests: List[Tuple[str, int, int]],
#         request_rate: float,
# ) -> AsyncGenerator[Tuple[str, int, int], None]:
#     input_requests = iter(input_requests)
#     for request in input_requests:
#         yield request
#
#         # if request_rate == float("inf"):
#         #     # If the request rate is infinity, then we don't need to wait.
#         #     continue
#         # # Sample the request interval from the exponential distribution.
#         # interval = np.random.exponential(1.0 / request_rate)
#         # # The next request will be sent after the interval.
#         # await asyncio.sleep(interval)


async def run_lightllm(requests: List[Tuple[str, int]]):
    # global isFirst
    # if isFirst:
    #     loop = asyncio.get_event_loop()
    #     loop.create_task(httpserver_manager.handle_loop())
    #     isFirst = False
    loop = asyncio.get_event_loop()
    loop.create_task(httpserver_manager.handle_loop())

    requests = AsyncData(requests)
    output_num_tokens = []
    tasks: List[asyncio.Task] = []

    async def generate(request: Tuple[str, int], request_id):
        prompt, output_len = request
        para = SamplingParams(
            do_sample=False,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,  # -1 is for all
            ignore_eos=False,
            max_new_tokens=output_len,
            stop_sequences=[]
        )
        generator = httpserver_manager.generate(prompt, para, request_id=request_id,
                                                multimodal_params=MultimodalParams())
        output_num_tokens.append(len([_ async for _ in generator]))

    start = time.perf_counter()
    async for i, r in requests:
        tasks.append(asyncio.create_task(generate(r, i)))
    await asyncio.gather(*tasks)

    # async for i, (prompt, output_len) in requests:
    #     para = SamplingParams(
    #         do_sample=False,
    #         presence_penalty=0.0,
    #         frequency_penalty=0.0,
    #         repetition_penalty=1.0,
    #         temperature=1.0,
    #         top_p=1.0,
    #         top_k=-1,  # -1 is for all
    #         ignore_eos=False,
    #         max_new_tokens=output_len,
    #         stop_sequences=[]
    #     )
    #     generator = httpserver_manager.generate(prompt, para, request_id=i, multimodal_params=MultimodalParams())
    #     generators.append(generator)
    # output_num_tokens.append(len([_ async for _ in generator]) for generator in generators) error

    end = time.perf_counter()
    # input_num_tokens
    return end - start, output_num_tokens

    # return ret
    # {'generated_text': ['\n\n### 1'], 'count_output_tokens': 6, 'finish_reason': 'length', 'tokens': [{'id': 13, 'logprob': -0.5493996143341064, 'prompt_tokens': 422, 'text': '\n'}, {'id': 13, 'logprob': -0.4337013363838196, 'prompt_tokens': 422, 'text': '\n'}, {'id': 2277, 'logprob': -0.7054842710494995, 'prompt_tokens': 422, 'text': '##'}, {'id': 29937, 'logprob': -1.5623289346694946, 'prompt_tokens': 422, 'text': '#'}, {'id': 29871, 'logprob': -2.4111223220825195, 'prompt_tokens': 422, 'text': ' '}, {'id': 29896, 'logprob': -1.5127394199371338, 'prompt_tokens': 422, 'text': '1'}]}


def main_lightllm(args: argparse.Namespace):
    filename = os.path.basename(__file__).split(".")[0]
    log = get_log(filename)
    log.info(f"\n{os.popen('nvidia-smi').read()}")
    random.seed(args.seed)

    model_dir = args.model
    num_gpu = torch.cuda.device_count()

    torch.multiprocessing.set_start_method('spawn')
    # this code will not be ok for settings to fork to subprocess

    args.host = "127.0.0.1"
    args.port = 8000
    args.model_dir = model_dir
    args.tokenizer_mode = "auto"
    args.load_way = "HF"
    args.max_total_token_num = 6000  # ???
    args.batch_max_tokens = 10000
    args.eos_id = 2
    args.running_max_req_size = 1000
    args.tp = num_gpu
    args.max_req_input_len = 1024
    args.max_req_total_len = 1024 + 1776  # in scrambled_sampled_dataset.json
    args.nccl_port = 28765
    args.mode = ["triton_gqa_attention", "triton_flashdecoding", "triton_gqa_flashdecoding"]
    #     ModuleNotFoundError: No module named 'lightllm_ppl_fp16_kernel'
    """
    Model mode: [triton_int8kv | ppl_int8kv | ppl_fp16 | triton_flashdecoding | triton_gqa_attention | triton_gqa_flashdecoding] 
    [triton_int8weight | triton_int4weight | lmdeploy_int4weight | ppl_int4weight], 
    triton_flashdecoding mode is for long context, current support llama llama2 qwen;
    triton_gqa_attention and triton_gqa_flashdecoding is fast kernel for model which use GQA;
    triton_int8kv mode use int8 to store kv cache, can increase token capacity, use triton kernel;
    ppl_int8kv mode use int8 to store kv cache, and use ppl fast kernel;
    ppl_fp16 mode use ppl fast fp16 decode attention kernel;
    triton_int8weight and triton_int4weight and lmdeploy_int4weight or ppl_int4weight mode use int8 and int4 to store weights;
    you need to read source code to make sure the supported detail mode for all models
    """
    args.trust_remote_code = False
    args.disable_log_stats = False
    args.log_stats_interval = 10
    args.router_token_ratio = 0.5
    # req_queue.py
    # 判断当前服务是否处于token使用率过高的状态，过高的情况下，调度要偏向保守
    # cur_token_ratio = cur_all_used_tokens / self.max_total_tokens
    # is_busy = cur_token_ratio >= self.router_token_ratio
    args.router_max_new_token_len = 17760
    args.no_skipping_special_tokens = False
    args.no_spaces_between_special_tokens = False
    args.splitfuse_mode = True
    args.splitfuse_block_size = 256
    args.prompt_cache_strs = []
    args.enable_multimodal = False
    args.cache_capacity = 200
    args.cache_reserved_ratio = 0.5
    args.return_all_prompt_logprobs = False
    args.long_truncation_mode = None

    # 非splitfuse 模式，不支持 prompt cache 特性
    if not args.splitfuse_mode:
        assert len(args.prompt_cache_strs) == 0

    assert args.max_req_input_len < args.max_req_total_len
    assert args.max_req_total_len <= args.max_total_token_num

    if not args.splitfuse_mode:
        # 普通模式下
        if args.batch_max_tokens is None:
            batch_max_tokens = int(1 / 6 * args.max_total_token_num)
            batch_max_tokens = max(batch_max_tokens, args.max_req_total_len)
            args.batch_max_tokens = batch_max_tokens
        else:
            assert (
                    args.batch_max_tokens >= args.max_req_total_len
            ), "batch_max_tokens must >= max_req_total_len"
    else:
        # splitfuse 模式下
        # assert args.batch_max_tokens is not None, "need to set by yourself"
        if args.batch_max_tokens is None:
            batch_max_tokens = int(1 / 6 * args.max_total_token_num)
            batch_max_tokens = max(batch_max_tokens, args.splitfuse_block_size)
            args.batch_max_tokens = batch_max_tokens

    can_use_ports = alloc_can_use_network_port(
        num=5 + args.tp, used_nccl_port=args.nccl_port
    )
    router_port, detokenization_port, httpserver_port, visual_port, cache_port = can_use_ports[0:5]
    model_rpc_ports = can_use_ports[5:]

    global httpserver_manager
    httpserver_manager = MyHttpServerManager(
        args,
        router_port=router_port,
        cache_port=cache_port,
        visual_port=visual_port,
        httpserver_port=httpserver_port,
        enable_multimodal=False,
    )

    start_submodule_processes(start_funcs=[start_router_process, start_detokenization_process],
                              start_args=[(args, router_port, detokenization_port, model_rpc_ports),
                                          (args, detokenization_port, httpserver_port)])

    full_requests = get_full_data(args.dataset, args.num_samples)
    requests = [(r[0], r[2]) for r in full_requests]

    log.info(f"\n"
             f"args: {args} \n"
             f"request_length: {len(requests)} \n")

    elapsed_time, output_num_tokens = asyncio.run(run_lightllm(requests))

    input_num_tokens = [r[1] for r in full_requests]
    prompt_num_tokens = sum(input_num_tokens)
    total_num_tokens = sum(output_num_tokens) + prompt_num_tokens

    log.info(f"\n"
             f"Elapsed time: {elapsed_time:.2f} s \n"
             f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
             f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
             f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
             f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")

    # elap asyncio.run(run_lightllm(requests))


if __name__ == '__main__':
    args = argparse.Namespace()
    args.model = "/root/autodl-tmp/llama7hf"
    args.dataset = "../data/scrambled_sampled_dataset.json"
    args.num_samples = 10
    main_lightllm(args)
# (19.186401706188917, [6, 24, 498, 251, 16])
# (19.124050848186016, [6, 24, 498, 251, 16])
