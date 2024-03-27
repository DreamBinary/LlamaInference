# -*- coding:utf-8 -*-
# @FileName : main.py.py
# @Time : 2023/12/30 20:12
# @Author :fiv
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset.", required=True)
    parser.add_argument("--model", type=str, default="meta/llama2-70b", required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for each request")
    parser.add_argument("--output-len", type=int, default=None, help="Output length for each request")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of first few samples used for inference test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='trust remote code from huggingface'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'bfloat16'],
        help='FP16 or BF16 should be used during the inference processing, FP8, INT8 or INT4 are not allowed'
    )
    parser.add_argument(
        '--run',
        type=str,
        default="baseline",
        choices=["baseline", "vllm", "mii", "vllm_async"]
    )

    parser.add_argument(
        '--redis_password',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--num_gpus',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--master_addr',
        type=str,
        default=None,
    )

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None
        assert args.output_len is None

    if args.run == "baseline":
        from baseline import main_baseline

        main_baseline(args)
    elif args.run == "vllm":
        from infer_vllm_engine import main_vllm

        main_vllm(args)

    elif args.run == "vllm_async":
        from infer_vllm_async import main_vllm

        main_vllm(args)
    elif args.run == "mii":
        from infer_mii import main_mii

        main_mii(args)
