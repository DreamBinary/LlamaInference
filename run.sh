#!/bin/bash

module load anaconda/2022.10
module load cuda/12.2
module load gcc/11.2

source activate vvv

deepspeed --num_gpus 2 script/main.py --dataset data/scrambled_sampled_dataset.json --model llama7hf --run mii
