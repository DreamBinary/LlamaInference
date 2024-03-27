#!/bin/bash


module load anaconda/2022.10
module load cuda/12.2

source activate vvv

python script/main.py --dataset data/scrambled_sampled_dataset.json --model llama7hf --run baseline
