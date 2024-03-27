#!/bin/bash

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --qos gpugpu
#SBATCH -N 2
#SBATCH --gres=gpu:8
#SBATCH -p gpu_4090
#SBATCH --exclusive
### Give all resources to a single Ray task, ray can manage the resources internally


# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
# conda activate ${CONDA_ENV}

module load anaconda/2022.10
module load cuda/12.2
source activate aa

X_LOG_DIR="log_${SLURM_JOB_ID}"
X_GPU_LOG="${X_LOG_DIR}/gpu.log"
mkdir "${X_LOG_DIR}"
function gpus_collection(){
   sleep 15
   process=`ps -ef | grep python | grep $USER | grep -v "grep" | wc -l`
   while [[ "${process}" > "0" ]]; do
      sleep 1
      nvidia-smi >> "${X_GPU_LOG}" 2>&1
      echo "process num:${process}" >> "${X_GPU_LOG}" 2>&1
      process=`ps -ef | grep python | grep $USER | grep -v "grep" | wc -l`
   done
}
gpus_collection &


### 启用IB通信
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3

# ===== DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING =====
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=5241590000000000
export redis_password

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
node_cnt = $(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" \
  ray start --head --node-ip-address="$ip" --port=$port --redis-password="$redis_password" --block &
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" ray start --address "$ip_head" --redis-password="$redis_password" --block &
  sleep 5
done

# ===== Call your code below =====
python script/main.py --dataset data/scrambled_sampled_dataset.json --model Llama-2-70b-hf --run vllm --redis_password "$redis_password" --num_gpus 16

