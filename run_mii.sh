#!/bin/bash
#SBATCH -p gpu_4090
#SBATCH -N 2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --qos gpugpu

module load anaconda/2022.10
module load cuda/12.2
module load gcc/11.2
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


export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3

### 获取节点主机名
rm -rf hostfile
export HOSTFILE="./hostfile"

for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  echo "${host[$k]} slots=8" >> $HOSTFILE
done

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29500
echo "MASTER_ADDR=${MASTER_ADDR}"

ZERO_STAGE=3

deepspeed --num_nodes $SLURM_NNODES \
   --num_gpus 8 \
   --master_addr $MASTER_ADDR \
   --master_port $MASTER_PORT \
   --hostfile $HOSTFILE \
   --no_ssh_check \
   --launcher SLURM \
   --force_multi \
   script/main.py \
   --dataset data/scrambled_sampled_dataset.json \
   --model Llama-2-70b-hf   \
   --run mii \
   --master_addr $MASTER_ADDR
