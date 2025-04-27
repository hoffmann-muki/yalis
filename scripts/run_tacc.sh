#!/bin/bash
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

# source /work/10651/nkoleyumd/vista/lab/yalis-venv/bin/activate
module load gcc cuda/12.8

export PATH="$CONDA_PREFIX/bin:$PATH"
conda activate yalis-thresh
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# echo "python -> $(which python)"
# echo "libstdc++ -> $(ldd $(which python) | grep libstdc++)"

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"
export YALIS_CACHE="$WORK/lab/yalis/yalis/external"

export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_CONFIG_DIR="$SCRATCH/.cache/wandb_config"

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=0
export NCCL_CROSS_NIC=1
# export NCCL_SOCKET_IFNAME=hsn
export MPICH_GPU_SUPPORT_ENABLED=0
export NCCL_DEBUG=TRACE
# export TORCHINDUCTOR_COMPILE_THREADS=1

SCRIPT="threshold-attention/run_tasks.py --use-wandb"
export PYTHONPATH="$PYTHONPATH:."
chmod +x scripts/get_rank.sh
run_cmd="NCCL_CUMEM_ENABLE=0 TORCH_NCCL_AVOID_RECORD_STREAMS=1 srun -p gh -N 1 -n 1 --cpu-bind=cores ./scripts/get_rank.sh python -u $SCRIPT"

echo $run_cmd
eval $run_cmd