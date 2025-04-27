#!/bin/bash
#SBATCH -J {experiment}
#SBATCH --output={output_file}
#SBATCH --error={output_file}
#SBATCH -p gh
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node={gpus}
#SBATCH --time={time}

# activate your venv
module load gcc cuda/12.8
export PATH="$CONDA_PREFIX/bin:$PATH"
conda activate yalis-thresh

# Scratch and cache paths
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"
export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_CONFIG_DIR="$SCRATCH/.cache/wandb_config"
export HF_TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"
export YALIS_CACHE="$WORK/lab/yalis/yalis/external"

export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_CONFIG_DIR="$SCRATCH/.cache/wandb_config"

# Distributed env
NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS

## nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=0
export NCCL_CROSS_NIC=1
# export NCCL_SOCKET_IFNAME=hsn
export MPICH_GPU_SUPPORT_ENABLED=0
export NCCL_DEBUG=TRACE
# export TORCHINDUCTOR_COMPILE_THREADS=1

export PYTHONPATH="$PYTHONPATH:."
chmod +x scripts/get_rank.sh

echo "=== Thresh‑Attn Job Yalis ==="
echo "Command: {cmd}"
echo "Output: {output_file}"
echo

srun NCCL_CUMEM_ENABLE=0 TORCH_NCCL_AVOID_RECORD_STREAMS=1 srun -p gh -N 1 -n 1 --cpu-bind=cores ./scripts/get_rank.sh python -u {cmd}

echo "Finished at: $(date)"