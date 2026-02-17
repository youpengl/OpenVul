#!/bin/bash
#SBATCH --partition=
#SBATCH --qos=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=[4 A100 80g]
#SBATCH --cpus-per-task=64
#SBATCH --mem=480gb
#SBATCH --time=5-00:00:00
#SBATCH --job-name=judge_server
#SBATCH --output=logs/judge_server.out
#SBATCH --error=logs/judge_server.err

PORT=8002
SINGULARITY_IMAGE="vllm-v0.11.0.sif"
WORK_DIR=""
MODEL_NAME="openai/gpt-oss-120b"

module load singularity

singularity exec --nv --pwd $WORK_DIR $SINGULARITY_IMAGE bash -c "
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 4 \
    --max_num_batched_tokens 8192
"
