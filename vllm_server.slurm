#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --partition=      
#SBATCH --qos=               
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --gres=[2 A100 80g]
#SBATCH --mem=240gb             
#SBATCH --time=5-00:00:00       
#SBATCH --job-name=vllm_server     
#SBATCH --output=logs/vllm_server.out
#SBATCH --error=logs/vllm_server.err

source ~/.bashrc

cd "please input your local path here"

source .venv/bin/activate

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

VLLM_NODE="${NODELIST[0]}"

# --- Run vLLM server ---
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE \
    python -m trl.scripts.vllm_serve \
    --model "Leopo1d/OpenVul-Qwen3-4B-SFT-ep3" \
    --tensor_parallel_size 2 \
    --max_model_len 32768 \
    --data_parallel_size 1 &

wait