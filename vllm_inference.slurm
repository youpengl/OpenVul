#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --partition=         
#SBATCH --qos=           
#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=1             
#SBATCH --gres=[2 A100 80g]       
#SBATCH --mem=240gb                   
#SBATCH --time=5-00:00:00             
#SBATCH --job-name=vllm_inference        
#SBATCH --output=logs/vllm_inference.out     
#SBATCH --error=logs/vllm_inference.err      

source ~/.bashrc

cd "please input your local path here"

source .venv/bin/activate

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

node1="${NODELIST[0]}"
model="anony3849/Qwen3-4B-SFT-ep3"

tp=2
dp=1
python -m trl.scripts.vllm_serve \
    --model ${model} \
    --tensor_parallel_size ${tp} \
    --max_model_len 32768 \
    --port 8002 \
    --data_parallel_size ${dp} &

sleep 30

python -u vllms.py --model ${model} --name "Qwen3-4B-SFT-ep3" --mode test --port "8002"