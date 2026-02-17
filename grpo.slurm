#!/bin/bash
#SBATCH --cpus-per-task=64
#SBATCH --partition=
#SBATCH --qos=            
#SBATCH --nodes=2               
#SBATCH --ntasks-per-node=1     
#SBATCH --gres=[4 A100 80g]
#SBATCH --mem=480gb             
#SBATCH --time=5-00:00:00       
#SBATCH --job-name=grpo
#SBATCH --output=logs/grpo.out
#SBATCH --error=logs/grpo.err

source ~/.bashrc

cd "please input your local path here"

source .venv/bin/activate

NUM_NODES=2
NUM_GPUS=$((4*NUM_NODES))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
TRAIN_NODE1="${NODELIST[0]}"
TRAIN_NODE2="${NODELIST[1]}" 

VLLM_NODE=
JUDGE_NODE=

for ((rank=0; rank<NUM_NODES; rank++)); do

    node_num=$((rank + 1))
    
    node_var_name="TRAIN_NODE${node_num}"

    srun --nodes=1 --ntasks=1 --nodelist="${!node_var_name}" --output="logs/grpo_${node_num}.out" --error="logs/grpo_${node_num}.err" \
    bash grpo.sh \
    "$TRAIN_NODE1" \
    "$rank" \
    "$NUM_NODES" \
    "$NUM_GPUS" \
    "$VLLM_NODE" \
    "$JUDGE_NODE" &

done

wait