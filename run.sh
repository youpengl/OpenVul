# Train

## Cold Start Stage (SFT, 4x A100 80g)
sbatch sft.slurm


## Preference Optimizaiton (DPO, ORPO, 4x A100 80g)
sbatch dpo.slurm
sbatch orpo.slurm


## RL Stage (GRPO, 8x A100 80g for training, 4x A100 80g for judge server, 2x A100 80g for vllm server)

### Step 1: Run judge server and vllm server
sbatch judge_server.slurm
sbatch vllm_server.slurm

### Step 2: GRPO Training
#### Please switch between ['detection', 'prediction', 'reasoning', 'specification'] to change the reward system in the file `grpo.sh`.
#### For specification-based reward, please modify: --reward_weights 1.0 1.0 1.0 1.0.
#### By defualt, the reasoning-based reward is recommended to use to balance model performance and training stability.
sbatch grpo.slurm

#---------------------------------------

# Test

## LLM Inference via vLLM
sbatch vllm_inference.slurm


## Output Judge
python LLM_judge_for_vulnerability_detection.py --gpu XXX --name XXX 


## Metric Calculation
python calculate_metrics.py
