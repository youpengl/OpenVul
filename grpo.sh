lr=1e-6
ls=linear
ep=1
save_steps=25
decay=0
wr=0.1
pbz=1
ng=8
gc=32
beta=0
norm=0.5
name="Qwen3-4B-GRPO"

TRAIN_NODE1=$1
rank=$2
NUM_NODES=$3
NUM_GPUS=$4
VLLM_NODE=$5
JUDGE_NODE=$6

accelerate launch \
    --config_file examples/accelerate_configs/grpo_zero3.yaml \
    --main_process_ip $TRAIN_NODE1 \
    --main_process_port 32768 \
    --machine_rank $rank \
    --num_machines $NUM_NODES \
    --num_processes $NUM_GPUS\
    trl/scripts/grpo.py \
    --vllm_server_host $VLLM_NODE \
    --judge_server_host $JUDGE_NODE \
    --model_name_or_path "Leopo1d/OpenVul-Qwen3-4B-SFT-ep3" \
    --run_name "${name}" \
    --output_dir "outputs/${name}" \
    --dataset_name "Leopo1d/OpenVul_Vulnerability_Query_Dataset_for_RL" \
    --learning_rate $lr \
    --lr_scheduler_type $ls \
    --warmup_ratio $wr \
    --weight_decay $decay \
    --num_train_epochs $ep \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --attn_implementation "flash_attention_2" \
    --max_prompt_length 16384 \
    --max_completion_length 16384 \
    --num_generations $ng \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --min_p 0 \
    --repetition_penalty 1.0 \
    --per_device_train_batch_size $pbz \
    --per_device_eval_batch_size $pbz \
    --gradient_accumulation_steps $gc \
    --eval_accumulation_steps $gc \
    --eval_strategy "no" \
    --save_steps $save_steps \
    --logging_steps 1 \
    --save_strategy "steps" \
    --log_level "info" \
    --dataset_train_split "train" \
    --report_to "wandb" \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --use_vllm \
    --log_completions \
    --level reasoning  \
    --reward_weights 1.0 1.0 \
    --name "${name}" \
    --scale_rewards "none" \
    --beta $beta \
    --max_grad_norm $norm \
    --save_total_limit 3