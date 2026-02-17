lr=1e-6
ls=linear
bz=32
gc=8
ep=5
steps=50
size=4B
decay=0
wr=0.1
name=Qwen3-4B-DPO
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml trl/scripts/dpo.py \
    --model_name_or_path Leopo1d/OpenVul-Qwen3-4B-SFT-ep5 \
    --run_name "${name}" \
    --output_dir "outputs/${name}" \
    --dataset_name "Leopo1d/OpenVul_Vulnerability_Preference_Dataset_for_DPO" \
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
    --max_length 32768 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gc \
    --eval_accumulation_steps $gc \
    --eval_strategy "no" \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --log_level "info" \
    --dataset_train_split "train" \
    --report_to "wandb" \
    --use_liger_kernel \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --precompute_ref_log_probs true \
    --beta 0.1