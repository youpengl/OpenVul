lr=1e-5
ls=linear
bz=32
gc=8
ep=5
steps=50
decay=0.01
wr=0.1
name="Qwen3-4B-SFT"

accelerate launch --config_file examples/accelerate_configs/sft_zero3.yaml trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --run_name ${name} \
    --output_dir "outputs/${name}" \
    --dataset_name Leopo1d/OpenVul_Rejection_Sampling_based_Vulnerability_Reasoning_Dataset_for_SFT \
    --learning_rate $lr \
    --lr_scheduler_type linear \
    --warmup_ratio $wr \
    --weight_decay $decay \
    --num_train_epochs $ep \
    --bf16 true \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --max_length 32768 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gc \
    --eval_accumulation_steps $gc \
    --eos_token '<|im_end|>' \
    --eval_strategy no \
    --save_strategy epoch \
    --logging_steps 1 \
    --log_level info \
    --dataset_train_split train \
    --report_to wandb \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --use_liger_kernel
