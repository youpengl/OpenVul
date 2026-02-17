lr=1e-6
ls=linear
ep=5
size=4B
decay=0
wr=0.1
bz=32
gc=8
name=Qwen3-4B-ORPO

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2_offload.yaml trl/scripts/orpo.py \
  --model_name_or_path "Leopo1d/OpenVul-Qwen3-4B-SFT-ep5" \
  --output_dir "outputs/${name}" \
  --dataset_name Leopo1d/OpenVul_Vulnerability_Preference_Dataset_for_ORPO \
  --run_name "${name}" \
  --learning_rate $lr \
  --num_train_epochs $ep \
  --warmup_ratio $wr \
  --weight_decay $decay \
  --torch_dtype bfloat16 \
  --bf16 true \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --max_length 32768 \
  --max_prompt_length 16384 \
  --max_completion_length 16384 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps $gc \
  --eval_accumulation_steps $gc \
  --eval_strategy no \
  --logging_steps 1 \
  --save_strategy epoch \
  --log_level info \
  --dataset_train_split train \
  --report_to wandb \
  --use_liger_kernel \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --beta 0.1