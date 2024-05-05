# nohup ./train.sh > mylog_240430.log 2>&1 &
DATE=240430
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --finetuning_type lora \
    --model_name_or_path /home/dyf/model/Phi-3-mini-128k-instruct-modified \
    --template phi_self \
    --output_dir /home/finetune/phi_rog_${DATE}_01 \
    --dataset_dir data \
    --dataset rog_train_all_shuffled \
    --cutoff_len 2048 \
    --preprocessing_num_workers 1 \
    --num_train_epochs 3.0 \
    --max_samples 120000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 25 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-05 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --optim adamw_torch \
    --report_to none \
    --bf16 True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target all \
    --plot_loss True \
    --use_fast_tokenizer False \
    --resize_vocab True \
    --split_special_tokens True \
    --additional_target embed_tokens,lm_head,norm \
    --use_cache False \
    --flash_attn fa2 \
    | tee -a mylog_tee_${DATE}.log

#    --lora_dropout 0.05 \
#    --resize_vocab True \
#    --rope_scaling none \
#    --flash_attn fa2 \
#    --use_unsloth True \
