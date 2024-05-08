# nohup ./my/train.sh > mylog_240508.log 2>&1 &
DATE=240508
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target all \
    --template llama2_self \
    --model_name_or_path /home/dyf/model/Llama-2-7b-chat-hf-modified \
    --output_dir /home/finetune/rog_${DATE}_1 \
    --overwrite_output_dir True \
    --dataset_dir data \
    --dataset rog_train_all_shuffled \
    --cutoff_len 2048 \
    --preprocessing_num_workers 1 \
    --num_train_epochs 3.0 \
    --max_samples 118000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 2e-05 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --optim adamw_torch \
    --report_to none \
    --bf16 \
    --additional_target embed_tokens,lm_head,norm \
    --plot_loss True \
    --use_fast_tokenizer False \
    --resize_vocab True \
    --split_special_tokens True \
    --use_cache False \
    --flash_attn fa2 \
    --upcast_layernorm True

#    --lora_dropout 0.05 \
#    --resize_vocab True \
#    --rope_scaling none \
#    --flash_attn fa2 \
#    --use_unsloth True \

#    | tee -a mylog_tee_${DATE}.log

