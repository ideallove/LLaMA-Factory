# nohup ./train.sh > mylog_240420.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path /home/dyf/model/Llama-2-7b-chat-hf-modified \
    --finetuning_type lora \
    --template self_rog_llama2 \
    --dataset_dir data \
    --dataset rog_train_all_shuffled \
    --cutoff_len 4096 \
    --preprocessing_num_workers 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --max_samples 250000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 200 \
    --warmup_ratio 0.03 \
    --optim adamw_torch \
    --resize_vocab True \
    --report_to none \
    --bf16 True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target q_proj,v_proj \
    --plot_loss True \
    --use_fast_tokenizer False \
    --output_dir saves/LLaMA2-7B-Chat/lora/rog_240420_latest \
    | tee -a mylog_tee_240420.log

#    --rope_scaling none \
#    --flash_attn True \
#    --use_unsloth True \
