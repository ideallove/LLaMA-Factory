CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file my/accelerate/single_config.yaml \
    src/train.py my/llama3_lora_sft.yaml
