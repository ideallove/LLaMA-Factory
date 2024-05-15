# nohup ./my/single_node.sh > mylog_240515.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file my/accelerate/single_config.yaml \
    src/train.py my/llama3_lora_sft.yaml
