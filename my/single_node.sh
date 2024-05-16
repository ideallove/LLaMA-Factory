# nohup ./my/single_node.sh > mylog_240516.log 2>&1 &
# ./my/single_node.sh > mylog_240516.log

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file my/single_config.yaml \
    src/train.py my/train_sft_lora.yaml
