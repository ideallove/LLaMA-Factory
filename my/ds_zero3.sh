#!/bin/bash

NPROC_PER_NODE=2
NNODES=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/train.py my/sft_lora_train.yaml
