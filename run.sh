#!/bin/bash

rm -rf ./logs-Tacotron/

CUDA_VISIBLE_DEVICES=''  python train.py \
       --model=Tacotron \
       --tacotron_train_steps=40 \
       --lognode_time=False \
       --allow_shared=True \
       --build_cost_model=40 \
       --build_cost_model_after=10 \
       --hparams tacotron_num_gpus=1,tacotron_batch_size=32 \
       --target="grpc://localhost:29990" 
