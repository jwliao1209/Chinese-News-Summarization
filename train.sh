#!/bin/bash

python train.py \
        --epoch 25 \
        --batch_size 16 \
        --accum_grad_step 1 \
        --lr 1e-4 \
        --weight_decay 1e-5 \
        --warm_up_step 0 \
        --num_beams 5 \
        --top_p 0 \
        --top_k 0 \
        --temperature 1 \
        --device_id 0
