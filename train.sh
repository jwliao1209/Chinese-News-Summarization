#!/bin/bash

python train.py \
        --epoch 15 \
        --batch_size 2 \
        --accum_grad_step 32 \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --warm_up_step 0 \
        --num_beams 5 \
        --top_p 0 \
        --top_k 0 \
        --temperature 1 \
        --device_id 0
