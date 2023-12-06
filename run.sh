#!/bin/bash

if [ ! -d data ]; then
    unzip data.zip
fi

if [ ! -d best_checkpoint ]; then
    unzip best_checkpoint.zip
fi

wait

python infer.py --data_path "${1}" --output_path "${2}" --batch_size 10
