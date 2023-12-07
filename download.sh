#!/bin/bash

if [ ! -d data ]; then
    gdown https://drive.google.com/uc?id=10J_nF3pfthiHac7b1uGVNJE0MYGcqpQS -O data.zip
fi

if [ ! -d data ]; then
    unzip data.zip
fi

if [ ! -d best_checkpoint ]; then
    gdown https://drive.google.com/uc?id=1jpTbxkwClEOh5Au6TiCI3ZnfAQ9-FDgy -O best_checkpoint.zip
fi

if [ ! -d best_checkpoint ]; then
    unzip best_checkpoint.zip
fi
