#!/bin/bash

if [ ! -d data ]; then
    gdown https://drive.google.com/uc?id=10J_nF3pfthiHac7b1uGVNJE0MYGcqpQS -O data.zip
fi

if [ ! -d checkpoint ]; then
    gdown https://drive.google.com/uc?id=1oofRhAizRfcs8Cs81gp-Silia427HqmZ -O checkpoint.zip
fi
