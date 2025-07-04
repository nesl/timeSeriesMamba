#!/bin/bash

# Set the arguments
#MODEL_TYPE="mamba2"
MODEL_TYPE="gpt2"
#MODEL_NAME="state-spaces/mamba-130m-hf"
MODEL_NAME="gpt2"
#DATASET="openwebtext"
DATASET="pile"

# Set CUDA_VISIBLE_DEVICES
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1 #we used 1 and 2

# Run the Python script with the specified arguments
#python languagePretrain.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --dataset $DATASET

python localPretrain.py \
  --model_name Mamba2 \
  --dataset openwebtext \
  --total_steps 320000 \
  --save_every 10000
