#!/bin/bash

# Set the arguments
#MODEL_TYPE="mamba2"
MODEL_TYPE="gpt2"
#MODEL_NAME="state-spaces/mamba-130m-hf"
MODEL_NAME="gpt2"
#DATASET="openwebtext"
DATASET="pile"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=3

# Run the Python script with the specified arguments
#python languagePretrain.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --dataset $DATASET

python mambaPretrain.py \
  --model_name GPT2Local \
  --dataset pile \
  --total_steps 320000 \
  --save_every 10000
