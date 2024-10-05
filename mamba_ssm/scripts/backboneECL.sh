#!/bin/bash
model_name=BackboneModel
train_epochs=3
learning_rate=0.01
llm_layers=6

# Default values for variables
master_port_base=1051  # Base for master port calculation
batch_size=16
d_ff=128
num_params='2.7b'
llm_dim=768
gpu_id=0  # Default GPU
d_model=256

# Function to display usage information
usage() {
  echo "Usage: $0 -l <llm_layers> -e <train_epochs> -c <save_checkpoints> -m <llm_model> -g <gpu_id>"
  exit 1
}

# Parse command-line arguments
while getopts "l:e:c:m:g:" opt; do
  case $opt in
    l) llm_layers=$OPTARG ;;
    e) train_epochs=$OPTARG ;;
    c) save_checkpoints=$OPTARG ;;
    m) llm_model=$OPTARG ;;
    g) gpu_id=$OPTARG ;;  # Capture the GPU ID
    *) usage ;;
  esac
done

# Check if required arguments are provided
if [ -z "$llm_layers" ] || [ -z "$train_epochs" ] || [ -z "$save_checkpoints" ] || [ -z "$llm_model" ] || [ -z "$gpu_id" ]; then
  usage
fi

# Set CUDA_VISIBLE_DEVICES based on the gpu_id flag
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Running on GPU $CUDA_VISIBLE_DEVICES"

# Dynamically set the master port based on the GPU ID
master_port=$((master_port_base + (gpu_id % 10)))
echo "Using master_port $master_port"


# Print the values to verify
echo "Setting llm_layers to $llm_layers"
echo "Setting d_model to $d_model"
echo "Setting train_epochs to $train_epochs"
echo "Setting save_checkpoints to $save_checkpoints"
echo "Setting llm_model to $llm_model"

og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}"


# Redirect output to a file named after the comment variable
tag="backbone_ECL_${og_tag}"
for seed in {1..5}; do
  comment="checkpoints/${tag}_seed${seed}"
  log_file="results/${tag}_seed${seed}.txt"
  exec > "$log_file" 2>&1

  accelerate launch --mixed_precision bf16 --num_processes 1 --gpu_ids $gpu_id --main_process_port $master_port train.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_512_96 \
    --model $model_name \
    --data ECL \
    --features M \
    --seq_len 512 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --n_layer $llm_layers \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --save_checkpoints $save_checkpoints \
    --llm_model $llm_model \
    --llm_dim $d_model \
    --num_params $num_params \
    --use_wandb 1 \
    --seed $seed


  echo "ECL completed, saved to $comment"
done 
