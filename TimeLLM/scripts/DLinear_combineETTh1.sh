#!/bin/bash

model_name="DLinear"
train_epochs=3
learning_rate=0.01
batch_size=16
d_ff=128
num_process=1

# Function to display usage information
usage() {
  echo "Usage: $0 -e <train_epochs> -c <col_percent> [-f <downsampling_factor>] -g <gpu_id> [-t <percent>] [-p <master_port>]"
  exit 1
}

# Default values
master_port_base=1180
downsampling_factor=1  # Default downsampling factor is 1
percent=100            # Default percent is 100 (no columns removed)
col_percent=100
save_checkpoints=0

# Parse command-line arguments
while getopts "e:c:f:g:t:p:" opt; do
  case $opt in
    e) train_epochs=$OPTARG ;;
    c) col_percent=$OPTARG ;;
    f) downsampling_factor=$OPTARG ;;  # Optional downsampling factor
    g) gpu_id=$OPTARG ;;
    t) percent=$OPTARG ;;              # Optional percent argument
    p) master_port=$OPTARG ;;          # Optional master port argument
    *) usage ;;
  esac
done

# Check if required arguments are provided
if [ -z "$train_epochs" ] || [ -z "$gpu_id" ]; then
  usage
fi

# Set default master_port if not provided
if [ -z "$master_port" ]; then
  master_port="${master_port_base}${gpu_id}"
fi

# Export GPU ID
export CUDA_VISIBLE_DEVICES=$((gpu_id % 4))

# Print the values to verify
echo "Setting train_epochs to $train_epochs"
echo "Setting downsampling_factor to $downsampling_factor"
echo "Setting percent to $percent"
echo "Setting col_percent to $col_percent"
echo "Setting master_port to $master_port"

og_tag="e${train_epochs}_f${downsampling_factor}_t${percent}_c${col_percent}"

seq_len=$((512 / downsampling_factor))

# Define trials for different pred_len values
for pred_len in $((96 / downsampling_factor)) $((192 / downsampling_factor)) $((336 / downsampling_factor)) $((720 / downsampling_factor)) ; do
  for seed in {1..10}; do
    tag="DLinear_ETTh1_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}"
    comment="checkpoints/${tag}"
    log_file="results/ETTh1/${tag}.txt"
    exec > "$log_file" 2>&1

    accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_process.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${seq_len}_${pred_len} \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --pred_len $pred_len \
      --dsampfactor $downsampling_factor \
      --percent $percent \
      --col_percent $col_percent \
      --des 'Exp' \
      --itr 1 \
      --d_ff 32 \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --model_comment "$comment" \
      --seed $seed

    echo "DLinear ETTh1 with pred_len $pred_len and seed $seed completed, saved to $comment"
  done
done
