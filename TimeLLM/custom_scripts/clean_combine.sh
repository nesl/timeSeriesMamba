#!/bin/bash

model_name="TimeLLM"
train_epochs=3
learning_rate=0.02
llm_layers=6

num_params='2.8b'
batch_size=16
d_model=32
d_ff=128
num_process=1

# Function to display usage information
usage() {
  echo "Usage: $0 -l <llm_layers> -d <d_model> -e <train_epochs> -n <num_params> -c <save_checkpoints> -m <llm_model> -f <downsampling_factor> -g <gpu_id>"
  exit 1
}

# Parse command-line arguments
while getopts "l:d:e:n:c:m:f:g:" opt; do
  case $opt in
    l) llm_layers=$OPTARG ;;
    d) d_model=$OPTARG ;;
    e) train_epochs=$OPTARG ;;
    n) num_params=$OPTARG ;;
    c) save_checkpoints=$OPTARG ;;
    m) llm_model=$OPTARG ;;
    f) downsampling_factor=$OPTARG ;;
    g) gpu_id=$OPTARG ;;
    *) usage ;;
  esac
done

# Check if required arguments are provided
if [ -z "$llm_layers" ] || [ -z "$d_model" ] || [ -z "$train_epochs" ] || [ -z "$num_params" ] || [ -z "$save_checkpoints" ] || [ -z "$llm_model" ] || [ -z "$downsampling_factor" ] || [ -z "$gpu_id" ]; then
  usage
fi

# Export GPU ID and adjust master_port
export CUDA_VISIBLE_DEVICES=$((gpu_id % 4))
master_port_base=01180
master_port="${master_port_base}${gpu_id}"

# Print the values to verify
echo "Setting llm_layers to $llm_layers"
echo "Setting d_model to $d_model"
echo "Setting train_epochs to $train_epochs"
echo "Setting num_params to $num_params"
echo "Setting save_checkpoints to $save_checkpoints"
echo "Setting llm_model to $llm_model"
echo "Setting downsampling_factor to $downsampling_factor"
echo "Setting master_port to $master_port"

og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}"

llm_dim=10
if [ "$num_params" = "130m" ]; then
  llm_dim=768
elif [[ "$num_params" == "2.7b" || "$num_params" == "2.8b" ]]; then
  llm_dim=2560
elif [ "$num_params" = "7b" ]; then
  llm_dim=4096
elif [ "$num_params" = "1b" ]; then
  llm_dim=2048
fi

# Define trials for different pred_len values
for pred_len in 96 192 336 720 ; do
  for seed in {1..10}; do
    tag="ETTh1_${og_tag}_pred${pred_len}_seed${seed}"
    comment="checkpoints/${tag}"
    log_file="results/${tag}.txt"
    exec > "$log_file" 2>&1

    accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_process.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id "ETTh1_512_${pred_len}" \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 512 \
      --label_len 48 \
      --pred_len $pred_len \
      --dsampfactor $downsampling_factor \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff 32 \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --llm_layers $llm_layers \
      --train_epochs $train_epochs \
      --model_comment "$comment" \
      --llm_model $llm_model \
      --llm_dim $llm_dim \
      --num_params $num_params \
      --seed $seed

    echo "ETTh1 with pred_len $pred_len and seed $seed completed, saved to $comment"
  done
done
