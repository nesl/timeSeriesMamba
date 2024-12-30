#!/bin/bash

model_name="TimeLLM"
train_epochs=3
learning_rate=0.01
llm_layers=6

num_params='2.8b'
batch_size=16
d_model=32
d_ff=128
num_process=1  # Ensure this variable is defined

# Function to display usage information
usage() {
  echo "Usage: $0 -l <llm_layers> -d <d_model> -e <train_epochs> -n <num_params> -c <save_checkpoints> -m <llm_model> -p <period_of_interest> -g <gpu_id>"
  exit 1
}

# Parse command-line arguments
while getopts "l:d:e:n:c:m:p:g:" opt; do
  case $opt in
    l) llm_layers=$OPTARG ;;
    d) d_model=$OPTARG ;;
    e) train_epochs=$OPTARG ;;
    n) num_params=$OPTARG ;;
    c) save_checkpoints=$OPTARG ;;
    m) llm_model=$OPTARG ;;
    p) period_of_interest=$OPTARG ;;
    g) gpu_id=$OPTARG ;;
    *) usage ;;
  esac
done

# Check if required arguments are provided
if [ -z "$llm_layers" ] || [ -z "$d_model" ] || [ -z "$train_epochs" ] || [ -z "$num_params" ] || [ -z "$save_checkpoints" ] || [ -z "$llm_model" ] || [ -z "$period_of_interest" ] || [ -z "$gpu_id" ]; then
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
echo "Setting period_of_interest to $period_of_interest"
echo "Setting master_port to $master_port"  # Print master_port for debugging

og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_p${period_of_interest// /_}"

llm_dim=0
if [ "$num_params" = "130m" ]; then
  llm_dim=768
fi
if [[ "$num_params" == "2.7b" || "$num_params" == "2.8b" ]]; then
  llm_dim=2560
fi
if [ "$num_params" = "7b" ]; then
  llm_dim=4096
fi

# Define trials for different pred_len values
for seed in {1..3}; do
  tag="ETTh2_${og_tag}_pred${pred_len}_seed${seed}"
  comment="checkpoints/${tag}"
  log_file="results/${tag}.txt"
  exec > "$log_file" 2>&1

  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_process.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id "ETTh2_512_96" \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 512 \
    --label_len 48 \
    --pred_len 96 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --llm_layers $llm_layers \
    --train_epochs $train_epochs \
    --model_comment "$comment" \
    --llm_model $llm_model \
    --llm_dim $llm_dim \
    --num_params $num_params \
    --period_of_interest "None" \
    --seed $seed

  echo "ETTh2 with seed $seed completed, saved to $comment"
done

