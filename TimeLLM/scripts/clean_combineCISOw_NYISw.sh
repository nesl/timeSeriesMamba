#!/bin/bash

model_name="TimeLLM"
train_epochs=3
learning_rate=0.01
llm_layers=0
rand_init=0 # Default value for rand_init

num_params='2.8b'
batch_size=16
d_model=32
d_ff=128
num_process=1

# Function to display usage information
usage() {
  echo "Usage: $0 -l <llm_layers> -d <d_model> -e <train_epochs> -n <num_params> -c <col_percent> -m <llm_model> [-f <downsampling_factor>] -g <gpu_id> [-t <percent>] [-p <master_port>] [-r <rand_init>]"
  exit 1
}

# Default values
master_port_base=01180
downsampling_factor=1  # Default downsampling factor is 1
percent=100            # Default percent is 100 (no columns removed)
col_percent=100
save_checkpoints=0

# Parse command-line arguments
while getopts "l:d:e:n:c:m:f:g:t:p:r:" opt; do
  case $opt in
    l) llm_layers=$OPTARG ;;
    d) d_model=$OPTARG ;;
    e) train_epochs=$OPTARG ;;
    n) num_params=$OPTARG ;;
    c) col_percent=$OPTARG ;;
    m) llm_model=$OPTARG ;;
    f) downsampling_factor=$OPTARG ;;  # Optional downsampling factor
    g) gpu_id=$OPTARG ;;
    t) percent=$OPTARG ;;              # Optional percent argument
    p) master_port=$OPTARG ;;          # Optional master port argument
    r) rand_init=$OPTARG ;;            # Optional rand_init argument
    *) usage ;;
  esac
done

# Check if required arguments are provided
if [ -z "$llm_layers" ] || [ -z "$d_model" ] || [ -z "$train_epochs" ] || [ -z "$num_params" ] || [ -z "$save_checkpoints" ] || [ -z "$llm_model" ] || [ -z "$gpu_id" ]; then
  usage
fi

# Update model_name based on llm_model
if [ "$llm_model" == "ARIMA" ]; then
  model_name="ARIMA"
fi

# Set default master_port if not provided
if [ -z "$master_port" ]; then
  master_port="${master_port_base}${gpu_id}"
fi

# Export GPU ID
export CUDA_VISIBLE_DEVICES=$((gpu_id % 4))

# Print the values to verify
echo "Setting llm_layers to $llm_layers"
echo "Setting d_model to $d_model"
echo "Setting train_epochs to $train_epochs"
echo "Setting num_params to $num_params"
echo "Setting llm_model to $llm_model"
echo "Setting downsampling_factor to $downsampling_factor"
echo "Setting percent to $percent"
echo "Setting col_percent to $col_percent"
echo "Setting master_port to $master_port"
echo "Setting rand_init to $rand_init"

og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"

llm_dim=10
if [ "$num_params" == "130m" ]; then
  llm_dim=768
elif [[ "$num_params" == "2.7b" || "$num_params" == "2.8b" ]]; then
  llm_dim=2560
elif [ "$num_params" == "7b" ]; then
  llm_dim=4096
elif [[ "$num_params" == "1b" || "$num_params" == "1.3b" ]]; then
  llm_dim=2048
fi

seq_len=$((512 / downsampling_factor))

# Define trials for different pred_len values
for pred_len in $((96 / downsampling_factor)) ; do # $((192 / downsampling_factor)) $((336 / downsampling_factor)) $((720 / downsampling_factor)) ; do
#for pred_len in 336; do
  for seed in {1..10}; do

    for init_seed in {11..20}; do
      tag="NYISwtrain_CISOwtest_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
      comment="checkpoints/${tag}"
      log_file="results/NYISwtrain_CISOwtest/${tag}.txt"
      exec > "$log_file" 2>&1

      accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_process.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/CarbonCast/ \
        --data_path NYIS_aggregated_weather_data.csv \
        --data_path_test CISO_aggregated_weather_data.csv \
        --model_id NYISwtrain_CISOwtest_${seq_len}_${pred_len} \
        --model $model_name \
        --data CarbonCastw \
        --data_pretrain CarbonCastw \
        --pretrain 1 \
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
        --rand_init $rand_init \
        --seed $seed \
        --init_seed $init_seed

      echo "CISOw pretrained on NYISw with init_seed $init_seed and seed $seed completed, saved to $comment"
      if [[ "$rand_init" -eq 0 ]]; then
          break
      fi
    done
  done
done
