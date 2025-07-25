#!/bin/bash

model_name="TimeLLM"

llm_layers=0
rand_init=0 # Default value for rand_init
num_params='2.8b'
d_model=32
d_ff=128
num_process=1

heldout=""       # held‑out region (e.g. CISO)

# Function to display usage information
usage() {
  echo "Usage: $0 -l <llm_layers> -d <d_model> -n <num_params> -c <col_percent> -m <llm_model> [-f <downsampling_factor>] -g <gpu_id> [-t <percent>] [-p <master_port>] [-r <rand_init>]"
  exit 1
}

# Default values
master_port_base=01180
downsampling_factor=1  # Default downsampling factor is 1
percent=100            # Default percent is 100 (no columns removed)
col_percent=100

# Parse command-line arguments
while getopts "l:d:n:c:m:f:g:t:p:r:h:e:" opt; do
  case $opt in
    l) llm_layers=$OPTARG ;;
    d) d_model=$OPTARG ;;
    n) num_params=$OPTARG ;;
    c) col_percent=$OPTARG ;;
    m) llm_model=$OPTARG ;;
    f) downsampling_factor=$OPTARG ;;  # Optional downsampling factor
    g) gpu_id=$OPTARG ;;
    t) percent=$OPTARG ;;              # Optional percent argument
    p) master_port=$OPTARG ;;          # Optional master port argument
    r) rand_init=$OPTARG ;;            # Optional rand_init argument
    h) heldout=$OPTARG ;; 
    e) train_epochs=$OPTARG ;; 
    *) usage ;;
  esac
done

# Check if required arguments are provided
if [ -z "$llm_layers" ] || [ -z "$d_model" ] || [ -z "$num_params" ] || [ -z "$llm_model" ] || [ -z "$gpu_id" ] || [ -z "$heldout" ]; then
  usage
fi

# Update model_name based on llm_model
if [ "$llm_model" == "ARIMA" ]; then
  model_name="ARIMA"
elif [ "$llm_model" == "Ridge" ]; then
  model_name="Ridge"
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
echo "Setting num_params to $num_params"
echo "Setting llm_model to $llm_model"
echo "Setting downsampling_factor to $downsampling_factor"
echo "Setting percent to $percent"
echo "Setting col_percent to $col_percent"
echo "Setting master_port to $master_port"
echo "Setting rand_init to $rand_init"
echo "Setting train_epochs to $train_epochs"

#MAKE SURE THIS MATCHES YOUR INTENDED CHECKPOINT 
og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
if [ "$llm_model" == "DLinear" ]; then
  model_name="DLinear"
  og_tag="DLinear_l${llm_layers}_d${d_model}__e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
fi
if [ "$llm_model" == "Ridge" ]; then
  og_tag="Ridge_l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
fi


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
for pred_len in $((96 / downsampling_factor)); do
  for seed in {1..3}; do
    for init_seed in {11..13}; do
  
  #for seed in $((1)); do
    #for init_seed in $((11)); do
      tag="testing_NYIStrain_${heldout}test_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
      checkpoint_tag="NYIStrain_CISOtest_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
      log_file="results/eval_NYIStrain/${heldout}test_${tag}.txt"
      exec > "$log_file" 2>&1

      accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_evaluate.py \
        --task_name long_term_forecast \
        --model_id NYIStrain_${heldout}_heldout_${seq_len}_${pred_len} \
        --model $model_name \
        --data CarbonCast \
        --root_path ./dataset/CarbonCast/ \
        --data_path_test ${heldout}_clean.csv \
        --features M \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --factor 3 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --d_model $d_model \
        --d_ff 32 \
        --llm_layers $llm_layers \
        --llm_model $llm_model \
        --llm_dim $llm_dim \
        --num_params $num_params \
        --checkpoint_path checkpoints/${checkpoint_tag}/checkpoint \
        --seed $seed \
        --init_seed $init_seed \
        --use_wandb 1 \
        --visualize \

      echo "Evaluation for ${heldout} heldout with init_seed $init_seed and seed $seed completed, loaded from checkpoints/${tag}/checkpoint"
      if [[ "$rand_init" -eq 0 ]]; then
        break
      fi
    done
  done
done