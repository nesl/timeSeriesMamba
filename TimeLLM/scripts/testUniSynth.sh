#!/bin/bash

model_name="TimeLLM"

train_epochs=10
learning_rate=0.01
llm_layers=0
rand_init=0
num_params='2.8b'
batch_size=16
d_model=32
d_ff=32
num_process=1

heldout=""
source_type=""
# Default seed ranges: seed 1-3, init_seed 11-13
seed_ranges="1-3"
init_seed_ranges="11-13"

# Default values
master_port_base=01180
downsampling_factor=1
percent=100
col_percent=100
save_checkpoints=0

# Usage function
usage() {
  echo "Usage: $0 -n <num_params> -m <llm_model> -g <gpu_id> [-p <master_port>] [-r <rand_init>] -z <seed_ranges> [-i <init_seed_ranges>] -h <heldout> -s <source_type>"
  echo "Example: '-z 1-3' and '-i 11-13'"
  exit 1
}

# Parse args
while getopts "n:m:g:p:r:z:i:h:s:" opt; do
  case $opt in
    n) num_params=$OPTARG ;;  
    m) llm_model=$OPTARG ;;  
    g) gpu_id=$OPTARG ;;  
    p) master_port=$OPTARG ;;  
    r) rand_init=$OPTARG ;;  
    z) seed_ranges=$OPTARG ;;  
    i) init_seed_ranges=$OPTARG ;;  
    h) heldout=$OPTARG ;;  
    s) source_type=$OPTARG ;;  
    *) usage ;;
  esac
done

# Required arg check
if [ -z "$llm_model" ] || [ -z "$gpu_id" ] || [ -z "$source_type" ]; then
  usage
fi

# Set model name override
if [ "$llm_model" == "ARIMA" ]; then
  model_name="ARIMA"
fi

# Master port
if [ -z "$master_port" ]; then
  master_port="${master_port_base}${gpu_id}"
fi

export CUDA_VISIBLE_DEVICES=$((gpu_id % 4))

echo "Using model: $llm_model"
echo "Heldout: To Be Looped"
echo "Source type: $source_type"
echo "Num params: $num_params"
echo "Master port: $master_port"
echo "Rand init: $rand_init"
echo "Seed ranges: $seed_ranges"
echo "Init seed ranges: $init_seed_ranges"

og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
if [ "$llm_model" == "DLinear" ]; then
  model_name="DLinear"
  og_tag="DLinear_l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
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

# Parse seed ranges helper
parse_seed_ranges() {
  local ranges=$1
  local seed_list=()
  IFS=',' read -ra range_array <<< "$ranges"
  for range in "${range_array[@]}"; do
    if [[ $range =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start=${BASH_REMATCH[1]}
      end=${BASH_REMATCH[2]}
      for ((i=start; i<=end; i++)); do
        seed_list+=("$i")
      done
    else
      echo "Invalid range format: $range. Expected start-end."
      exit 1
    fi
  done
  echo "${seed_list[@]}"
}

# Generate arrays for seed and init_seed
seed_array=($(parse_seed_ranges "$seed_ranges"))
init_seed_array=($(parse_seed_ranges "$init_seed_ranges"))

# Outer loop for heldout values
for heldout in 200 300 400 500 600 700 800; do
  echo "Running heldout $heldout"
  for pred_len in $((96 / downsampling_factor)); do
    for seed in "${seed_array[@]}"; do
      for init_seed in "${init_seed_array[@]}"; do
        tag="testing_uniSynthPSD_${og_tag}_h${heldout}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
        checkpoint_tag="uniSynthPSD_${og_tag}_h200_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
        #comment="checkpoints/${tag}"
        log_file="results/uniSynthPSD_eval/${tag}.txt"
        exec > "$log_file" 2>&1

        data_path="train.csv"
        data_path_test="region_test_om0p${heldout}.csv"

        accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_evaluate.py \
          --task_name long_term_forecast \
          --root_path ./dataset/synthetic_data/psd_synth/ \
          --data_path $data_path \
          --data_path_test $data_path_test \
          --model_id ${heldout}_heldout_${seq_len}_${pred_len} \
          --model $model_name \
          --data Synthetic \
          --features M \
          --seq_len $seq_len \
          --label_len 48 \
          --factor 3 \
          --enc_in 1 \
          --dec_in 1 \
          --c_out 1 \
          --pred_len $pred_len \
          --d_model $d_model \
          --d_ff 32 \
          --llm_layers $llm_layers \
          --llm_model $llm_model \
          --llm_dim $llm_dim \
          --num_params $num_params \
          --rand_init $rand_init \
          --checkpoint_path checkpoints/${checkpoint_tag}/checkpoint \
          --seed $seed \
          --init_seed $init_seed \
          --visualize \
          --source $source_type \
          --use_wandb 1 \
          --heldout $heldout

        echo "eval of ${heldout} heldout (${source_type}) with init_seed $init_seed and seed $seed completed, saved to $comment"
        if [[ "$rand_init" -eq 0 ]]; then
            break
        fi
      done
    done
  done
done
