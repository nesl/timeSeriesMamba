#!/bin/bash

model_name="TimeLLM"

train_epochs=10
learning_rate=0.01
batch_size=16
llm_layers=0
rand_init=0
num_params='2.8b'
d_model=32
d_ff=32
num_process=1

heldout=""
source_type=""
seed_ranges="1-3"
init_seed_ranges="11-13"

master_port_base=01180
downsampling_factor=1
percent=100
col_percent=100
save_checkpoints=1

usage() {
  echo "Usage: $0 -n <num_params> -m <llm_model> -g <gpu_id> [-p <master_port>] [-r <rand_init>] [-z <seed_ranges>] [-i <init_seed_ranges>] -h <heldout> -s <source_type>"
  echo "Optional: -z <seed_ranges> (default: 1-3), -i <init_seed_ranges> (default: 11-13)"
  exit 1
}

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

if [ -z "$llm_model" ] || [ -z "$gpu_id" ] || [ -z "$heldout" ] || [ -z "$source_type" ]; then
  usage
fi

if [ "$llm_model" == "ARIMA" ]; then
  model_name="ARIMA"
fi

if [ -z "$master_port" ]; then
  master_port="${master_port_base}${gpu_id}"
fi

export CUDA_VISIBLE_DEVICES=$((gpu_id % 4))

echo "Using model: $llm_model"
echo "Heldout: $heldout"
echo "Source type: $source_type"
echo "Num params: $num_params"
echo "Seed ranges: $seed_ranges"
echo "Init seed ranges: $init_seed_ranges"

og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
if [ "$llm_model" == "DLinear" ]; then
  model_name="DLinear"
  og_tag="DLinear_l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_r${rand_init}"
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

seed_array=($(parse_seed_ranges "$seed_ranges"))
init_seed_array=($(parse_seed_ranges "$init_seed_ranges"))

for pred_len in $((96 / downsampling_factor)); do
  for seed in "${seed_array[@]}"; do
    for init_seed in "${init_seed_array[@]}"; do
      tag="testing_univariate_${heldout}_${source_type}_heldout_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
      comment="checkpoints/${tag}"
      log_file="results/univariate/heldout_${heldout}/${tag}.txt"
      exec > "$log_file" 2>&1
    
      checkpoint_tag="univariate_${heldout}_${source_type}_heldout_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"

      data_path="univariate/heldout_${heldout}_${source_type}.csv"
      data_path_test="univariate/${heldout}_${source_type}.csv"

      accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_evaluate.py \
        --task_name long_term_forecast \
        --root_path ./dataset/CarbonCast/ \
        --data_path $data_path \
        --data_path_test $data_path_test \
        --model_id ${heldout}_heldout_${seq_len}_${pred_len} \
        --model $model_name \
        --data CarbonCast \
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
        --boundary_file "dataset/CarbonCast/heldout_${heldout}_combined_clean_boundaries.json" \
        --rand_init $rand_init \
        --checkpoint_path checkpoints/${checkpoint_tag}/checkpoint \
        --seed $seed \
        --init_seed $init_seed \
        --visualize \
        --source $source_type

      echo "${heldout} heldout (${source_type}) with init_seed $init_seed and seed $seed completed, saved to $comment"
      if [[ "$rand_init" -eq 0 ]]; then
        break
      fi
    done
  done
done
