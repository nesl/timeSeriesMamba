#!/bin/bash

model_name="TimeLLM"

train_epochs=10
learning_rate=0.01
llm_layers=0
rand_init=0
num_params='2.8b'
batch_size=16
d_model=32
d_ff=128
num_process=1

heldout=""
source_type=""

# Default values
master_port_base=01180
downsampling_factor=1
percent=100
col_percent=100
save_checkpoints=0

# Usage function
usage() {
  echo "Usage: $0 -n <num_params> -m <llm_model> -g <gpu_id> [-p <master_port>] [-r <rand_init>] -h <heldout> -s <source_type>"
  exit 1
}

# Parse args
while getopts "n:m:g:p:r:h:s:" opt; do
  case $opt in
    n) num_params=$OPTARG ;;
    m) llm_model=$OPTARG ;;
    g) gpu_id=$OPTARG ;;
    p) master_port=$OPTARG ;;
    r) rand_init=$OPTARG ;;
    h) heldout=$OPTARG ;;
    s) source_type=$OPTARG ;;
    *) usage ;;
  esac
done

# Required arg check
if [ -z "$llm_model" ] || [ -z "$gpu_id" ] || [ -z "$heldout" ] || [ -z "$source_type" ]; then
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
echo "Heldout: $heldout"
echo "Source type: $source_type"
echo "Num params: $num_params"
echo "Master port: $master_port"
echo "Rand init: $rand_init"

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

for pred_len in $((96 / downsampling_factor)) ; do
  for seed in {4..5}; do
    for init_seed in {14..15}; do
      tag="uniSynth_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
      comment="checkpoints/${tag}"
      log_file="results/uniSynth/${tag}.txt"
      exec > "$log_file" 2>&1

      data_path="train_val.csv"
      data_path_test="test.csv"

      accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_process.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/synthetic_data/ \
        --data_path $data_path \
        --data_path_test $data_path_test \
        --model_id ${heldout}_heldout_${seq_len}_${pred_len} \
        --model $model_name \
        --data CarbonCast \
        --data_pretrain CarbonCast \
        --pretrain 1 \
        --features M \
        --seq_len $seq_len \
        --label_len 48 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
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
        --boundary_file "dataset/synthetic_data/train_boundaries.json" \
        --rand_init $rand_init \
        --seed $seed \
        --init_seed $init_seed \
        --save_checkpoints 0 \
        --visualize \
        --source $source_type \
        --univar 1

      echo "${heldout} heldout (${source_type}) with init_seed $init_seed and seed $seed completed, saved to $comment"
      if [[ "$rand_init" -eq 0 ]]; then
          break
      fi
    done
  done
done
