#!/usr/bin/env bash
set -euo pipefail

model_name="TimeLLM"

# Core training knobs
train_epochs=10
learning_rate=0.01
llm_layers=0
rand_init=0
num_params='2.8b'
batch_size=16
d_model=32
d_ff=32
num_process=1

# Defaults
master_port_base=01180
downsampling_factor=1
percent=100
col_percent=100
save_checkpoints=1

# New: data/source settings
dataset_dir="dataset/fitbit/fitbit_ds_v2/"           # pipeline output dir (has train.csv, boundaries.json, heldout/)
heldout="high"               # e.g., U001
data_name="Fitbit"       # change if your code expects something else
source="hr"              # <- per your request; kept and passed via --source

usage() {
  cat <<USAGE
Usage: $0 -d <dataset_dir> -h <heldout_alias> -m <llm_model> -g <gpu_id> [-n <num_params>] [-p <master_port>] [-r <rand_init>]
  -d  Path to pipeline directory (e.g., dataset/fitbit/fitbit_ds_v1)
  -h  Heldout alias (e.g., U001)
  -m  LLM backbone (e.g., GPT2, LLaMA, DLinear, ARIMA)
  -g  GPU id
  -n  Num params tag (default: 2.8b)
  -p  Master port (default: \$master_port_base\$gpu_id)
  -r  rand_init (0/1) default: 0
USAGE
  exit 1
}

# Parse args
while getopts "d:h:m:g:n:p:r:" opt; do
  case $opt in
    d) dataset_dir=$OPTARG ;;
    h) heldout=$OPTARG ;;
    m) llm_model=$OPTARG ;;
    g) gpu_id=$OPTARG ;;
    n) num_params=$OPTARG ;;
    p) master_port=$OPTARG ;;
    r) rand_init=$OPTARG ;;
    *) usage ;;
  esac
done

# Required checks
[[ -z "${dataset_dir}" || -z "${heldout}" || -z "${llm_model}" || -z "${gpu_id:-}" ]] && usage
[[ ! -f "${dataset_dir}/train.csv" ]] && { echo "ERROR: ${dataset_dir}/train.csv not found"; exit 2; }
[[ ! -f "${dataset_dir}/boundaries.json" ]] && { echo "ERROR: ${dataset_dir}/boundaries.json not found"; exit 2; }
[[ ! -f "${dataset_dir}/test_${heldout}.csv" ]] && { echo "ERROR: ${dataset_dir}/test_${heldout}.csv not found"; exit 2; }

# Model name overrides
[[ "$llm_model" == "ARIMA"   ]] && model_name="ARIMA"
[[ "$llm_model" == "DLinear" ]] && model_name="DLinear"

# Master port default
master_port="${master_port:-${master_port_base}${gpu_id}}"

# GPU selection
export CUDA_VISIBLE_DEVICES="${gpu_id}"

echo "Dataset dir : $dataset_dir"
echo "Using model : $llm_model ($model_name)"
echo "Heldout     : $heldout"
echo "Source      : $source"
echo "Num params  : $num_params"
echo "Master port : $master_port"
echo "Rand init   : $rand_init"

# Derived dims for LLM
llm_dim=10
case "$num_params" in
  130m) llm_dim=768 ;;
  2.7b|2.8b) llm_dim=2560 ;;
  7b) llm_dim=4096 ;;
  1b|1.3b) llm_dim=2048 ;;
esac

seq_len=$((512 / downsampling_factor))
pred_base=$((96 / downsampling_factor))

# Tagging
og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
[[ "$model_name" == "DLinear" ]] && og_tag="DLinear_${og_tag}"

# Paths
root_path="${dataset_dir}"
data_path="train.csv"
data_path_test="test_${heldout}.csv"
boundary_file="${dataset_dir}/boundaries.json"

mkdir -p "results/fitbit" "checkpoints"

# Univariate settings
features="M"
enc_in=1
dec_in=1
c_out=1

for pred_len in ${pred_base}; do
  for seed in {1..3}; do
    for init_seed in {11..13}; do
      tag="fitbit_${source}_${heldout}_heldout_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
      comment="checkpoints/${tag}"
      log_file="results/fitbit/${tag}.txt"
      exec > "$log_file" 2>&1

      accelerate launch --mixed_precision bf16 --num_processes ${num_process} --main_process_port ${master_port} seed_process.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "${root_path}/" \
        --data_path "${data_path}" \
        --data_path_test "${data_path_test}" \
        --model_id "fitbit_${source}_${heldout}_heldout" \
        --model "${model_name}" \
        --data "${data_name}" \
        --data_pretrain "${data_name}" \
        --pretrain 1 \
        --features "${features}" \
        --seq_len ${seq_len} \
        --label_len 48 \
        --factor 3 \
        --enc_in ${enc_in} \
        --dec_in ${dec_in} \
        --c_out ${c_out} \
        --pred_len ${pred_len} \
        --dsampfactor ${downsampling_factor} \
        --percent ${percent} \
        --col_percent ${col_percent} \
        --des 'Exp' \
        --itr 1 \
        --d_model ${d_model} \
        --d_ff ${d_ff} \
        --batch_size ${batch_size} \
        --learning_rate ${learning_rate} \
        --llm_layers ${llm_layers} \
        --train_epochs ${train_epochs} \
        --model_comment "${comment}" \
        --llm_model "${llm_model}" \
        --llm_dim ${llm_dim} \
        --num_params "${num_params}" \
        --boundary_file "${boundary_file}" \
        --rand_init ${rand_init} \
        --seed ${seed} \
        --init_seed ${init_seed} \
        --save_checkpoints ${save_checkpoints} \
        --source "${source}" \
        --use_wandb 1 \

      echo "Heldout ${heldout} (source=${source}) with init_seed ${init_seed} and seed ${seed} done → ${comment}"
      [[ "${rand_init}" -eq 0 ]] && break
    done
  done
done
