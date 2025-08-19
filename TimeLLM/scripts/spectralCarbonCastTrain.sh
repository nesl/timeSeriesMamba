#!/usr/bin/env bash
set -euo pipefail

# ---- Defaults you can tweak ----
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

# spectral data roots
SPECTRAL_DIR="dataset/CarbonCast/spectral"
TRAIN_DATA="$SPECTRAL_DIR/train_val_no_region.csv"
BOUNDARY_FILE="$SPECTRAL_DIR/train_val_boundaries.json"

# infra defaults
master_port_base=01180
downsampling_factor=1
percent=100
col_percent=100
save_checkpoints=1

usage() {
  cat <<USAGE
Usage: $0 -m <llm_model> -g <gpu_id> -t <test_csv> [-n <num_params>] [-p <master_port>] [-r <rand_init>]
  -t <test_csv> should be a path like:
     dataset/CarbonCast/spectral/heldout/CISO__solar__p05.csv
USAGE
  exit 1
}

llm_model=""; gpu_id=""; master_port=""; test_csv="dataset/CarbonCast/spectral/heldout/CISO_solar.csv"
while getopts "m:g:t:n:p:r:" opt; do
  case $opt in
    m) llm_model=$OPTARG ;;
    g) gpu_id=$OPTARG ;;
    t) test_csv=$OPTARG ;;
    n) num_params=$OPTARG ;;
    p) master_port=$OPTARG ;;
    r) rand_init=$OPTARG ;;
    *) usage ;;
  esac
done

# Required args
if [[ -z "$llm_model" || -z "$gpu_id" || -z "$test_csv" ]]; then
  usage
fi

# Model name override for baselines
if [[ "$llm_model" == "ARIMA" || "$llm_model" == "DLinear" ]]; then
  model_name="$llm_model"
fi

# Master port
if [[ -z "$master_port" ]]; then
  master_port="${master_port_base}${gpu_id}"
fi

export CUDA_VISIBLE_DEVICES=$((gpu_id % 4))

heldout_region="CISO"
source_type="solar"
percentile="05"

# LLM hidden size from param class
llm_dim=10
case "$num_params" in
  "130m") llm_dim=768 ;;
  "2.7b"|"2.8b") llm_dim=2560 ;;
  "7b") llm_dim=4096 ;;
  "1b"|"1.3b") llm_dim=2048 ;;
esac

seq_len=$((512 / downsampling_factor))
pred_len=$((96 / downsampling_factor))

# Tags & output dirs
og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
if [[ "$llm_model" == "DLinear" ]]; then
  og_tag="DLinear_l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
fi

tag="spectralUni_${heldout_region}_${source_type}_p${percentile}_${og_tag}_seq${seq_len}_pred${pred_len}"
comment_dir="checkpoints/${tag}"
log_dir="results/spectralUni/heldout_${heldout_region}"
mkdir -p "$comment_dir" "$log_dir"

echo "=== Config ==="
echo "Model:        $llm_model ($model_name)"
echo "Params:       $num_params (llm_dim=$llm_dim)"
echo "GPU:          $gpu_id (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Train data:   $TRAIN_DATA"
echo "Boundary:     $BOUNDARY_FILE"
echo "Test file:    $test_csv  (region=$heldout_region, source=$source_type, p=$percentile)"
echo "Seq/Pred:     $seq_len / $pred_len"
echo "Checkpoint:   $comment_dir"
echo "=============="

# One-pass training+single-file eval with fixed seeds (matches your prior convention)
for seed in 1 2 3; do
  for init_seed in 11 12 13; do
    this_tag="${tag}_seed${seed}_initseed${init_seed}"
    this_comment="${comment_dir}/s${seed}_i${init_seed}"
    this_log="${log_dir}/${this_tag}.txt"
    mkdir -p "$(dirname "$this_log")"
    exec > "$this_log" 2>&1

    # IMPORTANT: root_path="." so we can pass full relative paths for spectral files
    accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_process.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "." \
      --data_path "$TRAIN_DATA" \
      --data_path_test "$test_csv" \
      --model_id "${heldout_region}_p${percentile}_${seq_len}_${pred_len}" \
      --model "$model_name" \
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
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --llm_layers $llm_layers \
      --train_epochs $train_epochs \
      --model_comment "$this_comment" \
      --llm_model $llm_model \
      --llm_dim $llm_dim \
      --num_params $num_params \
      --boundary_file "$BOUNDARY_FILE" \
      --rand_init $rand_init \
      --seed $seed \
      --init_seed $init_seed \
      --save_checkpoints $save_checkpoints \
      --visualize \
      --source "$source_type"

    echo "[DONE] ${heldout_region}/${source_type} p${percentile} seed=${seed} init_seed=${init_seed} → $this_comment"

    # If rand_init==0, you only want the first init_seed — keep your prior behavior
    if [[ "$rand_init" -eq 0 ]]; then
      break
    fi
  done
done
