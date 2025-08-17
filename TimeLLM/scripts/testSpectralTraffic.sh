#!/usr/bin/env bash
# run_pems_eval.sh — TimeLLM eval vs DLinear train, on PeMS spectral-entropy outputs

set -euo pipefail

# ---------------- defaults ----------------
model_name="TimeLLM"

train_epochs=10
learning_rate=0.01
llm_layers=0
rand_init=0
num_params='2.8b'
d_model=32
d_ff=32
num_process=1
batch_size=16

# Master port base (string concat with gpu_id)
master_port_base=1180
downsampling_factor=1

# Required
heldout=""
source_type="PEMS"   # train/eval tag lineage

# Seeds
seed_ranges="1-3"
init_seed_ranges="11-13"

usage() {
  echo "Usage: $0 -n <num_params> -m <llm_model> -g <gpu_id> -h <heldout:{low|medium|high}> -s <source_type> [-p <master_port>] [-r <rand_init>] -z <seed_ranges> [-i <init_seed_ranges>]"
  echo "Example: -z 1-3 -i 11-13"
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

# Required arg check
if [ -z "${llm_model:-}" ] || [ -z "${gpu_id:-}" ] || [ -z "${heldout:-}" ] || [ -z "${source_type:-}" ]; then
  usage
fi
if [[ "$heldout" != "low" && "$heldout" != "medium" && "$heldout" != "high" ]]; then
  echo "heldout must be one of: low | medium | high"; exit 1
fi

# Model alias
if [ "$llm_model" == "ARIMA" ]; then
  model_name="ARIMA"
elif [ "$llm_model" == "DLinear" ]; then
  model_name="DLinear"
else
  model_name="TimeLLM"
fi

# Master port
if [ -z "${master_port:-}" ]; then
  master_port="${master_port_base}${gpu_id}"
fi
export CUDA_VISIBLE_DEVICES=$((gpu_id % 4))

echo "Using model: $llm_model  (dispatch: $model_name)"
echo "Heldout: $heldout"
echo "Source type: $source_type"
echo "Num params: $num_params"
echo "Master port: $master_port"
echo "Rand init: $rand_init"
echo "Seed ranges: $seed_ranges"
echo "Init seed ranges: $init_seed_ranges"

# LLM embedding dim hint
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

# Seq/pred config (hourly)
seq_len=$((512 / downsampling_factor))
pred_len=$((96 / downsampling_factor))

# Paths
ROOT="dataset/traffic/outputs_pems_hourly"
TRAIN_CSV="${ROOT}/train_univariate.csv"
BOUNDARY_JSON="${ROOT}/train_univariate_boundary.json"
TEST_CSV="${ROOT}/test_${heldout}.csv"

# Existence checks (TRAIN/BOUNDARY needed only for DLinear; TEST always needed)
if [ "$model_name,${model_name}" == "DLinear,DLinear" ]; then
  for f in "$TRAIN_CSV" "$BOUNDARY_JSON" "$TEST_CSV"; do
    if [ ! -f "$f" ]; then echo "Missing $f"; exit 1; fi
  done
else
  if [ ! -f "$TEST_CSV" ]; then echo "Missing $TEST_CSV"; exit 1; fi
fi

# Channel inference (for DLinear training)
enc_in=1; dec_in=1; c_out=1
if [ "$model_name" == "DLinear" ]; then
  # Count non-index columns assuming first column is time/index
  # NF-1: exclude index; this matches your earlier script
  col_count=$(head -n 1 "$TRAIN_CSV" | awk -F',' '{print NF-1}')
  if ! [[ "$col_count" =~ ^[0-9]+$ ]] || [ "$col_count" -le 0 ]; then
    echo "Failed to infer channels from $TRAIN_CSV header"; exit 1
  fi
  enc_in=$col_count; dec_in=$col_count; c_out=$col_count
fi

# Base tag
og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t100_c100_r${rand_init}"
if [ "$model_name" == "DLinear" ]; then
  og_tag="DLinear_d${d_model}_e${train_epochs}_f${downsampling_factor}_t100_c100_r${rand_init}"
fi

# Parse "a-b[,c-d,...]" into discrete seeds
parse_seed_ranges() {
  local ranges=$1
  local seed_list=()
  IFS=',' read -ra range_array <<< "$ranges"
  for range in "${range_array[@]}"; do
    if [[ $range =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local start=${BASH_REMATCH[1]}; local end=${BASH_REMATCH[2]}
      for ((i=start; i<=end; i++)); do seed_list+=("$i"); done
    else
      echo "Invalid range format: $range. Expected start-end."; exit 1
    fi
  done
  echo "${seed_list[@]}"
}

seed_array=($(parse_seed_ranges "$seed_ranges"))
init_seed_array=($(parse_seed_ranges "$init_seed_ranges"))

mkdir -p results/pems_eval results/pems checkpoints

for pl in $pred_len; do
  for seed in "${seed_array[@]}"; do
    for init_seed in "${init_seed_array[@]}"; do

      # checkpoint tag for eval models
      checkpoint_tag="pems_${source_type}_${heldout}_${og_tag}_seq${seq_len}_pred${pl}_seed${seed}_init${init_seed}"
      CKPT_PATH="checkpoints/${checkpoint_tag}/checkpoint"

      # per-run labels + logs
      if [ "$model_name" == "DLinear" ]; then
        tag="pems_${source_type}_${heldout}_${og_tag}_seq${seq_len}_pred${pl}_seed${seed}_init${init_seed}"
        log_file="results/pems_eval/${tag}.txt"
      else
        tag="pems_testing_${heldout}_${og_tag}_seq${seq_len}_pred${pl}_seed${seed}_init${init_seed}"
        log_file="results/pems_eval/${tag}.txt"
      fi
      mkdir -p "$(dirname "$log_file")"

      # ---- dispatch ----
      if [ "$model_name" == "DLinear" ]; then
        # Quick train DLinear on TRAIN_CSV then (internally) eval on TEST_CSV via seed_process.py
        {
          echo "[Run] DLinear train+eval"
          echo "Logging to $log_file"

          accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_process.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path "$ROOT" \
            --data_path "$(basename "$TRAIN_CSV")" \
            --data_path_test "$(basename "$TEST_CSV")" \
            --model_id "spectralTraffic_${heldout}_heldout" \
            --model "$model_name" \
            --data Traffic \
            --data_pretrain Traffic \
            --pretrain 1 \
            --features M \
            --seq_len $seq_len \
            --label_len 48 \
            --factor 3 \
            --enc_in $enc_in \
            --dec_in $dec_in \
            --c_out $c_out \
            --pred_len $pl \
            --dsampfactor $downsampling_factor \
            --percent 100 \
            --col_percent 100 \
            --des 'Exp' \
            --itr 1 \
            --d_model $d_model \
            --d_ff $d_ff \
            --batch_size $batch_size \
            --learning_rate $learning_rate \
            --llm_layers $llm_layers \
            --train_epochs $train_epochs \
            --model_comment "checkpoints/${tag}" \
            --llm_model $llm_model \
            --llm_dim $llm_dim \
            --num_params $num_params \
            --boundary_file "$BOUNDARY_JSON" \
            --rand_init $rand_init \
            --seed $seed \
            --init_seed $init_seed \
            --save_checkpoints 1 \
            --source $source_type
        } > "$log_file" 2>&1

        echo "DLinear ${heldout} (${source_type}) seed $seed init_seed $init_seed -> checkpoints/${tag}"

      else
        # Eval-only for TimeLLM (and others) via seed_evaluate.py, requires checkpoint
        if [ ! -f "$CKPT_PATH" ]; then
          echo "[ERROR] Missing checkpoint for eval: $CKPT_PATH"
          exit 1
        fi

        {
          echo "[Run] Eval-only ($model_name) with checkpoint $CKPT_PATH"
          echo "Logging to $log_file"

          accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_evaluate.py \
            --task_name long_term_forecast \
            --model_id "spectralTraffic_${heldout}_heldout_${seq_len}_${pl}" \
            --model "$model_name" \
            --data Traffic \
            --root_path "$ROOT" \
            --data_path_test "$(basename "$TEST_CSV")" \
            --features M \
            --seq_len $seq_len \
            --label_len 48 \
            --pred_len $pl \
            --factor 3 \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --d_model $d_model \
            --d_ff 32 \
            --llm_layers $llm_layers \
            --llm_model "${llm_model:-NA}" \
            --llm_dim $llm_dim \
            --num_params $num_params \
            --rand_init $rand_init \
            --checkpoint_path "$CKPT_PATH" \
            --seed $seed \
            --init_seed $init_seed \
            --use_wandb 1 \
            --visualize \
            --source $source_type \
            --heldout $heldout
        } > "$log_file" 2>&1

        echo "Eval ${model_name} ${heldout} seed $seed init_seed $init_seed completed"

      fi

      # If weights are fixed and not randomly re-initialized, break inner loop
      if [[ "$rand_init" -eq 0 ]]; then
        break
      fi
    done
  done
done
