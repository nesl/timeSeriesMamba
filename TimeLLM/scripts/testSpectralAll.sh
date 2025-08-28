#!/usr/bin/env bash
# run_all_eval.sh — unified runner (pems|unisynth|carboncast|fitbit)
# Checkpoints for DLinear encode the *training* heldout (train-heldout),
# while eval can target any heldout.

set -euo pipefail

########################################
# Global defaults
########################################
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

master_port_base=1180
mixed_precision="bf16"

downsampling_factor=1
percent=100
col_percent=100
save_checkpoints=1

# seeds
seed_ranges="1-3"
init_seed_ranges="11-13"

# selection
suite=""                 # pems|unisynth|carboncast|fitbit
heldout=""               # EVAL heldout (varies by suite)
source_type=""           # domain/source tag
data_name=""

# DLinear TRAIN heldout (controls ckpt path + train TEST file)
train_heldout=""         # default per suite if not provided

# Fitbit dir fixed
fitbit_dataset_dir="dataset/fitbit/fitbit_ds_v2"  # REQUIRED (train.csv, boundaries.json, test_*.csv)

stage="both"             # train|eval|both
dry_run=0
force_mp="--mixed_precision bf16"

usage() {
  cat <<USAGE
Usage:
  $0 -u <suite:{pems|unisynth|carboncast|fitbit}> -m <model> -g <gpu_id>
     [-h <heldout EVAL: name or path>] [-s <source_type>] [-n <num_params>] [-p <master_port>]
     [-r <rand_init>] [-z <seed_range>] [-i <init_seed_range>]
     [--train-heldout <value>] [--stage train|eval|both] [--dry-run] [-x <no>]

Notes:
- DLinear: trains with seed_process.py using --train-heldout ONLY,
           saves ckpts/logs with *train-heldout* in path,
           eval can target any --heldout (e.g., 651 -> test_651.csv or a direct CSV path).
- TimeLLM/others: eval-only with your fixed checkpoint paths.
- Fitbit dir fixed: ${fitbit_dataset_dir}
- Precision: pass -x no to force fp32 (default bf16).
USAGE
  exit 1
}

########################################
# Args
########################################
master_port=""
gpu_id=""
llm_model=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -u) suite="$2"; shift 2 ;;
    -m) llm_model="$2"; shift 2 ;;
    -g) gpu_id="$2"; shift 2 ;;
    -h) heldout="$2"; shift 2 ;;           # EVAL heldout
    -s) source_type="$2"; shift 2 ;;
    -n) num_params="$2"; shift 2 ;;
    -p) master_port="$2"; shift 2 ;;
    -r) rand_init="$2"; shift 2 ;;
    -z) seed_ranges="$2"; shift 2 ;;
    -i) init_seed_ranges="$2"; shift 2 ;;
    --train-heldout) train_heldout="$2"; shift 2 ;;   # TRAIN heldout (for ckpt + train-time test)
    --stage) stage="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift 1 ;;
    -x) [[ "${2:-}" == "no" ]] && force_mp="--mixed_precision no"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -z "${suite}" || -z "${llm_model}" || -z "${gpu_id}" ]] && usage
master_port="${master_port:-${master_port_base}${gpu_id}}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# Model dispatch
case "$llm_model" in
  DLinear) model_name="DLinear" ;;
  ARIMA)   model_name="ARIMA" ;;
  *)       model_name="TimeLLM" ;;
esac

# LLM dim hint
llm_dim=10
case "$num_params" in
  130m) llm_dim=768 ;;
  2.7b|2.8b) llm_dim=2560 ;;
  7b) llm_dim=4096 ;;
  1b|1.3b) llm_dim=2048 ;;
esac

seq_len=$((512 / downsampling_factor))
pred_len=$((96 / downsampling_factor))

parse_seed_ranges() {
  local ranges=$1; local out=()
  IFS=',' read -ra arr <<< "$ranges"
  for r in "${arr[@]}"; do
    if [[ $r =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local a=${BASH_REMATCH[1]} b=${BASH_REMATCH[2]}
      for ((i=a;i<=b;i++)); do out+=("$i"); done
    else
      echo "Bad range: $r (expect start-end)"; exit 2
    fi
  done
  echo "${out[@]}"
}
seed_array=($(parse_seed_ranges "$seed_ranges"))
init_seed_array=($(parse_seed_ranges "$init_seed_ranges"))

echo "Suite       : $suite"
echo "Model       : $llm_model (dispatch=$model_name)"
echo "GPU         : $gpu_id (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Heldout(E)  : ${heldout:-<none>}"
echo "Train-heldout(T): ${train_heldout:-<auto>}"
echo "Source type : ${source_type:-<none>}"
echo "Stage       : $stage"
echo "Seeds       : $seed_ranges ; init=$init_seed_ranges"
echo "Port        : $master_port"
echo "Precision   : ${force_mp#*= }"

run_or_echo() { [[ $dry_run -eq 1 ]] && echo "DRY-RUN >> $*" || eval "$@"; }

########################################
# Suites
########################################

run_pems() {
  local ROOT="dataset/traffic/outputs_pems_hourly"
  local TRAIN_CSV="${ROOT}/train_univariate.csv"
  local BOUNDARY_JSON="${ROOT}/train_univariate_boundary.json"

  # default TRAIN heldout if not provided
  local T_H="${train_heldout:-high}"
  local E_H="${heldout:?need -h <low|medium|high> for eval}"

  local TEST_TRAIN="${ROOT}/test_${T_H}.csv"   # used only during TRAIN (for DLinear)
  local TEST_EVAL="${ROOT}/test_${E_H}.csv"    # used during EVAL

  [[ "$model_name" == "DLinear" ]] && [[ ! -f "$TRAIN_CSV" || ! -f "$BOUNDARY_JSON" || ! -f "$TEST_TRAIN" ]] && { echo "Missing PeMS train assets"; exit 2; }
  [[ ! -f "$TEST_EVAL" ]] && { echo "Missing PeMS eval file: $TEST_EVAL"; exit 2; }

  local features="M"; local enc_in=1; local dec_in=1; local c_out=1
  if [[ "$model_name" == "DLinear" ]]; then
    local col_count; col_count=$(head -n 1 "$TRAIN_CSV" | awk -F',' '{print NF-1}')
    [[ "$col_count" =~ ^[0-9]+$ && "$col_count" -gt 0 ]] || { echo "Channel infer failed"; exit 2; }
    enc_in=$col_count; dec_in=$col_count; c_out=$col_count
  fi

  for seed in "${seed_array[@]}"; do
    for init_seed in "${init_seed_array[@]}"; do
      # tag encodes TRAIN heldout (T_H), NOT EVAL heldout
      local tag="pems_${source_type}_TRAIN${T_H}_dlin_e${train_epochs}_f${downsampling_factor}_r${rand_init}_seq${seq_len}_pred${pred_len}_s${seed}_i${init_seed}"
      local ckpt_dir="checkpoints/${tag}"
      mkdir -p results/pems_train results/pems_eval checkpoints

      if [[ "$model_name" == "DLinear" && "$stage" != "eval" ]]; then
        run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_process.py \
          --task_name long_term_forecast --is_training 1 \
          --root_path \"$ROOT\" \
          --data_path \"$(basename "$TRAIN_CSV")\" \
          --data_path_test \"$(basename "$TEST_TRAIN")\" \
          --model_id \"spectralTraffic_${T_H}_trainheldout\" \
          --model \"$model_name\" --data Traffic --data_pretrain Traffic --pretrain 1 \
          --features \"$features\" --seq_len ${seq_len} --label_len 48 --factor 3 \
          --enc_in ${enc_in} --dec_in ${dec_in} --c_out ${c_out} \
          --pred_len ${pred_len} --dsampfactor ${downsampling_factor} \
          --percent ${percent} --col_percent ${col_percent} \
          --des 'Exp' --itr 1 --d_model ${d_model} --d_ff ${d_ff} \
          --batch_size ${batch_size} --learning_rate ${learning_rate} \
          --llm_layers ${llm_layers} --train_epochs ${train_epochs} \
          --model_comment \"$ckpt_dir\" --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
          --boundary_file \"$BOUNDARY_JSON\" --rand_init ${rand_init} --seed ${seed} --init_seed ${init_seed} \
          --save_checkpoints ${save_checkpoints} --source \"$source_type\" \
          > \"results/pems_train/${tag}.txt\" 2>&1"
      fi

      if [[ "$stage" != "train" ]]; then
        if [[ "$model_name" == "DLinear" ]]; then
          local ckpt_path="${ckpt_dir}/checkpoint"
          run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast \
            --model_id \"spectralTraffic_${E_H}_eval_${seq_len}_${pred_len}\" \
            --model \"$model_name\" --data Traffic \
            --root_path \"$ROOT\" --data_path_test \"$(basename "$TEST_EVAL")\" \
            --features \"$features\" --seq_len ${seq_len} --label_len 48 --pred_len ${pred_len} --factor 3 \
            --enc_in ${enc_in} --dec_in ${dec_in} --c_out ${c_out} \
            --d_model ${d_model} --d_ff 32 --llm_layers ${llm_layers} \
            --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
            --rand_init ${rand_init} --checkpoint_path \"$ckpt_path\" \
            --seed ${seed} --init_seed ${init_seed} --use_wandb 1 --visualize \
            --source \"$source_type\" --heldout \"$E_H\" \
            > \"results/pems_eval/${tag/_TRAIN${T_H}_/}_EVAL${E_H}.txt\" 2>&1"
        elif [[ "$model_name" == "ARIMA" ]]; then
          run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast \
            --model_id \"pems_arima_${E_H}_eval_${seq_len}_${pred_len}\" \
            --model \"ARIMA\" --data Traffic \
            --root_path \"$ROOT\" --data_path_test \"$(basename "$TEST_EVAL")\" \
            --features \"$features\" --seq_len ${seq_len} --label_len 48 --pred_len ${pred_len} --factor 3 \
            --enc_in ${enc_in} --dec_in ${dec_in} --c_out ${c_out} \
            --d_model ${d_model} --d_ff 32 --llm_layers 0 \
            --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
            --rand_init ${rand_init} --checkpoint_path \"ARIMA\" \
            --seed ${seed} --init_seed ${init_seed} --use_wandb 1 --visualize \
            --source \"$source_type\" --heldout \"$E_H\" \
            > \"results/pems_eval/pems_ARIMA_${E_H}_seq${seq_len}_pred${pred_len}_seed${seed}_init${init_seed}.txt\" 2>&1"
        else
          local og="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t100_c100_r${rand_init}"
          local ckpt_path="checkpoints/pems_PEMS_high_${og}_seq${seq_len}_pred${pred_len}_seed${seed}_init${init_seed}/checkpoint"
          [[ -f "$ckpt_path" || $dry_run -eq 1 ]] || { echo "[ERROR] $ckpt_path missing"; exit 2; }
          run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast \
            --model_id \"spectralTraffic_${E_H}_eval_${seq_len}_${pred_len}\" \
            --model \"$model_name\" --data Traffic \
            --root_path \"$ROOT\" --data_path_test \"$(basename "$TEST_EVAL")\" \
            --features M --seq_len ${seq_len} --label_len 48 --pred_len ${pred_len} --factor 3 \
            --enc_in 1 --dec_in 1 --c_out 1 \
            --d_model ${d_model} --d_ff 32 --llm_layers ${llm_layers} \
            --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
            --rand_init ${rand_init} --checkpoint_path \"$ckpt_path\" \
            --seed ${seed} --init_seed ${init_seed} --use_wandb 1 --visualize \
            --source \"$source_type\" --heldout \"$E_H\" \
            > \"results/pems_eval/pems_testing_${E_H}_${og}_seq${seq_len}_pred${pred_len}_seed${seed}_init${init_seed}.txt\" 2>&1"
        fi
      fi

      [[ "$rand_init" -eq 0 ]] && break
    done
  done
}

run_unisynth() {
  local ROOT="dataset/synthetic_data/psd_synth"
  local TRAIN_CSV="${ROOT}/train.csv"

  local T_H="${train_heldout:-200}"           # TRAIN heldout for ckpt
  local E_list=()                             # EVAL heldouts
  if [[ -n "$heldout" ]]; then E_list=("$heldout"); else E_list=(200 300 400 500 600 700 800); fi

  local TEST_TRAIN="${ROOT}/region_test_om0p${T_H}.csv"
  [[ "$model_name" != "DLinear" || -f "$TEST_TRAIN" ]] || { echo "Missing uniSynth train test=${TEST_TRAIN}"; exit 2; }

  for seed in "${seed_array[@]}"; do
    for init_seed in "${init_seed_array[@]}"; do
      local tag="unisynth_TRAIN${T_H}_dlin_e${train_epochs}_f${downsampling_factor}_r${rand_init}_seq${seq_len}_pred${pred_len}_s${seed}_i${init_seed}"
      local ckpt_dir="checkpoints/${tag}"
      mkdir -p results/uniSynth_train results/uniSynth_eval checkpoints

      if [[ "$model_name" == "DLinear" && "$stage" != "eval" ]]; then
        run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_process.py \
          --task_name long_term_forecast --is_training 1 \
          --root_path \"$ROOT/\" \
          --data_path \"$(basename "$TRAIN_CSV")\" \
          --data_path_test \"$(basename "$TEST_TRAIN")\" \
          --model_id \"${T_H}_trainheldout\" \
          --model \"$model_name\" --data Synthetic --data_pretrain Synthetic --pretrain 1 \
          --features M --seq_len ${seq_len} --label_len 48 --factor 3 \
          --enc_in 1 --dec_in 1 --c_out 1 \
          --pred_len ${pred_len} --dsampfactor ${downsampling_factor} \
          --percent ${percent} --col_percent ${col_percent} \
          --des 'Exp' --itr 1 --d_model ${d_model} --d_ff ${d_ff} \
          --batch_size ${batch_size} --learning_rate ${learning_rate} \
          --llm_layers ${llm_layers} --train_epochs ${train_epochs} \
          --model_comment \"$ckpt_dir\" --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
          --rand_init ${rand_init} --seed ${seed} --init_seed ${init_seed} \
          --save_checkpoints ${save_checkpoints} --source \"uniSynth\" \
          > \"results/uniSynth_train/${tag}.txt\" 2>&1"
      fi

      if [[ "$stage" != "train" ]]; then
        if [[ "$model_name" == "DLinear" ]]; then
          local ckpt_path="${ckpt_dir}/checkpoint"
          for E_H in "${E_list[@]}"; do
            local TEST_EVAL="${ROOT}/region_test_om0p${E_H}.csv"
            [[ -f "$TEST_EVAL" ]] || { echo "Missing eval file: $TEST_EVAL"; exit 2; }
            run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
              --task_name long_term_forecast --root_path \"$ROOT/\" \
              --data_path_test \"$(basename "$TEST_EVAL")\" \
              --model_id \"${E_H}_eval_${seq_len}_${pred_len}\" \
              --model \"$model_name\" --data Synthetic --features M \
              --seq_len ${seq_len} --label_len 48 --factor 3 \
              --enc_in 1 --dec_in 1 --c_out 1 --pred_len ${pred_len} \
              --d_model ${d_model} --d_ff 32 --llm_layers ${llm_layers} \
              --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
              --rand_init ${rand_init} --checkpoint_path \"$ckpt_path\" \
              --seed ${seed} --init_seed ${init_seed} --visualize \
              --source \"uniSynth\" --use_wandb 1 --heldout \"$E_H\" \
              > \"results/uniSynth_eval/${tag/_TRAIN${T_H}_/}_EVAL${E_H}.txt\" 2>&1"
          done
        elif [[ "$model_name" == "ARIMA" ]]; then
          for E_H in "${E_list[@]}"; do
            local TEST_EVAL="${ROOT}/region_test_om0p${E_H}.csv"
            [[ -f "$TEST_EVAL" ]] || { echo "Missing eval file: $TEST_EVAL"; exit 2; }
            run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
              --task_name long_term_forecast --root_path \"$ROOT/\" \
              --data_path_test \"$(basename "$TEST_EVAL")\" \
              --model_id \"unisynth_arima_${E_H}_eval_${seq_len}_${pred_len}\" \
              --model \"ARIMA\" --data Synthetic --features M \
              --seq_len ${seq_len} --label_len 48 --factor 3 \
              --enc_in 1 --dec_in 1 --c_out 1 --pred_len ${pred_len} \
              --d_model ${d_model} --d_ff 32 --llm_layers 0 \
              --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
              --rand_init ${rand_init} --checkpoint_path \"ARIMA\" \
              --seed ${seed} --init_seed ${init_seed} --visualize \
              --source \"uniSynth\" --use_wandb 1 --heldout \"$E_H\" \
              > \"results/uniSynth_eval/unisynth_ARIMA_${E_H}_seq${seq_len}_pred${pred_len}_seed${seed}_init${init_seed}.txt\" 2>&1"
          done
        else
          local og="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
          local base_ckpt="checkpoints/uniSynthPSD_${og}_h200_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}/checkpoint"
          [[ -f "$base_ckpt" || $dry_run -eq 1 ]] || { echo "[ERROR] $base_ckpt missing"; exit 2; }
          for E_H in "${E_list[@]}"; do
            run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
              --task_name long_term_forecast --root_path \"$ROOT/\" \
              --data_path_test \"region_test_om0p${E_H}.csv\" \
              --model_id \"${E_H}_eval_${seq_len}_${pred_len}\" \
              --model \"$model_name\" --data Synthetic --features M \
              --seq_len ${seq_len} --label_len 48 --factor 3 \
              --enc_in 1 --dec_in 1 --c_out 1 --pred_len ${pred_len} \
              --d_model ${d_model} --d_ff 32 --llm_layers ${llm_layers} \
              --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
              --rand_init ${rand_init} --checkpoint_path \"$base_ckpt\" \
              --seed ${seed} --init_seed ${init_seed} --visualize \
              --source \"uniSynth\" --use_wandb 1 --heldout \"$E_H\" \
              > \"results/uniSynth_eval/testing_uniSynthPSD_${og}_h${E_H}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}.txt\" 2>&1"
          done
        fi
      fi


      [[ "$rand_init" -eq 0 ]] && break
    done
  done
}

run_carboncast() {
  local ROOT="dataset/CarbonCast"
  local data_name="CarbonCast"
  local features="M"; local enc_in=1; local dec_in=1; local c_out=1

  local T_H="${train_heldout:-high}"                   # training heldout for ckpt
  local data_path_train="spectral/heldout/${T_H}_${source_type}.csv"
  local data_path_eval="spectral/heldout/${heldout}_${source_type}.csv"  # eval heldout file

  [[ "$model_name" != "DLinear" || -f "$ROOT/$data_path_train" ]] || { echo "Missing CarbonCast TRAIN file: $ROOT/$data_path_train"; exit 2; }
  [[ -f "$ROOT/$data_path_eval" ]] || { echo "Missing CarbonCast EVAL file: $ROOT/$data_path_eval"; exit 2; }

  for seed in "${seed_array[@]}"; do
    for init_seed in "${init_seed_array[@]}"; do
      local tag="carbon_TRAIN${T_H}_${source_type}_dlin_e${train_epochs}_f${downsampling_factor}_r${rand_init}_seq${seq_len}_pred${pred_len}_s${seed}_i${init_seed}"
      local ckpt_dir="checkpoints/${tag}"
      mkdir -p results/carbon_train results/carbon_eval checkpoints

      if [[ "$model_name" == "DLinear" && "$stage" != "eval" ]]; then
        run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_process.py \
          --task_name long_term_forecast --is_training 1 \
          --root_path \"$ROOT/\" \
          --data_path \"$data_path_train\" \
          --data_path_test \"$data_path_train\" \
          --model_id \"${T_H}_${source_type}_trainheldout\" \
          --model \"$model_name\" --data \"$data_name\" --data_pretrain \"$data_name\" --pretrain 1 \
          --features \"$features\" --seq_len ${seq_len} --label_len 48 --factor 3 \
          --enc_in ${enc_in} --dec_in ${dec_in} --c_out ${c_out} \
          --pred_len ${pred_len} --dsampfactor ${downsampling_factor} \
          --percent ${percent} --col_percent ${col_percent} \
          --des 'Exp' --itr 1 --d_model ${d_model} --d_ff ${d_ff} \
          --batch_size ${batch_size} --learning_rate ${learning_rate} \
          --llm_layers ${llm_layers} --train_epochs ${train_epochs} \
          --model_comment \"$ckpt_dir\" --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
          --rand_init ${rand_init} --seed ${seed} --init_seed ${init_seed} \
          --save_checkpoints ${save_checkpoints} --source \"$source_type\" \
          > \"results/carbon_train/${tag}.txt\" 2>&1"
      fi

      if [[ "$stage" != "train" ]]; then
        if [[ "$model_name" == "DLinear" ]]; then
          local base_ckpt="checkpoints/carbon_TRAINCISO_solar_dlin_e10_f1_r0_seq${seq_len}_pred${pred_len}_s${seed}_i${init_seed}/checkpoint"
          run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast --root_path \"$ROOT/\" \
            --data_path_test \"$data_path_eval\" \
            --model_id \"${heldout}_${source_type}_eval_${seq_len}_${pred_len}\" \
            --model \"$model_name\" --data \"$data_name\" --features \"$features\" \
            --seq_len ${seq_len} --label_len 48 --factor 3 \
            --enc_in ${enc_in} --dec_in ${dec_in} --c_out ${c_out} \
            --pred_len ${pred_len} --d_model ${d_model} --d_ff 32 --llm_layers ${llm_layers} \
            --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
            --rand_init ${rand_init} --checkpoint_path \"$base_ckpt\" \
            --seed ${seed} --init_seed ${init_seed} --visualize --source \"$source_type\" --use_wandb 1 --heldout \"$heldout\" \
            > \"results/carbon_eval/${tag/_TRAIN${T_H}_/}_EVAL${heldout}.txt\" 2>&1"
        elif [[ "$model_name" == "ARIMA" ]]; then
          run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast --root_path \"$ROOT/\" \
            --data_path_test \"$data_path_eval\" \
            --model_id \"carbon_arima_${heldout}_${source_type}_eval_${seq_len}_${pred_len}\" \
            --model \"ARIMA\" --data \"$data_name\" --features \"$features\" \
            --seq_len ${seq_len} --label_len 48 --factor 3 \
            --enc_in ${enc_in} --dec_in ${dec_in} --c_out ${c_out} \
            --pred_len ${pred_len} --d_model ${d_model} --d_ff 32 --llm_layers 0 \
            --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
            --rand_init ${rand_init} --checkpoint_path \"ARIMA\" \
            --seed ${seed} --init_seed ${init_seed} --visualize --source \"$source_type\" --use_wandb 1 --heldout \"$heldout\" \
            > \"results/carbon_eval/carbon_ARIMA_${heldout}_${source_type}_seq${seq_len}_pred${pred_len}_seed${seed}_init${init_seed}.txt\" 2>&1"
        else
          local og="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
          local ckpt_path="checkpoints/spectralUni_CISO_solar_p05_${og}_seq${seq_len}_pred${pred_len}/s${seed}_i${init_seed}/checkpoint"
          [[ -f "$ckpt_path" || $dry_run -eq 1 ]] || { echo "[ERROR] $ckpt_path missing"; exit 2; }
          run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast --root_path \"$ROOT/\" \
            --data_path_test \"$data_path_eval\" \
            --model_id \"${heldout}_${source_type}_eval_${seq_len}_${pred_len}\" \
            --model \"$model_name\" --data \"$data_name\" --features \"$features\" \
            --seq_len ${seq_len} --label_len 48 --factor 3 \
            --enc_in 1 --dec_in 1 --c_out 1 --pred_len ${pred_len} \
            --d_model ${d_model} --d_ff 32 --llm_layers ${llm_layers} \
            --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
            --rand_init ${rand_init} --checkpoint_path \"$ckpt_path\" \
            --seed ${seed} --init_seed ${init_seed} --visualize --source \"$source_type\" \
            > \"results/spectralUniTest/spectralUniTest_${heldout}_${source_type}_heldout_${og}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}.txt\" 2>&1"
        fi
      fi


      [[ "$rand_init" -eq 0 ]] && break
    done
  done
}

run_fitbit() {
  local ROOT="${fitbit_dataset_dir}"
  local TRAIN_CSV="${ROOT}/train.csv"
  local BOUNDARY_JSON="${ROOT}/boundaries.json"
  local T_H="${train_heldout:-high}"
  local E_H="${heldout:?need -h <UXXX> for eval}"
  local TEST_TRAIN="${ROOT}/test_${T_H}.csv"
  local TEST_EVAL="${ROOT}/test_${E_H}.csv"
  local data_name="Fitbit"
  local features="M"; local enc_in=1; local dec_in=1; local c_out=1
  local source="hr"

  [[ -f "$TRAIN_CSV" && -f "$BOUNDARY_JSON" && -f "$TEST_TRAIN" && -f "$TEST_EVAL" ]] || { echo "Fitbit files missing"; exit 2; }

  for seed in "${seed_array[@]}"; do
    for init_seed in "${init_seed_array[@]}"; do
      local tag="fitbit_${source}_TRAIN${T_H}_dlin_e${train_epochs}_f${downsampling_factor}_r${rand_init}_seq${seq_len}_pred${pred_len}_s${seed}_i${init_seed}"
      local ckpt_dir="checkpoints/${tag}"
      mkdir -p results/fitbit_train results/fitbit_eval checkpoints

      if [[ "$model_name" == "DLinear" && "$stage" != "eval" ]]; then
        run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_process.py \
          --task_name long_term_forecast --is_training 1 \
          --root_path \"${ROOT}/\" \
          --data_path \"$(basename "$TRAIN_CSV")\" \
          --data_path_test \"$(basename "$TEST_TRAIN")\" \
          --model_id \"fitbit_${source}_${T_H}_trainheldout\" \
          --model \"$model_name\" --data \"$data_name\" --data_pretrain \"$data_name\" --pretrain 1 \
          --features \"$features\" --seq_len ${seq_len} --label_len 48 --factor 3 --freq t \
          --enc_in ${enc_in} --dec_in ${dec_in} --c_out ${c_out} \
          --pred_len ${pred_len} --dsampfactor ${downsampling_factor} \
          --percent ${percent} --col_percent ${col_percent} \
          --d_model ${d_model} --d_ff ${d_ff} --batch_size ${batch_size} --learning_rate ${learning_rate} \
          --llm_layers ${llm_layers} --train_epochs ${train_epochs} \
          --model_comment \"$ckpt_dir\" --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
          --boundary_file \"$BOUNDARY_JSON\" --rand_init ${rand_init} --seed ${seed} --init_seed ${init_seed} \
          --save_checkpoints ${save_checkpoints} --source \"$source\" \
          > \"results/fitbit_train/${tag}.txt\" 2>&1"
      fi

      if [[ "$stage" != "train" ]]; then
        if [[ "$model_name" == "DLinear" ]]; then
          local ckpt_path="${ckpt_dir}/checkpoint"
          run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast --root_path \"${ROOT}/\" \
            --data_path_test \"$(basename "$TEST_EVAL")\" \
            --model_id \"fitbit_${source}_${E_H}_eval\" \
            --model \"$model_name\" --data \"$data_name\" --features \"$features\" \
            --seq_len ${seq_len} --label_len 48 --factor 3 --freq t \
            --enc_in ${enc_in} --dec_in ${dec_in} --c_out ${c_out} \
            --pred_len ${pred_len} --dsampfactor ${downsampling_factor} \
            --percent ${percent} --col_percent ${col_percent} \
            --d_model ${d_model} --d_ff 32 --batch_size ${batch_size} \
            --llm_layers ${llm_layers} --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
            --checkpoint_path \"$ckpt_path\" --rand_init ${rand_init} \
            --seed ${seed} --init_seed ${init_seed} --source \"$source\" --use_wandb 1 --visualize \
            > \"results/fitbit_eval/${tag/_TRAIN${T_H}_/}_EVAL${E_H}.txt\" 2>&1"
        elif [[ "$model_name" == "ARIMA" ]]; then
          run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast --root_path \"${ROOT}/\" \
            --data_path_test \"$(basename "$TEST_EVAL")\" \
            --model_id \"fitbit_arima_${source}_${E_H}_eval\" \
            --model \"ARIMA\" --data \"$data_name\" --features \"$features\" \
            --seq_len ${seq_len} --label_len 48 --factor 3 --freq t \
            --enc_in ${enc_in} --dec_in ${dec_in} --c_out ${c_out} \
            --pred_len ${pred_len} --dsampfactor ${downsampling_factor} \
            --percent ${percent} --col_percent ${col_percent} \
            --d_model ${d_model} --d_ff 32 --batch_size ${batch_size} \
            --llm_layers 0 --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
            --rand_init ${rand_init} --checkpoint_path \"ARIMA\" \
            --seed ${seed} --init_seed ${init_seed} --source \"$source\" --use_wandb 1 --visualize \
            > \"results/fitbit_eval/fitbit_ARIMA_${source}_${E_H}_seq${seq_len}_pred${pred_len}_seed${seed}_init${init_seed}.txt\" 2>&1"
        else
          local og="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
          local ckpt_path="checkpoints/fitbit_${source}_high_heldout_${og}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}/checkpoint"
          [[ -f "$ckpt_path" || $dry_run -eq 1 ]] || { echo "[ERROR] $ckpt_path missing"; exit 2; }
          run_or_echo "accelerate launch $force_mp --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast --root_path \"${ROOT}/\" \
            --data_path_test \"$(basename "$TEST_EVAL")\" \
            --model_id \"fitbit_${source}_${E_H}_eval\" \
            --model \"$model_name\" --data \"$data_name\" --features \"$features\" \
            --seq_len ${seq_len} --label_len 48 --factor 3 --freq t \
            --enc_in ${enc_in} --dec_in ${dec_in} --c_out ${c_out} \
            --pred_len ${pred_len} --dsampfactor ${downsampling_factor} \
            --percent ${percent} --col_percent ${col_percent} \
            --d_model ${d_model} --d_ff 32 --batch_size ${batch_size} \
            --llm_layers ${llm_layers} --llm_model \"$llm_model\" --llm_dim ${llm_dim} --num_params \"$num_params\" \
            --checkpoint_path \"$ckpt_path\" --rand_init ${rand_init} \
            --seed ${seed} --init_seed ${init_seed} --source \"$source\" --use_wandb 1 --visualize \
            > \"results/fitbit_eval/test_fitbit_${source}_${E_H}_heldout_${og}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}.txt\" 2>&1"
        fi
      fi


      [[ "$rand_init" -eq 0 ]] && break
    done
  done
}

########################################
# Dispatch
########################################
case "$suite" in
  pems)       run_pems ;;
  unisynth)   run_unisynth ;;
  carboncast) run_carboncast ;;
  fitbit)     run_fitbit ;;
  *) usage ;;
esac

