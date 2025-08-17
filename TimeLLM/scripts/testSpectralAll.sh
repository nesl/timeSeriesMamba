#!/usr/bin/env bash
# run_all_eval.sh — unified driver for spectralTraffic (PeMS), uniSynth, CarbonCast, Fitbit
# DLinear => seed_process.py (train+eval); others => seed_evaluate.py (eval-only, fixed checkpoint paths)

set -euo pipefail

# ---------------- defaults (shared) ----------------
suite=""                   # one of: pems | unisynth | carboncast | fitbit (REQUIRED)
model_name="TimeLLM"       # dispatch alias; do not pass directly, use -m
llm_model=""               # e.g. TimeLLM, DLinear, ARIMA (REQUIRED)
gpu_id=""                  # REQUIRED

train_epochs=10
learning_rate=0.01
llm_layers=0
rand_init=0
num_params='2.8b'
d_model=32
d_ff=32
num_process=1
batch_size=16

master_port_base=1180
downsampling_factor=1
percent=100
col_percent=100
save_checkpoints=1
mixed_precision="bf16"

# seeds
seed_ranges="1-3"
init_seed_ranges="11-13"

# dry-run
dry_run=0

# per-suite knobs (override via flags where applicable)
# -- pems
pems_source_type="PEMS"
pems_heldout=""             # low|medium|high (REQUIRED for pems)
# -- unisynth
unisynth_source_type="uniSynth"
unisynth_heldouts_default="200 300 400 500 600 700 800"
unisynth_one_heldout=""     # if set, only run that heldout (e.g., -H 500)
# -- carboncast
carbon_source_type=""       # e.g., CISO_solar_p05 (REQUIRED for carboncast)
carbon_heldout=""           # REQUIRED (filename stem)
# -- fitbit
fitbit_dataset_dir="dataset/fitbit/fitbit_ds_v2"       # REQUIRED (path with train.csv, boundaries.json, test_*.csv)
fitbit_heldout=""           # REQUIRED (e.g., low)
fitbit_source="hr"          # passed through --source

usage() {
  cat <<USAGE
Usage:
  $0 -u <suite:{pems|unisynth|carboncast|fitbit}> -m <llm_model> -g <gpu_id> [common opts] [suite opts]

Common opts:
  -n <num_params>        (default: 2.8b)
  -p <master_port>       (default: base ${master_port_base} + gpu_id)
  -r <rand_init>         (default: 0)
  -z <seed_ranges>       (default: 1-3)  e.g. "1-2,5-6"
  -i <init_seed_ranges>  (default: 11-13)
  -x <mixed_precision>   (default: bf16) pass "no" to disable
  -y                     dry-run (print commands only)
  --dry-run              same as -y

Suite-specific:
  pems:
    -h <heldout:{low|medium|high}>   (REQUIRED)
    -s <source_type>                 (default: ${pems_source_type})

  unisynth:
    -H <one_heldout>   (optional; run just one of: 200 300 400 500 600 700 800)
    -s <source_type>   (tag only; default: ${unisynth_source_type})

  carboncast:
    -h <heldout>             (REQUIRED, filename stem)
    -s <source_type>         (REQUIRED, e.g., CISO_solar_p05)

  fitbit:
    -d <dataset_dir>         (REQUIRED; has train.csv, boundaries.json, test_*.csv)
    -h <heldout_alias>       (REQUIRED; e.g., U001)

Examples:
  PEMS TimeLLM eval:  $0 -u pems -m LLAMA3.2 -g 0 -h low -s PEMS
  PEMS DLinear train: $0 -u pems -m DLinear -g 0 -h low -s PEMS
  uniSynth all:       $0 -u unisynth -m LLAMA3.2 -g 0
  uniSynth one:       $0 -u unisynth -m DLinear -g 0 -H 500
  CarbonCast eval:    $0 -u carboncast -m LLAMA3.2 -g 0 -h 2020 -s CISO_solar_p05
  Fitbit eval:        $0 -u fitbit -m LLAMA3.2 -g 0 -d dataset/fitbit/fitbit_ds_v2 -h low
USAGE
  exit 1
}

# ---------------- parse args (short flags) ----------------
while getopts "u:m:g:n:p:r:z:i:x:h:H:s:d:y" opt; do
  case $opt in
    u) suite=$OPTARG ;;
    m) llm_model=$OPTARG ;;
    g) gpu_id=$OPTARG ;;
    n) num_params=$OPTARG ;;
    p) master_port=$OPTARG ;;
    r) rand_init=$OPTARG ;;
    z) seed_ranges=$OPTARG ;;
    i) init_seed_ranges=$OPTARG ;;
    x) mixed_precision=$OPTARG ;;
    y) dry_run=1 ;;
    h) # overloaded: pems/carbon/fitbit heldout
       if [[ "${suite}" == "pems" ]]; then pems_heldout=$OPTARG
       elif [[ "${suite}" == "carboncast" ]]; then carbon_heldout=$OPTARG
       elif [[ "${suite}" == "fitbit" ]]; then fitbit_heldout=$OPTARG
       else echo "Flag -h not applicable to suite=${suite}"; usage; fi ;;
    H) unisynth_one_heldout=$OPTARG ;;
    s) # source type: used by pems (tag), unisynth (tag), carboncast (file)
       if [[ "${suite}" == "pems" ]]; then pems_source_type=$OPTARG
       elif [[ "${suite}" == "unisynth" ]]; then unisynth_source_type=$OPTARG
       elif [[ "${suite}" == "carboncast" ]]; then carbon_source_type=$OPTARG
       else echo "Flag -s not applicable to suite=${suite}"; usage; fi ;;
    d) fitbit_dataset_dir=$OPTARG ;;
    *) usage ;;
  esac
done

# accept --dry-run long flag
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && dry_run=1
done

# ---------------- required checks ----------------
[[ -z "${suite}" || -z "${llm_model}" || -z "${gpu_id}" ]] && usage
case "${suite}" in
  pems)
    [[ -z "${pems_heldout}" ]] && usage
    [[ "${pems_heldout}" =~ ^(low|medium|high)$ ]] || { echo "PEMS heldout must be low|medium|high"; exit 2; }
    ;;
  unisynth)
    : ;;
  carboncast)
    [[ -z "${carbon_heldout}" || -z "${carbon_source_type}" ]] && usage
    ;;
  fitbit)
    [[ -z "${fitbit_dataset_dir}" || -z "${fitbit_heldout}" ]] && usage
    if (( ! dry_run )); then
      [[ -f "${fitbit_dataset_dir}/train.csv" ]] || { echo "ERROR: ${fitbit_dataset_dir}/train.csv not found"; exit 2; }
      [[ -f "${fitbit_dataset_dir}/boundaries.json" ]] || { echo "ERROR: ${fitbit_dataset_dir}/boundaries.json not found"; exit 2; }
    fi
    ;;
  *) usage ;;
esac

# ---------------- model alias ----------------
if [[ "${llm_model}" == "ARIMA" ]]; then
  model_name="ARIMA"
elif [[ "${llm_model}" == "DLinear" ]]; then
  model_name="DLinear"
else
  model_name="TimeLLM"
fi

# master port
master_port="${master_port:-${master_port_base}${gpu_id}}"
export CUDA_VISIBLE_DEVICES=$((gpu_id % 4))

# ---------------- derived dims ----------------
llm_dim=10
case "$num_params" in
  130m) llm_dim=768 ;;
  2.7b|2.8b) llm_dim=2560 ;;
  7b) llm_dim=4096 ;;
  1b|1.3b) llm_dim=2048 ;;
esac

seq_len=$((512 / downsampling_factor))
pred_base=$((96 / downsampling_factor))

# seed parsing
parse_seed_ranges() {
  local ranges=$1; local seed_list=()
  IFS=',' read -ra arr <<< "$ranges"
  for range in "${arr[@]}"; do
    if [[ $range =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local start=${BASH_REMATCH[1]}; local end=${BASH_REMATCH[2]}
      for ((i=start; i<=end; i++)); do seed_list+=("$i"); done
    else
      echo "Invalid range: $range (expected start-end)"; exit 3
    fi
  done
  echo "${seed_list[@]}"
}
seed_array=($(parse_seed_ranges "$seed_ranges"))
init_seed_array=($(parse_seed_ranges "$init_seed_ranges"))

echo "Suite: ${suite} | Model: ${llm_model} (dispatch: ${model_name}) | GPU: ${gpu_id}"
echo "Seeds: ${seed_ranges} | InitSeeds: ${init_seed_ranges} | Port: ${master_port} | MP: ${mixed_precision}"
(( dry_run )) && echo "[DRY-RUN] enabled — will not execute commands or check files."

# helpers
enc_dec_from_csv_header() {
  local csv=$1
  local c
  c=$(head -n 1 "$csv" | awk -F',' '{print NF-1}')
  if ! [[ "$c" =~ ^[0-9]+$ ]] || [ "$c" -le 0 ]; then
    echo "1 1 1"; return 0
  fi
  echo "$c $c $c"
}

run_cmd() {
  local log="$1"; shift
  if (( dry_run )); then
    echo "[DRY-RUN] would log to: $log"
    echo "[DRY-RUN] $*"
  else
    mkdir -p "$(dirname "$log")"
    "$@" > "$log" 2>&1
  fi
}

# ---------------- per-suite runners ----------------
run_pems() {
  local ROOT="dataset/traffic/outputs_pems_hourly"
  local TRAIN_CSV="${ROOT}/train_univariate.csv"
  local BOUNDARY_JSON="${ROOT}/train_univariate_boundary.json"
  local TEST_CSV="${ROOT}/test_${pems_heldout}.csv"

  if (( ! dry_run )) && [[ "${model_name}" == "DLinear" ]]; then
    [[ -f "$TRAIN_CSV" && -f "$BOUNDARY_JSON" && -f "$TEST_CSV" ]] || { echo "PEMS missing train/boundary/test"; exit 4; }
  fi
  if (( ! dry_run )) && [[ "${model_name}" != "DLinear" ]]; then
    [[ -f "$TEST_CSV" ]] || { echo "PEMS missing $TEST_CSV"; exit 4; }
  fi

  local enc_in=1 dec_in=1 c_out=1
  if [[ "${model_name}" == "DLinear" ]] && (( ! dry_run )); then
    read enc_in dec_in c_out < <(enc_dec_from_csv_header "$TRAIN_CSV")
  fi

  local og_tag
  if [[ "${model_name}" == "DLinear" ]]; then
    og_tag="DLinear_d${d_model}_e${train_epochs}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
  else
    og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
  fi

  mkdir -p results/pems_eval results/pems checkpoints

  for pred_len in ${pred_base}; do
    for seed in "${seed_array[@]}"; do
      for init_seed in "${init_seed_array[@]}"; do
        local checkpoint_tag="pems_${pems_source_type}_${pems_heldout}_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_init${init_seed}"
        if [[ "${model_name}" == "DLinear" ]]; then
          local tag="${checkpoint_tag}"
          local log_file="results/pems_eval/${tag}.txt"
          run_cmd "$log_file" accelerate launch --mixed_precision "${mixed_precision}" --num_processes ${num_process} --main_process_port ${master_port} seed_process.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path "${ROOT}" \
            --data_path "$(basename "$TRAIN_CSV")" \
            --data_path_test "$(basename "$TEST_CSV")" \
            --model_id "spectralTraffic_${pems_heldout}_heldout" \
            --model "${model_name}" \
            --data Traffic \
            --data_pretrain Traffic \
            --pretrain 1 \
            --features M \
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
            --model_comment "checkpoints/${tag}" \
            --llm_model "${llm_model}" \
            --llm_dim ${llm_dim} \
            --num_params "${num_params}" \
            --boundary_file "${BOUNDARY_JSON}" \
            --rand_init ${rand_init} \
            --seed ${seed} \
            --init_seed ${init_seed} \
            --save_checkpoints ${save_checkpoints} \
            --source "${pems_source_type}"
        else
          local CKPT="checkpoints/${checkpoint_tag}/checkpoint"   # fixed path (as provided)
          if (( ! dry_run )) && [[ ! -f "$CKPT" ]]; then echo "[ERROR] Missing checkpoint: $CKPT"; exit 5; fi
          local tag="pems_testing_${pems_heldout}_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_init${init_seed}"
          local log_file="results/pems_eval/${tag}.txt"
          run_cmd "$log_file" accelerate launch --mixed_precision "${mixed_precision}" --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast \
            --model_id "spectralTraffic_${pems_heldout}_heldout_${seq_len}_${pred_len}" \
            --model "${model_name}" \
            --data Traffic \
            --root_path "${ROOT}" \
            --data_path_test "$(basename "$TEST_CSV")" \
            --features M \
            --seq_len ${seq_len} \
            --label_len 48 \
            --pred_len ${pred_len} \
            --factor 3 \
            --enc_in 1 --dec_in 1 --c_out 1 \
            --d_model ${d_model} \
            --d_ff 32 \
            --llm_layers ${llm_layers} \
            --llm_model "${llm_model}" \
            --llm_dim ${llm_dim} \
            --num_params "${num_params}" \
            --rand_init ${rand_init} \
            --checkpoint_path "${CKPT}" \
            --seed ${seed} \
            --init_seed ${init_seed} \
            --use_wandb 1 \
            --visualize \
            --source "${pems_source_type}" \
            --heldout "${pems_heldout}"
        fi
        [[ "${rand_init}" -eq 0 ]] && break
      done
    done
  done
}

run_unisynth() {
  local ROOT="dataset/synthetic_data/psd_synth"
  local TRAIN_CSV="${ROOT}/train.csv"
  if (( ! dry_run )); then [[ -f "$TRAIN_CSV" ]] || { echo "uniSynth missing $TRAIN_CSV"; exit 6; }; fi

  local heldouts
  if [[ -n "${unisynth_one_heldout}" ]]; then
    heldouts="${unisynth_one_heldout}"
  else
    heldouts="${unisynth_heldouts_default}"
  fi

  local og_tag
  if [[ "${model_name}" == "DLinear" ]]; then
    og_tag="DLinear_d${d_model}_e${train_epochs}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
  else
    og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
  fi

  mkdir -p results/uniSynthPSD_eval checkpoints

  local enc_in=1 dec_in=1 c_out=1
  if [[ "${model_name}" == "DLinear" ]] && (( ! dry_run )); then
    read enc_in dec_in c_out < <(enc_dec_from_csv_header "$TRAIN_CSV")
  fi

  for heldout in ${heldouts}; do
    local TEST_CSV="${ROOT}/region_test_om0p${heldout}.csv"
    if (( ! dry_run )); then [[ -f "$TEST_CSV" ]] || { echo "uniSynth missing $TEST_CSV"; exit 6; }; fi

    for pred_len in ${pred_base}; do
      for seed in "${seed_array[@]}"; do
        for init_seed in "${init_seed_array[@]}"; do
          local tag="testing_uniSynthPSD_${og_tag}_h${heldout}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
          local log_file="results/uniSynthPSD_eval/${tag}.txt"

          if [[ "${model_name}" == "DLinear" ]]; then
            run_cmd "$log_file" accelerate launch --mixed_precision "${mixed_precision}" --num_processes ${num_process} --main_process_port ${master_port} seed_process.py \
              --task_name long_term_forecast \
              --is_training 1 \
              --root_path "${ROOT}/" \
              --data_path "$(basename "$TRAIN_CSV")" \
              --data_path_test "$(basename "$TEST_CSV")" \
              --model_id "${heldout}_heldout_${seq_len}_${pred_len}" \
              --model "${model_name}" \
              --data Synthetic \
              --features M \
              --seq_len ${seq_len} \
              --label_len 48 \
              --factor 3 \
              --enc_in ${enc_in} \
              --dec_in ${dec_in} \
              --c_out ${c_out} \
              --pred_len ${pred_len} \
              --d_model ${d_model} \
              --d_ff ${d_ff} \
              --batch_size ${batch_size} \
              --learning_rate ${learning_rate} \
              --llm_layers ${llm_layers} \
              --train_epochs ${train_epochs} \
              --model_comment "checkpoints/${tag}" \
              --llm_model "${llm_model}" \
              --llm_dim ${llm_dim} \
              --num_params "${num_params}" \
              --rand_init ${rand_init} \
              --seed ${seed} \
              --init_seed ${init_seed} \
              --source "${unisynth_source_type}" \
              --save_checkpoints ${save_checkpoints}
          else
            # fixed checkpoint pattern (as provided)
            local checkpoint_tag="uniSynthPSD_${og_tag}_h200_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
            local CKPT="checkpoints/${checkpoint_tag}/checkpoint"
            if (( ! dry_run )) && [[ ! -f "$CKPT" ]]; then echo "[ERROR] Missing checkpoint: $CKPT"; exit 5; fi
            run_cmd "$log_file" accelerate launch --mixed_precision "${mixed_precision}" --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
              --task_name long_term_forecast \
              --root_path "${ROOT}/" \
              --data_path "$(basename "$TRAIN_CSV")" \
              --data_path_test "$(basename "$TEST_CSV")" \
              --model_id "${heldout}_heldout_${seq_len}_${pred_len}" \
              --model "${model_name}" \
              --data Synthetic \
              --features M \
              --seq_len ${seq_len} \
              --label_len 48 \
              --factor 3 \
              --enc_in 1 --dec_in 1 --c_out 1 \
              --pred_len ${pred_len} \
              --d_model ${d_model} \
              --d_ff 32 \
              --llm_layers ${llm_layers} \
              --llm_model "${llm_model}" \
              --llm_dim ${llm_dim} \
              --num_params "${num_params}" \
              --rand_init ${rand_init} \
              --checkpoint_path "${CKPT}" \
              --seed ${seed} \
              --init_seed ${init_seed} \
              --visualize \
              --source "${unisynth_source_type}" \
              --use_wandb 1 \
              --heldout ${heldout}
          fi
          [[ "${rand_init}" -eq 0 ]] && break
        done
      done
    done
  done
}

run_carboncast() {
  local ROOT="dataset/CarbonCast"
  local TRAIN_CSV="${ROOT}/spectral/heldout/${carbon_heldout}_${carbon_source_type}.csv"
  local TEST_CSV="${ROOT}/spectral/heldout/${carbon_heldout}_${carbon_source_type}.csv"

  if (( ! dry_run )); then
    [[ -f "$TRAIN_CSV" && -f "$TEST_CSV" ]] || { echo "CarbonCast missing ${TRAIN_CSV} or ${TEST_CSV}"; exit 7; }
  fi

  local og_tag
  if [[ "${model_name}" == "DLinear" ]]; then
    og_tag="DLinear_l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_r${rand_init}"
  else
    og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
  fi

  local enc_in=1 dec_in=1 c_out=1
  if [[ "${model_name}" == "DLinear" ]] && (( ! dry_run )); then
    read enc_in dec_in c_out < <(enc_dec_from_csv_header "$TRAIN_CSV")
  fi

  mkdir -p results/spectralUniTest checkpoints

  for pred_len in ${pred_base}; do
    for seed in "${seed_array[@]}"; do
      for init_seed in "${init_seed_array[@]}"; do
        local tag="spectralUniTest_${carbon_heldout}_${carbon_source_type}_heldout_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
        local log_file="results/spectralUniTest/${tag}.txt"

        if [[ "${model_name}" == "DLinear" ]]; then
          run_cmd "$log_file" accelerate launch --mixed_precision "${mixed_precision}" --num_processes ${num_process} --main_process_port ${master_port} seed_process.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path "${ROOT}/" \
            --data_path "spectral/heldout/${carbon_heldout}_${carbon_source_type}.csv" \
            --data_path_test "spectral/heldout/${carbon_heldout}_${carbon_source_type}.csv" \
            --model_id "${carbon_heldout}_heldout_${seq_len}_${pred_len}" \
            --model "${model_name}" \
            --data CarbonCast \
            --features M \
            --seq_len ${seq_len} \
            --label_len 48 \
            --factor 3 \
            --enc_in ${enc_in} \
            --dec_in ${dec_in} \
            --c_out ${c_out} \
            --pred_len ${pred_len} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --batch_size ${batch_size} \
            --learning_rate ${learning_rate} \
            --llm_layers ${llm_layers} \
            --train_epochs ${train_epochs} \
            --model_comment "checkpoints/${tag}" \
            --llm_model "${llm_model}" \
            --llm_dim ${llm_dim} \
            --num_params "${num_params}" \
            --rand_init ${rand_init} \
            --seed ${seed} \
            --init_seed ${init_seed} \
            --save_checkpoints ${save_checkpoints}
        else
          # fixed checkpoint pattern (as provided)
          local checkpoint_tag="spectralUni_CISO_solar_p05_${og_tag}_seq${seq_len}_pred${pred_len}/s${seed}_i${init_seed}"
          local CKPT="checkpoints/${checkpoint_tag}/checkpoint"
          if (( ! dry_run )) && [[ ! -f "$CKPT" ]]; then echo "[ERROR] Missing checkpoint: $CKPT"; exit 5; fi
          run_cmd "$log_file" accelerate launch --mixed_precision "${mixed_precision}" --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast \
            --root_path "${ROOT}/" \
            --data_path "spectral/heldout/${carbon_heldout}_${carbon_source_type}.csv" \
            --data_path_test "spectral/heldout/${carbon_heldout}_${carbon_source_type}.csv" \
            --model_id "${carbon_heldout}_heldout_${seq_len}_${pred_len}" \
            --model "${model_name}" \
            --data CarbonCast \
            --features M \
            --seq_len ${seq_len} \
            --label_len 48 \
            --factor 3 \
            --enc_in 1 --dec_in 1 --c_out 1 \
            --pred_len ${pred_len} \
            --d_model ${d_model} \
            --d_ff 32 \
            --llm_layers ${llm_layers} \
            --llm_model "${llm_model}" \
            --llm_dim ${llm_dim} \
            --num_params "${num_params}" \
            --rand_init ${rand_init} \
            --checkpoint_path "${CKPT}" \
            --seed ${seed} \
            --init_seed ${init_seed} \
            --visualize
        fi
        [[ "${rand_init}" -eq 0 ]] && break
      done
    done
  done
}

run_fitbit() {
  local ROOT="${fitbit_dataset_dir}"
  local TRAIN_CSV="${ROOT}/train.csv"
  local BOUNDARY_JSON="${ROOT}/boundaries.json"
  local TEST_CSV="${ROOT}/test_${fitbit_heldout}.csv"
  if (( ! dry_run )); then
    [[ -f "$TRAIN_CSV" && -f "$BOUNDARY_JSON" && -f "$TEST_CSV" ]] || { echo "Fitbit missing train/boundaries/test"; exit 8; }
  fi

  local og_tag
  if [[ "${model_name}" == "DLinear" ]]; then
    og_tag="DLinear_l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_r${rand_init}"
  else
    og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
  fi

  local enc_in=1 dec_in=1 c_out=1
  if [[ "${model_name}" == "DLinear" ]] && (( ! dry_run )); then
    read enc_in dec_in c_out < <(enc_dec_from_csv_header "$TRAIN_CSV")
  fi

  mkdir -p results/fitbit_eval checkpoints

  for pred_len in ${pred_base}; do
    for seed in "${seed_array[@]}"; do
      for init_seed in "${init_seed_array[@]}"; do
        local tag="test_fitbit_${fitbit_source}_${fitbit_heldout}_heldout_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
        local log_file="results/fitbit_eval/${tag}.txt"

        if [[ "${model_name}" == "DLinear" ]]; then
          run_cmd "$log_file" accelerate launch --mixed_precision "${mixed_precision}" --num_processes ${num_process} --main_process_port ${master_port} seed_process.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path "${ROOT}/" \
            --data_path "train.csv" \
            --data_path_test "test_${fitbit_heldout}.csv" \
            --model_id "fitbit_${fitbit_source}_${fitbit_heldout}_heldout" \
            --model "${model_name}" \
            --data Fitbit \
            --features M \
            --seq_len ${seq_len} \
            --label_len 48 \
            --factor 3 \
            --freq t \
            --enc_in ${enc_in} \
            --dec_in ${dec_in} \
            --c_out ${c_out} \
            --pred_len ${pred_len} \
            --dsampfactor ${downsampling_factor} \
            --percent ${percent} \
            --col_percent ${col_percent} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --batch_size ${batch_size} \
            --learning_rate ${learning_rate} \
            --llm_layers ${llm_layers} \
            --train_epochs ${train_epochs} \
            --model_comment "checkpoints/${tag}" \
            --llm_model "${llm_model}" \
            --llm_dim ${llm_dim} \
            --num_params "${num_params}" \
            --boundary_file "${BOUNDARY_JSON}" \
            --rand_init ${rand_init} \
            --seed ${seed} \
            --init_seed ${init_seed} \
            --source "${fitbit_source}" \
            --save_checkpoints ${save_checkpoints}
        else
          # fixed checkpoint pattern (as provided)
          local checkpoint_tag="fitbit_${fitbit_source}_high_heldout_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_initseed${init_seed}"
          local CKPT="checkpoints/${checkpoint_tag}/checkpoint"
          if (( ! dry_run )) && [[ ! -f "$CKPT" ]]; then echo "[ERROR] Missing checkpoint: $CKPT"; exit 5; fi
          run_cmd "$log_file" accelerate launch --mixed_precision "${mixed_precision}" --num_processes ${num_process} --main_process_port ${master_port} seed_evaluate.py \
            --task_name long_term_forecast \
            --root_path "${ROOT}/" \
            --data_path_test "test_${fitbit_heldout}.csv" \
            --model_id "fitbit_${fitbit_source}_${fitbit_heldout}_heldout" \
            --model "${model_name}" \
            --data Fitbit \
            --features M \
            --seq_len ${seq_len} \
            --label_len 48 \
            --factor 3 \
            --freq t \
            --enc_in 1 --dec_in 1 --c_out 1 \
            --pred_len ${pred_len} \
            --dsampfactor ${downsampling_factor} \
            --percent ${percent} \
            --col_percent ${col_percent} \
            --d_model ${d_model} \
            --d_ff ${d_ff} \
            --batch_size ${batch_size} \
            --llm_layers ${llm_layers} \
            --llm_model "${llm_model}" \
            --llm_dim ${llm_dim} \
            --num_params "${num_params}" \
            --checkpoint_path "${CKPT}" \
            --rand_init ${rand_init} \
            --seed ${seed} \
            --init_seed ${init_seed} \
            --source "${fitbit_source}" \
            --use_wandb 1 \
            --visualize
        fi
        [[ "${rand_init}" -eq 0 ]] && break
      done
    done
  done
}

# ---------------- dispatch ----------------
case "${suite}" in
  pems)       run_pems ;;
  unisynth)   run_unisynth ;;
  carboncast) run_carboncast ;;
  fitbit)     run_fitbit ;;
  *) usage ;;
esac
