  #!/bin/bash
  # run_pems_hourly.sh — launch TimeLLM/DLinear/ARIMA on outputs from pems_entropy_pipeline.py

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

  # Required: -m <llm_model> -g <gpu_id> -h <heldout:{low_entropy|high_entropy}> -s <source_type:{R1|R3|R6}>
  # Optional: -n <num_params> -p <master_port> -r <rand_init>
  usage() {
    echo "Usage: $0 -m <llm_model> -g <gpu_id> -h <heldout:{low_entropy|high_entropy}> -s <source_type:{R1|R3|R6}> [-n <num_params>] [-p <master_port>] [-r <rand_init>]"
    exit 1
  }

  # Defaults
  master_port_base=1180          # avoid leading zero
  downsampling_factor=1
  percent=100
  col_percent=100
  save_checkpoints=1

  heldout=""
  source_type="PEMS"                 
  llm_model=""

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

  # Required checks
  if [ -z "$llm_model" ] || [ -z "$gpu_id" ] || [ -z "$heldout" ] || [ -z "$source_type" ]; then
    usage
  fi
  if [[ "$heldout" != "low" && "$heldout" != "medium" && "$heldout" != "high" ]]; then
    echo "heldout must be one of: low | medium | high"; exit 1
  fi
  # Model name normalization
  if [ "$llm_model" == "DLinear" ]; then
    model_name="DLinear"
  elif [ "$llm_model" == "ARIMA" ]; then
    model_name="ARIMA"
  fi

  # Master port
  if [ -z "$master_port" ]; then
    master_port="${master_port_base}${gpu_id}"
  fi
  export CUDA_VISIBLE_DEVICES=$((gpu_id % 4))

  echo "Using model: $llm_model"
  echo "Heldout: $heldout"
  echo "Train source: $source_type"
  echo "Num params: $num_params"
  echo "Master port: $master_port"
  echo "Rand init: $rand_init"

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

  # Seq/pred config (hourly data)
  seq_len=$((512 / downsampling_factor))
  pred_len=$((96 / downsampling_factor))

  # Paths to your generated files
  ROOT="dataset/traffic/outputs_pems_hourly"
  TRAIN_CSV="${ROOT}/train_univariate.csv"                 
  BOUNDARY_JSON="${ROOT}/train_univariate_boundary.json"   
  TEST_CSV="${ROOT}/test_${heldout}.csv"                       # e.g., test_low.csv

  if [ ! -f "$TRAIN_CSV" ]; then echo "Missing $TRAIN_CSV"; exit 1; fi
  if [ ! -f "$BOUNDARY_JSON" ]; then echo "Missing $BOUNDARY_JSON"; exit 1; fi
  if [ ! -f "$TEST_CSV" ]; then echo "Missing $TEST_CSV"; exit 1; fi

  # Infer channel count from CSV header (number of columns)
  col_count=$(head -n 1 "$TRAIN_CSV" | awk -F',' '{print NF-1}')
  # If index is written as first column, we need actual sensor columns:
  # try to detect if first header cell looks like a datetime label
  first_header=$(head -n 1 "$TRAIN_CSV" | awk -F',' '{print $1}')
  if [[ "$first_header" =~ date|timestamp|time|^Unnamed ]]; then
    enc_in=$((col_count))   # NF-1 already excluded index; keep as-is
  else
    # If index is in column 1 as an unlabeled index, still treat NF-1 as channels
    enc_in=$((col_count))
  fi
  dec_in=$enc_in
  c_out=$enc_in

  echo "Detected channels: enc_in=dec_in=c_out=${enc_in}"

  # Tagging
  og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
  if [ "$llm_model" == "DLinear" ]; then
    og_tag="DLinear_l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}_f${downsampling_factor}_t${percent}_c${col_percent}_r${rand_init}"
  fi

  for seed in {1..3}; do
    for init_seed in {11..13}; do
      tag="pems_${source_type}_${heldout}_${og_tag}_seq${seq_len}_pred${pred_len}_seed${seed}_init${init_seed}"
      comment="checkpoints/${tag}"
      log_file="results/pems/${tag}.txt"
      mkdir -p "$(dirname "$log_file")" "$(dirname "$comment")"

      echo "Logging to $log_file"
      exec > "$log_file" 2>&1

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
        --model_comment "$comment" \
        --llm_model $llm_model \
        --llm_dim $llm_dim \
        --num_params $num_params \
        --boundary_file $BOUNDARY_JSON \
        --rand_init $rand_init \
        --seed $seed \
        --init_seed $init_seed \
        --save_checkpoints $save_checkpoints \
        --source $source_type

      echo "${heldout} heldout (${source_type}) with init_seed $init_seed and seed $seed completed, saved to $comment"

      # If weights are fixed and not randomly re-initialized, break inner loop
      if [[ "$rand_init" -eq 0 ]]; then
        break
      fi
    done
  done
