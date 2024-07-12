model_name=TimeLLM
train_epochs=3
learning_rate=0.01
llm_layers=6

master_port=01097
num_params=1
batch_size=16
d_model=32
d_ff=128
num_params='2.8b'

# Function to display usage information
usage() {
  echo "Usage: $0 -l <llm_layers> -d <d_model> -e <train_epochs> -n <num_params> -c <save_checkpoints> -m <llm_model>"
  exit 1
}

# Parse command-line arguments
while getopts "l:d:e:n:c:m:" opt; do
  case $opt in
    l) llm_layers=$OPTARG ;;
    d) d_model=$OPTARG ;;
    e) train_epochs=$OPTARG ;;
    n) num_params=$OPTARG ;;
    c) save_checkpoints=$OPTARG ;;
    m) llm_model=$OPTARG ;;
    *) usage ;;
  esac
done


# Check if required arguments are provided
if [ -z "$llm_layers" ] || [ -z "$d_model" ] || [ -z "$train_epochs" ] || [ -z "$num_params" ] || [ -z "$save_checkpoints" ] || [ -z "$llm_model" ]; then
  usage
fi
 
# Print the values to verify
echo "Setting llm_layers to $llm_layers"
echo "Setting d_model to $d_model"
echo "Setting train_epochs to $epochs"
echo "Setting num_params to $num_params"
echo "Setting save_checkpoints to $save_checkpoints"
echo "Setting llm_model to $llm_model"

og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}"

llm_dim=0
if [ "$llm_model" = "Mamba" ]; then
  llm_dim=768
fi
if [ "$llm_model" = "Mamba2" ]; then
  llm_dim=2560
fi

# Redirect output to a file named after the comment variable

tag="ETTh1_${og_tag}"
comment="checkpoints/${tag}"
log_file="results/${tag}.txt"
exec > "$log_file" 2>&1

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port $master_port combine_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoints $save_checkpoints \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --num_params $num_params

echo "ETTh1 completed, saved to $comment"


tag="ETTh2_${og_tag}"
comment="checkpoints/${tag}"
log_file="results/${tag}.txt"
exec > "$log_file" 2>&1

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port $master_port combine_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoints $save_checkpoints \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --num_params $num_params

echo "ETTh2 completed, saved to $comment"

tag="ETTm1_${og_tag}"
comment="checkpoints/${tag}"
log_file="results/${tag}.txt"
exec > "$log_file" 2>&1

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port $master_port combine_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoints $save_checkpoints \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --num_params $num_params

echo "ETTm1 completed, saved to $comment"


tag="ETTm2_${og_tag}"
comment="checkpoints/${tag}"
log_file="results/${tag}.txt"
exec > "$log_file" 2>&1

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port $master_port combine_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoints $save_checkpoints \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --num_params $num_params

echo "ETTm2 completed, saved to $comment"


tag="ECL_${og_tag}"
comment="checkpoints/${tag}"
log_file="results/${tag}.txt"
exec > "$log_file" 2>&1

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port $master_port combine_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_96 \
  --model $model_name \
  --data ECL \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment\
  --save_checkpoints $save_checkpoints \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --num_params $num_params

echo "ECL completed, saved to $comment"


tag="Weather_${og_tag}"
comment="checkpoints/${tag}"
log_file="results/${tag}.txt"
exec > "$log_file" 2>&1

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port $master_port combine_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_96 \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
  --save_checkpoints $save_checkpoints \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --num_params $num_params

echo "Weather completed, saved to $comment"


tag="Traffic_${og_tag}"
comment="checkpoints/${tag}"
log_file="results/${tag}.txt"
exec > "$log_file" 2>&1

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port $master_port combine_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_512_96 \
  --model $model_name \
  --data Traffic \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --save_checkpoints $save_checkpoints \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --num_params $num_params

echo "Traffic completed, saved to $comment"

