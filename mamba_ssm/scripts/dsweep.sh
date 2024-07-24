model_name=BackboneModel
train_epochs=3
learning_rate=0.01
llm_layers=6

master_port=01096
batch_size=16
d_ff=128
num_params='2.7b'

llm_dim=768

gpu_id=0

# Function to display usage information
usage() {
  echo "Usage: $0 -l <llm_layers> -e <train_epochs> -n <num_params> -c <save_checkpoints> -m <llm_model>"
  exit 1
}

# Parse command-line arguments
while getopts "l:e:n:c:m:" opt; do
  case $opt in
    l) llm_layers=$OPTARG ;;
    e) train_epochs=$OPTARG ;;
    n) num_params=$OPTARG ;;
    c) save_checkpoints=$OPTARG ;;
    m) llm_model=$OPTARG ;;
    *) usage ;;
  esac
done


# Check if required arguments are provided
if [ -z "$llm_layers" ] || [ -z "$train_epochs" ] || [ -z "$num_params" ] || [ -z "$save_checkpoints" ] || [ -z "$llm_model" ]; then
  usage
fi
# Array of d_model values
d_model_values=(256 512 1024 2048)

# Loop through each value in the array
for d_model in "${d_model_values[@]}"
do

  # Print the values to verify
  echo "Setting llm_layers to $llm_layers"
  echo "Setting d_model to $d_model"
  echo "Setting train_epochs to $epochs"
  echo "Setting num_params to $num_params"
  echo "Setting save_checkpoints to $save_checkpoints"
  echo "Setting llm_model to $llm_model"

  og_tag="l${llm_layers}_d${d_model}_e${train_epochs}_m${llm_model}_n${num_params}"


  # Redirect output to a file named after the comment variable

  tag="dsweep_${og_tag}"
  comment="checkpoints/${tag}"
  log_file="results/${tag}.txt"
  exec > "$log_file" 2>&1

  accelerate launch --mixed_precision bf16 --num_processes 1 --gpu_ids $gpu_id --main_process_port $master_port train.py \
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
    --n_layer $llm_layers \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --save_checkpoints $save_checkpoints \
    --llm_model $llm_model \
    --llm_dim $d_model \
    --num_params $num_params \
    --use_wandb 1


  echo "ETTh1 completed, saved to $comment"
done 