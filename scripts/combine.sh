model_name=TimeLLM
train_epochs=3
learning_rate=0.01
llm_layers=6

master_port=01097
num_process=1
#2
batch_size=16
d_model=32
d_ff=128


# Function to display usage information
usage() {
  echo "Usage: $0 -l <llm_layers> -d <d_model> -e <train_epochs> -n <num_process> -c <save_checkpoints> -m <llm_model>"
  exit 1
}

# Parse command-line arguments
while getopts "l:d:e:n:c:m:" opt; do
  case $opt in
    l) llm_layers=$OPTARG ;;
    d) d_model=$OPTARG ;;
    e) train_epochs=$OPTARG ;;
    n) num_process=$OPTARG ;;
    c) save_checkpoints=$OPTARG ;;
    m) llm_model=$OPTARG ;;
    *) usage ;;
  esac
done


# Check if required arguments are provided
if [ -z "$llm_layers" ] || [ -z "$d_model" ] || [ -z "$train_epochs" ] || [ -z "$num_process" ] || [ -z "$save_checkpoints" ] || [ -z "$llm_model" ]; then
  usage
fi
 
# Print the values to verify
echo "Setting llm_layers to $llm_layers"
echo "Setting d_model to $d_model"
echo "Setting train_epochs to $epochs"
echo "Setting num_process to $num_process"
echo "Setting save_checkpoints to $save_checkpoints"
echo "Setting llm_model to $llm_model"

tag="combine_l${llm_layers}_d${d_model}_e${train_epochs}_n${num_process}_m${llm_model}"
comment="checkpoints/${tag}"
# Redirect output to a file named after the comment variable

log_file="results/${tag}.txt"
exec > "$log_file" 2>&1

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port combine_main.py \
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
  --llm_model $llm_model


echo "The script has completed. Output has been saved to $comment"