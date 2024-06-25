model_name=TimeLLM
train_epochs=3
learning_rate=0.01
llama_layers=6

master_port=01097
num_process=1
#2
batch_size=16
d_model=32
d_ff=128

# Function to display usage information
usage() {
  echo "Usage: $0 -l <llama_layers> -d <d_model> -e <epochs> -n <num_process>"
  exit 1
}

# Parse command-line arguments
while getopts "l:d:e:n:" opt; do
  case $opt in
    l) llama_layers=$OPTARG ;;
    d) d_model=$OPTARG ;;
    e) epochs=$OPTARG ;;
    n) num_process=$OPTARG ;;
    *) usage ;;
  esac
done

# Check if required arguments are provided
if [ -z "$llama_layers" ] || [ -z "$d_model" ] || [ -z "$epochs" ] || [ -z "$num_process" ]; then
  usage
fi
 
# Print the values to verify
echo "Setting llama_layers to $llama_layers"
echo "Setting d_model to $d_model"
echo "Setting epochs to $epochs"
echo "Setting num_process to $num_process"

comment="checkpoints/combine_l${llama_layers}_d${d_model}_e${epochs}_n${num_process}"
# Redirect output to a file named after the comment variable

log_file="results/combine_l${llama_layers}_d${d_model}_e${epochs}_n${num_process}.txt"
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
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment


echo "The script has completed. Output has been saved to $comment"