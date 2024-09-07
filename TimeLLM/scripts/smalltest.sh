model_name=TimeLLM
train_epochs=1
learning_rate=0.01
llama_layers=2

master_port=01073
num_process=1
batch_size=16
d_model=32
d_ff=128

comment='checkpoints/smallTest'
gpu_id=3
export CUDA_VISIBLE_DEVICES=$gpu_id

 accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_process.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id "ETTh1_128_48" \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 128 \
      --label_len 24 \
      --pred_len 64 \
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
      --llm_layers 2 \
      --train_epochs $train_epochs \
      --model_comment "$comment" \
      --llm_model Mamba2 \
      --llm_dim 768 \
      --period_of_interest "1 month" \
      --use_wandb 0
: ' 
accelerate launch --mixed_precision bf16 --num_processes 1 --gpu_ids $gpu_id --main_process_port $master_port seed_process.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_512_96 \
  --model $model_name \
  --data Traffic \
  --features M \
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
  --early_break 1 \
  --llm_model Mamba2 \
  --llm_dim 768 \
  --period_of_interest "1 month" \
  --use_wandb 0


accelerate launch --mixed_precision bf16 --num_processes 1 --gpu_ids $gpu_id --main_process_port $master_port seed_process.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_96_24 \
  --model $model_name \
  --data Illness \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --freq '10m' \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --early_break 1 \
  --llm_model Mamba2 \
  --llm_dim 768 \
  --use_wandb 0
'

: '
accelerate launch --mixed_precision bf16 --num_processes 1 --gpu_ids $gpu_id --main_process_port $master_port seed_process.py \
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
  --model_comment $comment \
  --early_break 1 \
  --llm_model Mamba2 \
  --llm_dim 768 \
  --use_wandb 0
'
: '
accelerate launch --mixed_precision bf16 --num_processes 1 --gpu_ids $gpu_id --main_process_port $master_port seed_process.py \
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
  --model_comment $comment \
  --early_break 1 \
  --llm_model Mamba2 \
  --llm_dim 2560 \
  --use_wandb 0
  #--use_amp 

'