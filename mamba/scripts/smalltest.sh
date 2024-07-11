model_name=TimeLLM
train_epochs=1
learning_rate=0.01
llama_layers=2

master_port=01097
num_process=1
#2
batch_size=16
d_model=32
d_ff=128



comment='checkpoints/smallTest'


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
  --model_comment $comment \
  --early_break 1
  #--use_amp 

