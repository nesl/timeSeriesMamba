model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

master_port=01099
num_process=1
batch_size=24
d_model=32
d_ff=128

comment='checkpoints/smallTest'
gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_process.py \
   --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./dataset/ETT-small/ \
   --data_path ETTh1.csv \
   --model_id ETTh1_512_96 \
   --model $model_name \
   --data ETTh1 \
   --features M \
   --seq_len 512 \
   --label_len 96 \
   --pred_len 96 \
   --factor 3 \
   --enc_in 7 \
   --dec_in 7 \
   --c_out 7 \
   --des 'Exp' \
   --itr 1 \
   --llm_layers $llama_layers \
   --d_ff $d_ff \
   --batch_size $batch_size \
   --learning_rate $learning_rate \
   --train_epochs $train_epochs \
   --patience 10 \
   --model_comment $comment \
   --save_checkpoints 0 \
   --llm_model LLAMA \
   --llm_dim 4096 \
   --num_params "none" \
   --use_wandb 0 \
   --dsampfactor 1 \
   --percent 100 \
   --col_percent 100 \
   --rand_init 0 \
   --seed 2021 \
   --verbose 1