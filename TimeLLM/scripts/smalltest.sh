model_name=TimeLLM
train_epochs=3
learning_rate=0.01
llm_layers=0

master_port=01180
num_process=1
batch_size=16
d_model=32
d_ff=32
comment='checkpoints/debugTest'
gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port seed_process.py \
   --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./dataset/traffic/ \
   --data_path traffic_train.csv \
   --data_path_test traffic_test.csv \
   --data_path_val traffic_val.csv \
   --model_id debug_run \
   --model $model_name \
   --data Traffic \
   --features M \
   --seq_len 512 \
   --label_len 48 \
   --pred_len 96 \
   --factor 3 \
   --enc_in 862 \
   --dec_in 862 \
   --c_out 862 \
   --des 'Exp' \
   --itr 1 \
   --llm_layers $llm_layers \
   --d_ff $d_ff \
   --batch_size $batch_size \
   --learning_rate $learning_rate \
   --train_epochs $train_epochs \
   --patience 10 \
   --model_comment $comment \
   --save_checkpoints 0 \
   --llm_model "Mamba2" \
   --llm_dim 768 \
   --num_params "130m" \
   --use_wandb 0 \
   --dsampfactor 1 \
   --percent 100 \
   --col_percent 100 \
   --rand_init 0 \
   --seed 42 \
   --verbose 1 \
   --rand_init 0 \
   #--split_type 'temporal'
