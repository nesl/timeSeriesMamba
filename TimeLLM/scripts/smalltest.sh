model_name=TimeLLM
train_epochs=1
learning_rate=0.01
llama_layers=2

master_port=01079
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
    --train_epochs $train_epochs \
    --model_comment $comment \
    --save_checkpoints 0 \
    --llm_model LLAMA3.1 \
    --llm_dim $d_model \
    --num_params "7b" \
    --use_wandb 0 \
    --seed 1