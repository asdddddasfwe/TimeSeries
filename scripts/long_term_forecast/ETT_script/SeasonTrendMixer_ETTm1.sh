#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

model_name=SeasonTrendMixer
run_tag=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}

seq_len=96
# Number of patch-token Transformer layers per fine seasonal scale.
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.001
dropout=0.2
weight_decay=0.0001
d_model=32
d_ff=64
n_heads=4
top_k=3
batch_size=32
train_epochs=30
patience=10
num_workers=4

# run.py sets the actual Python / NumPy / PyTorch seed to 2021.
# --seed is an augmentation argument, not the training seed.
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id "STMpatchconv_${run_tag}_ETTm1_${seq_len}_96" \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "STMpatchconv_${run_tag}" \
  --itr 1 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --moving_avg 25 \
  --decomp_method moving_avg \
  --use_norm 1 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --dropout $dropout \
  --weight_decay $weight_decay \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id "STMpatchconv_${run_tag}_ETTm1_${seq_len}_192" \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "STMpatchconv_${run_tag}" \
  --itr 1 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --moving_avg 25 \
  --decomp_method moving_avg \
  --use_norm 1 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --dropout $dropout \
  --weight_decay $weight_decay \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id "STMpatchconv_${run_tag}_ETTm1_${seq_len}_336" \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "STMpatchconv_${run_tag}" \
  --itr 1 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --moving_avg 25 \
  --decomp_method moving_avg \
  --use_norm 1 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --dropout $dropout \
  --weight_decay $weight_decay \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id "STMpatchconv_${run_tag}_ETTm1_${seq_len}_720" \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "STMpatchconv_${run_tag}" \
  --itr 1 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --moving_avg 25 \
  --decomp_method moving_avg \
  --use_norm 1 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --dropout $dropout \
  --weight_decay $weight_decay \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
