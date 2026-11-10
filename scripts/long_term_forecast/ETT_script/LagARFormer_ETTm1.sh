#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

model_name=LagARFormer
run_tag=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}

# -1 restores the original shared decay for a matched baseline.
lag_relation_weight_decay=${LAG_RELATION_WEIGHT_DECAY:-0}
optimizer=${OPTIMIZER:-adam}
seq_len=96
# Four resolutions; native-rate lag messages; learned scale fusion.
# d_ff is Q/K width; n_heads is unused by the upstream axial modules.
e_layers=2
learning_rate=0.01
dropout=0.2
weight_decay=0.0001
d_model=32
d_ff=64
top_k=5
batch_size=64
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
  --model_id "LARmulti_v5_${optimizer}_acfwd${lag_relation_weight_decay}_${run_tag}_ETTm1_${seq_len}_96" \
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
  --des "LARmulti_v5_${optimizer}_acfwd${lag_relation_weight_decay}_${run_tag}" \
  --itr 1 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --top_k $top_k \
  --moving_avg 25 \
  --decomp_method moving_avg \
  --use_norm 1 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --dropout $dropout \
  --optimizer $optimizer \
  --weight_decay $weight_decay \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lag_max 0 \
  --lag_layers 2 \
  --lag_chunk_size 8 \
  --lag_relation_weight_decay $lag_relation_weight_decay \
  --lag_monitor \
  --lag_monitor_samples 16

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id "LARmulti_v5_${optimizer}_acfwd${lag_relation_weight_decay}_${run_tag}_ETTm1_${seq_len}_192" \
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
  --des "LARmulti_v5_${optimizer}_acfwd${lag_relation_weight_decay}_${run_tag}" \
  --itr 1 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --top_k $top_k \
  --moving_avg 25 \
  --decomp_method moving_avg \
  --use_norm 1 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --dropout $dropout \
  --optimizer $optimizer \
  --weight_decay $weight_decay \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lag_max 0 \
  --lag_layers 2 \
  --lag_chunk_size 8 \
  --lag_relation_weight_decay $lag_relation_weight_decay \
  --lag_monitor \
  --lag_monitor_samples 16

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id "LARmulti_v5_${optimizer}_acfwd${lag_relation_weight_decay}_${run_tag}_ETTm1_${seq_len}_336" \
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
  --des "LARmulti_v5_${optimizer}_acfwd${lag_relation_weight_decay}_${run_tag}" \
  --itr 1 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --top_k $top_k \
  --moving_avg 25 \
  --decomp_method moving_avg \
  --use_norm 1 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --dropout $dropout \
  --optimizer $optimizer \
  --weight_decay $weight_decay \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lag_max 0 \
  --lag_layers 2 \
  --lag_chunk_size 8 \
  --lag_relation_weight_decay $lag_relation_weight_decay \
  --lag_monitor \
  --lag_monitor_samples 16

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id "LARmulti_v5_${optimizer}_acfwd${lag_relation_weight_decay}_${run_tag}_ETTm1_${seq_len}_720" \
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
  --des "LARmulti_v5_${optimizer}_acfwd${lag_relation_weight_decay}_${run_tag}" \
  --itr 1 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --top_k $top_k \
  --moving_avg 25 \
  --decomp_method moving_avg \
  --use_norm 1 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --dropout $dropout \
  --optimizer $optimizer \
  --weight_decay $weight_decay \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lag_max 0 \
  --lag_layers 2 \
  --lag_chunk_size 8 \
  --lag_relation_weight_decay $lag_relation_weight_decay \
  --lag_monitor \
  --lag_monitor_samples 16
