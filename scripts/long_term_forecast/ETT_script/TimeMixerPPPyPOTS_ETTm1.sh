set -e

export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixerPPPyPOTS
run_tag=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

seq_len=96
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
num_kernels=3
batch_size=64
train_epochs=10
patience=3
num_workers=4

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id TMpp_best_${run_tag}'_ETTm1_'$seq_len'_'96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des TMpp_best_${run_tag} \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --num_kernels $num_kernels \
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
  --model_id TMpp_best_${run_tag}'_ETTm1_'$seq_len'_'192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des TMpp_best_${run_tag} \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --num_kernels $num_kernels \
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
  --model_id TMpp_best_${run_tag}'_ETTm1_'$seq_len'_'336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des TMpp_best_${run_tag} \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --num_kernels $num_kernels \
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
  --model_id TMpp_best_${run_tag}'_ETTm1_'$seq_len'_'720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des TMpp_best_${run_tag} \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --num_kernels $num_kernels \
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
