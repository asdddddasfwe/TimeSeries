export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixerPP

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.02
d_model=32
d_ff=64
n_heads=4
top_k=3
batch_size=128
train_epochs=10
patience=3
num_workers=4

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id TimeMixerPP_ETTm1_$seq_len'_'96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'TimeMixerPP' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --channel_independence 0 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id TimeMixerPP_ETTm1_$seq_len'_'192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'TimeMixerPP' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --channel_independence 0 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id TimeMixerPP_ETTm1_$seq_len'_'336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'TimeMixerPP' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --channel_independence 0 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id TimeMixerPP_ETTm1_$seq_len'_'720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers $e_layers \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'TimeMixerPP' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --top_k $top_k \
  --channel_independence 0 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj cosine \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window
