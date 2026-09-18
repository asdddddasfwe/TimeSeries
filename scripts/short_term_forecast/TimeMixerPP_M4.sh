export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixerPP

e_layers=2
down_sampling_layers=1
down_sampling_window=2
learning_rate=0.001
d_model=32
d_ff=64
n_heads=4
top_k=3
batch_size=16
train_epochs=20
patience=5
num_workers=4

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns 'Yearly' \
  --model_id TimeMixerPP_M4_Yearly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
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
  --down_sampling_window $down_sampling_window \
  --loss 'SMAPE'


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns 'Quarterly' \
  --model_id TimeMixerPP_M4_Quarterly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
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
  --down_sampling_window $down_sampling_window \
  --loss 'SMAPE'


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns 'Monthly' \
  --model_id TimeMixerPP_M4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
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
  --down_sampling_window $down_sampling_window \
  --loss 'SMAPE'


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns 'Weekly' \
  --model_id TimeMixerPP_M4_Weekly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
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
  --down_sampling_window $down_sampling_window \
  --loss 'SMAPE'


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns 'Daily' \
  --model_id TimeMixerPP_M4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
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
  --down_sampling_window $down_sampling_window \
  --loss 'SMAPE'


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns 'Hourly' \
  --model_id TimeMixerPP_M4_Hourly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
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
  --down_sampling_window $down_sampling_window \
  --loss 'SMAPE'
