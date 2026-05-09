mask_ratios=(0.25 0.5 0.75)
gpu=1


model_name=TimeMixer

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=16
train_epochs=20
patience=10


for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_mask \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len 96 \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size 128 \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --gpu $gpu \
    --mask_ratio $mr 
done

