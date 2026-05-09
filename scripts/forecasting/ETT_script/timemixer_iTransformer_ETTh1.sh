mask_ratios=(0 0.25 0.5 0.75)
gpu=0
model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=16

for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path  ./dataset/ETT-small/\
    --data_path ETTh1.csv \
    --model_id ETTh1_$seq_len'_'192 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len 192 \
    --e_layers $e_layers \
    --enc_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 128 \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --gpu $gpu \
    --mask_ratio $mr 
done


model_name=iTransformer

for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512\
    --d_ff 512 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --itr 1
    --mask_ratio $mr \
    --gpu $gpu 
done