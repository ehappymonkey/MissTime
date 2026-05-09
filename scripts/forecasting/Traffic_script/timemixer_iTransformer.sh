
mask_ratios=(0.75)
gpu=3

model_name=TimeMixer

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64
batch_size=8

# for mr in "${mask_ratios[@]}"; do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --model_id Traffic_$seq_len'_'96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len 96 \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --itr 1 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --down_sampling_layers $down_sampling_layers \
#     --down_sampling_method avg \
#     --down_sampling_window $down_sampling_window \
#     --gpu $gpu \
#     --mask_ratio $mr 
# done


model_name=iTransformer


for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 4 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --itr 1 \
    --gpu $gpu \
    --mask_ratio $mr 
done


