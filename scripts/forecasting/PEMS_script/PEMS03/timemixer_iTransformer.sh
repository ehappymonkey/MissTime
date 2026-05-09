mask_ratios=(0 0.25 0.5 0.75)
gpu=4


model_name=TimeMixer

seq_len=96
pred_len=12
down_sampling_layers=1
down_sampling_window=2
learning_rate=0.003
d_model=128
d_ff=256
batch_size=16
train_epochs=10
patience=10


# for mr in "${mask_ratios[@]}"; do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS03.npz \
#     --model_id PEMS03 \
#     --model $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --e_layers 5 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 358 \
#     --dec_in 358 \
#     --c_out 358 \
#     --des 'Exp' \
#     --itr 1 \
#     --use_norm 0 \
#     --channel_independence 0 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --batch_size 32 \
#     --learning_rate $learning_rate \
#     --train_epochs $train_epochs \
#     --patience $patience \
#     --down_sampling_layers $down_sampling_layers \
#     --down_sampling_method avg \
#     --down_sampling_window $down_sampling_window \
#     --freq m \
#     --gpu $gpu \
#     --mask_ratio $mr 
# done


model_name=iTransformer


for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 4 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --learning_rate 0.0005 \
    --itr 1 \
    --use_norm 0 \
    --freq m \
    --gpu $gpu \
    --mask_ratio $mr 
done