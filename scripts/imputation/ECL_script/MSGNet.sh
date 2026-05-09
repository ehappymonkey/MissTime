
mask_ratios=(0.25 0.5 0.75)

seq_len=96
label_len=48
model_name=MSGNet
pred_len=96
gpu=4


for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id electricity_mask \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
    --target 'OT' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 384 \
    --d_ff 64 \
    --top_k 3 \
    --conv_channel 8 \
    --skip_channel 16 \
    --batch_size 128 \
    --itr 1 \
    --gpu $gpu \
    --mask_ratio $mr 
done