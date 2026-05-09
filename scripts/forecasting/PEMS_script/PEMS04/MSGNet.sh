
mask_ratios=(0 0.25 0.5 0.75)

seq_len=96
label_len=48
model_name=MSGNet
pred_len=12
gpu=6


for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --freq m \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 32 \
    --top_k 3 \
    --conv_channel 8 \
    --skip_channel 16 \
    --batch_size 128 \
    --itr 1 \
    --gpu $gpu \
    --mask_ratio $mr 
done