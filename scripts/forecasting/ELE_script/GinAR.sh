

mask_ratios=(0 0.25 0.5 0.75)
model_name=MSGNet
gpu=1

for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_model 256 \
    --d_ff 512 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --conv_channel 16 \
     --node_dim 100 \
    --mask_ratio $mr \
    --gpu $gpu
done