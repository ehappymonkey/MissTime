  
  
mask_ratios=(0 0.25 0.5 0.75)
seq_len=96
pred_len=96
model_name=TimeFilter
gpu=3

for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic \
    --data_path traffic.csv \
    --model_id traffic_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --patch_len 96 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 2048 \
    --dropout 0.3 \
    --top_p 0.0 \
    --pos 0 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 30 \
    --itr 1 \
    --mask_ratio $mr \
    --gpu $gpu 
done