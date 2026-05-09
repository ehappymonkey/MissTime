  
mask_ratios=(0.25 0.5 0.75)
seq_len=96
pred_len=96
model_name=TimeFilter
gpu=3


for mr in "${mask_ratios[@]}"; do
  python -u run.py \
    --task_name imputation \
    --is_training 2 \
    --root_path ./dataset/weather \
    --data_path weather.csv \
    --model_id weather_mask \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --patch_len 48 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.3 \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --itr 1 \
    --mask_ratio $mr \
    --gpu $gpu 
done
