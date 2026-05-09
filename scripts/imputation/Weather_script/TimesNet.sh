mask_ratios=(0 0.25 0.5 0.75)
model_name=TimesNet
gpu=1


for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_mask_0.125 \
    --mask_rate 0.125 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --batch_size 16 \
    --d_model 64 \
    --d_ff 64 \
    --des 'Exp' \
    --itr 1 \
    --top_k 3 \
    --learning_rate 0.001 \
    --mask_ratio $mr \
    --gpu $gpu 
done
