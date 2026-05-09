mask_ratios=(0.25 0.5 0.75)
model_name=iTransformer
gpu=7


model_name=iTransformer

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
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --d_model 512\
    --d_ff 512\
    --itr 1 \
    --gpu $gpu \
    --mask_ratio $mr 
done

