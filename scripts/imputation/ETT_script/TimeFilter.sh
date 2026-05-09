
mask_ratios=(0.25 0.5 0.75)
seq_len=96
pred_len=96
model_name=TimeFilter
gpu=1


for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1_mask \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --dropout 0.8 \
    --patch_len 2 \
    --pos 0 \
    --des 'Exp' \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --train_epochs 10 \
    --d_model 128 \
    --d_ff 256 \
    --itr 1 \
    --mask_ratio $mr \
    --gpu $gpu 
done