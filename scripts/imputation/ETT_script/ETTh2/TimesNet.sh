mask_ratios=(0.25 0.5 0.75)
model_name=TimesNet
gpu=6


for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_mask_0.125 \
    --mask_rate 0.125 \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --batch_size 16 \
    --d_model 32 \
    --d_ff 32 \
    --des 'Exp' \
    --itr 1 \
    --top_k 3 \
    --learning_rate 0.001 \
    --mask_ratio $mr \
    --gpu $gpu
done