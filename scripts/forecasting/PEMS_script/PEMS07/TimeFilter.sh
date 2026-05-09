mask_ratios=(0 0.25 0.5 0.75)
seq_len=96
pred_len=12
model_name=TimeFilter
gpu=6

for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS \
    --data_path PEMS07.npz \
    --model_id PEMS07_$seq_len'_'$pred_len \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --patch_len 96 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 1024 \
    --dropout 0.1 \
    --top_p 0.0 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 20 \
    --itr 1 \
    --use_norm 0 \
    --mask_ratio $mr \
    --gpu $gpu 
done