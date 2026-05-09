mask_ratios=(0 0.25 0.5 0.75)
seq_len=96
pred_len=12
model_name=TimeFilter
gpu=4


for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS \
    --data_path PEMS03.npz \
    --model_id PEMS03_$seq_len'_'$pred_len \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --patch_len 48 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 1024 \
    --dropout 0.1 \
    --top_p 0.0 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 20 \
    --itr 1 \
    --mask_ratio $mr \
    --gpu $gpu 
done

