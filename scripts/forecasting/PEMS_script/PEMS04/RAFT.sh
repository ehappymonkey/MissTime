mask_ratios=(0 0.25 0.5 0.75)
model_name=RAFT
gpu=5

for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 4 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 96 \
    --d_ff 128 \
    --learning_rate 0.0005 \
    --itr 1 \
    --use_norm 0 \
    --topm 100 \
    --temperature 1000 \
    --mask_ratio $mr \
    --gpu $gpu \
    --freq m 
done