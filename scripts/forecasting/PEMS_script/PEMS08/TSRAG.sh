
mask_ratios=(0 0.25 0.5 0.75)
model_name=TSRAG
gpu=7

for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id PEMS08_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 2 \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 256 \
    --itr 1 \
    --use_norm 1 \
    --freq m \
    --mask_ratio $mr \
    --gpu $gpu \
    --rag_type no_rag \
    --freq m
done