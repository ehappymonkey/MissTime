
mask_ratios=(0 0.25 0.5 0.75)
model_name=saits
gpu=0


for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 2 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --itr 1 \
    --use_norm 0 \
    --mask_ratio $mr \
    --gpu $gpu \
    --rag_type no_rag \
    --freq m
done