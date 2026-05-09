mask_ratios=(0.2 0.4 0.6 0.8)
model_name="TimesNet"
gpu=7

for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --itr 1 \
    --top_k 5 \
    --mask_ratio $mr \
    --gpu $gpu \
    --batch_size 128
done


# for mr in "${mask_ratios[@]}"; do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_96_96 \
#     --model $model_name \
#     --data ETTh1 \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --d_model 16 \
#     --d_ff 32 \
#     --des 'Exp' \
#     --itr 1 \
#     --top_k 5 \
#     --mask_ratio $mr \
#     --gpu $gpu \
#     --use_full_retrieval
# done

# for mr in "${mask_ratios[@]}"; do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_96_96 \
#     --model $model_name \
#     --data ETTh1 \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --d_model 16 \
#     --d_ff 32 \
#     --des 'Exp' \
#     --itr 1 \
#     --top_k 5 \
#     --mask_ratio $mr \
#     --gpu $gpu \
#     --use_latent_retrieval
# done