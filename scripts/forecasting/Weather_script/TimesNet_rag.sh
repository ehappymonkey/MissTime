mask_ratios=(0.25 0.5 0.75)
model_name="TimesNet"
gpu=2


for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_96 \
    --model $model_name \
    --data weather \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --d_model 32 \
    --d_ff 32 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --mask_ratio $mr \
    --gpu $gpu \
    --rag_type feature_rag \
    --retrieve_encoder Typology \
    --latent_dim 512 \
    --contrastive_loss normal
done