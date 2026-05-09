
mask_ratios=(0.25 0.5 0.75)
model_name="TimesNet"
gpu=7



for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data traffic \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --d_model 256 \
    --d_ff 256 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --mask_ratio $mr \
    --gpu $gpu \
    --rag_type latent_rag \
    --retrieve_encoder iTransformer \
    --contrastive_loss hard_negative \
    --contrastive_batch 64 \
    --encoder_epochs 50 \
    --gamma 0.5

    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data traffic \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --d_model 256 \
    --d_ff 256 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --mask_ratio $mr \
    --gpu $gpu \
    --rag_type latent_rag \
    --retrieve_encoder iTransformer \
    --contrastive_loss normal \
    --contrastive_batch 64 \
    --encoder_epochs 50 \
    --gamma 0.5

    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data traffic \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --d_model 256 \
    --d_ff 256 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --mask_ratio $mr \
    --gpu $gpu \
    --rag_type latent_rag \
    --retrieve_encoder Typology \
    --contrastive_loss normal \
    --contrastive_batch 64 \
    --encoder_epochs 50 \
    --gamma 0.5
done

