mask_ratios=(0.25 0.5 0.75)
gpu=1


for mr in "${mask_ratios[@]}"; do
  python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path ./dataset/SMD \
    --model_id SMD \
    --model TimesNet \
    --data SMD \
    --features M \
    --seq_len 100 \
    --pred_len 0 \
    --d_model 64 \
    --d_ff 64 \
    --e_layers 2 \
    --enc_in 38 \
    --c_out 38 \
    --top_k 5 \
    --anomaly_ratio 0.5 \
    --batch_size 128 \
    --train_epochs 10 \
    --gpu $gpu \
    --mask_ratio $mr \
    --rag_type latent_rag \
    --retrieve_encoder Typology \
    --contrastive_loss hard_negative
done