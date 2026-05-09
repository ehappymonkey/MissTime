
mask_ratios=(0.75)
model_name=TimesNet
gpu=0

for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 4 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --itr 1 \
    --mask_ratio $mr \
    --gpu $gpu \
    --freq m \
    --rag_type latent_rag \
    --retrieve_encoder iTransformer \
    --contrastive_loss hard_negative \
    --contrastive_batch 64 \
    --encoder_epochs 50 \
    --gamma 0.1


    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 4 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --itr 1 \
    --mask_ratio $mr \
    --gpu $gpu \
    --freq m \
    --rag_type latent_rag \
    --retrieve_encoder iTransformer \
    --contrastive_loss normal \
    --contrastive_batch 64 \
    --encoder_epochs 50 \
    --gamma 0.5
done
