mask_ratios=(0.25 0.5 0.75)
model_name=TimesNet
gpu=5

for mr in "${mask_ratios[@]}"; do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_mask \
    --mask_rate 0.125 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --batch_size 16 \
    --d_model 64 \
    --d_ff 64 \
    --des 'Exp' \
    --itr 1 \
    --top_k 3 \
    --learning_rate 0.001 \
    --mask_ratio $mr \
    --gpu $gpu \
    --rag_type latent_rag \
    --retrieve_encoder Typology \
    --contrastive_loss hard_negative \
    --encoder_epochs 50 \
    --gamma 1
done


for mr in "${mask_ratios[@]}"; do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_mask \
    --mask_rate 0.125 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --batch_size 16 \
    --d_model 64 \
    --d_ff 64 \
    --des 'Exp' \
    --itr 1 \
    --top_k 3 \
    --learning_rate 0.001 \
    --mask_ratio $mr \
    --gpu $gpu \
    --rag_type latent_rag \
    --retrieve_encoder iTransformer \
    --contrastive_loss normal \
    --encoder_epochs 50 \
    --gamma 1
done

