mask_ratios=(0 0.2 0.4 0.6 0.8)
model_name=TimesNet
gpu=2

# for mr in "${mask_ratios[@]}"; do
#   python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTm1.csv \
#     --model_id ETTm1_96_96 \
#     --model $model_name \
#     --data ETTm1 \
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
#     --des 'Exp' \
#     --d_model 64 \
#     --d_ff 64 \
#     --top_k 5 \
#     --itr 1 \
#     --mask_ratio $mr \
#     --gpu $gpu \
#     --rag_type no_rag 
# done

# for mr in "${mask_ratios[@]}"; do
#   python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTm1.csv \
#     --model_id ETTm1_96_96 \
#     --model $model_name \
#     --data ETTm1 \
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
#     --des 'Exp' \
#     --d_model 64 \
#     --d_ff 64 \
#     --top_k 5 \
#     --itr 1 \
#     --mask_ratio $mr \
#     --gpu $gpu \
#     --rag_type feature_rag 
# done


  # python -u run.py \
  #   --task_name long_term_forecast \
  #   --is_training 1 \
  #   --root_path ./dataset/ETT-small/ \
  #   --data_path ETTm1.csv \
  #   --model_id ETTm1_96_192 \
  #   --model $model_name \
  #   --data ETTm1 \
  #   --features M \
  #   --seq_len 96 \
  #   --label_len 48 \
  #   --pred_len 192 \
  #   --e_layers 2 \
  #   --d_layers 1 \
  #   --factor 3 \
  #   --enc_in 7 \
  #   --dec_in 7 \
  #   --c_out 7 \
  #   --des 'Exp' \
  #   --d_model 64 \
  #   --d_ff 64 \
  #   --top_k 5 \
  #   --itr 1 \
  #   --mask_ratio $mr \
  #   --gpu $gpu \
  #   --use_latent_retrieval


  # python -u run.py \
  #   --task_name long_term_forecast \
  #   --is_training 1 \
  #   --root_path ./dataset/ETT-small/ \
  #   --data_path ETTm1.csv \
  #   --model_id ETTm1_96_336 \
  #   --model $model_name \
  #   --data ETTm1 \
  #   --features M \
  #   --seq_len 96 \
  #   --label_len 48 \
  #   --pred_len 336 \
  #   --e_layers 2 \
  #   --d_layers 1 \
  #   --factor 3 \
  #   --enc_in 7 \
  #   --dec_in 7 \
  #   --c_out 7 \
  #   --des 'Exp' \
  #   --d_model 16 \
  #   --d_ff 32 \
  #   --top_k 5 \
  #   --itr 1 \
  #   --train_epochs 3 \
  #   --mask_ratio $mr \
  #   --gpu $gpu \
  #   --use_latent_retrieval

for mr in "${mask_ratios[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_96 \
    --model $model_name \
    --data ETTm1 \
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
    --des 'Exp' \
    --d_model 16 \
    --d_ff 32 \
    --top_k 5 \
    --itr 1 \
    --mask_ratio $mr \
    --gpu $gpu \
    --rag_type latent_rag \
    --retrieve_encoder Typology \
    --contrastive_loss normal
done