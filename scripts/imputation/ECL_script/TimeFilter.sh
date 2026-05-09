
mask_ratios=(0.25 0.5 0.75)
seq_len=96
pred_len=96
model_name=TimeFilter
gpu=2



for mr in "${mask_ratios[@]}"; do
  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity.csv \
  --model_id ECL_mask \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --patch_len 32 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --batch_size 16 \
  --train_epochs 15 \
  --d_model 512\
  --d_ff 512\
  --dropout 0.5 \
  --itr 1 \
  --mask_ratio $mr \
  --gpu $gpu 
done