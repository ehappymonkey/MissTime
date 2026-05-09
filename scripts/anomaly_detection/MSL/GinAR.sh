mask_ratios=(0 0.2 0.4 0.6 0.8)
model_name=GinAR
gpu=0

for mr in "${mask_ratios[@]}"; do
  python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path ./dataset/MSL \
    --model_id MSL \
    --model $model_name \
    --data MSL \
    --features M \
    --seq_len 100 \
    --pred_len 0 \
    --d_model 8 \
    --d_ff 16 \
    --e_layers 1 \
    --enc_in 55 \
    --c_out 55 \
    --top_k 3 \
    --anomaly_ratio 1 \
    --batch_size 128 \
    --train_epochs 1 \
    --mask_ratio $mr \
    --gpu $gpu 
done
