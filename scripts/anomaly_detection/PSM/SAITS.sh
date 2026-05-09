mask_ratios=(0 0.2 0.4 0.6 0.8)
model_name=saits
gpu=3


for mr in "${mask_ratios[@]}"; do
  python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path ./dataset/PSM \
    --model_id PSM \
    --model $model_name \
    --data PSM \
    --features M \
    --seq_len 100 \
    --pred_len 0 \
    --d_model 64 \
    --d_ff 64 \
    --e_layers 2 \
    --enc_in 25 \
    --c_out 25 \
    --top_k 3 \
    --anomaly_ratio 1 \
    --batch_size 128 \
    --train_epochs 3 \
    --mask_ratio $mr \
    --gpu $gpu 
done
