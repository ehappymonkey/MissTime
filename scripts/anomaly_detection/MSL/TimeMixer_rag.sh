mask_ratios=(0.25 0.5 0.75)
gpu=3

model_name=TimeMixer2
seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.001
d_model=16
d_ff=32
patience=10

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
    --d_model $d_model \
    --learning_rate $learning_rate \
    --patience $patience \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --gpu $gpu \
    --mask_ratio $mr \
    --rag_type latent_rag \
    --retrieve_encoder Typology \
    --contrastive_loss hard_negative \
    --encoder_epochs 1
done




# for mr in "${mask_ratios[@]}"; do
#     python -u run.py \
#     --task_name imputation \
#     --is_training 1 \
#     --root_path ./dataset/electricity/ \
#     --data_path electricity.csv \
#     --model_id ECL_mask \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len 96 \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des 'Exp' \
#     --itr 1 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --train_epochs $train_epochs \
#     --patience $patience \
#     --down_sampling_layers $down_sampling_layers \
#     --down_sampling_method avg \
#     --down_sampling_window $down_sampling_window \
#     --gpu $gpu \
#     --mask_ratio $mr \
#     --rag_type latent_rag \
#     --retrieve_encoder Typology \
#     --contrastive_loss hard_negative
# done
