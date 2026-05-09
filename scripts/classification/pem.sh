mask_ratios=(0 0.2 0.4 0.6 0.8)
model="TimesNet"
gpu=1

# for mr in "${mask_ratios[@]}"; do
#     python -u run.py \
#     --task_name classification \
#     --is_training 1 \
#     --root_path ./dataset/PEMS-SF/ \
#     --model_id PEMS-SF \
#     --model $model \
#     --data UEA \
#     --e_layers 3 \
#     --batch_size 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --top_k 3 \
#     --des 'Exp' \
#     --itr 1 \
#     --learning_rate 0.001 \
#     --train_epochs 100 \
#     --patience 10 \
#     --mask_ratio $mr \
#     --gpu $gpu
# done

# for mr in "${mask_ratios[@]}"; do
#     python -u run.py \
#     --task_name classification \
#     --is_training 1 \
#     --root_path ./dataset/PEMS-SF/ \
#     --model_id PEMS-SF \
#     --model $model \
#     --data UEA \
#     --e_layers 3 \
#     --batch_size 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --top_k 3 \
#     --des 'Exp' \
#     --itr 1 \
#     --learning_rate 0.001 \
#     --train_epochs 100 \
#     --patience 10 \
#     --mask_ratio $mr \
#     --gpu $gpu \
#     --use_retrieval
# done

for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./dataset/PEMS-SF/ \
    --model_id PEMS-SF \
    --model $model \
    --data UEA \
    --e_layers 3 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --top_k 3 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --train_epochs 100 \
    --patience 10 \
    --mask_ratio $mr \
    --gpu $gpu \
    --use_latent_retrieval
done