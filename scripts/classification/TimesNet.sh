
mask_ratios=(0.25 0.5 0.75)
model_name=TimesNet
gpu=0
for mr in "${mask_ratios[@]}"; do
    python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./dataset/SelfRegulationSCP1/ \
    --model_id SelfRegulationSCP1 \
    --model $model_name \
    --data UEA \
    --e_layers 3 \
    --batch_size 16 \
    --d_model 16 \
    --d_ff 32 \
    --top_k 3 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --train_epochs 30 \
    --patience 10 \
    --mask_ratio $mr \
    --gpu $gpu 
done
