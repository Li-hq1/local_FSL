PRETRIAN_MODEL_CHECKPOINT='output/pretrain/MINI_1600_512d8b/checkpoint-1599.pth'
DATASET_NAME='mini' #'mini', 'FC100', 'tiered', 'CUB', 'CIFAR_FS'
OUTPUT_DIR='output/finetune/localization/standard'
GPUS='7' # we use sinle GPU
python main_finetune_localization.py \
    --blr 5e-4 \
    --seed 0 \
    --gpus ${GPUS} \
    --model vit_base_patch16 \
    --dataset_name ${DATASET_NAME} \
    --finetune ${PRETRIAN_MODEL_CHECKPOINT} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --epochs 100 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --smoothing 0.0
    
    
    # --reprob 0.25 --mixup 0.8 --cutmix 1.0 