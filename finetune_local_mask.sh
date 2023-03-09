PRETRIAN_MODEL_CHECKPOINT='output/pretrain/MINI_1600_512d8b/checkpoint-1599.pth'
DATASET_NAME='mini' #'mini', 'FC100', 'tiered', 'CUB', 'CIFAR_FS'
OUTPUT_DIR='output/finetune/no_focal/local/0.2_1ratio'
GPUS='5' # we use sinle GPU
python main_finetune_local_mask.py \
    --blr 7e-4 \
    --seed 0 \
    --pooling nextvlad \
    --mask_ratio 0.7 \
    --batch_size 128 \
    --model vit_base_patch16 \
    --gpus ${GPUS} \
    --dataset_name ${DATASET_NAME} \
    --finetune ${PRETRIAN_MODEL_CHECKPOINT} \
    --output_dir ${OUTPUT_DIR} \
    --meta_val \
    --epochs 100 \
    --focal_gamma 0 \

    # --layer_decay 0.65