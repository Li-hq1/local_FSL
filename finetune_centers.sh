# [centers]
PRETRIAN_MODEL_CHECKPOINT='output/pretrain/MINI_1600_512d8b/checkpoint-1599.pth'
DATASET_NAME='mini' #'mini', 'FC100', 'tiered', 'CUB', 'CIFAR_FS'
OUTPUT_DIR='output/finetune/no_focal/centers/0.01mcr_128cluster'
GPUS='6' # we use sinle GPU
python main_finetune_centers.py \
    --blr 7e-4 \
    --seed 0 \
    --gpus ${GPUS} \
    --model vit_base_patch16 \
    --dataset_name ${DATASET_NAME} \
    --finetune ${PRETRIAN_MODEL_CHECKPOINT} \
    --output_dir ${OUTPUT_DIR} \
    --meta_val \
    --batch_size 128 \
    --epochs 100 \
    --pooling nextvlad_centers \
    --mask_ratio 0.7 \
    --focal_gamma 0 \
    --nextvlad_cluster 128
