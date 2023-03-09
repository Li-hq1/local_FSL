# meta test
MODEL_PATH='output/finetune/no_focal/centers/1mcr_128cluster/checkpoint-best_meta_val.pth'
DATASET_NAME='mini'  #'mini', 'FC100', 'tiered', 'CUB', 'CIFAR_FS'
OUTPUT_DIR='output/test'
GPUS='2'
python main_finetune.py \
    --seed 0 \
    --gpus ${GPUS} \
    --model vit_base_patch16 \
    --dataset_name ${DATASET_NAME} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --epochs 100 \
    --pooling nextvlad_centers \
    --mask_ratio 0 \
    --meta_test \
    --meta_val \
    --nextvlad_cluster 128

