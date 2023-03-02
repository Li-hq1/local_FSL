# meta test
MODEL_PATH='output/finetune/mini1600/MINI_1600_0.75+-0.1mask/checkpoint-best_meta_val.pth'
DATASET_NAME='mini'  #'mini', 'FC100', 'tiered', 'CUB', 'CIFAR_FS'
OUTPUT_DIR='output/test'
GPUS='5'
python main_finetune.py \
    --seed 0 \
    --gpus ${GPUS} \
    --model vit_base_patch16 \
    --dataset_name ${DATASET_NAME} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --epochs 100 \
    --pooling nextvlad \
    --mask_ratio 0 \
    --meta_test \
    --meta_val \

