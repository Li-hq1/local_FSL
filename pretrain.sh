# OUTPUT_DIR='output/pretrain/MINI_1600'
# DATASET_NAME='mini'  #'mini', 'FC100', 'tiered', 'CUB', 'CIFAR_FS'
# GPUS='0,1'
# DIST_URL='tcp://127.0.0.1:6666'
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 run_pretrain.py \
#         --dist_url ${DIST_URL} \
#         --gpus ${GPUS} \
#         --dataset_name ${DATASET_NAME} \
#         --mask_ratio 0.75 \
#         --model pretrain_mae_base_patch16_224 \
#         --batch_size 128 \
#         --opt adamw \
#         --opt_betas 0.9 0.95 \
#         --warmup_epochs 40 \
#         --epochs 1600 \
#         --output_dir ${OUTPUT_DIR}


# OUTPUT_DIR='output/pretrain/MINI_1600_384d8b_nonorm'
# DATASET_NAME='mini'  #'mini', 'FC100', 'tiered', 'CUB', 'CIFAR_FS'
# GPUS='0,5'
# DIST_URL='tcp://127.0.0.1:6668'
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6668 main_pretrain.py \
#         --dist_url ${DIST_URL} \
#         --gpus ${GPUS} \
#         --dataset_name ${DATASET_NAME} \
#         --output_dir ${OUTPUT_DIR} \
#         --blr 1.5e-4 \
#         --weight_decay 0.05 \
#         --mask_ratio 0.75 \
#         --model mae_vit_base_patch16_dec384d8b \
#         --batch_size 128 \
#         --warmup_epochs 40 \
#         --epochs 1600 \
#         --save_freq 50 \
#         --norm_pix_loss


OUTPUT_DIR='output/pretrain/MINI_1600_512d8b_addcls'
DATASET_NAME='mini'  #'mini', 'FC100', 'tiered', 'CUB', 'CIFAR_FS'
GPUS='1,2'
DIST_URL='tcp://127.0.0.1:6666'
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 main_pretrain_addcls.py \
        --dist_url ${DIST_URL} \
        --gpus ${GPUS} \
        --dataset_name ${DATASET_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --blr 1.5e-4 \
        --weight_decay 0.05 \
        --mask_ratio 0.75 \
        --model mae_vit_base_patch16_dec512d8b \
        --batch_size 128 \
        --warmup_epochs 40 \
        --epochs 1600 \
        --save_freq 50 \
        --norm_pix_loss
        