#!/bin/bash
# =============================================================================
# FSAS + CA + CBAM + ViT 模型训练脚本
# =============================================================================
# 数据集: /data/gqzhu/data/dataset/10-new_with_noise_and_quant
# 训练模式: 5 PSF图像 -> 77 Zernike系数
# 环境: zgq conda environment
# =============================================================================
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate zgq

# -----------------------------------------------------------------------------
# Small 模型 (CA + FSAS + CBAM 实验)
# -----------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0,1,2,3 python /data/gqzhu/data/ca_fcas_cbam_vit/train_fcas_vit.py \
    --train_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/train" \
    --val_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
    --test_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
    --checkpoint_dir "/data/gqzhu/data/ca_fcas_cbam_vit/result/fcas_cbam_vit-ca-5psf-7x11-small" \
    --input_size 112 \
    --patch_size 8 \
    --fsas_patch_size 8 \
    --cbam_reduction 16 \
    --cbam_kernel_size 7 \
    --ca_reduction 32 \
    --num_images 5 \
    --num_coefficients 77 \
    --variant "small" \
    --dropout 0.1 \
    --batch_size 512 \
    --epochs 200 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --num_workers 8 \
    --use_amp \
    --patience 200

# -----------------------------------------------------------------------------
# Base 模型 (推荐配置)
# -----------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0,1,2,3 python /data/gqzhu/data/ca_fcas_cbam_vit/train_fcas_vit.py \
#     --train_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/train" \
#     --val_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
#     --test_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
#     --checkpoint_dir "/data/gqzhu/data/ca_fcas_cbam_vit/result/fcas_cbam_vit-5psf-7x11-base" \
#     --input_size 112 \
#     --patch_size 8 \
#     --fsas_patch_size 8 \
#     --cbam_reduction 16 \
#     --cbam_kernel_size 7 \
#     --num_images 5 \
#     --num_coefficients 77 \
#     --variant "base" \
#     --dropout 0.1 \
#     --batch_size 256 \
#     --epochs 400 \
#     --lr 1e-4 \
#     --weight_decay 0.05 \
#     --max_grad_norm 1.0 \
#     --num_workers 8 \
#     --use_amp \
#     --patience 400

# -----------------------------------------------------------------------------
# Large 模型 (最大容量)
# -----------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0,1,2,3 python /data/gqzhu/data/ca_fcas_cbam_vit/train_fcas_vit.py \
#     --train_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/train" \
#     --val_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
#     --test_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
#     --checkpoint_dir "/data/gqzhu/data/ca_fcas_cbam_vit/result/fcas_cbam_vit-5psf-7x11-large" \
#     --input_size 112 \
#     --patch_size 8 \
#     --fsas_patch_size 8 \
#     --cbam_reduction 16 \
#     --cbam_kernel_size 7 \
#     --num_images 5 \
#     --num_coefficients 77 \
#     --variant "large" \
#     --dropout 0.1 \
#     --batch_size 128 \
#     --epochs 400 \
#     --lr 1e-4 \
#     --weight_decay 0.05 \
#     --max_grad_norm 1.0 \
#     --num_workers 8 \
#     --use_amp \
#     --patience 400

# -----------------------------------------------------------------------------
# Tiny 模型 (最小资源)
# -----------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0,1 python /data/gqzhu/data/ca_fcas_cbam_vit/train_fcas_vit.py \
#     --train_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/train" \
#     --val_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
#     --test_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
#     --checkpoint_dir "/data/gqzhu/data/ca_fcas_cbam_vit/result/fcas_cbam_vit-5psf-7x11-tiny" \
#     --input_size 112 \
#     --patch_size 8 \
#     --fsas_patch_size 8 \
#     --cbam_reduction 16 \
#     --cbam_kernel_size 7 \
#     --num_images 5 \
#     --num_coefficients 77 \
#     --variant "tiny" \
#     --dropout 0.1 \
#     --batch_size 1024 \
#     --epochs 200 \
#     --lr 1e-4 \
#     --weight_decay 0.05 \
#     --max_grad_norm 1.0 \
#     --num_workers 8 \
#     --use_amp \
#     --patience 200

# -----------------------------------------------------------------------------
# 25系数模式 (4 PSF图像)
# -----------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0,1,2,3 python /data/gqzhu/data/ca_fcas_cbam_vit/train_fcas_vit.py \
#     --train_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/train" \
#     --val_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
#     --test_dir "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
#     --checkpoint_dir "/data/gqzhu/data/ca_fcas_cbam_vit/result/fcas_cbam_vit-4psf-25-base" \
#     --input_size 112 \
#     --patch_size 8 \
#     --fsas_patch_size 8 \
#     --cbam_reduction 16 \
#     --cbam_kernel_size 7 \
#     --num_images 4 \
#     --num_coefficients 25 \
#     --variant "base" \
#     --dropout 0.1 \
#     --batch_size 256 \
#     --epochs 400 \
#     --lr 1e-4 \
#     --weight_decay 0.05 \
#     --max_grad_norm 1.0 \
#     --num_workers 8 \
#     --use_amp \
#     --patience 400

# =============================================================================
# 示例: 运行测试
# =============================================================================
# 测试 Small 模型
# CUDA_VISIBLE_DEVICES=0 python /data/gqzhu/data/ca_fcas_cbam_vit/get_test_rmse.py \
#     --input_path "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
#     --checkpoint_path "/data/gqzhu/data/ca_fcas_cbam_vit/result/fcas_cbam_vit-ca-5psf-7x11-small/model_best.pth" \
#     --output_dir "/data/gqzhu/data/ca_fcas_cbam_vit/test" \
#     --patch_size 8 \
#     --fsas_patch_size 8 \
#     --cbam_reduction 16 \
#     --cbam_kernel_size 7 \
#     --ca_reduction 32 \
#     --img_size 112 \
#     --num_threads 64 \
#     --device cuda
