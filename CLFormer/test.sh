#!/bin/bash
# =============================================================================
# FSAS + CA + CBAM + ViT 模型测试脚本
# =============================================================================
# 使用 get_test_rmse.py 在测试集上评估模型性能
# 环境: zgq conda environment
# =============================================================================
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate zgq

# -----------------------------------------------------------------------------
# 测试 CA + FSAS + CBAM Small 模型
# -----------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python /data/gqzhu/data/ca_fcas_cbam_vit/get_test_rmse.py \
    --input_path "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
    --checkpoint_path "/data/gqzhu/data/ca_fcas_cbam_vit/result/fcas_cbam_vit-ca-5psf-7x11-small/model_best.pth" \
    --Zseg_path "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/Zseg/Zeg.npy" \
    --output_dir "/data/gqzhu/data/ca_fcas_cbam_vit/result/fcas_cbam_vit-ca-5psf-7x11-small-test" \
    --patch_size 8 \
    --fsas_patch_size 8 \
    --cbam_reduction 16 \
    --cbam_kernel_size 7 \
    --ca_reduction 32 \
    --img_size 112 \
    --num_threads 64 \
    --device cuda

# -----------------------------------------------------------------------------
# 测试 Base 模型
# -----------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 python /data/gqzhu/data/fcas_cbam_vit/get_test_rmse.py \
#     --input_path "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
#     --checkpoint_path "/data/gqzhu/data/fcas_cbam_vit/result/fcas_cbam_vit-5psf-7x11-base/model_best.pth" \
#     --Zseg_path "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/Zseg/Zeg.npy" \
#     --output_dir "/data/gqzhu/data/fcas_cbam_vit/result/fcas_cbam_vit-5psf-7x11-base-test" \
#     --patch_size 8 \
#     --fsas_patch_size 8 \
#     --cbam_reduction 16 \
#     --cbam_kernel_size 7 \
#     --img_size 112 \
#     --num_threads 64 \
#     --device cuda

# -----------------------------------------------------------------------------
# 测试 Large 模型
# -----------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 python /data/gqzhu/data/fcas_cbam_vit/get_test_rmse.py \
#     --input_path "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/test" \
#     --checkpoint_path "/data/gqzhu/data/fcas_cbam_vit/result/fcas_cbam_vit-5psf-7x11-large/model_best.pth" \
#     --Zseg_path "/data/gqzhu/data/dataset/10-new_with_noise_and_quant/Zseg/Zeg.npy" \
#     --output_dir "/data/gqzhu/data/fcas_cbam_vit/result/fcas_cbam_vit-5psf-7x11-large-test" \
#     --patch_size 8 \
#     --fsas_patch_size 8 \
#     --cbam_reduction 16 \
#     --cbam_kernel_size 7 \
#     --img_size 112 \
#     --num_threads 64 \
#     --device cuda
