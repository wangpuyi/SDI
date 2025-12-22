#!/bin/bash
#SBATCH --job-name=Flowers-generate        # 作业名称
#SBATCH --output=/grp01/cs_hszhao/cs002u03/log_notImportant/Flowers_%j_output.log  # 输出日志文件，使用数组作业ID和任务ID命名
#SBATCH --error=/grp01/cs_hszhao/cs002u03/log_notImportant/Flowers_%j_error.log    # 错误日志文件
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=48           
#SBATCH --gres=gpu:3                   # 每个任务使用1个GPU
#SBATCH --partition=gpu_shared             # 提交到的分区名称
#SBATCH --time=4-00:00:00

# # 激活你的 conda 环境（假设环境名为 myenv）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py39

# 运行 Python 脚本，指定每个任务的 GPU ID
# python SEI_Generate.py \
#     --guidance_scale=7.5 \
#     --pt_root="/grp01/cs_hszhao/cs002u03/output/ProjTensor/Flowers" \
#     --save_root="/grp01/cs_hszhao/cs002u03/output/SEI/Flowers-Gaussian-7.5" \
#     --data_root="/grp01/cs_hszhao/cs002u03/dataset/Flower102/train" \
#     --expanded_number_per_sample=40 \
#     --sample_func="Gaussian"
# python SEI_Generate.py \
#     --guidance_scale=2.0 \
#     --pt_root="/grp01/cs_hszhao/cs002u03/output/ProjTensor/Flowers" \
#     --save_root="/grp01/cs_hszhao/cs002u03/output/SEI/Flowers-GMM3-2.0-5x" \
#     --data_root="/grp01/cs_hszhao/cs002u03/dataset/Flower102/train" \
#     --expanded_number_per_sample=5 \
#     --sample_func="GMM3"

# python SEI_Generate.py \
#     --guidance_scale=2.0 \
#     --pt_root="/grp01/cs_hszhao/cs002u03/output/ProjTensor/Flowers" \
#     --save_root="/grp01/cs_hszhao/cs002u03/output/SEI/Flowers-GMM5-2.0-5x" \
#     --data_root="/grp01/cs_hszhao/cs002u03/dataset/Flower102/train" \
#     --expanded_number_per_sample=5 \
#     --sample_func="GMM5"
python SEI_Generate.py \
    --guidance_scale=2.0 \
    --pt_root="/grp01/cs_hszhao/cs002u03/output/ProjTensor/Flowers" \
    --save_root="/grp01/cs_hszhao/cs002u03/output/SEI/Flowers-GMM2-2.0-5x" \
    --data_root="/grp01/cs_hszhao/cs002u03/dataset/Flower102/train" \
    --expanded_number_per_sample=5 \
    --sample_func="GMM2"