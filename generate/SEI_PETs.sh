#!/bin/bash
#SBATCH --job-name=PETs-generate        # 作业名称
#SBATCH --output=/grp01/cs_hszhao/cs002u03/log_notImportant/PETs%j_output.log  # 输出日志文件，使用数组作业ID和任务ID命名
#SBATCH --error=/grp01/cs_hszhao/cs002u03/log_notImportant/PETs%j_error.log    # 错误日志文件
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=36           
#SBATCH --gres=gpu:4                   # 每个任务使用1个GPU
#SBATCH --partition=gpu_shared             # 提交到的分区名称
#SBATCH --mem=150G                      # 每个任务所需内存
#SBATCH --time=4-00:00:00

# # 激活你的 conda 环境（假设环境名为 myenv）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py39

# 运行 Python 脚本，指定每个任务的 GPU ID
python SEI_Generate.py \
    --guidance_scale=2.0 \
    --pt_root="/grp01/cs_hszhao/cs002u03/output/ProjTensor/PETs" \
    --save_root="/grp01/cs_hszhao/cs002u03/output/SEI/PETs-Gaussian-2.0" \
    --data_root="/grp01/cs_hszhao/cs002u03/dataset/PETs/trainval" \
    --expanded_number_per_sample=60 \
    --sample_func="Gaussian"
