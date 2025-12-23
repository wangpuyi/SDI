#!/bin/bash
#SBATCH --job-name=ProjectIN200-S   # 作业名称
#SBATCH -p gpu               # 分区名称
#SBATCH --output=/grp01/cs_hszhao/cs002u03/log_notImportant/trainProjector/IN200sub_%j_output.log  # 输出日志文件，使用数组作业ID和任务ID命名
#SBATCH --error=/grp01/cs_hszhao/cs002u03/log_notImportant/trainProjector/IN200sub_%j_error.log
#SBATCH --gres=gpu:4                    # 请求N个GPU
#SBATCH -N 1                            # 使用1个节点
#SBATCH --cpus-per-task=48              # 每个任务使用16个CPU核心
#SBATCH --mem=128G                      # 每个任务所需内存


source ~/anaconda3/etc/profile.d/conda.sh
conda activate py39

# 运行训练脚本
python train_Projector.py \
    --pretrained_model_name_or_path="/grp01/cs_hszhao/cs002u03/ckpt/stable-diffusion-v1-5" \
    --data_path="/grp01/cs_hszhao/cs002u03/dataset/IN200_S/train" \
    --clip_path="/grp01/cs_hszhao/cs002u03/ckpt/clip-vit-large-patch14" \
    --mixed_precision="fp16" \
    --resolution=512 \
    --train_batch_size=16 \
    --dataloader_num_workers=16 \
    --learning_rate=0.0005 \
    --weight_decay=0.01 \
    --port=17449 \
    --output_dir="/grp01/cs_hszhao/cs002u03/ckpt/adapter/IN200_S"