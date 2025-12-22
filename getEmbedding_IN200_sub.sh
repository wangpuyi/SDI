#!/bin/bash
#SBATCH -p gpu                   # Partition to submit to
#SBATCH --job-name=getEmbed              # Job name
#SBATCH --gres=gpu:1                   # Request 1 GPU per task
#SBATCH --ntasks=1                     # Number of tasks per split
#SBATCH --cpus-per-task=8              # Number of CPU cores per task (adjust as needed)
#SBATCH --nodes=1                      # Ensure each task runs on 1 node
#SBATCH --array=0                    # Job array with 4 tasks (0 to 3)
#SBATCH --output=log_notImportant/split_%A_%a.out  # Standard output log (%A for array job ID, %a for task ID)
#SBATCH --error=log_notImportant/split_%A_%a.err   # Standard error log

# Load your environment if needed
source activate py39  # Activate your conda environment if necessary

# Run your Python script, passing the split ID as an argument
python getEmbedding.py --total_split 1 --split $SLURM_ARRAY_TASK_ID \
 --projector_ckpt="/grp01/cs_hszhao/cs002u03/ckpt/adapter/IN200_S/bs64-lr_0.0005-20241009-223420/checkpoint-99.pt" \
 --data_root="/grp01/cs_hszhao/cs002u03/dataset/IN200_S/train" \
 --save_root="/grp01/cs_hszhao/cs002u03/output/ProjTensor/IN200_sub" \