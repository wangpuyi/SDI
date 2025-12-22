#!/bin/bash
#SBATCH -p gpu                  # Partition to submit to
#SBATCH --job-name=inference              # Job name
#SBATCH --gres=gpu:1                   # Request 1 GPU per task
#SBATCH --ntasks=1                     # Number of tasks per split
#SBATCH --cpus-per-task=8              # Number of CPU cores per task (adjust as needed)
#SBATCH --nodes=1                      # Ensure each task runs on 1 node
#SBATCH --output=log_notImportant/inference_%A.out  # Standard output log (%A for array job ID, %a for task ID)
#SBATCH --error=log_notImportant/inference_%A.err   # Standard error log

# Load your environment if needed
source activate py39  # Activate your conda environment if necessary

# python inference.py
python inference_more.py