#!/bin/bash
#SBATCH -p gpu                  # Partition to submit to
#SBATCH --job-name=Sample2img               # Job name
#SBATCH --gres=gpu:1                   # Request 2 GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=8              # Number of CPU cores per task (adjust if needed)
#SBATCH --output=log_notImportant/sample2img_%j.out        # Standard output log (%x is the job name, %j is the job ID)
#SBATCH --error=log_notImportant/sample2img_%j.err         

# Load any necessary modules or activate your environment
source activate py39  # Activate your conda environment if needed

# Run the Python script with the required arguments
python Sample2img.py
# python Sample2imgSeed.py