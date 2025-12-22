#!/bin/bash
#SBATCH -p gpu_shared                  # Partition to submit to
#SBATCH --job-name=res50               # Job name
#SBATCH --gres=gpu:2                   # Request n GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=36              # Number of CPU cores per task (adjust if needed)
#SBATCH --output=sbatch.log/Generalization/%x_%j.out        # Standard output log (%x is the job name, %j is the job ID)
#SBATCH --error=sbatch.log/Generalization/%x_%j.err         
#SBATCH --mem=100G                     # Memory per node (adjust as needed)
#SBATCH --time=4-00:00                 # Walltime/duration of the job

# Load any necessary modules or activate your environment
source activate py39  # Activate your conda environment if needed

# Run the Python script with the required arguments
# python main_DDP_top5.py --config configs/IN200_generalization/ImageNet-a.yaml --port="16233"
python main_DDP_top5.py --config configs/IN200_generalization/ImageNet-sketch.yaml --port="16233"
