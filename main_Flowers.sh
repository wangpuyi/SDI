#!/bin/bash
#SBATCH -p gpu_shared                  # Partition to submit to
#SBATCH --job-name=res50               # Job name
#SBATCH --gres=gpu:4                   # Request n GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=36              # Number of CPU cores per task (adjust if needed)
#SBATCH --output=sbatch.log/Flowers/%x_%j.out        # Standard output log (%x is the job name, %j is the job ID)
#SBATCH --error=sbatch.log/Flowers/%x_%j.err         
#SBATCH --mem=150G                     # Memory per node (adjust as needed)
#SBATCH --time=4-00:00                 # Walltime/duration of the job

# Load any necessary modules or activate your environment
source activate py39  # Activate your conda environment if needed

# Run the Python script with the required arguments
# python main_DDP.py --config configs/Flowers102/GIF.yaml --port="16503"
# python main_DDP.py --config configs/Flowers102/SEI_Noseed-7.5.yaml --port="16500"
# python main_DDP.py --config configs/Flowers102/SEI_Noseed-2.0.yaml --port="16502"
# python main_DDP.py --config configs/Flowers102/Ablation_Gaussian.yaml --port="16502"
# python main_DDP.py --config configs/Flowers102/Ablation_GMM3.yaml --port="16503"
# python main_DDP.py --config configs/Flowers102/Ablation_GMM5.yaml --port="16504"
# python main_DDP.py --config configs/Flowers102/aug.yaml --port="16508"
# python main_DDP.py --config configs/Flowers102/SD.yaml --port="16509"
# python main_DDP.py --config configs/Flowers102/Ablation_GMM2.yaml --port="16509"

