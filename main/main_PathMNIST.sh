#!/bin/bash
#SBATCH -p gpu_shared                  # Partition to submit to
#SBATCH --job-name=res50               # Job name
#SBATCH --gres=gpu:4                   # Request n GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=36              # Number of CPU cores per task (adjust if needed)
#SBATCH --output=sbatch.log/PathMNIST/%x_%j.out        # Standard output log (%x is the job name, %j is the job ID)
#SBATCH --error=sbatch.log/PathMNIST/%x_%j.err         
#SBATCH --time=4-00:00                 # Walltime/duration of the job
#SBATCH --mem=200G                      # Memory per node in GB. Also see --mem-per-cpu

# Load any necessary modules or activate your environment
source activate py39  # Activate your conda environment if needed

# Run the Python script with the required arguments
python main_DDP.py --config configs/PathMNIST/SEI_Gaussian-2.0.yaml --port="14321"
# python main_DDP.py --config configs/PathMNIST/SEI_Gaussian-7.5.yaml --port="14341"