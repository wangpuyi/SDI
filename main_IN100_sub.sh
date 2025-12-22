#!/bin/bash
#SBATCH -p gpu_shared                  # Partition to submit to
#SBATCH --job-name=res50               # Job name
#SBATCH --gres=gpu:4                   # Request n GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=36              # Number of CPU cores per task (adjust if needed)
#SBATCH --output=sbatch.log/IN100-sub/%x_%j.out        # Standard output log (%x is the job name, %j is the job ID)
#SBATCH --error=sbatch.log/IN100-sub/%x_%j.err         
#SBATCH --mem=150G                     # Memory per node (adjust as needed)
#SBATCH --time=4-00:00                 # Walltime/duration of the job

# Load any necessary modules or activate your environment
source activate py39  # Activate your conda environment if needed

# Run the Python script with the required arguments
# python main_DDP.py --config configs/IN100_sub/base.yaml --port="16533"
# python main_DDP.py --config configs/IN100_sub/GIF.yaml --port="16733"
# python main_DDP.py --config configs/IN100_sub/SEI_Gaussian_7.5.yaml --port="16520"
python main_DDP.py --config configs/IN100_sub/SEI_Gaussian_2.0.yaml --port="16541"
# python main_DDP.py --config configs/IN100_sub/SEI_Gaussian_7.5-seed.yaml --port="16543"
# python main_DDP.py --config configs/IN100_sub/aug.yaml --port="16536"
# python main_DDP.py --config configs/IN100_sub/SD_1_5.yaml --port="16537"
# python main_DDP.py --config configs/IN100_sub/MAE.yaml --port="16538"
# python main_DDP.py --config configs/IN100_sub/DALLE.yaml --port="16539"

