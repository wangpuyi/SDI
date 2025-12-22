#!/bin/bash
#SBATCH -p gpu                  # Partition to submit to
#SBATCH --job-name=res50               # Job name
#SBATCH --gres=gpu:4                   # Request n GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=36              # Number of CPU cores per task (adjust if needed)
#SBATCH --output=sbatch.log/CUB/%x_%j.out        # Standard output log (%x is the job name, %j is the job ID)
#SBATCH --error=sbatch.log/CUB/%x_%j.err         
#SBATCH --mem=150G                     # Memory per node (adjust as needed)
#SBATCH --time=4-00:00                 # Walltime/duration of the job

# Load any necessary modules or activate your environment
source activate py39  # Activate your conda environment if needed

# Run the Python script with the required arguments
# python main_DDP.py --config configs/Cub2011/GIF.yaml --port="12340"
python main_DDP.py --config configs/Cub2011/SEI_Gaussian-2.0.yaml --port="12440"
# python main_DDP.py --config configs/Cub2011/SEI_Gaussian-7.5.yaml --port="12440"
# python main_DDP.py --config configs/Cub2011/base.yaml --port="12944"
# python main_DDP.py --config configs/Cub2011/aug.yaml --port="13946"
# python main_DDP.py --config configs/Cub2011/SD_1_5.yaml --port="13947"
# python main_DDP.py --config configs/Cub2011/MAE.yaml --port="13948"
# python main_DDP.py --config configs/Cub2011/NoAligned.yaml --port="13949"
# python main_DDP.py --config configs/Cub2011/DALLE.yaml --port="13949"
# python main_DDP.py --config configs/Cub2011/SEI_Gaussian_ration_1.5.yaml --port="13950"
# python main_DDP.py --config configs/Cub2011/SEI_Gaussian_ration_3.yaml --port="13951"
# python main_DDP.py --config configs/Cub2011/SEI_Gaussian_ration_4.yaml --port="13952"
