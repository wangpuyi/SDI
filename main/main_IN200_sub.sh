#!/bin/bash
#SBATCH -p gpu_shared                  # Partition to submit to
#SBATCH --job-name=res50               # Job name
#SBATCH --gres=gpu:4                   # Request n GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=36              # Number of CPU cores per task (adjust if needed)
#SBATCH --output=sbatch.log/IN200-sub/%x_%j.out        # Standard output log (%x is the job name, %j is the job ID)
#SBATCH --error=sbatch.log/IN200-sub/%x_%j.err         
#SBATCH --mem=150G                     # Memory per node (adjust as needed)
#SBATCH --time=4-00:00                 # Walltime/duration of the job

# Load any necessary modules or activate your environment
source activate py39  # Activate your conda environment if needed

# Run the Python script with the required arguments
# python main_DDP.py --config configs/IN100_sub/base.yaml --port="16533"
# python main_DDP.py --config configs/IN200_S/GIF.yaml --port="16733"
python main_DDP.py --config configs/IN200_S/SEI_Gaussian_2.0.yaml --port="16541"
# python main_DDP.py --config configs/IN200_S/Ablation_Gaussian.yaml --port="16563"
# python main_DDP.py --config configs/IN200_S/Ablation_GMM3.yaml --port="16564"
# python main_DDP.py --config configs/IN200_S/Ablation_GMM5.yaml --port="16565"
# python main_DDP.py --config configs/IN200_S/NoAlign.yaml --port="16565"
# python main_DDP.py --config configs/IN200_S/SD_1_5.yaml --port="16566"
# python main_DDP.py --config configs/IN200_S/MAE.yaml --port="16567"
# python main_DDP.py --config configs/IN200_S/base.yaml --port="16568"
# python main_DDP.py --config configs/IN200_S/aug.yaml --port="16571"
# python main_DDP.py --config configs/IN200_S/DALLE.yaml --port="16572"
# python main_DDP.py --config configs/IN200_S/Ablation_GMM2.yaml --port="16564"
# python main_DDP.py --config configs/IN200_S/SEI_ratio_1.5.yaml --port="16565" 
# python main_DDP.py --config configs/IN200_S/SEI_ratio_3.yaml --port="16566"
# python main_DDP.py --config configs/IN200_S/SEI_ratio_4.yaml --port="16567"


