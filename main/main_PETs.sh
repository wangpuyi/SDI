#!/bin/bash
#SBATCH -p gpu_shared                  # Partition to submit to
#SBATCH --job-name=res50               # Job name
#SBATCH --gres=gpu:4                   # Request n GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=16              # Number of CPU cores per task (adjust if needed)
#SBATCH --output=sbatch.log/PETs/%x_%j.out        # Standard output log (%x is the job name, %j is the job ID)
#SBATCH --error=sbatch.log/PETs/%x_%j.err         
#SBATCH --mem=100G                     # Memory per node (adjust as needed)
#SBATCH --time=4-00:00                 # Walltime/duration of the job

# Load any necessary modules or activate your environment
source activate py39  # Activate your conda environment if needed

# Run the Python script with the required arguments
# python main_DDP.py --config configs/PETs/GIF.yaml --port="13519"
python main_DDP.py --config configs/PETs/SEI_Gaussian.yaml --port="13520"
# python main_DDP.py --config configs/PETs/Ablation_Gaussian.yaml --port="13521"
# python main_DDP.py --config configs/PETs/Ablation_GMM3.yaml --port="13522"
# python main_DDP.py --config configs/PETs/Ablation_GMM5.yaml --port="13523"
# python main_DDP.py --config configs/PETs/SD_1_5.yaml --port="13519"
# python main_DDP.py --config configs/PETs/Ablation_GMM2.yaml --port="13522"
# python main_DDP.py --config configs/PETs/SEI_Gaussian_ration_1.5.yaml --port="13523"
# python main_DDP.py --config configs/PETs/SEI_Gaussian_ration_3.yaml --port="13524"
# python main_DDP.py --config configs/PETs/SEI_Gaussian_ration_4.yaml --port="13525"

