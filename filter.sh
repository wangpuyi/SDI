#!/bin/bash
#SBATCH -p gpu                  # Partition to submit to
#SBATCH --job-name=filter              # Job name
#SBATCH --gres=gpu:1                   # Request 1 GPU per task
#SBATCH --ntasks=1                     # Number of tasks per split
#SBATCH --cpus-per-task=8              # Number of CPU cores per task (adjust as needed)
#SBATCH --nodes=1                      # Ensure each task runs on 1 node
#SBATCH --output=log_notImportant/filter_%A.out  # Standard output log (%A for array job ID, %a for task ID)
#SBATCH --error=log_notImportant/filter_%A.err   # Standard error log

# Load your environment if needed
source activate py39  # Activate your conda environment if necessary

# Run your Python script, passing the split ID as an argument
# python filtering/filter.py --config configs/IN100_sub/SEI_Gaussian_2.0.yaml filter.load=false
# python filtering/filter.py --config configs/IN100_sub/SEI_Gaussian_7.5.yaml
# python filtering/filter.py --config configs/IN100_sub/SEI_Gaussian_7.5-seed.yaml
# python filtering/filter_Cars.py --config configs/Cars196/SEI_seedImage-7.5.yaml
# python filtering/filter_Cars.py --config configs/Cars196/SEI_Noseed-7.5.yaml
# python filtering/filter_Cars_softmax.py --config configs/Cars196/SEI_seedImage-7.5.yaml

# python filtering/filter_CUB.py --config configs/Cub2011/SEI_Gaussian-7.5.yaml
# python filtering/filter_CUB.py --config configs/Cub2011/SEI_Gaussian-2.0.yaml
# python filtering/filter_CUB.py --config configs/Caltech101/SEI-Gaussian-7.5.yaml
# python filtering/filter_CUB.py --config configs/Caltech101/SEI-Gaussian-2.0.yaml
# python filtering/filter_CUB.py --config configs/Cars196/SEI_Noseed-2.0.yaml
# python filtering/filter_CUB.py --config configs/PETs/SEI_Gaussian.yaml
# python filtering/filter_CUB.py --config configs/PETs/Ablation_Gaussian.yaml
# python filtering/filter_CUB.py --config configs/PETs/Ablation_GMM3.yaml
# python filtering/filter_CUB.py --config configs/PETs/Ablation_GMM2.yaml
# python filtering/filter_CUB.py --config configs/PETs/Ablation_GMM5.yaml
# python filtering/filter_CUB.py --config configs/Flowers102/SEI_Noseed-2.0.yaml
# python filtering/filter_CUB.py --config configs/Cub2011/NoAligned.yaml
# python filtering/filter_CUB.py --config configs/DTD/SEI_Noseed-2.0.yaml



# python filtering/filter_IN200S.py --config configs/IN200_S/SEI_Gaussian_2.0.yaml
# python filtering/filter_IN200S.py --config configs/IN200_S/Ablation_GMM3.yaml
# python filtering/filter_IN200S.py --config configs/IN200_S/Ablation_GMM5.yaml
# python filtering/filter_IN200S.py --config configs/IN200_S/Ablation_GMM2.yaml

# python filtering/filter_ratio.py --config configs/Cub2011/SEI_Gaussian_ration_1.5.yaml
# python filtering/filter_ratio.py --config configs/Cub2011/SEI_Gaussian_ration_3.yaml
# python filtering/filter_ratio.py --config configs/Cub2011/SEI_Gaussian_ration_4.yaml

# python filtering/filter_ratio.py --config configs/IN200_S/SEI_ratio_1.5.yaml
# python filtering/filter_ratio.py --config configs/IN200_S/SEI_ratio_3.yaml
# python filtering/filter_ratio.py --config configs/IN200_S/SEI_ratio_4.yaml

# python filtering/filter_ratio.py --config configs/PETs/SEI_Gaussian_ration_1.5.yaml
# python filtering/filter_ratio.py --config configs/PETs/SEI_Gaussian_ration_3.yaml
python filtering/filter_ratio.py --config configs/PETs/SEI_Gaussian_ration_4.yaml
