#!/bin/bash
#SBATCH --job-name=graf_uniform
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=jfb4@princeton.edu
#SBATCH --output=/tigress/jfb4/graf/out/uniform/slurm.out

module load anaconda3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jfb4/.conda/envs/graf/lib/
conda activate graf

python3 /home/jfb4/SeeingWithSound/code/graf/train.py \
/home/jfb4/SeeingWithSound/code/graf/configs/carla_uniform.yaml
