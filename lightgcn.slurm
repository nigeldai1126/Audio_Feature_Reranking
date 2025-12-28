#!/bin/bash
#SBATCH --job-name=lightgcn_run
#SBATCH --output=/scratch/jd5316/Capstone/Capstone/logs/lightgcn_%j.out
#SBATCH --error=/scratch/jd5316/Capstone/Capstone/logs/lightgcn_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jd5316@nyu.edu

# ==========================
#  Environment setup
# ==========================

# Load the conda module
module load anaconda3/2024.02

# Initialize conda in this shell
source ~/.bashrc

# Activate your environment
conda activate /scratch/jd5316/capstone_env

# Change to your working directory
cd /scratch/jd5316/Capstone/Capstone/notebooks

# (Optional) print files to confirm script visibility
ls -l *lightgcn_baseline*

# Run your Python script (no .py extension)
python /scratch/jd5316/Capstone/Capstone/notebooks/train_lightgcn_baseline
