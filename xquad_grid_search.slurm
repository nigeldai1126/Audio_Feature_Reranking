#!/bin/bash
#SBATCH --job-name=xquad_grid_search
#SBATCH --output=/scratch/jd5316/Capstone/Capstone/logs/xquad_grid_search_%j.out
#SBATCH --error=/scratch/jd5316/Capstone/Capstone/logs/xquad_grid_search_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jd5316@nyu.edu

# Load the conda module
module load anaconda3/2024.02


# Initialize conda in this shell
source ~/.bashrc

# Activate your environment
conda activate /scratch/jd5316/capstone_env

# Change to your working directory (update if needed)
cd /scratch/jd5316/Capstone/Capstone/notebooks


# Run your Python script or notebook
python /scratch/jd5316/Capstone/Capstone/notebooks/grid_search/xquad_grid_search