#!/bin/bash
#SBATCH --job-name=audioprefsim
#SBATCH --output=/scratch/jd5316/Capstone/Capstone/logs/audioprefsim_%j.out
#SBATCH --error=/scratch/jd5316/Capstone/Capstone/logs/audioprefsim_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jd5316@nyu.edu

module load python/3.9.12
conda activate /scratch/jd5316/capstone_env

python /scratch/jd5316/Capstone/Capstone/notebooks/audioprefsim
echo "Job finished at $(date)"

