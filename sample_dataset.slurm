#!/bin/bash
#SBATCH --job-name=sample_dataset
#SBATCH --output=/scratch/jd5316/Capstone/Capstone/logs/sample_dataset_%j.out
#SBATCH --error=/scratch/jd5316/Capstone/Capstone/logs/sample_dataset_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jd5316@nyu.edu

module load python/3.9.12
conda activate /scratch/jd5316/capstone_env

python /scratch/jd5316/Capstone/Capstone/notebooks/sample_dataset
echo "Job finished at $(date)"

