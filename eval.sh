#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o "/scratch/inf0/user/gtiwari/slurm-%A.out"
#SBATCH -e "/scratch/inf0/user/gtiwari/slurm-%A.err"
#SBATCH --gres gpu:1

echo "canonical pose data for whole body"
cd /BS/garvita/work/code/sizer
source /BS/garvita/static00/software/miniconda3/etc/profile.d/conda.sh
conda activate pytorch3d

python trainer.py
