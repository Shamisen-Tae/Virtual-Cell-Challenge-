#!/bin/bash
#SBATCH --job-name=vcc_randomforest
#SBATCH --account=st-evanesce-1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --output=slurm/cv_randomforest_%j.out
#SBATCH --error=slurm/cv_randomforest_%j.err

source ~/.bashrc
conda activate vcc_py3.10

export NUMBA_CACHE_DIR=/scratch/st-evanesce-1/vivian/.numba_cache
export MPLCONFIGDIR=/scratch/st-evanesce-1/vivian/.matplotlib_cache

python eda_cv.py --model randomforest

