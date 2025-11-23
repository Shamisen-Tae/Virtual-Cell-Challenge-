#!/bin/bash
#SBATCH --job-name=vcc_baseline
#SBATCH --account=st-evanesce-1
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --output=slurm/cv_baseline_%j.out
#SBATCH --error=slurm/cv_baseline_%j.err

source ~/.bashrc
conda activate vcc_py3.10

export NUMBA_CACHE_DIR=/scratch/st-evanesce-1/vivian/.numba_cache
export MPLCONFIGDIR=/scratch/st-evanesce-1/vivian/.matplotlib_cache

python eda_cv.py --model baseline

