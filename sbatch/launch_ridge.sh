#!/bin/bash
#SBATCH --job-name=vcc_ridge
#SBATCH --account=st-evanesce-1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --output=slurm/ridge_%j.out
#SBATCH --error=slurm/ridge_%j.err

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate vcc_py3.10

export NUMBA_CACHE_DIR=/scratch/st-evanesce-1/vivian/.numba_cache
export MPLCONFIGDIR=/scratch/st-evanesce-1/vivian/.matplotlib_cache

python run_cv.py --model ridge

