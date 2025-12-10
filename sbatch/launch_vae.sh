#!/bin/bash
#SBATCH --job-name=vcc_vae
#SBATCH --account=st-evanesce-1-gpu
#SBATCH --gpus=1
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --output=slurm/vae_%j.out
#SBATCH --error=slurm/vae_%j.err

source ~/.bashrc
conda activate vcc_py3.10

export NUMBA_CACHE_DIR=/scratch/st-evanesce-1/vivian/.numba_cache
export MPLCONFIGDIR=/scratch/st-evanesce-1/vivian/.matplotlib_cache

python run_cv.py --model vae

