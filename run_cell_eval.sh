#!/bin/bash
#SBATCH --job-name=run_cell_eval_prep
#SBATCH --output=/home/axu03/vcc_scripts/cell_eval_prep.out
#SBATCH --error=/home/axu03/vcc_scripts/cell_eval_prep.err
#SBATCH --account=st-jimsun-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00

module load miniconda3
conda activate vcc_py3.10

cell-eval prep \
    -i /home/axu03/vcc_scripts/adata_pred_with_ntc.h5ad \
    -g /home/axu03/vcc_scripts/expected_genes.txt \
    -o /home/axu03/vcc_scripts/submission.vcc

