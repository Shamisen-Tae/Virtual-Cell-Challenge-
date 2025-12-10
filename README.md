## CPSC 545 Final Project: Benchmarking Machine Learning Models for Single-Cell Perturbation Response Prediction
This repository contains code and experiments for benchmarking a variety of machine learning models (linear models, ensemble methods, and variational autoencoder) on single-cell perturbation response prediction for the 2025 ARC Virtual Cell Challenge.
### Directory Structure:
* `figures/` contains our preprocessing figures (computed from `notebooks/eda.ipynb`)
* `notebooks/` contains our Jupyter notebook used for exploratory data analysis and preprocessing.
* `results/` contains the CSV files with our results: MAE, PDS, DES scores for each model. `*cv_detailed_celleval.csv` contains the scores at each CV fold, and `*cv_summary_celleval.csv` contains the scores averaged together with standard deviation.
* `sbatch/` contains the SBATCH scripts we used to launch experiments on the UBC sockeye cluster. These show the resources requested for each run.
* `scripts/` contains the Python script used to run our experiments and our VAE implementation.

### How To Run Experiments:
#### Recommended packages (Python >=3.10 recommended):
`pip install requirements.txt`
* cell-eval                 0.6.6
* anndata                   0.11.3
* ipykernel                 6.31.0
* ipython                   8.30.0
* jupyterlab                4.4.7
* matplotlib                3.10.7
* numpy                     2.2.6
* pandas                    2.3.3
* scanpy                    1.11.5
* scikit-learn              1.7.2
* scipy                     1.15.2
* seaborn                   0.13.2
* torch                     2.9.1

#### Script:
`scripts/run_cv.py` is the main experiment script. 

#### Required arguments: 
* the model type to run (choose from `baseline`, `ridge`, `elasticnet`, `randomforest`, `gradientboost`, and `vae`)
* the number of folds for cross-validation (default is `5`)
* the location of the VCC training file.


#### Example
```
python scripts/run_cv.py \
  --model ridge \
  --folds 5 \
  --train_path data/vcc_train.h5ad
```
#### Running on UBC Sockeye
The SLURM submission scripts in `sbatch/` show the resource configurations used for each model.
After updating them to match your account configuration, jobs can be submitted like so:
```
sbatch sbatch/launch_ridge.sh
```

### References
Y. H. Roohani, T. J. Hua, P.-Y. Tung, L. R. Bounds, F. B. Yu, A. Dobin, N. Teyssier, A. Adduri,
A. Woodrow, B. S. Plosky, R. Mehta, B. Hsu, J. Sullivan, C. Ricci-Tam, N. Li, J. Kazaks, L. A.
Gilbert, S. Konermann, P. D. Hsu, H. Goodarzi, and D. P. Burke. Virtual cell challenge: Toward a
turing test for the virtual cell. Cell, 188(13):3370–3374, 2025. ISSN 00928674. doi: 10.1016/j.cell.
2025.06.008.
