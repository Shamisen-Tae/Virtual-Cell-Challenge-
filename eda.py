import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor

print("Loading data...")
adata = sc.read_h5ad("/scratch/st-evanesce-1/vivian/vcc_data/adata_Training.h5ad")

output_h5ad = "/scratch/st-evanesce-1/vivian/adata_elastic_pred.h5ad"
test_csv = "/scratch/st-evanesce-1/vivian/vcc_data/pert_counts_Test.csv"
test_set = pd.read_csv(test_csv)
expected_perturbations = test_set["target_gene"].tolist()

genes = adata.var_names
expected_genes_file = "/scratch/st-evanesce-1/vivian/expected_genes.txt"
np.savetxt(expected_genes_file, genes, fmt="%s")


# one-hot encoding for training features
train_target_genes = adata.obs["target_gene"].values
unique_genes = sorted(set(train_target_genes))
gene_to_idx = {g: i for i, g in enumerate(unique_genes)}

X_train = np.zeros((len(train_target_genes), len(unique_genes)))
for i, gene in enumerate(train_target_genes):
    X_train[i, gene_to_idx[gene]] = 1

y_train = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

print("Training ElasticNet model...")
model = MultiOutputRegressor(
    ElasticNet(
        alpha=1.0,
        l1_ratio=0.5,
    ),
    n_jobs=24,
)

model.fit(X_train, y_train)

# encoding test features
X_test = np.zeros((len(expected_perturbations), len(unique_genes)))
for i, target_gene in enumerate(expected_perturbations):
    if target_gene in gene_to_idx:
        X_test[i, gene_to_idx[target_gene]] = 1

predictions = model.predict(X_test)
pred_df = pd.DataFrame(predictions, columns=genes)
pred_df.insert(0, "target_gene", expected_perturbations)


# Add negative control cells
ntc_cells = adata[adata.obs["target_gene"] == "non-targeting"]
n_ntc = min(1000, ntc_cells.n_obs)
sample_indices = np.random.choice(ntc_cells.n_obs, n_ntc, replace=False)
ntc_sample = ntc_cells[sample_indices]

ntc_df = pd.DataFrame(
    ntc_sample.X.toarray() if hasattr(ntc_sample.X, "toarray") else ntc_sample.X,
    columns=ntc_sample.var_names
)
ntc_df.insert(0, "target_gene", ["non-targeting"] * n_ntc)

# Merge predictions and controls
pred_df_full = pd.concat([pred_df, ntc_df], ignore_index=True)

# Clip negative values in case elasticnet results in any
pred_df_full_values = pred_df_full[genes].values.astype(np.float32)
pred_df_full_values = np.maximum(pred_df_full_values, 0)

adata_pred = sc.AnnData(X=pred_df_full_values)
adata_pred.var_names = genes
adata_pred.obs["target_gene"] = pred_df_full["target_gene"].values.astype(str)

adata_pred.X = np.log1p(adata_pred.X)

adata_pred.write(output_h5ad)

print(f"Saved prediction AnnData to {output_h5ad}")

