import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
import subprocess
import os
import json
from cell_eval import MetricsEvaluator

parser = argparse.ArgumentParser(description='Run CV evaluation for vcc models')
parser.add_argument('--model', type=str, required=True, choices=['baseline', 'elasticnet', 'rf', 'vae']) #TODO: implement RF and VAE
parser.add_argument('--n_folds', type=int, default=5, help='number of CV folds (default 5)')
args = parser.parse_args()

train_file = "/scratch/st-evanesce-1/vivian/vcc_data/adata_Training.h5ad"

print("\nLoading training data...")
adata_train = sc.read_h5ad(train_file)

# Prepare data
train_target_genes = adata_train.obs["target_gene"].values
unique_genes = sorted(set(train_target_genes))
gene_to_idx = {g: i for i, g in enumerate(unique_genes)}

# Creating one-hot encoded features for the predictive models
X = np.zeros((len(train_target_genes), len(unique_genes)))
for i, gene in enumerate(train_target_genes):
    X[i, gene_to_idx[gene]] = 1

y = adata_train.X.toarray() if hasattr(adata_train.X, "toarray") else adata_train.X


# apply log1p transformation
y_raw = adata_train.X.toarray() if hasattr(adata_train.X, "toarray") else adata_train.X
y = np.log1p(y_raw)


def create_anndata_for_celleval(predictions, target_genes, gene_names):
    """
    Create AnnData object in format expected by cell-eval
    
    Args:
        predictions: expression matrix (n_cells, n_genes)
        target_genes: array of target gene names for each cell
        gene_names: list of all gene names
    """

    # Clip negative values
    predictions = np.maximum(predictions, 0)
    
    # Create AnnData
    adata = sc.AnnData(X=predictions.astype(np.float32))
    adata.var_names = gene_names
    adata.obs["target_gene"] = target_genes.astype(str)
    
    return adata


def run_celleval_metrics(adata_pred, adata_truth):
    """
    Run cell-eval to compute VCC metrics
    
    Returns dict with: mae, des, pds
    """
    try:
        evaluator = MetricsEvaluator(
            adata_pred=adata_truth,
            adata_real=adata_real,
            num_threads=24,
        )
        
        results, agg_results = evaluator.compute()
        
        # Extract mean row 
        mean_row = agg_results.filter(agg_results['statistic'] == 'mean')
        
        metrics = {
            'mae': mean_row['mae'][0] if 'mae' in agg_results.columns else np.nan,
            'des': mean_row['overlap_at_N'][0] if 'overlap_at_N' in agg_results.columns else np.nan,
            'pds': mean_row['discrimination_score_l1'][0] if 'discrimination_score_l1' in agg_results.columns else np.nan,
        }
        
        return metrics
    except Exception as e:
        print(f"Error running cell-eval: {e}")
        import traceback
        traceback.print_exc()
        return {'mae': np.nan, 'des': np.nan, 'pds': np.nan}

print(f"Running {args.n_folds}-Fold Cross-Validation")



kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)

print(f"\nRunning model: {args.model}")

# Store results for each fold
fold_results = {
    'train_mae': [], 'val_mae': [], 'train_mse': [], 'val_mse': [],
    'celleval_mae': [], 'celleval_des': [], 'celleval_pds': []
}

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold_idx + 1}/{args.n_folds}")
    
    # Split data
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    train_genes_fold = train_target_genes[train_idx]
    val_genes_fold = train_target_genes[val_idx]
    
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    
    # Create ground truth AnnData for validation set
    adata_truth = create_anndata_for_celleval(
        y_val_fold, val_genes_fold, adata_train.var_names
    )
    
    # BASELINE MODEL
    if args.model == 'baseline':
        print("\n[Baseline] Computing mean expression...")
        baseline_pred = np.asarray(y_train_fold.mean(axis=0)).ravel()
        y_train_pred_baseline = np.tile(baseline_pred, (len(train_idx), 1))
        y_val_pred_baseline = np.tile(baseline_pred, (len(val_idx), 1))
        fold_results['train_mae'].append(
            mean_absolute_error(y_train_fold, y_train_pred_baseline)
        )
        fold_results['val_mae'].append(
            mean_absolute_error(y_val_fold, y_val_pred_baseline)
        )
    
        # Save predictions for cell-eval
        adata_pred = create_anndata_for_celleval(
            y_val_pred_baseline, val_genes_fold, adata_train.var_names
        )
    
        # Run cell-eval
        print("[Baseline] Running cell-eval...")
        celleval_metrics = run_celleval_metrics(adata_pred, adata_truth)
        fold_results['celleval_mae'].append(celleval_metrics.get('mae', np.nan))
        fold_results['celleval_des'].append(celleval_metrics.get('des', np.nan))
        fold_results['celleval_pds'].append(celleval_metrics.get('pds', np.nan))
    
        print(f"  Train MAE: {fold_results['train_mae'][-1]:.4f}")
        print(f"  Val MAE: {fold_results['val_mae'][-1]:.4f}")
        print(f"  cell-eval MAE: {celleval_metrics.get('mae', 'N/A')}")
        print(f"  cell-eval DES: {celleval_metrics.get('des', 'N/A')}")
        print(f"  cell-eval PDS: {celleval_metrics.get('pds', 'N/A')}")
    
    
    # ELASTICNET MODEL
    if args.model == 'elasticnet':
        print("\n[ElasticNet] Training model...")
        elasticnet_model = MultiOutputRegressor(
            ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-3, selection='random'),
            n_jobs=24
        )
        elasticnet_model.fit(X_train_fold, y_train_fold)
    
        print("[ElasticNet] Predicting...")
        y_train_pred_en = elasticnet_model.predict(X_train_fold)
        y_val_pred_en = elasticnet_model.predict(X_val_fold)
    
        y_train_pred_en = np.maximum(y_train_pred_en, 0)
        y_val_pred_en = np.maximum(y_val_pred_en, 0)
    
        fold_results['train_mae'].append(
            mean_absolute_error(y_train_fold, y_train_pred_en)
        )
        fold_results['val_mae'].append(
            mean_absolute_error(y_val_fold, y_val_pred_en)
        )
    
        # Save predictions for cell-eval
        adata_pred = create_anndata_for_celleval(
            y_val_pred_en, val_genes_fold, adata_train.var_names
        )
    
        # Run cell-eval
        print("[ElasticNet] Running cell-eval...")
        celleval_metrics = run_celleval_metrics(adata_pred, adata_truth)
        fold_results['celleval_mae'].append(celleval_metrics.get('mae', np.nan))
        fold_results['celleval_des'].append(celleval_metrics.get('des', np.nan))
        fold_results['celleval_pds'].append(celleval_metrics.get('pds', np.nan))
    
        print(f"  Train MAE: {fold_results['train_mae'][-1]:.4f}")
        print(f"  Val MAE: {fold_results['val_mae'][-1]:.4f}")
        print(f"  cell-eval MAE: {celleval_metrics.get('mae', 'N/A')}")
        print(f"  cell-eval DES: {celleval_metrics.get('des', 'N/A')}")
        print(f"  cell-eval PDS: {celleval_metrics.get('pds', 'N/A')}")


        # TODO: RANDOM FOREST & VAE MODELS


# AGGREGATE RESULTS

def format_metric(values):
    valid_values = [v for v in values if not np.isnan(v)]
    if len(valid_values) == 0:
        return "N/A"
    return f"{np.mean(valid_values):.4f}±{np.std(valid_values):.4f}"

summary = pd.DataFrame({
    'Model': [f'{args.model}'],
    'Val MAE (sklearn)': [
        format_metric(fold_results['val_mae']),
    ],
    'MAE (cell-eval)': [
        format_metric(fold_results['celleval_mae']),
    ],
    'DES (cell-eval)': [
        format_metric(fold_results['celleval_des']),
    ],
    'PDS (cell-eval)': [
        format_metric(fold_results['celleval_pds']),
    ]
})

print("\n" + summary.to_string(index=False))

# Save results
summary.to_csv(f'{args.model}_cv_summary_celleval.csv', index=False)

# Save detailed results
detailed_df = pd.DataFrame({
    'Fold': list(range(1, args.n_folds + 1)),
    f'{args.model}_MAE': fold_results['celleval_mae'],
    f'{args.model}_DES': fold_results['celleval_des'],
    f'{args.model}_PDS': fold_results['celleval_pds'],
})
detailed_df.to_csv(f'{args.model}_cv_detailed_celleval.csv', index=False)

print(f"\nSaved: {args.model}_cv_summary_celleval.csv")
print(f"\nSaved: {args.model}_detailed_celleval.csv")

