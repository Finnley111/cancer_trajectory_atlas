"""Quantitative batch-mixing diagnostic: kNN section-purity on a saved embedding.

Pure post-hoc analysis over an existing run's adata_full.h5ad. Does NOT
re-run PCA, Harmony, scVI, clustering, or DPT — it reads whichever of
obsm['X_pca_original'] (pre-correction), obsm['X_pca_harmony']
(post-Harmony), and obsm['X_scvi'] (post-scVI) are present and scores how
well-mixed obs['section_number'] is in each, using the same k/metric the
live pipeline uses for its Leiden neighbor graph.
"""

import inspect

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .clustering import cluster_leiden


def get_pipeline_k_and_metric():
    """Read the Leiden neighbor-graph k/metric from cluster_leiden's actual
    defaults, since run_all.py's call site never overrides them — this is
    the literal (k, metric) the live pipeline uses today."""
    params = inspect.signature(cluster_leiden).parameters
    return params["n_neighbors"].default, params["metric"].default


def chance_baseline(section_labels: np.ndarray) -> float:
    """Prevalence-weighted null: P(two distinct random patches share a section).

    sum_s n_s*(n_s-1) / (N*(N-1)) — the correct chance baseline when the
    section split is imbalanced, not a naive 0.5.
    """
    _, counts = np.unique(section_labels, return_counts=True)
    n = counts.sum()
    same_pairs = (counts * (counts - 1)).sum()
    return float(same_pairs / (n * (n - 1)))


def knn_batch_purity(X: np.ndarray, section_labels: np.ndarray, k: int, metric: str) -> float:
    """Cohort-mean fraction of each patch's k nearest neighbors (excluding
    itself) that share its section_number."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    indices = indices[:, 1:]  # drop self (first column, distance 0)

    neighbor_sections = section_labels[indices]
    own_section = section_labels[:, None]
    same_section_frac = (neighbor_sections == own_section).mean(axis=1)
    return float(same_section_frac.mean())


def compute_batch_mixing_report(adata_path) -> dict:
    adata = ad.read_h5ad(adata_path)

    if "section_number" not in adata.obs:
        raise ValueError(
            f"adata.obs has no 'section_number' column. Found: {list(adata.obs.columns)}"
        )
    section_labels = adata.obs["section_number"].astype(str).to_numpy()
    k, metric = get_pipeline_k_and_metric()

    # For harmony/scVI runs the pre-correction PCA is stored as X_pca_original.
    # For no-correction runs adata.X IS the PCA embedding (X_embed == X_pca).
    if "X_pca_original" in adata.obsm:
        X_raw = np.asarray(adata.obsm["X_pca_original"])
        print(f"  Using obsm['X_pca_original'] for raw_pca score.")
    elif adata.X is not None:
        X_raw = np.asarray(adata.X)
        print(f"  X_pca_original absent (no-correction run) — using adata.X as raw_pca.")
    else:
        raise ValueError(
            f"adata.obsm has no 'X_pca_original' and adata.X is None. "
            f"Found obsm keys: {list(adata.obsm.keys())}"
        )
    raw_pca_score = knn_batch_purity(X_raw, section_labels, k, metric)

    harmony_score = None
    if "X_pca_harmony" in adata.obsm:
        X_harmony = np.asarray(adata.obsm["X_pca_harmony"])
        harmony_score = knn_batch_purity(X_harmony, section_labels, k, metric)

    scvi_score = None
    if "X_scvi" in adata.obsm:
        X_scvi = np.asarray(adata.obsm["X_scvi"])
        scvi_score = knn_batch_purity(X_scvi, section_labels, k, metric)

    sections, counts = np.unique(section_labels, return_counts=True)
    per_section_counts = {s: int(c) for s, c in zip(sections, counts)}

    return {
        "raw_pca": raw_pca_score,
        "harmony": harmony_score,
        "scvi": scvi_score,
        "k_used": k,
        "n_patches": int(section_labels.shape[0]),
        "per_section_counts": per_section_counts,
        "chance_baseline": chance_baseline(section_labels),
    }
