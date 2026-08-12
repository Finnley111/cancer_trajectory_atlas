"""ARCHIVED 2026-08-12 (Phase 5) — extracted verbatim from analysis/diffusion.py.

Dead code. `run_diffusion_pseudotime` was a single-root convenience wrapper
chaining build_adata -> compute_diffusion_map -> compute_dpt. Its only caller was
the top-level `run_train_test.py`, which was deleted before this cleanup began
(the sole surviving trace was a stale __pycache__/run_train_test.cpython-310.pyc
still referencing the symbol). Zero callers remained in any .py, .sh, or .md.

It is superseded in substance by `compute_dpt_multi_root`, which the atlas
pipeline uses: that runs DPT from 20 density-ranked roots and median-aggregates,
rather than from a single cluster-anchored root.

This file is NOT importable as-is — it is a verbatim record of the removed
function, kept so the deletion is reversible. The names it calls (build_adata,
compute_diffusion_map, compute_dpt) still exist in analysis/diffusion.py; to
restore, paste the function back there rather than importing this module.
"""

# --- verbatim extract from analysis/diffusion.py, immediately before removal ---

# ── Convenience wrapper ──────────────────────────────────────────────

def run_diffusion_pseudotime(
    X_pca: np.ndarray,
    cluster_labels: np.ndarray,
    slide_ids: np.ndarray,
    root_cluster: str,
    X_umap: Optional[np.ndarray] = None,
    n_neighbors: int = 30,
    n_comps: int = 10,
) -> "ad.AnnData":
    """
    Full Phase 4 pipeline: build AnnData → diffusion map → DPT.

    Returns the fully populated AnnData object.
    """
    adata = build_adata(X_pca, cluster_labels, slide_ids, X_umap)
    compute_diffusion_map(adata, n_neighbors=n_neighbors, n_comps=n_comps)
    compute_dpt(adata, root_cluster=root_cluster)
    return adata
