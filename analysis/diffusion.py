"""Diffusion pseudotime utilities built on scanpy."""

import numpy as np
from typing import Optional, Tuple

def _require_scanpy():
    try:
        import scanpy as sc
        import anndata as ad
        # Let's force an igraph import test here just to be safe
        import igraph
    except ImportError as e:
        print(f"\n==================================================")
        print(f"CRITICAL DEPENDENCY ERROR DETECTED ON COMPUTE NODE")
        print(f"==================================================")
        print(f"The actual missing module is: {e.name}")
        print(f"Full error message: {e.msg}")
        print(f"==================================================\n")
        import traceback
        traceback.print_exc()
        raise e


def build_adata(
    X_pca: np.ndarray,
    cluster_labels: np.ndarray,
    slide_ids: np.ndarray,
    X_umap: Optional[np.ndarray] = None,
) -> "ad.AnnData":
    """Create an AnnData object from PCA features and metadata."""
    _require_scanpy()
    import anndata as ad # dynamic import

    adata = ad.AnnData(X=X_pca.astype(np.float32))
    adata.obs["cluster"] = cluster_labels.astype(str)
    adata.obs["slide_id"] = slide_ids.astype(str)

    if X_umap is not None:
        adata.obsm["X_umap"] = X_umap.astype(np.float32)

    return adata


def compute_diffusion_map(
    adata: "ad.AnnData",
    n_neighbors: int = 30,
    n_comps: int = 10,
) -> "ad.AnnData":
    """Build the neighbor graph and compute diffusion map components."""
    _require_scanpy()
    import scanpy as sc # dynamic import

    print(f"  Building neighbor graph (k={n_neighbors})...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")

    print(f"  Computing diffusion map ({n_comps} components)...")
    sc.tl.diffmap(adata, n_comps=n_comps)

    return adata


def choose_root_cell(
    adata: "ad.AnnData",
    root_cluster: str,
) -> int:
    """
    Select the root cell as the patch closest to the centroid of root_cluster.

    The root cluster should be the most well-organized, low-density,
    morphologically regular cluster — identified by visual inspection
    in Phase 3.

    Args:
        adata: AnnData with cluster labels in adata.obs['cluster'].
        root_cluster: String label of the root cluster.

    Returns:
        Global index of the root cell.
    """
    mask = adata.obs["cluster"].values == str(root_cluster)
    if mask.sum() == 0:
        raise ValueError(f"Root cluster '{root_cluster}' not found in data.")

    cluster_features = adata.X[mask]
    centroid = cluster_features.mean(axis=0)
    distances = np.linalg.norm(cluster_features - centroid, axis=1)
    nearest_local = int(np.argmin(distances))
    root_global = int(np.where(mask)[0][nearest_local])

    print(f"  Root cell: index {root_global} (cluster '{root_cluster}', "
          f"{mask.sum()} patches in cluster)")
    return root_global


def compute_dpt(
    adata: "ad.AnnData",
    root_cluster: Optional[str] = None,
    root_index: Optional[int] = None,
) -> "ad.AnnData":
    """
    Compute Diffusion Pseudotime from a biologically anchored root.

    Either specify root_cluster (recommended — will auto-select the centroid
    patch) or root_index (if you already know which patch to use).

    Results stored in adata.obs['dpt_pseudotime'] and adata.obs['pseudotime']
    (the latter normalized to [0, 1]).
    """
    _require_scanpy()
    import scanpy as sc # dynamic import

    if root_cluster is not None:
        root_idx = choose_root_cell(adata, root_cluster)
    elif root_index is not None:
        root_idx = root_index
    else:
        raise ValueError("Must specify either root_cluster or root_index.")

    adata.uns["iroot"] = root_idx

    print("  Computing diffusion pseudotime...")
    sc.tl.dpt(adata)

    # Normalize to [0, 1]
    pt = adata.obs["dpt_pseudotime"].values.copy()

    # Handle infinities (disconnected components)
    finite_mask = np.isfinite(pt)
    if not finite_mask.all():
        n_inf = (~finite_mask).sum()
        print(f"  WARNING: {n_inf} patches have infinite DPT (disconnected). "
              f"Clamping to max finite value.")
        pt[~finite_mask] = pt[finite_mask].max()

    pt_min, pt_max = pt.min(), pt.max()
    if pt_max - pt_min < 1e-10:
        print("  WARNING: DPT range is near-zero — no trajectory detected.")
        adata.obs["pseudotime"] = np.zeros(len(pt))
    else:
        adata.obs["pseudotime"] = (pt - pt_min) / (pt_max - pt_min)

    print(f"  Pseudotime range: [{pt_min:.4f}, {pt_max:.4f}] → normalized [0, 1]")
    return adata


def compute_dpt_multi_root(
    adata: "ad.AnnData",
    nuclear_density: np.ndarray,
    n_roots: int = 20,
) -> "ad.AnnData":
    """Run DPT from n_roots lowest-cellularity root candidates; median-aggregate.

    For each root candidate r:
        pt_r = scanpy DPT with iroot=r
    Final pseudotime = median(pt_matrix, axis=0), normalized to [0, 1].
    Uncertainty = std(pt_matrix, axis=0), stored as pseudotime_std (un-normalized).

    Results stored in adata.obs['pseudotime'] and adata.obs['pseudotime_std'].
    """
    _require_scanpy()
    import scanpy as sc

    n_patches = len(adata)
    n_roots = min(n_roots, n_patches)
    root_candidates = np.argsort(nuclear_density)[:n_roots].tolist()

    print(f"  Multi-root DPT: {n_roots} root candidates "
          f"(nuclear density range [{nuclear_density[root_candidates].min():.4f}, "
          f"{nuclear_density[root_candidates[-1]]:.4f}])")

    pt_matrix = np.zeros((n_roots, n_patches), dtype=np.float64)

    for r_i, root_idx in enumerate(root_candidates):
        adata_tmp = adata.copy()
        adata_tmp.uns["iroot"] = int(root_idx)
        sc.tl.dpt(adata_tmp)
        pt = adata_tmp.obs["dpt_pseudotime"].values.copy()

        finite_mask = np.isfinite(pt)
        if not finite_mask.all():
            pt[~finite_mask] = pt[finite_mask].max() if finite_mask.any() else 0.0

        pt_matrix[r_i] = pt

    pseudotime_median = np.median(pt_matrix, axis=0)
    pseudotime_std    = np.std(pt_matrix, axis=0)

    pt_min, pt_max = pseudotime_median.min(), pseudotime_median.max()
    if pt_max - pt_min < 1e-10:
        print("  WARNING: DPT range is near-zero — no trajectory detected.")
        adata.obs["pseudotime"] = np.zeros(n_patches)
    else:
        adata.obs["pseudotime"] = (pseudotime_median - pt_min) / (pt_max - pt_min)

    adata.obs["pseudotime_std"] = pseudotime_std

    print(f"  Pseudotime median range: [{pt_min:.4f}, {pt_max:.4f}] → normalized [0, 1]")
    print(f"  Pseudotime std range:    [{pseudotime_std.min():.4f}, {pseudotime_std.max():.4f}]")
    return adata


def compute_paga_topology(
    adata: "ad.AnnData",
    groups: str = "cluster",
    threshold: float = 0.05,
) -> Tuple[int, "ad.AnnData"]:
    """Run PAGA and report connected-component count of the cluster graph.

    Requires that sc.pp.neighbors has already been called (compute_diffusion_map).

    Returns:
        n_components: Number of connected components in the thresholded cluster graph.
        adata: AnnData with paga results added to adata.uns['paga'].
    """
    _require_scanpy()
    import scanpy as sc
    from scipy.sparse.csgraph import connected_components as sp_connected_components
    from scipy.sparse import csr_matrix

    # PAGA requires a categorical obs column.
    if not hasattr(adata.obs[groups], "cat"):
        adata.obs[groups] = adata.obs[groups].astype("category")

    print(f"  Running PAGA (groups='{groups}')...")
    sc.tl.paga(adata, groups=groups)

    conn = adata.uns["paga"]["connectivities"]
    conn_thresh = (conn > threshold).astype(np.float32)
    n_components, _ = sp_connected_components(
        csr_matrix(conn_thresh), directed=False
    )

    print(f"  PAGA topology: {n_components} connected component(s) "
          f"(threshold={threshold:.2f})")
    if n_components == 1:
        print("  → SINGLE component. Manifold is connected; DPT is valid.")
    else:
        print(f"  → {n_components} DISCONNECTED components. "
              f"Check qc_umap_section_vs_cluster.png to see if they map onto section_number.")

    return n_components, adata


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
