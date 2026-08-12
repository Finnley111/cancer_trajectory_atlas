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
    """Build the neighbor graph and compute diffusion map components.

    THIS IS NOT THE CLUSTERING GRAPH. It is a second, independent k-NN graph
    over the same ``X_embed`` matrix, and it differs from the Leiden graph in
    both k and metric:

        Leiden      k=15, cosine     (analysis/clustering.py:cluster_leiden)
        this graph  k=30, EUCLIDEAN  (here)

    **The euclidean metric is not a choice — it is scanpy's default.** The
    ``sc.pp.neighbors`` call below passes no ``metric`` argument, so scanpy
    supplies ``metric='euclidean'``. There is no CLI flag for it; changing it
    means editing this line. ``n_neighbors`` IS configurable, via
    ``--diffmap-neighbors`` (default 30).

    See the module docstring of ``analysis/clustering.py`` for the full
    three-graph picture and why cluster membership and pseudotime position are
    answering different geometric questions.

    ``use_rep="X"`` makes scanpy read ``adata.X`` directly rather than
    recomputing a PCA, so the diffusion map sees exactly the matrix
    ``build_adata`` was handed — post-batch-correction when one is active.

    This function must run before ``compute_paga_topology`` and
    ``compute_dpt_multi_root``; both consume the graph it writes into
    ``adata.uns['neighbors']``.
    """
    _require_scanpy()
    import scanpy as sc # dynamic import

    print(f"  Building neighbor graph (k={n_neighbors})...")
    # No metric= here: scanpy defaults to euclidean. Deliberate to leave as-is
    # (every published run used it), but it is NOT the cosine metric Leiden uses.
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

    # Root candidates must come from patches whose density was actually MEASURED.
    # nan means extraction failed (validation/morphological_features.py); such a
    # patch has no density, so it cannot be "the least cellular". np.argsort does
    # place nan last, which happens to give the right answer here, but relying on
    # that is implicit and silently breaks if the sort or dtype ever changes —
    # and the cost of being wrong is that the pseudotime ORIGIN is anchored on
    # whichever patches crashed the segmenter. So mask explicitly.
    finite = np.isfinite(nuclear_density)
    n_excluded = int((~finite).sum())
    if n_excluded:
        print(f"  Excluding {n_excluded} patch(es) with non-finite nuclear density "
              "from root candidates (failed extraction).")
    finite_idx = np.flatnonzero(finite)
    if finite_idx.size == 0:
        raise ValueError(
            "No patch has a finite nuclear density — every extraction failed, so "
            "DPT roots cannot be selected. Check the feature-extraction log."
        )

    n_roots = min(n_roots, finite_idx.size)
    order = finite_idx[np.argsort(nuclear_density[finite_idx])]
    root_candidates = order[:n_roots].tolist()

    print(f"  Multi-root DPT: {n_roots} root candidates "
          f"(nuclear density range [{nuclear_density[root_candidates].min():.4f}, "
          f"{nuclear_density[root_candidates[-1]]:.4f}])")

    # Persist the roots. Without this, the root set cannot be recovered from the
    # run afterwards — it has to be re-derived from a rule applied to a different
    # array than the one actually used, which is not verifiable.
    adata.uns["dpt_root_candidates"] = np.asarray(root_candidates, dtype=np.int64)
    adata.uns["dpt_n_roots_excluded_nonfinite"] = n_excluded

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

    THIS GATE MIXES BOTH NEIGHBOUR GRAPHS, which is what makes its verdict easy
    to misread. It groups patches by ``adata.obs['cluster']`` — Leiden labels
    from the k=15 COSINE graph — but the connectivities PAGA computes come from
    ``adata.uns['neighbors']``, the k=30 EUCLIDEAN diffusion graph.

    So "SINGLE component -> DPT is valid" means precisely: *the euclidean k=30
    manifold is connected between the cosine k=15 clusters*. It is not a
    statement about the clustering graph, and a disconnected result does not
    imply the Leiden clusters are separable. See ``analysis/clustering.py``'s
    module docstring for the three-graph layout.

    ``threshold=0.05`` is applied to PAGA connectivities before counting
    components, so the answer is threshold-dependent; a lower threshold merges
    components. It has never been exposed to the CLI.

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
