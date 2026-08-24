"""Diffusion pseudotime utilities built on scanpy."""

import numpy as np
from typing import Optional, Sequence, Tuple

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

    **The euclidean metric is not a choice.** It is scanpy's default. The
    ``sc.pp.neighbors`` call below passes no ``metric`` argument, so scanpy
    supplies ``metric='euclidean'``. There is no CLI flag for it; changing it
    means editing this line. ``n_neighbors`` IS configurable, via
    ``--diffmap-neighbors`` (default 30).

    See the module docstring of ``analysis/clustering.py`` for the full
    three-graph picture and why cluster membership and pseudotime position are
    answering different geometric questions.

    ``use_rep="X"`` makes scanpy read ``adata.X`` directly rather than
    recomputing a PCA, so the diffusion map sees exactly the matrix
    ``build_adata`` was handed, post-batch-correction when one is active.

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
    morphologically regular cluster, identified by visual inspection in
    Phase 3.

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

    SINGLE-ROOT PATH, used by run_individual.py ONLY. The atlas pipeline
    (run_all.py) does not call this; it uses compute_dpt_multi_root, which picks
    roots by nuclear density and median-aggregates over 20 of them. Do not
    "unify" the two: per-slide runs deliberately anchor on a cluster because a
    single slide has no cohort to rank densities against.

    Either specify root_cluster, which auto-selects the centroid patch and is
    the recommended form, or root_index when you already know the patch.

    Results stored in adata.obs['dpt_pseudotime'] and adata.obs['pseudotime']
    (the latter normalized to [0, 1]). Note this writes 'dpt_pseudotime' whereas
    compute_dpt_multi_root does not, and it does NOT write 'pseudotime_std'.

    Infinite DPT values are clamped to the maximum finite value with a warning.
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
    root_indices: Optional[Sequence[int]] = None,
) -> "ad.AnnData":
    """Run DPT from n_roots lowest-cellularity root candidates; median-aggregate.

    ``root_indices`` (default None) OVERRIDES the density ranking with a
    caller-supplied root set. Added for the v3 holeyness-rooted experiment
    (``analysis/holeyness_roots.py``), which anchors the axis on expert-annotated
    per-duct hole %, a quantity derived from hand annotation rather than from the
    pipeline's own pixels, so it removes the circularity of rooting on
    nuclear_density, which is simultaneously a validation feature and the
    cellularity-confound covariate.

    When it is None, which covers every production run to date, nothing below
    changes: the
    density rule, the non-finite masking, the clamping, the median aggregation
    and the min-max normalisation are all exactly as they were. When it is
    supplied, ONLY the root set changes; ``nuclear_density`` is then used solely
    for the diagnostic print and may legitimately be all-finite-but-unranked.

    A root-rule change is EXPECTED to alter the axis ORIENTATION and the root
    set, not the ordering: uniformly random 20-root sets already
    reproduce the production pseudotime at |rho| 0.78-0.89, so the manifold fixes
    the ordering and the roots fix only which end is zero.

    For each root candidate r:
        pt_r = scanpy DPT with iroot=r
    Final pseudotime = median(pt_matrix, axis=0), normalized to [0, 1].
    Uncertainty = std(pt_matrix, axis=0), stored as pseudotime_std (un-normalized).

    Results stored in adata.obs['pseudotime'] and adata.obs['pseudotime_std'].

    The root rule, stated exactly
    -----------------------------
    Non-finite densities are masked out FIRST, then the surviving indices are
    sorted by density and the lowest ``n_roots`` are taken::

        finite_idx = flatnonzero(isfinite(nuclear_density))
        roots      = finite_idx[argsort(nuclear_density[finite_idx])][:n_roots]

    This is NOT the same as ``argsort(nuclear_density)[:n_roots]``. The two agree
    only when every patch has a measured density. Several analysis and diagnostic
    modules quote the simpler form when re-deriving the root set. That happened
    to be correct for runs with zero extraction failures; it is not the rule.

    Anything needing the true root set should read ``adata.uns[...]`` rather than
    re-deriving it; see below.

    What is persisted
    -----------------
    ``adata.uns['dpt_root_candidates']``          int64 array, the actual roots used
    ``adata.uns['dpt_n_roots_excluded_nonfinite']`` count of patches masked out

    Persisting the roots is load-bearing, not a convenience: without it the root
    set can only be reconstructed by applying a rule to a density array that may
    not be the one the run actually used, which is unverifiable after the fact.
    Do not remove these writes.

    Note ``n_roots`` is silently clamped to the number of finite-density patches,
    so the realised root count can be lower than requested. Read
    ``len(adata.uns['dpt_root_candidates'])`` for the true count rather than
    assuming the CLI value.

    Infinite DPT values (disconnected manifold) are clamped per-root to that
    root's maximum finite value BEFORE aggregation, so a partially disconnected
    graph biases the median rather than propagating inf.
    """
    _require_scanpy()
    import scanpy as sc

    n_patches = len(adata)

    if root_indices is not None:
        root_candidates = [int(i) for i in root_indices]
        if not root_candidates:
            raise ValueError(
                "root_indices was supplied but empty — no DPT roots to run from. "
                "The caller must either supply at least one index or pass None to "
                "fall back to the nuclear-density rule."
            )
        bad = [i for i in root_candidates if i < 0 or i >= n_patches]
        if bad:
            raise ValueError(
                f"root_indices contains {len(bad)} out-of-range value(s) for an "
                f"adata with {n_patches} rows (first offenders: {bad[:5]}). This "
                "usually means the root set was derived against a DIFFERENT patch "
                "set than the one being run — check that the extraction settings "
                "match."
            )
        if len(set(root_candidates)) != len(root_candidates):
            raise ValueError(
                "root_indices contains duplicates. Each root must be distinct, "
                "otherwise the median across roots is silently weighted."
            )
        n_roots = len(root_candidates)
        n_excluded = 0
        print(f"  Multi-root DPT: {n_roots} CALLER-SUPPLIED root candidates "
              "(nuclear-density ranking BYPASSED)")
        nd_at_roots = nuclear_density[root_candidates]
        finite_nd = nd_at_roots[np.isfinite(nd_at_roots)]
        if finite_nd.size:
            print(f"    nuclear density at those roots: "
                  f"[{finite_nd.min():.4f}, {finite_nd.max():.4f}] "
                  f"(reported only; not used for selection)")
        adata.uns["dpt_root_source"] = "caller_supplied"
    else:
        adata.uns["dpt_root_source"] = "nuclear_density"
        root_candidates = None  # resolved by the density rule below

    if root_candidates is None:
        # Root candidates must come from patches whose density was actually MEASURED.
        # nan means extraction failed (validation/morphological_features.py); such a
        # patch has no density, so it cannot be "the least cellular". np.argsort does
        # place nan last, which happens to give the right answer here, but relying on
        # that is implicit and silently breaks if the sort or dtype ever changes.
        # The cost of being wrong is that the pseudotime ORIGIN gets anchored on
        # whichever patches crashed the segmenter, so mask explicitly instead.
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
    # run afterwards. It would have to be re-derived from a rule applied to a
    # different array than the one actually used, which is not verifiable.
    adata.uns["dpt_root_candidates"] = np.asarray(root_candidates, dtype=np.int64)
    adata.uns["dpt_n_roots_excluded_nonfinite"] = n_excluded

    pt_matrix = np.zeros((n_roots, n_patches), dtype=np.float64)
    # How many patches each root could not reach, BEFORE clamping. Previously
    # discarded, which made the clamp invisible: a root in a small component
    # returns inf for everything outside it, those become that root's own maximum,
    # and a near-constant vector enters the median. That is what produced Config
    # B's pseudotime_std at 30% of its range, and nothing in the output said so.
    n_nonfinite_per_root: list[int] = []

    for r_i, root_idx in enumerate(root_candidates):
        adata_tmp = adata.copy()
        adata_tmp.uns["iroot"] = int(root_idx)
        sc.tl.dpt(adata_tmp)
        pt = adata_tmp.obs["dpt_pseudotime"].values.copy()

        finite_mask = np.isfinite(pt)
        n_nonfinite_per_root.append(int((~finite_mask).sum()))
        if not finite_mask.all():
            pt[~finite_mask] = pt[finite_mask].max() if finite_mask.any() else 0.0

        pt_matrix[r_i] = pt

    nf = np.asarray(n_nonfinite_per_root, dtype=np.int64)
    adata.uns["dpt_n_nonfinite_per_root"] = nf
    n_roots_clamped = int((nf > 0).sum())
    if n_roots_clamped:
        print(f"  ⚠ CLAMPING FIRED: {n_roots_clamped}/{n_roots} root(s) could not reach "
              f"every patch (unreached counts: min {nf[nf>0].min()}, "
              f"max {nf.max()}, of {n_patches}).")
        print("    Each such root's unreachable patches were set to that root's OWN "
              "maximum, so it contributes a near-constant vector to the median and "
              "inflates pseudotime_std. Check graph connectivity before trusting the "
              "pseudotime: qc/graph_connectivity.py.")
    else:
        print(f"  All {n_roots} roots reached every patch — no clamping.")

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
    to misread. It groups patches by ``adata.obs['cluster']``, the Leiden labels
    from the k=15 COSINE graph, but the connectivities PAGA computes come from
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


# The single-root convenience wrapper `run_diffusion_pseudotime` lived here until
# Phase 5 (2026-08-12). It was dead: its only caller was run_train_test.py, which
# was deleted before this cleanup began. Moved verbatim to
# archive/analysis/diffusion_run_diffusion_pseudotime.py.
