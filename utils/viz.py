"""Plotting helpers for cluster, pseudotime, and validation figures.

Every function here writes a PNG and returns None. They are called for their
side effect, print the path they wrote, and create parent directories as needed.

Figures are DIAGNOSTIC OUTPUT, not analysis. Nothing downstream reads a PNG, and
no recorded number comes from one. The UMAP coordinates that several of these
functions plot are themselves display-only: the trajectory is computed on the
diffusion map, not on the UMAP embedding, so two clusters appearing adjacent in a
UMAP panel implies nothing about their pseudotime.

The backend is forced to Agg at import time because these run on compute nodes
with no display. That import must stay before ``pyplot``.

Two conventions worth knowing before comparing panels.

Cluster colours come from tab10/tab20 indexed by position in the SORTED cluster
list. Ids therefore keep a fixed colour within a run, but not across runs where
the cluster count differs, so the same colour in two runs is not the same
cluster.

Pseudotime colour scales are NOT uniform across this module. The per-slide
spatial panels pin ``vmin=0, vmax=1``, so colour means the same thing on every
slide. The UMAP pseudotime plots do not, and autoscale to the data range
instead. Do not read colour across the two kinds of figure.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, Optional


def _ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_umap_clusters(X_umap, cluster_labels, save_path, title="UMAP — Morphological Clusters"):
    """Scatter the UMAP embedding coloured by cluster.

    ``X_umap`` is (N, 2) and ``cluster_labels`` is (N,), aligned row-for-row.
    HDBSCAN noise (-1) is drawn like any other group and labelled "Noise".

    The legend is suppressed above 12 clusters, where it would take more space
    than the plot. Colours still come from tab20, so beyond 20 clusters they
    start repeating and the panel becomes unreadable.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    unique = sorted(set(cluster_labels))
    cmap = plt.cm.get_cmap("tab20", len(unique))
    for i, c in enumerate(unique):
        mask = cluster_labels == c
        label = f"Cluster {c}" if c != -1 else "Noise"
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], s=5, alpha=0.5,
                   color=cmap(i), label=label, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if len(unique) <= 12:
        ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cluster_patch_grid(
    patches, cluster_labels, cluster_centroids,
    save_path, n_per_cluster=5, title="Representative Patches per Cluster",
):
    """Grid of example patches per cluster, one cluster per row.

    WARNING: these are NOT representative patches, despite what this docstring
    said until 2026-08-24 and despite the figure's appearance. The rows are
    ``indices[:n_per_cluster]``, the first patches of each cluster in array
    order. Array order is extraction order, so a row shows whichever slide was
    processed first, sampled from its top-left corner. It is a spatially and
    per-slide biased sample of the cluster, not its centre.

    ``get_cluster_centroids`` does compute a nearest-to-centroid index and passes
    it in as ``cluster_centroids[c][1]``, but this function reads only the keys.
    Making the figure match its name means using that index; that would change
    every cluster grid produced so far, so it is a behaviour change rather than a
    fix to apply silently.

    Args:
        patches: (N, H, W, 3), indexed by the same positions as cluster_labels.
        cluster_labels: (N,) cluster assignments.
        cluster_centroids: {cluster_id: (centroid, nearest_global_idx)}. Only the
            keys are used, and they determine which clusters get a row and in
            what order.
        n_per_cluster: columns in the grid. Rows with fewer members than this
            leave blank cells rather than erroring.

    Noise (label -1) gets a row only if it appears in ``cluster_centroids``.
    """
    _ensure_dir(save_path)
    clusters = sorted(cluster_centroids.keys())
    n_clusters = len(clusters)

    fig, axes = plt.subplots(n_clusters, n_per_cluster, figsize=(n_per_cluster * 2, n_clusters * 2))
    if n_clusters == 1:
        axes = axes[np.newaxis, :]

    for row, c in enumerate(clusters):
        mask = cluster_labels == c
        indices = np.where(mask)[0]
        # First n by array index, which is extraction order, NOT proximity to the
        # centroid. See the docstring: the previous comment here claimed the
        # opposite and was wrong.
        sample = indices[:n_per_cluster]
        for col in range(n_per_cluster):
            ax = axes[row, col]
            if col < len(sample):
                ax.imshow(patches[sample[col]])
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"C{c}", fontsize=10, rotation=0, labelpad=30)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spatial_clusters(coords, cluster_labels, slide_ids, save_dir,
                          prefix="spatial_clusters", slide_name_map=None,
                          pseudotime=None):
    """Write one spatial map per slide, patches drawn at their slide coordinates.

    Writes ``{prefix}_{sid}.png`` into ``save_dir`` for every distinct value in
    ``slide_ids``, so this produces as many files as there are slides.

    Args:
        coords: (N, 2) patch positions in CROPPED-PNG pixel space, the same
            coordinates results.csv carries. Not ratio space.
        cluster_labels: (N,) cluster ids. Noise (-1) is excluded entirely, so
            noise patches leave gaps in the map rather than appearing in a
            colour of their own.
        slide_ids: (N,) integer slide index.
        slide_name_map: {int sid: display name}. Falls back to str(sid) when None
            or when a key is missing, which is why an unmapped slide shows a bare
            integer in its title.
        pseudotime: (N,) in [0, 1]. When given, each figure gains a second panel
            pinned to vmin=0/vmax=1 so colour is comparable across slides.

    The colour map is built once from the full label set before the loop, so a
    cluster keeps its colour on every slide even when absent from some of them.
    Building it per slide would silently recolour clusters between panels.

    Y is inverted on every axis because these are image coordinates with the
    origin at the top left. Without that the maps are vertically mirrored
    relative to the slide.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build a consistent color assignment from the full label set so cluster N
    # is always the same color regardless of which slide is being plotted.
    unique_clusters = sorted(int(c) for c in set(cluster_labels) if c != -1)
    n_clusters = len(unique_clusters)
    cmap = plt.cm.get_cmap("tab10" if n_clusters <= 10 else "tab20", n_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}

    for sid in np.unique(slide_ids):
        mask = slide_ids == sid
        slide_name = (slide_name_map or {}).get(int(sid), str(sid))

        n_panels = 2 if pseudotime is not None else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 10))
        if n_panels == 1:
            axes = [axes]

        # One scatter call per cluster, rather than a single call with a colour
        # array, so matplotlib builds a discrete legend entry per cluster.
        ax = axes[0]
        for c in unique_clusters:
            cmask = mask & (cluster_labels == c)
            if not cmask.any():
                continue
            ax.scatter(coords[cmask, 0], coords[cmask, 1],
                       color=cmap(cluster_to_idx[c]), s=15, marker="s", alpha=0.8,
                       label=f"Cluster {c}")
        ax.invert_yaxis()  # image coords: y increases downward (origin top-left)
        ax.set_aspect("equal")
        ax.set_title(f"Spatial Clusters — {slide_name}")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
                  markerscale=2, fontsize=9, title="Clusters")

        # vmin/vmax are pinned so this panel is comparable across slides. Letting
        # it autoscale would make each slide's own range fill the colour map and
        # every slide would look like it spans the full trajectory.
        if pseudotime is not None:
            ax2 = axes[1]
            sc = ax2.scatter(coords[mask, 0], coords[mask, 1], c=pseudotime[mask],
                             s=15, cmap="viridis", marker="s", vmin=0, vmax=1)
            ax2.invert_yaxis()
            ax2.set_aspect("equal")
            ax2.set_title(f"Spatial Pseudotime — {slide_name}")
            ax2.set_xlabel("X (pixels)")
            ax2.set_ylabel("Y (pixels)")
            plt.colorbar(sc, ax=ax2, label="Pseudotime", shrink=0.8, pad=0.02)

        plt.tight_layout()
        fname = save_dir / f"{prefix}_{sid}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


def plot_umap_pseudotime(X_umap, pseudotime, save_path, title="UMAP — Pseudotime"):
    """Scatter the UMAP embedding coloured by pseudotime.

    ``X_umap`` (N, 2) and ``pseudotime`` (N,) must be aligned row-for-row.

    The colour scale AUTOSCALES to the data, unlike the per-slide spatial panels
    which pin 0 to 1. Two of these figures from different runs are therefore not
    colour-comparable even though both look like they span the full range.

    Showing pseudotime on UMAP axes is a presentational convenience and nothing
    more. The pseudotime was computed on the diffusion map, so a smooth gradient
    here is not evidence the trajectory is smooth, and a discontinuity here is
    not evidence that it jumps.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=pseudotime, s=5,
                    cmap="viridis", alpha=0.6, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.colorbar(sc, ax=ax, label="Pseudotime")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_umap_pseudotime_std(X_umap, pseudotime_std, save_path,
                             title="UMAP — Pseudotime Uncertainty (std across roots)"):
    """Scatter the UMAP embedding coloured by per-patch pseudotime spread.

    ``pseudotime_std`` is the standard deviation across the n_roots separate DPT
    runs that were median-aggregated into the final pseudotime, so it measures
    how much the answer depended on which root was chosen. High values mark
    patches whose position in the trajectory the root set disagreed about.

    Bright regions are the ones to distrust. A patch with a large spread has a
    pseudotime that is an artifact of aggregation rather than a stable estimate.

    Only meaningful for a multi-root run. Single-root DPT gives an all-zero
    array, which plots as a uniform panel rather than as an error.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=pseudotime_std, s=5,
                    cmap="magma", alpha=0.6, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.colorbar(sc, ax=ax, label="Pseudotime std")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pseudotime_violins(pseudotime, cluster_labels, save_path,
                            title="Pseudotime Distribution by Cluster"):
    """Violin plot of the pseudotime distribution within each cluster.

    The most direct read on whether the clusters carry any trajectory signal: if
    every violin covers the same range, the partition and the pseudotime are
    describing different things.

    Noise (-1) is dropped. Clusters are ordered by id, not by median pseudotime,
    so a rising staircase across the x axis would be a coincidence of labelling
    rather than a result. Leiden ids carry no order.

    The y limit is fixed at (-0.05, 1.05) on the assumption that pseudotime is
    normalised to [0, 1]. Values outside that range are silently cropped from
    view rather than rescaling the axis.
    """
    _ensure_dir(save_path)
    unique = sorted(set(cluster_labels))
    if -1 in unique:
        unique.remove(-1)

    data = [pseudotime[cluster_labels == c] for c in unique]
    labels = [f"C{c}" for c in unique]

    fig, ax = plt.subplots(figsize=(max(8, len(unique) * 1.2), 6))
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    ax.set_xticks(range(1, len(unique) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Pseudotime")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_vs_pseudotime(
    pseudotime, morph_features, correlation_results, save_path,
    title="Morphological Features vs Pseudotime",
):
    """Scatter each morphological feature against pseudotime, one panel each.

    Args:
        pseudotime: (N,).
        morph_features: {feature name: (N,) values}. Panel order follows dict
            insertion order, which is the order the extractor produced.
        correlation_results: {feature name: {"rho": float, ...}}. Only "rho" is
            read, for the panel title. A feature missing from this dict gets a
            NaN in its title rather than being skipped.

    Non-finite values are dropped per panel, so different panels can be drawn
    from different numbers of patches. That is the intended behaviour: a failed
    feature is nan by convention, and nan means "not measured" rather than zero.
    The rho in the title comes from the caller and is not recomputed here, so it
    need not match a correlation computed on the points shown.

    Panels are laid out in rows of at most 3, and unused cells are hidden.
    """
    _ensure_dir(save_path)
    feature_names = list(morph_features.keys())
    n = len(feature_names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(feature_names):
        ax = axes[i]
        values = morph_features[name]
        valid = np.isfinite(values) & np.isfinite(pseudotime)

        ax.scatter(pseudotime[valid], values[valid], s=3, alpha=0.3, rasterized=True)
        rho = correlation_results.get(name, {}).get("rho", np.nan)
        ax.set_title(f"{name}\nρ = {rho:+.3f}", fontsize=10)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel(name)

    # Hide unused axes.
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_permutation_nulls(perm_results, save_path,
                           title="Permutation Test Null Distributions"):
    """Summarise each feature's permutation test as two vertical lines.

    WARNING: despite the title, this draws NO null distribution. The permutation
    results carry only summary statistics, not the null samples, so each panel
    shows the observed |rho| in red and the null 95th percentile in grey with
    nothing between them. A reader expecting a histogram will misread these
    panels as one whose bars failed to render.

    Args:
        perm_results: {feature name: {"real_rho", "null_95th", "perm_p_value"}}.
            Missing keys become NaN and plot as absent lines rather than raising.

    Read it as: red right of grey means the observed correlation beat 95% of
    permutations. The p value in the panel title is the real test result; the
    lines are an illustration of it.

    Drawing a genuine null would mean retaining the permutation samples in
    validation.json, which is a change to what the pipeline records.
    """
    _ensure_dir(save_path)
    feature_names = list(perm_results.keys())
    n = len(feature_names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(feature_names):
        ax = axes[i]
        result = perm_results[name]
        real_rho = result.get("real_rho", np.nan)
        null_95 = result.get("null_95th", np.nan)
        perm_p = result.get("perm_p_value", np.nan)

        # Two reference lines stand in for the histogram. The null samples are
        # not retained by the permutation test, so there is nothing to bin.
        ax.axvline(abs(real_rho), color="red", linewidth=2, label=f"|ρ| = {abs(real_rho):.3f}")
        ax.axvline(null_95, color="gray", linestyle="--", label=f"95th = {null_95:.3f}")
        ax.set_title(f"{name}\np = {perm_p:.4f}", fontsize=10)
        ax.set_xlabel("|ρ|")
        ax.legend(fontsize=8)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_test_projection(
    train_umap, train_pt, test_umap, test_pt, save_path,
    title="Test Slide Projection on Training UMAP",
):
    """Three panels comparing a projected slide against the training atlas.

    Training, test, and the two overlaid. All three share axis limits computed
    from both sets together, so positions are directly comparable between
    panels; without that each panel would autoscale to its own extent and a
    projected slide would appear to cover the whole atlas.

    Args:
        train_umap, test_umap: (N, 2) and (M, 2) in the SAME UMAP space. The
            test coordinates must come from transforming with the fitted
            training reducer, not from a fresh fit, or the comparison is
            meaningless while still rendering.
        train_pt, test_pt: (N,) and (M,) pseudotime in [0, 1].

    Pseudotime is pinned to 0..1 in all three panels, but train uses viridis and
    test uses plasma, so the two are distinguishable when overlaid and are not
    comparable by colour.

    What to look for: test points landing outside the training cloud are
    extrapolation, and their pseudotime is an extension of the model rather than
    a measurement.
    """
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    xlim = (min(train_umap[:, 0].min(), test_umap[:, 0].min()) - 0.5,
            max(train_umap[:, 0].max(), test_umap[:, 0].max()) + 0.5)
    ylim = (min(train_umap[:, 1].min(), test_umap[:, 1].min()) - 0.5,
            max(train_umap[:, 1].max(), test_umap[:, 1].max()) + 0.5)

    # Training
    ax = axes[0]
    sc = ax.scatter(train_umap[:, 0], train_umap[:, 1], c=train_pt, s=5,
                    cmap="viridis", alpha=0.5, vmin=0, vmax=1, rasterized=True)
    ax.set_title("Training")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    plt.colorbar(sc, ax=ax, label="Pseudotime")

    # Test
    ax = axes[1]
    sc = ax.scatter(test_umap[:, 0], test_umap[:, 1], c=test_pt, s=8,
                    cmap="plasma", alpha=0.7, vmin=0, vmax=1, rasterized=True)
    ax.set_title("Test")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    plt.colorbar(sc, ax=ax, label="Pseudotime")

    # Overlay
    ax = axes[2]
    ax.scatter(train_umap[:, 0], train_umap[:, 1], c="lightgrey", s=3,
               alpha=0.3, label="Train", rasterized=True)
    sc = ax.scatter(test_umap[:, 0], test_umap[:, 1], c=test_pt, s=8,
                    cmap="plasma", alpha=0.8, vmin=0, vmax=1)
    ax.set_title("Overlay")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.legend(loc="upper right", fontsize=9)
    plt.colorbar(sc, ax=ax, label="Test Pseudotime")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spatial_pseudotime(coords, pseudotime, slide_ids, save_dir, prefix="spatial_pt"):
    """Write one pseudotime map per slide at the patches' slide coordinates.

    Produces ``{prefix}_{sid}.png`` per distinct slide id. ``coords`` is in
    cropped-PNG pixel space and y is inverted, matching
    :func:`plot_spatial_clusters`.

    Colour is pinned to 0..1 so slides are comparable, and uses plasma where
    ``plot_spatial_clusters``'s pseudotime panel uses viridis. The two show the
    same quantity in different colour maps.

    Unlike every other function here, this one does NOT print the paths it
    writes, so its output is silent in the job log.

    Titles show the integer slide id and take no name map, so identifying a
    slide means matching the id against the run's slide ordering.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for sid in np.unique(slide_ids):
        mask = slide_ids == sid
        fig, ax = plt.subplots(figsize=(10, 10))
        sc = ax.scatter(coords[mask, 0], coords[mask, 1], c=pseudotime[mask],
                        s=15, cmap="plasma", marker="s", vmin=0, vmax=1)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"Spatial Pseudotime — Slide {sid}")
        plt.colorbar(sc, ax=ax, label="Pseudotime")
        fname = save_dir / f"{prefix}_{sid}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()


def plot_umap_by_slide(X_umap, slide_ids, save_path, title="UMAP — Slide ID (Batch Check)"):
    """Scatter the UMAP embedding coloured by slide, as a batch-effect check.

    The question this answers: do patches separate by SLIDE rather than by
    morphology? Slides appearing as distinct blobs indicates a batch effect the
    clustering will then recover as biology. Well-mixed colours indicate the
    embedding is not dominated by slide identity.

    This is a visual check only. The quantitative version is the slide
    independence test in ``analysis/clustering.py``, which writes
    slide_independence.json, and that is the one to cite.

    The legend is dropped above 16 slides. Colours come from tab20, so a cohort
    larger than 20 slides reuses them.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_slides = np.unique(slide_ids)
    cmap = plt.cm.get_cmap("tab20", len(unique_slides))
    for i, sid in enumerate(unique_slides):
        mask = slide_ids == sid
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], s=5, alpha=0.4,
                   color=cmap(i), label=str(sid), rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if len(unique_slides) <= 16:
        ax.legend(markerscale=3, fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_paga(adata, save_path, title="PAGA — Cluster Connectivity"):
    """Draw scanpy's PAGA graph: one node per cluster, edges weighted by
    connectivity.

    Requires ``sc.tl.paga`` to have already run on ``adata``; this only renders
    what is in ``adata.uns['paga']`` and raises if that is absent.

    TRAP: the nodes are Leiden clusters from the k=15 COSINE graph, but the edge
    weights come from the diffusion neighbour graph, which is k=30 and
    EUCLIDEAN. The picture therefore mixes two different graphs, and the edges
    are not the ones that produced the clusters. See ``analysis/diffusion.py``.

    Read the layout as topology only. Node positions come from a force-directed
    layout with no relation to UMAP coordinates or to pseudotime, so a node
    drawn on the left is not "early".
    """
    _ensure_dir(save_path)
    # Imported here rather than at module scope. scanpy is slow to import and
    # every other function in this module works without it.
    import scanpy as sc
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.paga(adata, ax=ax, show=False, title=title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_umap_section_cluster(X_umap, section_labels, cluster_labels, save_path):
    """One figure, two panels: the same UMAP coloured by section, then by cluster.

    Built to be read as a pair. If the section panel and the cluster panel show
    the same partition, the clusters are tracking the batch rather than
    morphology, and that is visible immediately side by side in a way it is not
    across two separate files.

    ``section_labels`` are strings ("2M-1", "2M-2") and are coerced with
    ``np.asarray`` before masking, so a list works. ``cluster_labels`` must
    already be an array, since it is compared with ``==`` directly. Noise (-1)
    is drawn and labelled rather than dropped.

    The cluster legend is dropped above 12 clusters; the section legend is
    always drawn, on the assumption that sections are few.
    """
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax = axes[0]
    unique_sections = sorted(set(section_labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_sections))
    for i, sec in enumerate(unique_sections):
        mask = np.asarray(section_labels) == sec
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], s=5, alpha=0.5,
                   color=cmap(i), label=sec, rasterized=True)
    ax.set_title("UMAP — Section (batch)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=9, title="Section")

    ax = axes[1]
    unique_clusters = sorted(set(cluster_labels))
    cmap2 = plt.cm.get_cmap("tab20", len(unique_clusters))
    for i, c in enumerate(unique_clusters):
        mask = cluster_labels == c
        label = f"Cluster {c}" if c != -1 else "Noise"
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], s=5, alpha=0.5,
                   color=cmap2(i), label=label, rasterized=True)
    ax.set_title("UMAP — Leiden Cluster")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if len(unique_clusters) <= 12:
        ax.legend(markerscale=3, fontsize=8, title="Cluster")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_3d_manifold(diff_coords, color, save_path, title="3D Diffusion Manifold", cmap="plasma"):
    """Scatter the first three diffusion components in 3D, coloured by ``color``.

    Unlike the UMAP figures, these axes are the space the trajectory was actually
    computed in, so distance here does correspond to what DPT measured. It is
    the one plot in this module where apparent structure is the structure.

    Args:
        diff_coords: (N, >=3) diffusion map coordinates. Components beyond the
            third are ignored.
        color: (N,) values for the colour map, usually pseudotime.

    RETURNS EARLY, having written nothing, when fewer than three components are
    available. It prints a line and returns None, which is indistinguishable to
    the caller from success, so a missing 3D figure means this branch fired.

    The colour scale autoscales; it is not pinned to 0..1 like the spatial
    panels.
    """
    _ensure_dir(save_path)
    if diff_coords.shape[1] < 3:
        print(f"  Fewer than 3 diffusion components — skipping 3D plot.")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(diff_coords[:, 0], diff_coords[:, 1], diff_coords[:, 2],
                    c=color, cmap=cmap, s=10, alpha=0.6, depthshade=False)
    ax.set_xlabel("DC1"); ax.set_ylabel("DC2"); ax.set_zlabel("DC3")
    ax.set_title(title)
    ax.view_init(elev=30, azim=45)
    plt.colorbar(sc, ax=ax, label="Pseudotime", shrink=0.6)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
