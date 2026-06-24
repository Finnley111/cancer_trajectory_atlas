"""Plotting helpers for cluster, pseudotime, and validation figures."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, Optional


def _ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# UMAP cluster plot

def plot_umap_clusters(X_umap, cluster_labels, save_path, title="UMAP — Morphological Clusters"):
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


# Representative patch grid

def plot_cluster_patch_grid(
    patches, cluster_labels, cluster_centroids,
    save_path, n_per_cluster=5, title="Representative Patches per Cluster",
):
    """
    Show representative patches (nearest to centroid) for each cluster.

    Args:
        patches: (N, H, W, 3) array of all patches.
        cluster_labels: (N,) cluster assignments.
        cluster_centroids: {cluster_id: (centroid, nearest_global_idx)}.
        n_per_cluster: How many patches to show per cluster.
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
        # Patches are already ordered by proximity to the centroid.
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


# Spatial cluster maps

def plot_spatial_clusters(coords, cluster_labels, slide_ids, save_dir,
                          prefix="spatial_clusters", slide_name_map=None,
                          pseudotime=None):
    """Plot per-slide spatial cluster maps with a discrete legend.

    slide_name_map: optional dict mapping integer sid -> display name string.
                   Falls back to str(sid) if None or key missing.
    pseudotime:    if provided, appends a side-by-side pseudotime panel per slide.
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

        # Cluster panel — one scatter call per cluster for a clean discrete legend.
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

        # Pseudotime panel (optional side-by-side).
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


# UMAP pseudotime plot

def plot_umap_pseudotime(X_umap, pseudotime, save_path, title="UMAP — Pseudotime"):
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


# Pseudotime per cluster

def plot_pseudotime_violins(pseudotime, cluster_labels, save_path,
                            title="Pseudotime Distribution by Cluster"):
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


# Features vs pseudotime

def plot_feature_vs_pseudotime(
    pseudotime, morph_features, correlation_results, save_path,
    title="Morphological Features vs Pseudotime",
):
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


# Permutation null distributions

def plot_permutation_nulls(perm_results, save_path,
                           title="Permutation Test Null Distributions"):
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

        # We don't store the full null distribution in results, so plot a placeholder
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


# ── Fig 8: Test projection overlay ──────────────────────────────────

def plot_test_projection(
    train_umap, train_pt, test_umap, test_pt, save_path,
    title="Test Slide Projection on Training UMAP",
):
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


# ── Spatial pseudotime per slide ─────────────────────────────────────

def plot_spatial_pseudotime(coords, pseudotime, slide_ids, save_dir, prefix="spatial_pt"):
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


# ── UMAP colored by slide ID (QC check) ─────────────────────────────

def plot_umap_by_slide(X_umap, slide_ids, save_path, title="UMAP — Slide ID (Batch Check)"):
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


# ── PAGA cluster connectivity graph ─────────────────────────────────

def plot_paga(adata, save_path, title="PAGA — Cluster Connectivity"):
    _ensure_dir(save_path)
    import scanpy as sc
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.paga(adata, ax=ax, show=False, title=title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Side-by-side UMAP: section vs cluster ────────────────────────────

def plot_umap_section_cluster(X_umap, section_labels, cluster_labels, save_path):
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


# ── 3D diffusion manifold ───────────────────────────────────────────

def plot_3d_manifold(diff_coords, color, save_path, title="3D Diffusion Manifold", cmap="plasma"):
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
