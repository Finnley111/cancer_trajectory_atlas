"""Pure-plotting diagnostic: UMAP scatter colored by section_number.

Loads an existing run's adata_full.h5ad (already embedded/clustered/DPT'd)
and produces two PNGs showing whether 2M-1 and 2M-2 patches are
interspersed (Harmony aligned them) or still spatially segregated (Harmony
under-corrected). Does NOT re-run any pipeline stage.

Usage:
    python analysis/plot_umap_by_section.py <run_dir>

<run_dir> must contain adata_full.h5ad. Output PNGs are written to
<run_dir>/diagnostics/.
"""

import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

SECTION_COLORS = {"2M-1": "#1b9e77", "2M-2": "#d95f02"}


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_umap_by_section.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    h5ad_path = run_dir / "adata_full.h5ad"
    if not h5ad_path.exists():
        print(f"ERROR: {h5ad_path} does not exist.")
        sys.exit(1)

    run_name = run_dir.name
    adata = ad.read_h5ad(h5ad_path)

    if "X_umap" not in adata.obsm:
        print(f"ERROR: adata.obsm has no 'X_umap' key. Found: {list(adata.obsm.keys())}")
        sys.exit(1)
    if "section_number" not in adata.obs:
        print(f"ERROR: adata.obs has no 'section_number' column. Found: {list(adata.obs.columns)}")
        sys.exit(1)

    umap = np.asarray(adata.obsm["X_umap"])[:, :2]
    section = adata.obs["section_number"].astype(str).to_numpy()

    sections = sorted(SECTION_COLORS.keys())
    missing = set(np.unique(section)) - set(sections)
    if missing:
        print(f"ERROR: unexpected section_number values: {missing}. Expected only {sections}.")
        sys.exit(1)

    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    xlim = (umap[:, 0].min(), umap[:, 0].max())
    ylim = (umap[:, 1].min(), umap[:, 1].max())

    # Figure 1: overlaid scatter, draw order shuffled so neither section
    # is painted entirely on top of the other.
    rng = np.random.default_rng(0)
    order = rng.permutation(umap.shape[0])

    fig, axis = plt.subplots(figsize=(7, 6))
    colors = np.array([SECTION_COLORS[s] for s in section])
    axis.scatter(
        umap[order, 0], umap[order, 1],
        c=colors[order], s=3, alpha=0.4, linewidths=0,
    )
    for sec_name in sections:
        axis.scatter([], [], c=SECTION_COLORS[sec_name], label=sec_name, s=20, alpha=1.0)
    axis.legend(title="Section", loc="best")
    axis.set_xlabel("UMAP1")
    axis.set_ylabel("UMAP2")
    axis.set_title(f"UMAP by Section — {run_name}")
    fig.tight_layout()
    overlay_path = diagnostics_dir / "umap_by_section.png"
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)

    # Figure 2: side-by-side panels, one per section, shared axis limits.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for axis, sec_name in zip(axes, sections):
        mask = section == sec_name
        axis.scatter(
            umap[mask, 0], umap[mask, 1],
            c=SECTION_COLORS[sec_name], s=3, alpha=0.4, linewidths=0,
        )
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_xlabel("UMAP1")
        axis.set_title(sec_name)
    axes[0].set_ylabel("UMAP2")
    fig.suptitle(f"UMAP by Section (split) — {run_name}")
    fig.tight_layout()
    split_path = diagnostics_dir / "umap_by_section_split.png"
    fig.savefig(split_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {overlay_path}")
    print(f"Saved: {split_path}")


if __name__ == "__main__":
    main()
