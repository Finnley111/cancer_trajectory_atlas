"""Project a held-out slide onto a LOO-trained manifold and compare pseudotime distributions.

Run AFTER the LOO pipeline job completes for a given held-out slide.

Usage:
    python -m cancer_trajectory_atlas.analysis.loo_project \\
        --projector-dir  $SCRATCH/results/loo_SLIDE/projector \\
        --held-out-slide 6027-4L-2M-1_x5 \\
        --cache-dir      $SCRATCH/data/features_cache \\
        --full-run-dir   $SCRATCH/results/atlas_full_macenko \\
        --output-dir     $SCRATCH/results/loo_6027-4L-2M-1_x5
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import wasserstein_distance, ks_2samp, spearmanr

from ..features.patching import sample_patches


def load_cached_features(cache_dir: Path, slide_name: str) -> np.ndarray:
    feat_file = cache_dir / f"{slide_name}_features.npy"
    if not feat_file.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {feat_file}\n"
            "Run the full pipeline with --features-cache-dir first to populate the cache."
        )
    return np.load(feat_file)


def load_inmanifold_pseudotime(full_run_dir: Path, slide_name: str) -> np.ndarray:
    """Pull pseudotime for one slide from the full 16-slide run's results.csv.

    Returns values in patch extraction order, matching the feature cache order.
    """
    csv_path = full_run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found in {full_run_dir}")
    df = pd.read_csv(csv_path)
    slide_df = df[df["slide_name"] == slide_name].reset_index(drop=True)
    if len(slide_df) == 0:
        raise ValueError(
            f"Slide '{slide_name}' not found in {csv_path}.\n"
            f"Available: {df['slide_name'].unique().tolist()}"
        )
    return slide_df["pseudotime"].values.astype(float)


def main():
    parser = argparse.ArgumentParser(
        description="Project held-out slide onto LOO manifold; compare pseudotime distributions."
    )
    parser.add_argument("--projector-dir", type=Path, required=True,
                        help="Saved AtlasProjector directory from the LOO training run")
    parser.add_argument("--held-out-slide", type=str, required=True,
                        help="Slide stem, e.g. 6027-4L-2M-1_x5")
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="Per-slide feature cache directory ({slide}_features.npy)")
    parser.add_argument("--full-run-dir", type=Path, required=True,
                        help="Full 16-slide run directory (reference in-manifold pseudotime)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write loo_result_*.json and loo_distribution_*.png")
    parser.add_argument("--max-patches-per-slide", type=int, default=None,
                        help="Cap patches to this count (must match the cap used in training)")
    parser.add_argument("--patch-sample-seed", type=int, default=42,
                        help="Base seed for patch subsampling (must match the seed used in training)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slide_name = args.held_out_slide

    from .projector import AtlasProjector

    print(f"Loading projector from {args.projector_dir}...")
    projector = AtlasProjector.load(str(args.projector_dir))

    print(f"Loading cached features for '{slide_name}'...")
    raw_features = load_cached_features(args.cache_dir, slide_name)
    print(f"  Features shape: {raw_features.shape}")
    raw_features, _, _ = sample_patches(
        raw_features, np.arange(len(raw_features)),
        args.max_patches_per_slide, args.patch_sample_seed, slide_name,
    )
    if args.max_patches_per_slide is not None:
        print(f"  After sampling cap {args.max_patches_per_slide}: {len(raw_features)} patches")

    adata_proj = projector.project(raw_features, method="knn")
    projected_pt = adata_proj.obs["pseudotime"].values.astype(float)

    print(f"Loading in-manifold pseudotime from {args.full_run_dir}...")
    inmanifold_pt = load_inmanifold_pseudotime(args.full_run_dir, slide_name)
    n_proj = len(projected_pt)
    n_inm  = len(inmanifold_pt)
    print(f"  In-manifold patches: {n_inm}, Projected patches: {n_proj}")

    if n_proj != n_inm:
        raise ValueError(
            f"Patch count mismatch for {slide_name}: "
            f"cache has {n_proj} patches, results.csv has {n_inm}. "
            "Cache and reference run must use identical extraction settings."
        )

    # Primary metric: paired Spearman rho (patch i vs patch i)
    spearman_rho, spearman_p = spearmanr(inmanifold_pt, projected_pt)

    # Secondary: unpaired distribution metrics
    w_dist        = float(wasserstein_distance(inmanifold_pt, projected_pt))
    ks_stat, ks_p = ks_2samp(inmanifold_pt, projected_pt)

    result = {
        "slide_name":         slide_name,
        "n_patches":          int(n_proj),
        "spearman_rho":       float(spearman_rho),
        "spearman_p":         float(spearman_p),
        "wasserstein":        w_dist,
        "ks_stat":            float(ks_stat),
        "ks_pvalue":          float(ks_p),
        "projected_pt_mean":  float(projected_pt.mean()),
        "projected_pt_std":   float(projected_pt.std()),
        "inmanifold_pt_mean": float(inmanifold_pt.mean()),
        "inmanifold_pt_std":  float(inmanifold_pt.std()),
    }

    json_path = args.output_dir / f"loo_result_{slide_name}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {json_path.name}")
    print(f"  Spearman rho={spearman_rho:.4f}  p={spearman_p:.3e}")
    print(f"  Wasserstein={w_dist:.4f}  (secondary, distribution-level)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        ax.scatter(inmanifold_pt, projected_pt, s=1, alpha=0.3, rasterized=True)
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="y=x")
        ax.set_xlabel("In-manifold pseudotime")
        ax.set_ylabel("Projected pseudotime")
        ax.set_title(f"Paired patch pseudotime\nSpearman ρ={spearman_rho:.3f}  p={spearman_p:.2e}")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        x = np.linspace(0, 1, 300)
        ax2.fill_between(x, gaussian_kde(inmanifold_pt)(x), alpha=0.45,
                         label=f"In-manifold  n={n_inm}")
        ax2.fill_between(x, gaussian_kde(projected_pt)(x), alpha=0.45,
                         label=f"Projected    n={n_proj}")
        ax2.set_xlabel("Pseudotime")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Distribution comparison\nWasserstein={w_dist:.4f}")
        ax2.legend(fontsize=8)

        fig.suptitle(slide_name, fontsize=10)
        fig.tight_layout()
        fig_path = args.output_dir / f"loo_distribution_{slide_name}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {fig_path.name}")
    except Exception as exc:
        print(f"WARNING: Could not save distribution plot: {exc}")


if __name__ == "__main__":
    main()
