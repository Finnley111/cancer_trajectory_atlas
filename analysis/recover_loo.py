"""One-off recovery: build AtlasProjector from completed LOO Phase A artifacts,
then run Phase B (pseudotime projection + distribution comparison) for each slide.

Phase A (15-slide pipeline) already completed for every loo_* directory.
Phase B (loo_project.py) never ran because the projector code didn't exist
on Narval when the array job was submitted.

Usage (run from ~ on Narval, after git pull):
    python -m cancer_trajectory_atlas.analysis.recover_loo \
        --loo-root       $SCRATCH/results \
        --cache-dir      $SCRATCH/data/features_cache \
        --full-run-dir   $SCRATCH/results/atlas_none_harmony
"""
import argparse
import pickle
import re
import sys
import traceback
import numpy as np
from pathlib import Path

# Make cancer_trajectory_atlas importable when run as a plain script from any CWD.
# This replicates what `cd ~ && python -m cancer_trajectory_atlas...` does implicitly.
sys.path.insert(0, str(Path.home()))

# LOO slide directories look like loo_6027-4L-2M-1_x5 (mouse ID is 4 digits).
# This excludes loo_summary and any other loo_* directories that aren't slide runs.
_LOO_SLIDE_DIR = re.compile(r"^loo_\d{4}-")


def build_projector_from_artifacts(loo_dir: Path):
    """Reconstruct and save an AtlasProjector from an existing LOO run directory."""
    import anndata as ad
    from cancer_trajectory_atlas.analysis.projector import AtlasProjector
    from cancer_trajectory_atlas.analysis.clustering import get_cluster_centroids

    required = ["adata_full.h5ad", "scaler.pkl", "pca.pkl", "umap_reducer.pkl"]
    missing = [f for f in required if not (loo_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Phase A artifacts in {loo_dir.name}: {missing}")

    print(f"  Loading artifacts from {loo_dir.name}...")
    adata = ad.read_h5ad(loo_dir / "adata_full.h5ad")

    with open(loo_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(loo_dir / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(loo_dir / "umap_reducer.pkl", "rb") as f:
        umap_reducer = pickle.load(f)

    # Convert cluster labels from strings to ints for get_cluster_centroids.
    raw_labels = adata.obs["cluster"].values
    cluster_labels = np.array([int(c) for c in raw_labels])

    # X is Harmony-corrected PCA if Harmony was used; X_pca_original is raw PCA.
    # get_cluster_centroids works on whatever representation was used for clustering.
    X_for_centroids = adata.X

    centroids = get_cluster_centroids(X_for_centroids, cluster_labels)

    projector = AtlasProjector.from_training(scaler, pca, umap_reducer, adata, centroids)
    projector.save(loo_dir / "projector")
    print(f"  Projector saved.")


def run_phase_b(loo_dir: Path, slide_name: str, cache_dir: Path, full_run_dir: Path):
    """Project held-out slide and compare pseudotime against the full-run reference.

    Primary metric: Spearman rho between PAIRED patch-level pseudotime values.
    The feature cache preserves patch order, so index i in the cache corresponds
    to index i in results.csv for that slide. Comparing unpaired distributions
    (Wasserstein) is kept as a secondary check but is not the main result.
    """
    import json
    import pandas as pd
    from scipy.stats import wasserstein_distance, ks_2samp, spearmanr
    from cancer_trajectory_atlas.analysis.projector import AtlasProjector

    feat_file = cache_dir / f"{slide_name}_features.npy"
    if not feat_file.exists():
        raise FileNotFoundError(
            f"Feature cache missing: {feat_file}\n"
            "Run run_cache_population.sh first and verify all 16 .npy files exist."
        )

    csv_path = full_run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Full-run results.csv not found: {csv_path}")

    print(f"  Loading projector...")
    projector = AtlasProjector.load(str(loo_dir / "projector"))

    print(f"  Loading cached features ({slide_name})...")
    raw_features = np.load(feat_file)

    adata_proj = projector.project(raw_features, method="knn")
    projected_pt = adata_proj.obs["pseudotime"].values.astype(float)

    # Load in-manifold pseudotime in patch order matching the cache.
    # results.csv lists patches in the same extraction order as the cache.
    print(f"  Loading in-manifold pseudotime...")
    df = pd.read_csv(csv_path)
    slide_df = df[df["slide_name"] == slide_name].reset_index(drop=True)
    if len(slide_df) == 0:
        available = df["slide_name"].unique().tolist()
        raise ValueError(
            f"Slide '{slide_name}' not in {csv_path}.\n"
            f"Available: {available}"
        )
    inmanifold_pt = slide_df["pseudotime"].values.astype(float)

    n_proj = len(projected_pt)
    n_inm  = len(inmanifold_pt)
    print(f"  n_inmanifold={n_inm}, n_projected={n_proj}")

    if n_proj != n_inm:
        raise ValueError(
            f"Patch count mismatch for {slide_name}: "
            f"cache has {n_proj} patches, results.csv has {n_inm}. "
            "This means the cache was built with different extraction settings "
            "than the reference run. Cannot compute paired Spearman."
        )

    # Primary metric: paired Spearman rho (preserves ordering at patch level)
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

    json_path = loo_dir / f"loo_result_{slide_name}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {json_path.name}")
    print(f"  Spearman rho={spearman_rho:.4f}  p={spearman_p:.3e}")
    print(f"  Wasserstein={w_dist:.4f}  (distribution-level, secondary)")

    # Scatter: paired pseudotime comparison (in-manifold vs projected)
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
        fig.savefig(loo_dir / f"loo_distribution_{slide_name}.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"  WARNING: plot failed: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Recover LOO Phase B: build projectors and run projection for all loo_* dirs."
    )
    parser.add_argument("--loo-root", type=Path, required=True,
                        help="Parent directory containing loo_* subdirectories (e.g. $SCRATCH/results)")
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="Feature cache directory ({slide}_features.npy)")
    parser.add_argument("--full-run-dir", type=Path, required=True,
                        help="Full 16-slide reference run (for in-manifold pseudotime)")
    parser.add_argument("--skip-projector-build", action="store_true",
                        help="Skip projector build if projector/ already exists in a loo_* dir")
    args = parser.parse_args()

    loo_dirs = sorted(p for p in args.loo_root.iterdir()
                      if p.is_dir() and _LOO_SLIDE_DIR.match(p.name))

    if not loo_dirs:
        print(f"No loo_* directories found under {args.loo_root}")
        sys.exit(1)

    print(f"Found {len(loo_dirs)} LOO directories.\n")

    successes, failures = [], []

    for loo_dir in loo_dirs:
        # Derive slide name from directory name: loo_6027-4L-2M-1_x5 → 6027-4L-2M-1_x5
        slide_name = loo_dir.name[len("loo_"):]
        print(f"\n{'='*55}")
        print(f"Processing: {slide_name}")

        try:
            projector_dir = loo_dir / "projector"
            if projector_dir.exists() and args.skip_projector_build:
                print(f"  Projector exists — skipping build (--skip-projector-build).")
            else:
                build_projector_from_artifacts(loo_dir)

            result_file = loo_dir / f"loo_result_{slide_name}.json"
            if result_file.exists():
                print(f"  loo_result already exists — skipping Phase B.")
            else:
                run_phase_b(loo_dir, slide_name, args.cache_dir, args.full_run_dir)

            successes.append(slide_name)

        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            failures.append((slide_name, str(exc)))

    print(f"\n{'='*55}")
    print(f"Done. {len(successes)}/{len(loo_dirs)} succeeded.")
    if failures:
        print(f"\nFailed slides:")
        for name, err in failures:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
