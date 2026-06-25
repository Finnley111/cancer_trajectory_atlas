"""Aggregate scVI LOO projection results: stability + out-of-fold morphology survival.

Sibling of loo_summary.py (which is unchanged and still serves the Harmony
LOO cohort's CSV + Wasserstein-based figure). This script targets the
specific question for the scVI backend: is the pseudotime axis robust across
slides, AND does it retain biological signal (morphology correlations)
out-of-fold?

For each held-out slide, morphological features are pulled directly from the
full scVI reference run's results.csv — they are patch-intrinsic (computed
from patch images only) and do not need recomputing per fold. They are
joined against that slide's LOO-projected pseudotime (loo_projected_pt_*.npy,
saved by loo_project.py) to get out-of-fold correlations, which are compared
against the full run's in-sample correlations (validation.json) to flag
whether each feature's relationship survives outside the training fold.

Run after all 16 scVI LOO jobs (submit_loo_array_scvi.sh) have completed:

    python -m cancer_trajectory_atlas.analysis.loo_summary_scvi \\
        --loo-dirs $SCRATCH/results/loo_*_scvi \\
        --full-run-dir $SCRATCH/results/atlas_none_scvi \\
        --output-dir $SCRATCH/results/atlas_none_scvi/loo_scvi
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import save_json
from ..validation.correlations import correlate_features_with_pseudotime

MORPH_FEATURES = [
    "nuclear_density", "mean_nuclear_area", "nc_ratio",
    "texture_entropy", "h_intensity", "packing_irregularity",
]

# A feature's out-of-fold correlation is considered RETAINED if it keeps the
# same sign as the in-sample correlation AND its magnitude is at least this
# fraction of the in-sample |rho|. This is a simple, explicit judgment call,
# not a statistical test — it's meant to distinguish "weaker but still
# present" from "vanished" or "flipped".
RETENTION_MAGNITUDE_FRACTION = 0.5


def classify_survival(in_sample_rho: float, out_of_fold_rho: float) -> str:
    if not np.isfinite(in_sample_rho) or not np.isfinite(out_of_fold_rho):
        return "UNDEFINED"
    if np.sign(in_sample_rho) != np.sign(out_of_fold_rho):
        return "SIGN_FLIP"
    if abs(out_of_fold_rho) >= RETENTION_MAGNITUDE_FRACTION * abs(in_sample_rho):
        return "RETAINED"
    return "WEAKENED"


def load_slide_morph_features(results_csv: pd.DataFrame, slide_name: str) -> dict:
    slide_df = results_csv[results_csv["slide_name"] == slide_name]
    if len(slide_df) == 0:
        raise ValueError(f"Slide '{slide_name}' not found in full-run results.csv")
    return {feat: slide_df[feat].to_numpy(dtype=float) for feat in MORPH_FEATURES}


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate scVI LOO results: projection stability + morphology survival."
    )
    parser.add_argument("--loo-dirs", nargs="+", type=Path, required=True,
                        help="LOO output directories, each containing loo_result_*.json "
                             "and loo_projected_pt_*.npy")
    parser.add_argument("--full-run-dir", type=Path, required=True,
                        help="Full 16-slide scVI reference run directory "
                             "(results.csv for morphology, validation.json for in-sample rho)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination for loo_summary.json and loo_stability_figure.png")
    parser.add_argument("--outlier-threshold", type=float, default=0.6,
                        help="Flag slides with projection-stability Spearman rho below this "
                             "as outliers (default: 0.6)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_results_csv = pd.read_csv(args.full_run_dir / "results.csv")
    with open(args.full_run_dir / "validation.json") as f:
        full_validation = json.load(f)
    in_sample_corrs = full_validation["feature_correlations"]

    per_slide = []
    for loo_dir in sorted(args.loo_dirs):
        loo_dir = Path(loo_dir)
        for result_path in sorted(loo_dir.glob("loo_result_*.json")):
            with open(result_path) as f:
                result = json.load(f)
            slide_name = result["slide_name"]

            pt_path = loo_dir / f"loo_projected_pt_{slide_name}.npy"
            if not pt_path.exists():
                print(f"  WARNING: {pt_path} missing — skipping morphology check for {slide_name}")
                morph_corrs = {}
            else:
                projected_pt = np.load(pt_path)
                morph_features = load_slide_morph_features(full_results_csv, slide_name)
                raw_corrs = correlate_features_with_pseudotime(projected_pt, morph_features)
                morph_corrs = {
                    feat: {
                        "out_of_fold_rho": raw_corrs[feat]["rho"],
                        "out_of_fold_p": raw_corrs[feat]["p_value"],
                        "in_sample_rho": in_sample_corrs[feat]["rho"],
                        "survival": classify_survival(
                            in_sample_corrs[feat]["rho"], raw_corrs[feat]["rho"],
                        ),
                    }
                    for feat in MORPH_FEATURES
                }

            per_slide.append({
                "slide_name": slide_name,
                "n_patches": result["n_patches"],
                "spearman_rho": result["spearman_rho"],
                "spearman_p": result["spearman_p"],
                "is_outlier": result["spearman_rho"] < args.outlier_threshold,
                "morph_correlations": morph_corrs,
            })

    if not per_slide:
        print("No loo_result_*.json files found — check --loo-dirs.")
        return

    rhos = [s["spearman_rho"] for s in per_slide]
    cohort_mean_rho = float(np.mean(rhos))
    outlier_slides = [s["slide_name"] for s in per_slide if s["is_outlier"]]

    # Signal-survival summary for the two features flagged in the full scVI
    # run as weak-to-modest (h_intensity, texture_entropy): how many slides
    # retain each, out-of-fold.
    signal_survival = {}
    for feat in ("h_intensity", "texture_entropy"):
        survivals = [
            s["morph_correlations"][feat]["survival"]
            for s in per_slide if feat in s["morph_correlations"]
        ]
        signal_survival[feat] = {
            "in_sample_rho": in_sample_corrs[feat]["rho"],
            "n_retained": survivals.count("RETAINED"),
            "n_weakened": survivals.count("WEAKENED"),
            "n_sign_flip": survivals.count("SIGN_FLIP"),
            "n_slides": len(survivals),
        }

    summary = {
        "per_slide": per_slide,
        "cohort_mean_rho": cohort_mean_rho,
        "outlier_threshold": args.outlier_threshold,
        "outlier_slides": outlier_slides,
        "signal_survival": signal_survival,
    }

    out_path = args.output_dir / "loo_summary.json"
    save_json(summary, out_path)
    print(f"Saved: {out_path}")
    print(f"\nCohort mean Spearman rho: {cohort_mean_rho:.4f}  (n={len(per_slide)} slides)")
    if outlier_slides:
        print(f"Outlier slides (rho < {args.outlier_threshold}): {outlier_slides}")
    else:
        print(f"No outlier slides (threshold rho < {args.outlier_threshold}).")
    for feat, s in signal_survival.items():
        print(f"  {feat}: in-sample rho={s['in_sample_rho']:.3f}  "
              f"retained={s['n_retained']}/{s['n_slides']}  "
              f"weakened={s['n_weakened']}  sign_flip={s['n_sign_flip']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df_plot = pd.DataFrame(per_slide).sort_values("spearman_rho")
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#d62728" if r < args.outlier_threshold else "#1f77b4"
                  for r in df_plot["spearman_rho"]]
        ax.barh(df_plot["slide_name"], df_plot["spearman_rho"], color=colors)
        ax.axvline(args.outlier_threshold, color="red", linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"Threshold ({args.outlier_threshold})")
        ax.axvline(cohort_mean_rho, color="black", linestyle=":", linewidth=1.0,
                   alpha=0.7, label=f"Mean ({cohort_mean_rho:.3f})")
        ax.set_xlabel("Spearman ρ  (paired patch pseudotime: in-manifold vs. projected)")
        ax.set_title("scVI LOO Projection Stability")
        ax.legend()
        fig.tight_layout()

        fig_path = args.output_dir / "loo_stability_figure.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {fig_path.name}")
    except Exception as exc:
        print(f"WARNING: Could not save figure: {exc}")


if __name__ == "__main__":
    main()
