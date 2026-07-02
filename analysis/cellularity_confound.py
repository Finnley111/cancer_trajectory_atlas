"""Post-hoc analysis: how much of pseudotime is explained by gross cellularity?

Computes Spearman rho between pseudotime and a composite cellularity proxy
(PC1 of nuclear_density, mean_nuclear_area, nc_ratio), then checks whether
residual signal remains in texture_entropy, h_intensity, and packing_irregularity
after regressing out cellularity.

Usage:
    python -m cancer_trajectory_atlas.analysis.cellularity_confound \\
        --results-dirs $SCRATCH/results/atlas_full_macenko \\
                       $SCRATCH/results/atlas_full_reinhard \\
        --output-dir   $SCRATCH/results/confound_analysis
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA as SklearnPCA

CELLULARITY_FEATURES = ["nuclear_density", "mean_nuclear_area", "nc_ratio"]
OTHER_FEATURES = ["texture_entropy", "h_intensity", "packing_irregularity"]

# All 5 features to test when controlling for nuclear_density specifically
ALL_OTHER_FEATURES = [
    "mean_nuclear_area", "nc_ratio",
    "texture_entropy", "h_intensity", "packing_irregularity",
]


def compute_cellularity_proxy(obs_df):
    """First PC of the three nuclear features (scaled). Returns (proxy, pc1_variance_explained)."""
    X = obs_df[CELLULARITY_FEATURES].values.astype(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X_scaled = (X - mu) / sd

    pca = SklearnPCA(n_components=1)
    proxy = pca.fit_transform(X_scaled)[:, 0]

    # Orient so that higher proxy = denser cellularity (nuclear_density loads positively)
    if pca.components_[0, 0] < 0:
        proxy = -proxy

    return proxy, float(pca.explained_variance_ratio_[0])


def compute_residual_pseudotime(pseudotime, cellularity_proxy):
    """OLS residuals of pseudotime ~ cellularity_proxy."""
    X = np.column_stack([np.ones(len(cellularity_proxy)), cellularity_proxy])
    coeffs, _, _, _ = np.linalg.lstsq(X, pseudotime, rcond=None)
    return pseudotime - X @ coeffs


def analyze_run(results_dir: Path, output_dir: Path):
    """Analyze one pipeline run directory. Returns a summary dict, or None on failure."""
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata is required: pip install anndata")

    h5ad = results_dir / "adata_full.h5ad"
    if not h5ad.exists():
        print(f"  SKIP: {h5ad} not found")
        return None

    print(f"\n  Loading {results_dir.name}...")
    adata = ad.read_h5ad(h5ad)
    obs = adata.obs.copy()

    missing = [c for c in CELLULARITY_FEATURES + ["pseudotime"] if c not in obs.columns]
    if missing:
        print(f"  SKIP: missing columns {missing}")
        return None

    pseudotime = obs["pseudotime"].values.astype(float)
    proxy, pc1_var = compute_cellularity_proxy(obs)
    residuals = compute_residual_pseudotime(pseudotime, proxy)

    rho_cell, p_cell = spearmanr(pseudotime, proxy)

    row = {
        "run": results_dir.name,
        "n_patches": len(obs),
        "rho_cellularity": rho_cell,
        "p_cellularity": p_cell,
        "pc1_var_explained": pc1_var,
    }

    for feat in OTHER_FEATURES:
        if feat in obs.columns:
            vals = obs[feat].values.astype(float)
            rho_full, p_full   = spearmanr(pseudotime, vals)
            rho_resid, p_resid = spearmanr(residuals,  vals)
            row[f"rho_{feat}_full"]     = rho_full
            row[f"rho_{feat}_residual"] = rho_resid
            row[f"p_{feat}_full"]       = p_full
            row[f"p_{feat}_residual"]   = p_resid

    # Scatter: cellularity proxy vs pseudotime
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hexbin(proxy, pseudotime, gridsize=40, cmap="Blues", mincnt=1)
        ax.set_xlabel("Cellularity proxy (PC1 of nuclear features)")
        ax.set_ylabel("Pseudotime")
        ax.set_title(
            f"{results_dir.name}\nSpearman ρ = {rho_cell:.3f}  p = {p_cell:.2e}"
        )
        fig.tight_layout()
        out_png = output_dir / f"cellularity_scatter_{results_dir.name}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_png.name}")
    except Exception as exc:
        print(f"  WARNING: Could not save scatter plot: {exc}")

    return row


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Algebraic partial Spearman: rho(x, y | z).

    Uses the 3-variable formula:
        partial_rho = (rho_xy - rho_xz * rho_yz) /
                      sqrt((1 - rho_xz^2) * (1 - rho_yz^2))

    All inputs are raw values (not pre-ranked); spearmanr is applied internally.
    Returns nan if the denominator collapses (perfect collinearity).
    """
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if valid.sum() < 10:
        return float("nan")
    x, y, z = x[valid], y[valid], z[valid]
    rho_xy = float(spearmanr(x, y).statistic)
    rho_xz = float(spearmanr(x, z).statistic)
    rho_yz = float(spearmanr(y, z).statistic)
    denom = np.sqrt((1 - rho_xz ** 2) * (1 - rho_yz ** 2))
    if denom < 1e-10:
        return float("nan")
    return float((rho_xy - rho_xz * rho_yz) / denom)


def analyze_run_nuclear_density(results_dir: Path, n_permutations: int = 1000) -> dict:
    """Partial Spearman test: for each of the 5 non-density features, compute
    raw_rho and partial_rho controlling for nuclear_density, plus a permutation
    null on the partial correlation.

    Outputs land in results_dir/cellularity_confound/ (additive; never overwrites
    adata_full.h5ad, results.csv, or validation.json).

    Returns a summary dict with survive/collapse verdict per feature.
    """
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata is required: pip install anndata")

    h5ad = results_dir / "adata_full.h5ad"
    if not h5ad.exists():
        print(f"  SKIP: {h5ad} not found")
        return {}

    print(f"\n  Loading {results_dir.name} ...")
    adata = ad.read_h5ad(h5ad)
    obs = adata.obs.copy()

    required = ALL_OTHER_FEATURES + ["nuclear_density", "pseudotime"]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        print(f"  SKIP: missing columns {missing}")
        return {}

    pt = obs["pseudotime"].values.astype(float)
    nd = obs["nuclear_density"].values.astype(float)

    # Zero-order: how much does pseudotime track nuclear_density?
    rho_pt_nd = float(spearmanr(pt, nd).statistic)
    print(f"  rho(pseudotime, nuclear_density) = {rho_pt_nd:+.3f}")

    # Precompute rho(feature, nuclear_density) — constant across permutations
    rho_feat_nd = {
        feat: float(spearmanr(obs[feat].values.astype(float), nd).statistic)
        for feat in ALL_OTHER_FEATURES
    }

    rng = np.random.default_rng(42)
    perm_nulls = {feat: [] for feat in ALL_OTHER_FEATURES}

    print(f"  Running {n_permutations} permutations for partial-rho null ...")
    for _ in range(n_permutations):
        pt_shuf = rng.permutation(pt)
        for feat in ALL_OTHER_FEATURES:
            fvals = obs[feat].values.astype(float)
            valid = np.isfinite(pt_shuf) & np.isfinite(fvals) & np.isfinite(nd)
            if valid.sum() < 10:
                perm_nulls[feat].append(float("nan"))
                continue
            rho_xy = float(spearmanr(pt_shuf[valid], fvals[valid]).statistic)
            rho_xz = float(spearmanr(pt_shuf[valid], nd[valid]).statistic)
            rho_yz = rho_feat_nd[feat]
            denom = np.sqrt((1 - rho_xz ** 2) * (1 - rho_yz ** 2))
            pnull = float((rho_xy - rho_xz * rho_yz) / denom) if denom >= 1e-10 else float("nan")
            perm_nulls[feat].append(pnull)

    # Compute results per feature
    feature_results = {}
    survivors = []
    collapses = []

    print(f"\n  {'Feature':<26s}  {'raw_rho':>8s}  {'partial_rho':>11s}  {'delta':>7s}  {'perm_p':>7s}  Status")
    print("  " + "-" * 78)

    for feat in ALL_OTHER_FEATURES:
        fvals = obs[feat].values.astype(float)
        raw_rho = float(spearmanr(pt, fvals).statistic)
        prho = partial_spearman(pt, fvals, nd)
        delta = prho - raw_rho if not (np.isnan(prho) or np.isnan(raw_rho)) else float("nan")
        nulls = np.array([v for v in perm_nulls[feat] if not np.isnan(v)])
        perm_p = float(np.mean(np.abs(nulls) >= abs(prho))) if len(nulls) > 0 else float("nan")

        status = "SURVIVES" if abs(prho) >= 0.1 else "collapses"
        if abs(prho) >= 0.1:
            survivors.append(feat)
        else:
            collapses.append(feat)

        feature_results[feat] = {
            "raw_rho": raw_rho,
            "partial_rho": prho,
            "delta": delta,
            "partial_perm_p": perm_p,
        }
        print(f"  {feat:<26s}  {raw_rho:>+8.3f}  {prho:>+11.3f}  {delta:>+7.3f}  {perm_p:>7.4f}  {status}")

    output = {
        "run": results_dir.name,
        "control_feature": "nuclear_density",
        "rho_pt_vs_control": rho_pt_nd,
        "n_patches": int(len(pt)),
        "n_permutations": n_permutations,
        "features": feature_results,
        "summary": {
            "survivors": survivors,
            "collapses": collapses,
        },
    }

    out_dir = results_dir / "cellularity_confound"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "cellularity_confound.json"
    with open(json_path, "w") as f:
        import json as _json
        _json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Grouped bar figure: raw vs partial rho per feature
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        feats = ALL_OTHER_FEATURES
        x = np.arange(len(feats))
        raw_vals   = [feature_results[f]["raw_rho"]     for f in feats]
        part_vals  = [feature_results[f]["partial_rho"] for f in feats]
        w = 0.35

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - w / 2, raw_vals,  w, label="raw rho",     color="#4878CF", alpha=0.85)
        ax.bar(x + w / 2, part_vals, w, label="partial rho\n(ctrl: nuclear_density)",
               color="#D65F5F", alpha=0.85)
        ax.axhline( 0.1, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(-0.1, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(0,    color="k", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_", "\n") for f in feats], fontsize=9)
        ax.set_ylabel("Spearman ρ with pseudotime")
        ax.set_title(f"{results_dir.name} — cellularity confound\n"
                     f"rho(PT, nuclear_density) = {rho_pt_nd:+.3f}; "
                     f"dashed lines = ±0.1 threshold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig_path = out_dir / "raw_vs_partial_rho.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_path}")
    except Exception as exc:
        print(f"  WARNING: Could not save figure: {exc}")

    print(f"\n  SUMMARY [{results_dir.name}]: "
          f"survives={survivors or 'none'}, collapses={collapses or 'none'}")
    return output


def _decision_gate(rho):
    if rho < 0.3:
        return "SKIP Exp 3 — pseudotime barely tracks cellularity"
    if rho > 0.7:
        return "RECONSIDER — pseudotime is mainly a cellularity meter; reframe paper"
    return "RUN Exp 3 — partial confounding, residual signal worth exploring"


def main():
    parser = argparse.ArgumentParser(
        description="Cellularity confound test for atlas pipeline runs."
    )
    parser.add_argument(
        "--results-dirs", nargs="+", type=Path, required=True,
        help="Pipeline output directories containing adata_full.h5ad",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Destination for confound_summary.csv and scatter plots "
             "(required for --mode composite; ignored for --mode partial).",
    )
    parser.add_argument(
        "--mode", choices=["composite", "partial"], default="composite",
        help="'composite': original PC1-proxy test (default). "
             "'partial': partial Spearman controlling for nuclear_density per run dir.",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Permutation count for null distribution (--mode partial only; default: 1000).",
    )
    args = parser.parse_args()

    if args.mode == "partial":
        for d in args.results_dirs:
            analyze_run_nuclear_density(Path(d), n_permutations=args.n_permutations)
        return

    # ── composite mode (original behaviour) ──────────────────────────────────
    if args.output_dir is None:
        parser.error("--output-dir is required for --mode composite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in args.results_dirs:
        row = analyze_run(Path(d), args.output_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No valid results found — nothing to summarise.")
        return

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "confound_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSummary written: {csv_path}")

    display_cols = ["run", "n_patches", "rho_cellularity", "p_cellularity"]
    print(df[display_cols].to_string(index=False))

    print("\nDecision gates for Experiment 3:")
    for _, r in df.iterrows():
        print(f"  {r['run']}: ρ = {r['rho_cellularity']:.3f} → {_decision_gate(r['rho_cellularity'])}")


if __name__ == "__main__":
    main()
