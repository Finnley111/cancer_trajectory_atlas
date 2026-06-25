"""Post-processing analysis for the scVI atlas run.

Loads adata_full.h5ad and validation.json from a completed scVI run and
produces, into $RUN_DIR/postprocess/:

  1. umap_leiden_lowres.png    — UMAP coloured by a fresh low-resolution
                                 Leiden clustering (key: leiden_lowres).
                                 Requires adata.obsp["connectivities"].
                                 adata_full.h5ad is NEVER written.
  2. umap_by_section.png       — UMAP coloured by section_number (overlay).
  3. umap_by_section_split.png — same, one panel per section.
  4. morphology_correlations.csv / .png — Spearman rho + permutation p-value
                                 table with effect-size flags, loaded from
                                 the existing validation.json (no recompute).
  5. SUMMARY.txt               — cluster count + correlation summary.

Usage:
    python -m cancer_trajectory_atlas.visualize.scvi_postprocess \\
        --run-dir           $SCRATCH/results/atlas_none_scvi \\
        --output-dir        $SCRATCH/results/atlas_none_scvi/postprocess \\
        --leiden-resolution 0.4
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# Section colours matching analysis/plot_umap_by_section.py
SECTION_COLORS = {"2M-1": "#1b9e77", "2M-2": "#d95f02"}

# Effect-size thresholds (see SUMMARY note on biological interpretation)
_NEGLIGIBLE_RHO = 0.05   # below this: significant_but_negligible
_MEANINGFUL_RHO = 0.10   # at or above this: biologically meaningful


# ── Part B.1: Low-resolution Leiden + UMAP ──────────────────────────────────

def _run_leiden_lowres(adata, leiden_resolution: float) -> int:
    """Recompute Leiden at lower resolution on the stored neighbor graph.

    Writes leiden_lowres into adata.obs in memory only — h5ad is never touched.
    Returns the cluster count.
    """
    import scanpy as sc
    sc.tl.leiden(adata, resolution=leiden_resolution, key_added="leiden_lowres")
    n = adata.obs["leiden_lowres"].nunique()
    print(f"  leiden_lowres: {n} clusters at resolution={leiden_resolution}")
    return n


def _plot_leiden_lowres(adata, output_dir: Path) -> None:
    umap = np.asarray(adata.obsm["X_umap"])[:, :2]
    labels = adata.obs["leiden_lowres"].astype(str).values

    # Sort numerically (scanpy stores cluster labels as strings "0", "1", ...)
    unique_labels = sorted(set(labels), key=lambda x: int(x) if x.isdigit() else 0)
    n = len(unique_labels)
    cmap = plt.cm.get_cmap("tab20", max(n, 1))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, c in enumerate(unique_labels):
        mask = labels == c
        ax.scatter(umap[mask, 0], umap[mask, 1], s=5, alpha=0.5,
                   color=cmap(i), label=f"Cluster {c}", rasterized=True)
    ax.set_title(f"UMAP — leiden_lowres ({n} clusters)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if n <= 12:
        ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    out = output_dir / "umap_leiden_lowres.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Part B.2: Section-mixing figures ────────────────────────────────────────

def _section_color_map(observed_sections):
    """Return {section: colour} for all observed sections.

    Uses SECTION_COLORS for the two known values; falls back to tab10
    for any unexpected section labels so the figure still renders.
    """
    colors = dict(SECTION_COLORS)
    extra = sorted(s for s in observed_sections if s not in colors)
    cmap = plt.cm.get_cmap("tab10", max(len(extra), 1))
    for i, s in enumerate(extra):
        colors[s] = cmap(i)
    return colors


def _plot_section_mixing(adata, output_dir: Path, run_name: str) -> None:
    umap = np.asarray(adata.obsm["X_umap"])[:, :2]
    section = adata.obs["section_number"].astype(str).to_numpy()
    sections_sorted = sorted(set(section))
    colors = _section_color_map(sections_sorted)

    xlim = (float(umap[:, 0].min()), float(umap[:, 0].max()))
    ylim = (float(umap[:, 1].min()), float(umap[:, 1].max()))

    # Figure 1: overlaid scatter, draw order shuffled so neither section
    # is painted entirely on top of the other (matches plot_umap_by_section.py).
    rng = np.random.default_rng(0)
    order = rng.permutation(umap.shape[0])
    color_arr = np.array([colors[s] for s in section])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(umap[order, 0], umap[order, 1],
               c=color_arr[order], s=3, alpha=0.4, linewidths=0)
    for sec in sections_sorted:
        ax.scatter([], [], c=colors[sec], label=sec, s=20, alpha=1.0)
    ax.legend(title="Section", loc="best")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"UMAP by Section — {run_name}")
    fig.tight_layout()
    out = output_dir / "umap_by_section.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # Figure 2: side-by-side panels, one per section, shared axis limits.
    ncols = len(sections_sorted)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols + 1, 6),
                             sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, sec in zip(axes, sections_sorted):
        mask = section == sec
        ax.scatter(umap[mask, 0], umap[mask, 1],
                   c=colors[sec], s=3, alpha=0.4, linewidths=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("UMAP1")
        ax.set_title(sec)
    axes[0].set_ylabel("UMAP2")
    fig.suptitle(f"UMAP by Section (split) — {run_name}")
    fig.tight_layout()
    out = output_dir / "umap_by_section_split.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Part B.3: Morphology correlation table ──────────────────────────────────

def _effect_size_flag(perm_p, rho) -> str:
    if pd.isna(perm_p) or pd.isna(rho):
        return "insufficient_data"
    if perm_p < 0.05:
        return ("significant_but_negligible"
                if abs(rho) < _NEGLIGIBLE_RHO else "significant")
    return "not_significant"


def _load_and_flag_correlations(run_dir: Path, adata) -> pd.DataFrame:
    """Load feature–pseudotime correlations from validation.json and add
    effect-size flags.  Falls back to recomputing via validation.correlations
    functions if the JSON is absent.
    """
    json_path = run_dir / "validation.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        feat_corr = data.get("feature_correlations", {})
        perm_res = data.get("permutation_tests", {})
        features = sorted(feat_corr.keys())
        rows = []
        for feat in features:
            rho = feat_corr[feat].get("rho")
            perm_p = perm_res.get(feat, {}).get("perm_p_value")
            sig = perm_res.get(feat, {}).get("significant", False)
            rows.append({
                "feature":          feat,
                "rho":              rho,
                "perm_p_value":     perm_p,
                "significant":      sig,
                "effect_size_flag": _effect_size_flag(perm_p, rho),
            })
        print(f"  Loaded correlations from {json_path.name}")
        return pd.DataFrame(rows)

    # Fallback: recompute using existing validation functions.
    print(f"  WARNING: {json_path.name} not found; recomputing from results.csv.")
    from ..validation.correlations import (
        correlate_features_with_pseudotime,
        permutation_test,
    )
    STANDARD_COLS = {
        "x", "y", "slide_id", "slide_name", "cluster",
        "pseudotime", "pseudotime_std",
    }
    csv_path = run_dir / "results.csv"
    df = pd.read_csv(csv_path)
    morph_cols = [c for c in df.columns if c not in STANDARD_COLS]
    pseudotime = df["pseudotime"].values
    morph_features = {c: df[c].values for c in morph_cols}

    feat_corr = correlate_features_with_pseudotime(pseudotime, morph_features)
    perm_res = permutation_test(pseudotime, morph_features, n_permutations=200)

    rows = []
    for feat in sorted(feat_corr.keys()):
        rho = feat_corr[feat].get("rho")
        perm_p = perm_res.get(feat, {}).get("perm_p_value")
        sig = perm_res.get(feat, {}).get("significant", False)
        rows.append({
            "feature":          feat,
            "rho":              rho,
            "perm_p_value":     perm_p,
            "significant":      sig,
            "effect_size_flag": _effect_size_flag(perm_p, rho),
        })
    return pd.DataFrame(rows)


def _plot_corr_bar(corr_df: pd.DataFrame, output_dir: Path) -> None:
    # Sort by |rho| ascending so largest bars appear at top of the chart.
    df = corr_df.sort_values("rho", key=abs, ascending=True).reset_index(drop=True)

    FLAG_COLORS = {
        "significant":               "#2ca02c",
        "significant_but_negligible": "#ff7f0e",
        "not_significant":            "#aaaaaa",
        "insufficient_data":          "#dddddd",
    }
    bar_colors = [FLAG_COLORS.get(f, "#aaaaaa") for f in df["effect_size_flag"]]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.7 * len(df))))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["rho"].fillna(0), color=bar_colors, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"], fontsize=9)

    # Reference lines at zero, ±0.1 (bio threshold), ±0.3 (moderate threshold)
    ax.axvline(0,    color="black",   linewidth=1.0, linestyle="-")
    for xref in (-0.3, -0.1, 0.1, 0.3):
        ax.axvline(xref, color="#888888", linewidth=0.5, linestyle="--")

    # p-value annotations to the side of each bar
    for idx, row in df.iterrows():
        p = row["perm_p_value"]
        if pd.isna(p):
            continue
        label = f"p={p:.3f}"
        x = row["rho"] if not pd.isna(row["rho"]) else 0.0
        offset = 0.01 if x >= 0 else -0.01
        ha = "left" if x >= 0 else "right"
        ax.text(x + offset, idx, label, va="center", ha=ha, fontsize=7)

    # Dynamic x-axis limits
    max_abs = max((abs(v) for v in df["rho"].dropna()), default=0.35)
    xlim_bound = max(0.4, max_abs + 0.15)
    ax.set_xlim(-xlim_bound, xlim_bound)

    # Legend
    legend_elements = [
        Patch(facecolor=FLAG_COLORS["significant"],
              label="significant"),
        Patch(facecolor=FLAG_COLORS["significant_but_negligible"],
              label="significant_but_negligible"),
        Patch(facecolor=FLAG_COLORS["not_significant"],
              label="not_significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    ax.set_xlabel("Spearman ρ (vs pseudotime)")
    ax.set_title("Morphology–Pseudotime Correlations (scVI run)")
    plt.tight_layout()
    out = output_dir / "morphology_correlations.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Part B.4: SUMMARY.txt ───────────────────────────────────────────────────

def _write_summary(
    output_dir: Path,
    run_dir: Path,
    n_lowres_clusters,
    leiden_resolution: float,
    corr_df: pd.DataFrame,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "scVI Atlas Post-Processing Summary",
        f"Generated: {ts}",
        f"Run dir:   {run_dir}",
        "",
        "=== Low-resolution Leiden (leiden_lowres) ===",
    ]
    if n_lowres_clusters is not None:
        lines.append(
            f"Clusters at resolution={leiden_resolution}: {n_lowres_clusters}"
        )
    else:
        lines.append(
            "Skipped — adata.obsp['connectivities'] not found in adata_full.h5ad."
        )

    lines += [
        "",
        "=== Morphology–Pseudotime Correlations ===",
        f"{'Feature':<30s}  {'rho':>8s}  {'perm_p':>10s}  effect_size_flag",
        "-" * 72,
    ]
    for _, row in corr_df.sort_values("rho", key=abs, ascending=False).iterrows():
        rho_s = (f"{row['rho']:+.3f}" if not pd.isna(row["rho"]) else "   nan")
        p_s   = (f"{row['perm_p_value']:.4f}"
                 if not pd.isna(row["perm_p_value"]) else "     nan")
        lines.append(
            f"{row['feature']:<30s}  {rho_s:>8s}  {p_s:>10s}  {row['effect_size_flag']}"
        )

    # Biological interpretation notes
    meaningful = corr_df[
        corr_df["rho"].abs() >= _MEANINGFUL_RHO
    ]["feature"].tolist()
    negligible = corr_df[
        corr_df["effect_size_flag"] == "significant_but_negligible"
    ]["feature"].tolist()
    not_sig = corr_df[
        corr_df["effect_size_flag"] == "not_significant"
    ]["feature"].tolist()

    lines += [
        "",
        "=== Biological interpretation ===",
        (f"Biologically meaningful (|rho| >= {_MEANINGFUL_RHO}): "
         + (", ".join(meaningful) if meaningful else "none")),
        (f"Significant but negligible (|rho| < {_NEGLIGIBLE_RHO}): "
         + (", ".join(negligible) if negligible else "none")),
        (f"Not significant: "
         + (", ".join(not_sig) if not_sig else "none")),
        "",
        "=== Qualitative outputs (visual inspection required) ===",
        f"  Spatial pseudotime overlays:     {run_dir}/overlays/",
        f"  High-PT coherence contact sheet: {run_dir}/patch_export/high_pseudotime/",
    ]

    out = output_dir / "SUMMARY.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "scVI-specific post-processing: section-mixing figures, "
            "morphology correlation table, and low-res leiden_lowres UMAP."
        )
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to the completed scVI run directory (contains adata_full.h5ad)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write outputs (default: {run_dir}/postprocess)",
    )
    parser.add_argument(
        "--leiden-resolution", type=float, default=0.4,
        help="Leiden resolution for low-res re-clustering (default: 0.4)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "postprocess").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import scanpy as sc
    except ImportError:
        print("ERROR: scanpy is required.  pip install scanpy")
        raise SystemExit(1)

    h5ad_path = run_dir / "adata_full.h5ad"
    if not h5ad_path.exists():
        print(f"ERROR: {h5ad_path} not found.")
        raise SystemExit(1)

    print(f"Loading {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    run_name = run_dir.name
    print(f"  {adata.n_obs} patches | obsm: {list(adata.obsm.keys())}")

    # Startup checks — required for Parts B.2 and B.3 even if B.1 is skipped.
    missing = []
    if "section_number" not in adata.obs.columns:
        missing.append("obs['section_number']")
    if "X_umap" not in adata.obsm:
        missing.append("obsm['X_umap']")
    if missing:
        print(f"ERROR: missing required keys in {h5ad_path.name}: {missing}")
        raise SystemExit(1)

    # Part B.1 — leiden_lowres (requires stored neighbor graph)
    print("\n=== Part B.1: leiden_lowres ===")
    n_lowres_clusters = None
    if "connectivities" in adata.obsp:
        n_lowres_clusters = _run_leiden_lowres(adata, args.leiden_resolution)
        _plot_leiden_lowres(adata, output_dir)
    else:
        print(
            "WARNING: adata.obsp['connectivities'] not found.\n"
            "  The neighbor graph is not stored in adata_full.h5ad — cannot\n"
            "  recompute leiden_lowres without rebuilding it.  Skipping Part B.1."
        )

    # Part B.2 — Section-mixing figures
    print("\n=== Part B.2: section-mixing figures ===")
    _plot_section_mixing(adata, output_dir, run_name)

    # Part B.3 — Morphology correlation table
    print("\n=== Part B.3: morphology correlation table ===")
    corr_df = _load_and_flag_correlations(run_dir, adata)
    csv_out = output_dir / "morphology_correlations.csv"
    corr_df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")
    _plot_corr_bar(corr_df, output_dir)

    # Part B.4 — SUMMARY.txt
    print("\n=== Part B.4: SUMMARY.txt ===")
    _write_summary(
        output_dir, run_dir, n_lowres_clusters, args.leiden_resolution, corr_df
    )

    print(f"\nDone.  All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
