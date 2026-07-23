"""Publication figures for the cancer trajectory atlas paper.

Read-only plotting module: reads existing result files from completed runs
and produces figures for the manuscript. Does NOT re-run the pipeline,
retrain scVI, rebuild the feature cache, or modify any adata/results file.

Figures supported:
  3  Batch mixing across correction methods (bar chart)
  5  Leave-one-slide-out reproducibility per section (two-panel bar)
  6  Cross-section morphological correlates (grouped bars)
  7  Cellularity confound per section (grouped bars, two panels)
  2  Pooled no-correction panel assembly (A: PAGA, B: section UMAP)
  4  Per-section 2x4 panel composite

Out of scope (not implemented here):
  1  Workflow schematic — hand-drawn diagram
  8  Holey-ness validation — requires region-level aggregation not yet implemented

CLI:
  python -m cancer_trajectory_atlas.figures.make_paper_figures \\
      --scvi-run-dir     $SCRATCH/results/atlas_none_scvi \\
      --harmony-run-dir  $SCRATCH/results/runs_paga/all_harmony/full \\
      --section1-dir     $SCRATCH/results/per_section/atlas_2M-1 \\
      --section2-dir     $SCRATCH/results/per_section/atlas_2M-2 \\
      --output-dir       $SCRATCH/results/paper_figures \\
      --dpi              300 \\
      --figures          3,5,6,7
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

# ── Style constants ───────────────────────────────────────────────────────────
TICK_FS = 9
LABEL_FS = 10
LEGEND_FS = 9

# Okabe-Ito colorblind-safe palette
SECTION_COLORS = {"2M-1": "#0072B2", "2M-2": "#E69F00"}
# Batch mixing bars: Raw PCA (sky blue), Harmony (orange), scVI (bluish-green)
MIXING_COLORS = ["#56B4E9", "#E69F00", "#009E73"]
RAW_BAR_COLOR = "#56B4E9"
PARTIAL_BAR_COLOR = "#CC79A7"  # mauve/violet — distinguishable from blue/orange

MORPH_FEATURES = [
    "nuclear_density", "mean_nuclear_area", "nc_ratio",
    "texture_entropy", "h_intensity", "packing_irregularity",
]
# ALL_OTHER_FEATURES from cellularity_confound.py — excludes nuclear_density
NON_DENSITY_FEATURES = [
    "mean_nuclear_area", "nc_ratio",
    "texture_entropy", "h_intensity", "packing_irregularity",
]

VALID_FIGURES = {"2", "3", "4", "5", "6", "7"}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _require_file(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"ERROR: required input file not found:\n  {path}")
    return path


def _load_json(path: Path) -> dict:
    _require_file(path)
    with open(path) as f:
        return json.load(f)


def _save(fig: plt.Figure, stem: Path, dpi: int) -> None:
    """Save as PDF (vector) and PNG, then close the figure."""
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {stem}.pdf / .png")


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Figure 3 ─────────────────────────────────────────────────────────────────

def figure3_batch_mixing(
    scvi_dir: Path,
    harmony_dir: Path,
    out_stem: Path,
    dpi: int,
    harmony_purity_override: float | None = None,
) -> None:
    """Figure 3: single-panel bar chart of kNN batch purity across methods."""
    scvi_data = _load_json(scvi_dir / "batch_mixing.json")
    harmony_data = _load_json(harmony_dir / "batch_mixing.json")

    # raw_pca: use scVI run as canonical (most recent pooled run)
    raw_pca = scvi_data["raw_pca"]

    scvi_val = scvi_data.get("scvi")
    if scvi_val is None:
        sys.exit(
            f"ERROR: 'scvi' key is null in:\n  {scvi_dir / 'batch_mixing.json'}\n"
            "  Ensure run_batch_mixing was executed on the scVI run directory:\n"
            f"    python -m cancer_trajectory_atlas.analysis.run_batch_mixing {scvi_dir}"
        )

    harmony_val = harmony_data.get("harmony")
    if harmony_val is None:
        if harmony_purity_override is not None:
            harmony_val = harmony_purity_override
            print(f"  INFO: 'harmony' key is null in batch_mixing.json; "
                  f"using --harmony-purity override: {harmony_val}")
        else:
            sys.exit(
                f"ERROR: 'harmony' key is null in:\n  {harmony_dir / 'batch_mixing.json'}\n"
                "  Pass --harmony-purity <value> as a CLI fallback."
            )

    chance = scvi_data["chance_baseline"]
    n_patches = scvi_data["n_patches"]

    labels = ["Raw PCA", "Harmony", "scVI"]
    values = [raw_pca, harmony_val, scvi_val]

    print(f"\n[Figure 3] Batch mixing")
    print(f"  n_patches={n_patches}  chance_baseline={chance:.4g}")
    for lbl, val in zip(labels, values):
        print(f"  {lbl}: {val:.4g}")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=MIXING_COLORS, width=0.55, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.010,
            f"{val:.4g}",
            ha="center", va="bottom", fontsize=TICK_FS,
        )

    ax.axhline(chance, color="0.35", linestyle="--", linewidth=1.0, zorder=2)
    # Blended transform: axes-relative x, data-coordinate y
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.97, chance + 0.013,
        f"chance baseline ({chance:.3f})",
        transform=trans, ha="right", va="bottom",
        fontsize=TICK_FS - 1, color="0.35",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FS)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean kNN same-section fraction", fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.text(
        0.97, 0.03, "lower = better mixing",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=TICK_FS - 1, style="italic", color="0.45",
    )
    _despine(ax)
    fig.tight_layout()
    _save(fig, out_stem, dpi)


# ── Figure 5 ─────────────────────────────────────────────────────────────────

def figure5_loo(
    section1_dir: Path,
    section2_dir: Path,
    out_stem: Path,
    dpi: int,
) -> None:
    """Figure 5: LOO reproducibility, two panels side by side."""
    LOO_THRESHOLD = 0.6

    def _load(d: Path) -> pd.DataFrame:
        csv = _require_file(d / "loo_summary" / "loo_summary.csv")
        df = pd.read_csv(csv)
        if "spearman_rho" not in df.columns:
            sys.exit(
                f"ERROR: 'spearman_rho' column missing in {csv}\n"
                "  Check that loo_project.py wrote spearman_rho to loo_result_*.json."
            )
        return df.sort_values("spearman_rho", ascending=False).reset_index(drop=True)

    def _strip_x5(name: object) -> str:
        s = str(name)
        return s[:-3] if s.endswith("_x5") else s

    df1 = _load(section1_dir)
    df2 = _load(section2_dir)

    print(f"\n[Figure 5] LOO reproducibility")
    for label, df in [("2M-1", df1), ("2M-2", df2)]:
        mean_rho = df["spearman_rho"].mean()
        print(f"  Section {label}: n={len(df)}  mean_rho={mean_rho:.4f}  "
              f"min={df['spearman_rho'].min():.4f}  max={df['spearman_rho'].max():.4f}")
        for _, row in df.iterrows():
            print(f"    {row['slide_name']}: {row['spearman_rho']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

    for ax, df, section_label in zip(axes, [df1, df2], ["2M-1", "2M-2"]):
        tick_names = [_strip_x5(n) for n in df["slide_name"]]
        x = np.arange(len(tick_names))
        color = SECTION_COLORS[section_label]
        ax.bar(x, df["spearman_rho"].values, color=color, width=0.6, zorder=3)

        mean_rho = float(df["spearman_rho"].mean())
        ax.axhline(LOO_THRESHOLD, color="0.35", linestyle="--", linewidth=1.0, zorder=2)
        ax.axhline(mean_rho, color="0.15", linestyle=":", linewidth=1.2, zorder=2)

        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.97, LOO_THRESHOLD + 0.025, "threshold (0.6)",
                transform=trans, ha="right", va="bottom",
                fontsize=TICK_FS - 1, color="0.35")
        ax.text(0.97, mean_rho + 0.025, f"mean ({mean_rho:.3f})",
                transform=trans, ha="right", va="bottom",
                fontsize=TICK_FS - 1, color="0.15")

        ax.set_xticks(x)
        ax.set_xticklabels(tick_names, rotation=45, ha="right", fontsize=TICK_FS - 1)
        ax.set_title(f"Section {section_label}", fontsize=LABEL_FS, pad=4)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        _despine(ax)

    axes[0].set_ylabel("Spearman rho (in-manifold vs projected)", fontsize=LABEL_FS)
    fig.tight_layout()
    _save(fig, out_stem, dpi)


# ── Figure 6 ─────────────────────────────────────────────────────────────────

def figure6_morphology(
    section1_dir: Path,
    section2_dir: Path,
    out_stem: Path,
    dpi: int,
) -> None:
    """Figure 6: cross-section morphological correlates, grouped bars."""
    def _load_rhos(d: Path) -> dict:
        v = _load_json(d / "validation.json")
        fc = v.get("feature_correlations", {})
        missing = [f for f in MORPH_FEATURES if f not in fc]
        if missing:
            sys.exit(
                f"ERROR: features missing from {d / 'validation.json'}: {missing}\n"
                "  Expected under key: feature_correlations -> <feature> -> rho"
            )
        return {f: fc[f]["rho"] for f in MORPH_FEATURES}

    rhos_1 = _load_rhos(section1_dir)
    rhos_2 = _load_rhos(section2_dir)

    print(f"\n[Figure 6] Morphological correlates (Spearman rho)")
    print(f"  {'Feature':<26s}  {'2M-1':>8s}  {'2M-2':>8s}")
    for feat in MORPH_FEATURES:
        print(f"  {feat:<26s}  {rhos_1[feat]:>+8.4f}  {rhos_2[feat]:>+8.4f}")

    vals_1 = [rhos_1[f] for f in MORPH_FEATURES]
    vals_2 = [rhos_2[f] for f in MORPH_FEATURES]
    y_abs_max = max(abs(v) for v in vals_1 + vals_2)
    y_lim = max(0.5, y_abs_max * 1.25)

    x = np.arange(len(MORPH_FEATURES))
    w = 0.35
    feat_labels = [f.replace("_", "\n") for f in MORPH_FEATURES]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w / 2, vals_1, w, color=SECTION_COLORS["2M-1"], label="2M-1", zorder=3)
    ax.bar(x + w / 2, vals_2, w, color=SECTION_COLORS["2M-2"], label="2M-2", zorder=3)
    ax.axhline(0, color="0.2", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, fontsize=TICK_FS)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_ylabel("Signed Spearman rho", fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, frameon=False)
    _despine(ax)
    fig.tight_layout()
    _save(fig, out_stem, dpi)


# ── Figure 7 ─────────────────────────────────────────────────────────────────

def figure7_cellularity(
    section1_dir: Path,
    section2_dir: Path,
    out_stem: Path,
    dpi: int,
) -> None:
    """Figure 7: cellularity confound, raw vs partial rho, two panels."""
    def _load_confound(d: Path) -> tuple[dict, dict]:
        path = d / "cellularity_confound" / "cellularity_confound.json"
        data = _load_json(path)
        feats = data.get("features", {})
        missing = [f for f in NON_DENSITY_FEATURES if f not in feats]
        if missing:
            sys.exit(
                f"ERROR: features missing from {path}: {missing}\n"
                "  Expected under key: features -> <feature> -> raw_rho / partial_rho"
            )
        # Note: key is partial_perm_p (not perm_p) in the JSON schema
        raw = {f: feats[f]["raw_rho"] for f in NON_DENSITY_FEATURES}
        partial = {f: feats[f]["partial_rho"] for f in NON_DENSITY_FEATURES}
        return raw, partial

    raw1, partial1 = _load_confound(section1_dir)
    raw2, partial2 = _load_confound(section2_dir)

    print(f"\n[Figure 7] Cellularity confound (raw rho vs partial rho | nuclear_density)")
    print(f"  {'Feature':<26s}  {'raw 2M-1':>9s}  {'part 2M-1':>10s}  {'raw 2M-2':>9s}  {'part 2M-2':>10s}")
    for feat in NON_DENSITY_FEATURES:
        print(f"  {feat:<26s}  {raw1[feat]:>+9.4f}  {partial1[feat]:>+10.4f}"
              f"  {raw2[feat]:>+9.4f}  {partial2[feat]:>+10.4f}")

    all_finite = [
        v for d in [raw1, partial1, raw2, partial2]
        for v in d.values() if np.isfinite(v)
    ]
    y_lim = max(0.5, max(abs(v) for v in all_finite) * 1.25) if all_finite else 0.5

    x = np.arange(len(NON_DENSITY_FEATURES))
    w = 0.35
    feat_labels = [f.replace("_", "\n") for f in NON_DENSITY_FEATURES]
    raw_patch = mpatches.Patch(color=RAW_BAR_COLOR, label="raw rho")
    part_patch = mpatches.Patch(color=PARTIAL_BAR_COLOR,
                                label="partial rho\n(ctrl: nuclear_density)")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

    for ax, raw, partial, section_label in zip(
        axes, [raw1, raw2], [partial1, partial2], ["2M-1", "2M-2"]
    ):
        # NaN partial_rho (denominator collapse) rendered as zero bar
        rvals = [raw[f] if np.isfinite(raw[f]) else 0.0 for f in NON_DENSITY_FEATURES]
        pvals = [partial[f] if np.isfinite(partial[f]) else 0.0 for f in NON_DENSITY_FEATURES]
        ax.bar(x - w / 2, rvals, w, color=RAW_BAR_COLOR, zorder=3)
        ax.bar(x + w / 2, pvals, w, color=PARTIAL_BAR_COLOR, zorder=3)
        ax.axhline(0, color="0.2", linewidth=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(feat_labels, fontsize=TICK_FS)
        ax.set_title(f"Section {section_label}", fontsize=LABEL_FS, pad=4)
        ax.set_ylim(-y_lim, y_lim)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.legend(handles=[raw_patch, part_patch], fontsize=LEGEND_FS - 1, frameon=False)
        _despine(ax)

    axes[0].set_ylabel("Spearman rho with pseudotime", fontsize=LABEL_FS)
    fig.tight_layout()
    _save(fig, out_stem, dpi)


# ── Figure 2 ─────────────────────────────────────────────────────────────────

def figure2_panel(
    nocorrection_dir: Path,
    out_stem: Path,
    dpi: int,
) -> None:
    """Figure 2: pooled no-correction two-panel assembly (PAGA + section UMAP)."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("ERROR: Pillow is required for panel assembly. Install with: pip install Pillow")

    figs_dir = nocorrection_dir / "figures"
    paga_path = _require_file(figs_dir / "qc_paga_topology.png")
    umap_path = _require_file(figs_dir / "qc_umap_section_vs_cluster.png")

    img_a = np.array(Image.open(paga_path).convert("RGB"))
    img_b = np.array(Image.open(umap_path).convert("RGB"))

    print(f"\n[Figure 2]")
    print(f"  (A) {paga_path.name}  {img_a.shape[1]}x{img_a.shape[0]} px")
    print(f"  (B) {umap_path.name}  {img_b.shape[1]}x{img_b.shape[0]} px")

    # Pad the shorter image to a common height with white background
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    h = max(ha, hb)
    if ha < h:
        img_a = np.vstack([img_a, np.full((h - ha, wa, 3), 255, dtype=np.uint8)])
    if hb < h:
        img_b = np.vstack([img_b, np.full((h - hb, wb, 3), 255, dtype=np.uint8)])

    total_w = wa + wb
    fig_w = 10.0
    fig, axes = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_w * h / total_w * 0.90),
        gridspec_kw={"width_ratios": [wa, wb]},
    )
    for ax, img, letter in zip(axes, [img_a, img_b], ["A", "B"]):
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.01, 0.99, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")
    fig.tight_layout(pad=0.2)
    _save(fig, out_stem, dpi)


# ── Figure 4 ─────────────────────────────────────────────────────────────────

def figure4_panel(
    section1_dir: Path,
    section2_dir: Path,
    out_stem: Path,
    dpi: int,
) -> None:
    """Figure 4: 2-row x 4-column panel composite from per-section run dirs."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("ERROR: Pillow is required for panel assembly. Install with: pip install Pillow")

    COL_FILES = [
        "qc_paga_topology.png",
        "diffusion_3d.png",
        "fig4_umap_pseudotime.png",
        "fig5_pt_violins.png",
    ]

    print(f"\n[Figure 4]")
    images: list[list[np.ndarray]] = []
    for section_label, d in [("2M-1", section1_dir), ("2M-2", section2_dir)]:
        row: list[np.ndarray] = []
        for fname in COL_FILES:
            p = _require_file(d / "figures" / fname)
            img = np.array(Image.open(p).convert("RGB"))
            row.append(img)
            print(f"  ({section_label}) {fname}  {img.shape[1]}x{img.shape[0]} px")
        images.append(row)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    letters = list("ABCDEFGH")
    for row_idx, row_imgs in enumerate(images):
        for col_idx, img in enumerate(row_imgs):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis("off")
            ax.text(0.01, 0.99, letters[row_idx * 4 + col_idx],
                    transform=ax.transAxes, fontsize=13, fontweight="bold",
                    va="top", ha="left")
    fig.tight_layout(pad=0.2)
    _save(fig, out_stem, dpi)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication figures for the cancer trajectory atlas."
    )
    parser.add_argument("--scvi-run-dir", type=Path, required=True,
                        help="Pooled scVI run dir ($SCRATCH/results/atlas_none_scvi)")
    parser.add_argument("--harmony-run-dir", type=Path, required=True,
                        help="Pooled Harmony run dir ($SCRATCH/results/runs_paga/all_harmony/full)")
    parser.add_argument("--section1-dir", type=Path, required=True,
                        help="Per-section run dir for 2M-1 ($SCRATCH/results/per_section/atlas_2M-1)")
    parser.add_argument("--section2-dir", type=Path, required=True,
                        help="Per-section run dir for 2M-2 ($SCRATCH/results/per_section/atlas_2M-2)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination for all output PDF/PNG files")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG resolution (default: 300)")
    parser.add_argument("--figures", default="3,5,6,7",
                        help="Comma-separated figure numbers to generate (default: '3,5,6,7')")
    parser.add_argument("--harmony-purity", type=float, default=None,
                        help="Fallback Harmony purity if harmony_run_dir/batch_mixing.json "
                             "has a null 'harmony' key. Required only if that file is missing "
                             "the key and Figure 3 is requested.")
    parser.add_argument("--pooled-nocorrection-dir", type=Path, default=None,
                        help="Pooled no-correction run dir — required for Figures 2 and 4. "
                             "Expected: $SCRATCH/results/runs_paga/all_noharmony/full")
    args = parser.parse_args()

    requested = [f.strip() for f in args.figures.split(",")]
    invalid = [f for f in requested if f not in VALID_FIGURES]
    if invalid:
        sys.exit(
            f"ERROR: unknown figure number(s): {invalid}\n"
            f"  Valid: {sorted(VALID_FIGURES)}\n"
            "  Note: Figures 1 and 8 are out of scope (see module docstring)."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": TICK_FS,
        "axes.labelsize": LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": LEGEND_FS,
        "pdf.fonttype": 42,   # TrueType fonts embedded in PDF (journal requirement)
        "ps.fonttype": 42,
    })

    for fig_num in requested:
        if fig_num == "3":
            figure3_batch_mixing(
                args.scvi_run_dir, args.harmony_run_dir,
                args.output_dir / "fig3_batch_mixing",
                args.dpi, args.harmony_purity,
            )
        elif fig_num == "5":
            figure5_loo(
                args.section1_dir, args.section2_dir,
                args.output_dir / "fig5_loo_reproducibility",
                args.dpi,
            )
        elif fig_num == "6":
            figure6_morphology(
                args.section1_dir, args.section2_dir,
                args.output_dir / "fig6_morphology_correlates",
                args.dpi,
            )
        elif fig_num == "7":
            figure7_cellularity(
                args.section1_dir, args.section2_dir,
                args.output_dir / "fig7_cellularity_confound",
                args.dpi,
            )
        elif fig_num == "2":
            if args.pooled_nocorrection_dir is None:
                sys.exit(
                    "ERROR: --pooled-nocorrection-dir is required for Figure 2.\n"
                    "  Expected: $SCRATCH/results/runs_paga/all_noharmony/full\n"
                    "  If that run does not exist, Figure 2 cannot be assembled "
                    "(no corrected run substituted)."
                )
            figure2_panel(
                args.pooled_nocorrection_dir,
                args.output_dir / "fig2_pooled_nocorrection",
                args.dpi,
            )
        elif fig_num == "4":
            figure4_panel(
                args.section1_dir, args.section2_dir,
                args.output_dir / "fig4_per_section_panel",
                args.dpi,
            )

    print(f"\nAll requested figures written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
