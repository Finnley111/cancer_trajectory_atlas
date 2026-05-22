"""Post-hoc diagnostic for a single outlier slide in the LOO projection experiment.

Investigates five candidate explanations for why a target slide projects poorly:
  H1: Unusual patch count (noisy in-manifold distribution)
  H2: Unusual cluster composition (overrepresented in rare clusters)
  H3: Outlier in Phikon feature space (slide centroid distance)
  H4: Isolated position in UMAP space (patches in unpopulated regions)
  H5: Unusual annotation pattern (polygon count or area)

Usage:
    python -m cancer_trajectory_atlas.analysis.slide_diagnostics \\
        --adata-path         $SCRATCH/results/atlas_macenko_harmony/adata_full.h5ad \\
        --results-csv        $SCRATCH/results/atlas_macenko_harmony/results.csv \\
        --loo-dir            $SCRATCH/results/loo_phase_b \\
        --features-cache-dir $SCRATCH/features_cache \\
        --annotation-dir     $SCRATCH/data/annotations \\
        --target-slide       6028-4L-2M-2_x5 \\
        --output-dir         $SCRATCH/results/slide_diagnostics
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

_RED = "#e74c3c"
_GRAY = "#95a5a6"
_BLUE = "#3498db"


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def load_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    required = {"slide_name", "cluster", "pseudotime", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results.csv missing columns: {missing}")
    df["cluster"] = df["cluster"].astype(str)
    return df


def load_loo_summary(loo_dir: Path) -> pd.DataFrame:
    loo_dir = Path(loo_dir)
    summary_csv = loo_dir / "loo_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
    else:
        rows = []
        for p in sorted(loo_dir.glob("loo_result_*.json")):
            with open(p) as f:
                rows.append(json.load(f))
        if not rows:
            raise FileNotFoundError(f"No LOO results found in {loo_dir}")
        df = pd.DataFrame(rows)
    keep = [c for c in ["slide_name", "n_patches", "spearman_rho", "wasserstein"] if c in df.columns]
    return df[keep].copy()


def load_umap(adata_path: Path) -> np.ndarray:
    import scanpy as sc
    adata = sc.read_h5ad(str(adata_path))
    if "X_umap" not in adata.obsm:
        raise KeyError("X_umap not found in adata.obsm")
    return np.array(adata.obsm["X_umap"])


def load_slide_features(features_cache_dir: Path, slide_names) -> dict:
    features_cache_dir = Path(features_cache_dir)
    out = {}
    for name in slide_names:
        fp = features_cache_dir / f"{name}_features.npy"
        if fp.exists():
            out[name] = np.load(str(fp))
        else:
            print(f"  WARN: feature file not found: {fp}")
    return out


def _shoelace_area(ring) -> float:
    n = len(ring)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += ring[i][0] * ring[j][1]
        area -= ring[j][0] * ring[i][1]
    return abs(area) / 2.0


def load_annotations(annotation_dir: Path, slide_names) -> dict:
    annotation_dir = Path(annotation_dir)
    out = {}
    for name in slide_names:
        base = name.replace("_x5", "")
        ann_path = None
        for cand in [
            annotation_dir / f"{name}.json",
            annotation_dir / f"{base}.json",
            annotation_dir / f"{name}.geojson",
            annotation_dir / f"{base}.geojson",
        ]:
            if cand.exists():
                ann_path = cand
                break

        if ann_path is None:
            out[name] = {"n_polygons": 0, "total_area_ratio": None, "labels": []}
            continue

        with open(ann_path) as f:
            gj = json.load(f)

        if gj.get("type") == "FeatureCollection":
            features = gj.get("features", [])
        else:
            features = [gj]

        n_polys = 0
        total_area = 0.0
        labels = []
        for feat in features:
            geom = feat.get("geometry", {})
            gtype = geom.get("type", "")
            if gtype == "Polygon":
                rings = geom.get("coordinates", [])
                if rings:
                    total_area += _shoelace_area(rings[0])
                    n_polys += 1
            elif gtype == "MultiPolygon":
                for poly in geom.get("coordinates", []):
                    if poly:
                        total_area += _shoelace_area(poly[0])
                        n_polys += 1
            props = feat.get("properties", {})
            clf = props.get("classification", {})
            lname = clf.get("name", "") if isinstance(clf, dict) else str(clf)
            if lname:
                labels.append(lname)

        out[name] = {"n_polygons": n_polys, "total_area_ratio": total_area, "labels": labels}

    return out


# ─────────────────────────────────────────────────────────────
# H1: Patch counts
# ─────────────────────────────────────────────────────────────

def investigate_patch_counts(results_df, loo_df, target_slide, output_dir):
    counts = results_df.groupby("slide_name").size().rename("patch_count")
    df = loo_df.set_index("slide_name").join(counts).reset_index()
    df = df.sort_values("spearman_rho").reset_index(drop=True)

    median_count = df["patch_count"].median()
    target_idx = df["slide_name"].tolist().index(target_slide)
    target_count = int(df.iloc[target_idx]["patch_count"])
    rel_diff = abs(target_count - median_count) / median_count
    flag = rel_diff > 0.33
    flag_msg = (
        f"FLAGGED: target patch count {target_count:,} is {rel_diff*100:.0f}% away from cohort median {int(median_count):,}"
        if flag else
        f"OK: target patch count {target_count:,} is within 33% of cohort median {int(median_count):,}"
    )

    colors = [_RED if s == target_slide else _BLUE for s in df["slide_name"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(df)), df["patch_count"], color=colors, width=0.8)
    ax.axhline(median_count, color="black", linestyle="--", linewidth=1, label=f"Median ({int(median_count):,})")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(
        [f"{s.replace('_x5', '')} ({r:.2f})" for s, r in zip(df["slide_name"], df["spearman_rho"])],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_ylabel("Patch count")
    ax.set_title("H1: Patch count per slide (sorted by LOO ρ, worst → best)")
    ax.legend()
    target_y = target_count + df["patch_count"].max() * 0.04
    ax.annotate(
        f"TARGET\nρ={df.iloc[target_idx]['spearman_rho']:.2f}",
        xy=(target_idx, target_count),
        xytext=(target_idx, target_y),
        ha="center", fontsize=7, color=_RED,
        arrowprops=dict(arrowstyle="-", color=_RED, lw=0.8),
    )
    plt.tight_layout()
    fig.savefig(str(output_dir / "fig1_patch_counts.png"), dpi=150)
    plt.close(fig)

    return (
        {"flag": flag, "flag_msg": flag_msg, "patch_count": target_count, "median": int(median_count)},
        df[["slide_name", "patch_count"]],
    )


# ─────────────────────────────────────────────────────────────
# H2: Cluster composition
# ─────────────────────────────────────────────────────────────

def investigate_cluster_composition(results_df, loo_df, target_slide, output_dir):
    comp = pd.crosstab(results_df["slide_name"], results_df["cluster"], normalize="index")
    clusters = comp.columns.tolist()

    # sort by rho for display
    comp = comp.join(loo_df.set_index("slide_name")["spearman_rho"])
    comp = comp.sort_values("spearman_rho")
    rho_series = comp["spearman_rho"].copy()
    comp = comp.drop(columns=["spearman_rho"])

    cohort_means = comp.mean()
    overrepresented = []
    if target_slide in comp.index:
        target_row = comp.loc[target_slide]
        for c in clusters:
            if cohort_means[c] > 0 and target_row[c] > 2 * cohort_means[c]:
                overrepresented.append((str(c), float(target_row[c]), float(cohort_means[c])))

    flag = len(overrepresented) > 0
    if flag:
        parts = [f"cluster {c} ({f:.1%} vs mean {m:.1%})" for c, f, m in overrepresented]
        flag_msg = f"FLAGGED: target overrepresented in {', '.join(parts)}"
    else:
        flag_msg = "OK: no cluster is >2x overrepresented in target vs cohort mean"

    n_slides = len(comp)
    cmap = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    fig, ax = plt.subplots(figsize=(12, 5))
    bottoms = np.zeros(n_slides)
    for col, color in zip(clusters, cmap):
        ax.bar(range(n_slides), comp[col].values, bottom=bottoms, color=color, label=f"Cluster {col}", width=0.8)
        bottoms += comp[col].values

    if target_slide in comp.index:
        tidx = comp.index.tolist().index(target_slide)
        ax.bar(tidx, 1.02, bottom=0, fill=False, edgecolor=_RED, linewidth=2.5, width=0.85)
        ax.text(tidx, 1.04, "TARGET", ha="center", va="bottom", fontsize=7, color=_RED)

    tick_labels = [
        f"{s.replace('_x5', '')}\n(ρ={rho:.2f})"
        for s, rho in zip(comp.index, rho_series.values)
    ]
    ax.set_xticks(range(n_slides))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Fraction of patches")
    ax.set_ylim(0, 1.12)
    ax.set_title("H2: Cluster composition per slide (sorted by LOO ρ, worst → best)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(str(output_dir / "fig2_cluster_composition.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return (
        {"flag": flag, "flag_msg": flag_msg, "overrepresented_clusters": overrepresented},
        comp,
    )


# ─────────────────────────────────────────────────────────────
# H3: Feature space distance
# ─────────────────────────────────────────────────────────────

def investigate_feature_distances(features_dict, loo_df, target_slide, output_dir):
    slide_names = sorted(features_dict.keys())
    if target_slide not in slide_names:
        msg = f"SKIPPED: {target_slide} not in feature cache"
        return {"flag": False, "flag_msg": msg}, None

    centroids = np.array([features_dict[s].mean(axis=0) for s in slide_names])
    dist_matrix = cdist(centroids, centroids, metric="euclidean")
    dist_df = pd.DataFrame(dist_matrix, index=slide_names, columns=slide_names)

    # per-slide mean distance to all others (exclude diagonal)
    mean_dists = [
        dist_matrix[i, [j for j in range(len(slide_names)) if j != i]].mean()
        for i in range(len(slide_names))
    ]
    target_idx = slide_names.index(target_slide)
    target_mean = mean_dists[target_idx]
    p75 = float(np.percentile(mean_dists, 75))
    flag = target_mean > p75
    flag_msg = (
        f"FLAGGED: target mean centroid distance {target_mean:.2f} > 75th percentile {p75:.2f}"
        if flag else
        f"OK: target mean centroid distance {target_mean:.2f} <= 75th percentile {p75:.2f}"
    )

    short = [s.replace("_x5", "") for s in slide_names]
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(9, 8))
        mask = np.eye(len(slide_names), dtype=bool)
        sns.heatmap(
            dist_matrix, mask=mask,
            xticklabels=short, yticklabels=short,
            cmap="Blues", ax=ax, linewidths=0.3,
            cbar_kws={"label": "L2 distance between slide centroids"},
        )
    except ImportError:
        fig, ax = plt.subplots(figsize=(9, 8))
        masked = np.where(np.eye(len(slide_names), dtype=bool), np.nan, dist_matrix)
        im = ax.imshow(masked, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, label="L2 distance")
        ax.set_xticks(range(len(slide_names)))
        ax.set_yticks(range(len(slide_names)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)

    # red border around target row/col
    ax.axhline(target_idx, color=_RED, linewidth=2)
    ax.axhline(target_idx + 1, color=_RED, linewidth=2)
    ax.axvline(target_idx, color=_RED, linewidth=2)
    ax.axvline(target_idx + 1, color=_RED, linewidth=2)
    ax.set_title(
        "H3: Pairwise L2 distance between per-slide Phikon centroids\n"
        f"(red border = target; target mean dist = {target_mean:.2f}, 75th pct = {p75:.2f})"
    )
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(str(output_dir / "fig3_feature_distances.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return (
        {"flag": flag, "flag_msg": flag_msg, "target_mean_dist": float(target_mean), "p75": p75},
        dist_df,
    )


# ─────────────────────────────────────────────────────────────
# H4: UMAP position
# ─────────────────────────────────────────────────────────────

def investigate_umap_position(umap_coords, results_df, target_slide, output_dir):
    slide_names_vec = results_df["slide_name"].values
    target_mask = slide_names_vec == target_slide

    if target_mask.sum() == 0:
        msg = f"SKIPPED: {target_slide} not found in results_df"
        return {"flag": False, "flag_msg": msg}, None

    other_coords = umap_coords[~target_mask]
    target_coords = umap_coords[target_mask]

    # Isolation fraction: fraction of target patches in grid cells with zero other-slide patches
    grid_size = 50
    x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
    y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)

    other_hist, _, _ = np.histogram2d(other_coords[:, 0], other_coords[:, 1], bins=[x_bins, y_bins])
    target_xi = np.clip(np.digitize(target_coords[:, 0], x_bins) - 1, 0, grid_size - 1)
    target_yi = np.clip(np.digitize(target_coords[:, 1], y_bins) - 1, 0, grid_size - 1)
    isolation_frac = float(np.mean(other_hist[target_xi, target_yi] == 0))

    flag = isolation_frac > 0.2
    flag_msg = (
        f"FLAGGED: {isolation_frac:.1%} of target patches are in UMAP cells with zero other-slide patches (>20% threshold)"
        if flag else
        f"OK: only {isolation_frac:.1%} of target patches are in UMAP cells with zero other-slide patches"
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        other_coords[:, 0], other_coords[:, 1],
        c=_GRAY, s=1, alpha=0.12, rasterized=True, label="Other slides",
    )
    ax.scatter(
        target_coords[:, 0], target_coords[:, 1],
        c=_RED, s=3, alpha=0.6, label=f"{target_slide.replace('_x5', '')} (target)",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        f"H4: UMAP position of target slide\n"
        f"Isolation fraction: {isolation_frac:.1%} of patches in empty (other-slide) grid cells"
    )
    ax.legend(markerscale=4, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(str(output_dir / "fig4_umap_highlight.png"), dpi=150)
    plt.close(fig)

    return {"flag": flag, "flag_msg": flag_msg, "isolation_frac": isolation_frac}, None


# ─────────────────────────────────────────────────────────────
# H5: Annotation pattern
# ─────────────────────────────────────────────────────────────

def investigate_annotations(ann_dict, loo_df, target_slide, output_dir):
    slides = loo_df["slide_name"].tolist()
    rows = []
    for s in slides:
        info = ann_dict.get(s, {"n_polygons": 0, "total_area_ratio": None})
        rows.append({
            "slide_name": s,
            "n_polygons": info["n_polygons"],
            "total_area": info.get("total_area_ratio"),
        })
    df = pd.DataFrame(rows).merge(loo_df[["slide_name", "spearman_rho"]], on="slide_name")
    df = df.sort_values("spearman_rho").reset_index(drop=True)

    flag = False
    flag_parts = []
    for col in ["n_polygons", "total_area"]:
        valid = df[col].dropna()
        if len(valid) < 3:
            continue
        m, s = valid.mean(), valid.std()
        if s == 0:
            continue
        tv_series = df.loc[df["slide_name"] == target_slide, col]
        if tv_series.empty or pd.isna(tv_series.iloc[0]):
            continue
        tv = float(tv_series.iloc[0])
        if abs(tv - m) > 2 * s:
            flag = True
            flag_parts.append(f"{col}={tv:.4f} (mean={m:.4f}, σ={s:.4f})")

    flag_msg = (
        f"FLAGGED: target is >=2σ outlier on {', '.join(flag_parts)}"
        if flag else
        "OK: target annotation metrics within ±2σ of cohort"
    )

    colors = [_RED if s == target_slide else _BLUE for s in df["slide_name"]]
    short = [s.replace("_x5", "") for s in df["slide_name"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, ylabel in zip(
        axes,
        ["n_polygons", "total_area"],
        ["Number of annotation polygons", "Total annotated area (ratio units²)"],
    ):
        vals = df[col].fillna(0).values
        ax.bar(range(len(df)), vals, color=colors, width=0.8)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(
            [f"{n}\n(ρ={r:.2f})" for n, r in zip(short, df["spearman_rho"])],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylabel(ylabel)
    fig.suptitle("H5: Annotation pattern (sorted by LOO ρ, worst → best)", y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_dir / "fig5_annotation_pattern.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"flag": flag, "flag_msg": flag_msg}, df[["slide_name", "n_polygons", "total_area"]]


# ─────────────────────────────────────────────────────────────
# Diagnostic table
# ─────────────────────────────────────────────────────────────

def build_diagnostic_table(loo_df, patch_count_df, cluster_comp_df, dist_df, ann_df):
    tbl = loo_df[["slide_name", "spearman_rho", "n_patches"]].copy()

    if patch_count_df is not None:
        tbl = tbl.merge(patch_count_df.rename(columns={"patch_count": "patch_count_results"}), on="slide_name", how="left")

    if cluster_comp_df is not None:
        comp_reset = cluster_comp_df.reset_index()
        comp_reset.columns = (
            ["slide_name"] + [f"cluster_{c}_frac" for c in comp_reset.columns[1:]]
        )
        tbl = tbl.merge(comp_reset, on="slide_name", how="left")

    if dist_df is not None:
        mean_dists = []
        for s in dist_df.index:
            other = dist_df.loc[s, dist_df.columns != s]
            mean_dists.append({"slide_name": s, "centroid_dist_mean": float(other.mean())})
        tbl = tbl.merge(pd.DataFrame(mean_dists), on="slide_name", how="left")

    if ann_df is not None:
        tbl = tbl.merge(ann_df, on="slide_name", how="left")

    return tbl


# ─────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────

def generate_report(output_dir, target_slide, loo_df, metrics):
    target_rho_series = loo_df.loc[loo_df["slide_name"] == target_slide, "spearman_rho"]
    target_rho = float(target_rho_series.iloc[0]) if len(target_rho_series) > 0 else float("nan")
    others = loo_df.loc[loo_df["slide_name"] != target_slide, "spearman_rho"]
    cohort_mean = float(others.mean())
    cohort_min = float(others.min())
    cohort_max = float(others.max())

    # LOO table
    loo_sorted = loo_df.sort_values("spearman_rho").reset_index(drop=True)
    has_w = "wasserstein" in loo_sorted.columns
    header = "| Slide | ρ | N patches |" + (" Wasserstein |" if has_w else "")
    sep    = "|---|---|---|" + ("---|" if has_w else "")
    table_lines = [header, sep]
    for _, row in loo_sorted.iterrows():
        name = row["slide_name"].replace("_x5", "")
        rho = f"{row['spearman_rho']:.3f}"
        n = f"{int(row['n_patches']):,}"
        w = f"{row['wasserstein']:.3f}" if has_w else ""
        cells = f"| {name} | {rho} | {n} |" + (f" {w} |" if has_w else "")
        if row["slide_name"] == target_slide:
            cells = cells.replace(f"| {name}", f"| **{name}**").replace(f"| {rho}", f"| **{rho}**")
        table_lines.append(cells)
    loo_table = "\n".join(table_lines)

    hyp_labels = {
        "h1": "H1: Patch Count",
        "h2": "H2: Cluster Composition",
        "h3": "H3: Feature Space Distance",
        "h4": "H4: UMAP Position",
        "h5": "H5: Annotation Pattern",
    }
    fig_names = {
        "h1": "fig1_patch_counts.png",
        "h2": "fig2_cluster_composition.png",
        "h3": "fig3_feature_distances.png",
        "h4": "fig4_umap_highlight.png",
        "h5": "fig5_annotation_pattern.png",
    }
    h_sections = []
    for key in ["h1", "h2", "h3", "h4", "h5"]:
        if key not in metrics:
            continue
        m = metrics[key]
        h_sections.append(
            f"### {hyp_labels[key]}\n\n"
            f"![{hyp_labels[key]}]({fig_names[key]})\n\n"
            f"**Result:** {m['flag_msg']}\n"
        )

    supported = [hyp_labels[k] for k in ["h1", "h2", "h3", "h4", "h5"] if k in metrics and metrics[k].get("flag")]
    delta = cohort_mean - target_rho

    if not supported:
        synthesis = (
            f"None of the five hypotheses were flagged for the target slide. "
            f"The poor projection (ρ={target_rho:.3f}, delta={delta:.3f} below cohort mean {cohort_mean:.3f}) "
            f"may reflect stochastic variability in the LOO estimate rather than a structural outlier property."
        )
    elif len(supported) == 1:
        synthesis = (
            f"The evidence most strongly supports **{supported[0]}** as the primary explanation "
            f"for the target slide's poor projection (ρ={target_rho:.3f}, {delta:.3f} below cohort mean {cohort_mean:.3f}; "
            f"cohort range {cohort_min:.3f}–{cohort_max:.3f}). No secondary factors were flagged."
        )
    else:
        synthesis = (
            f"Multiple factors were flagged: **{', '.join(supported)}**. "
            f"The target slide's ρ={target_rho:.3f} is {delta:.3f} below the cohort mean of {cohort_mean:.3f} "
            f"(cohort range {cohort_min:.3f}–{cohort_max:.3f}). "
            f"Among the flagged hypotheses, H2 (cluster composition) and H4 (UMAP position) are most "
            f"directly mechanistically linked to projection quality: if the target slide's patches occupy "
            f"a different region of the manifold, the KNN regressor will extrapolate rather than interpolate, "
            f"degrading projection accuracy."
        )

    sections_text = "\n\n".join(h_sections)
    report = (
        f"# Diagnostic Report: {target_slide.replace('_x5', '')} LOO Projection\n\n"
        f"**Target slide:** `{target_slide}`  \n"
        f"**LOO Spearman ρ:** {target_rho:.3f}  \n"
        f"**Cohort mean ρ (excl. target):** {cohort_mean:.3f}  \n"
        f"**Cohort range:** {cohort_min:.3f} – {cohort_max:.3f}  \n\n"
        f"---\n\n"
        f"## LOO Projection Results (all slides)\n\n"
        f"{loo_table}\n\n"
        f"---\n\n"
        f"## Five-Hypothesis Investigation\n\n"
        f"{sections_text}\n\n"
        f"---\n\n"
        f"## Synthesis\n\n"
        f"{synthesis}\n\n"
        f"---\n\n"
        f"*Generated by `cancer_trajectory_atlas.analysis.slide_diagnostics`*\n"
    )

    report_path = output_dir / "slide_diagnostics_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report written: {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic report for an outlier slide in the LOO projection experiment."
    )
    parser.add_argument("--adata-path", type=Path, required=True,
                        help="Path to adata_full.h5ad from the pooled atlas run")
    parser.add_argument("--results-csv", type=Path, required=True,
                        help="Path to results.csv from the same pooled atlas run")
    parser.add_argument("--loo-dir", type=Path, required=True,
                        help="Directory containing loo_summary.csv or loo_result_*.json files")
    parser.add_argument("--features-cache-dir", type=Path, required=True,
                        help="Directory containing {slide}_features.npy files")
    parser.add_argument("--annotation-dir", type=Path, default=None,
                        help="Directory containing annotation JSON/GeoJSON files (optional; skips H5 if absent)")
    parser.add_argument("--target-slide", type=str, default="6028-4L-2M-2_x5",
                        help="Slide stem to diagnose (default: 6028-4L-2M-2_x5)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write figures, report, and diagnostic_table.csv")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    results_df = load_results(args.results_csv)
    loo_df = load_loo_summary(args.loo_dir)
    slide_names = loo_df["slide_name"].tolist()
    print(f"  {len(results_df):,} patches, {len(slide_names)} slides in LOO summary")

    if args.target_slide not in slide_names:
        raise ValueError(
            f"Target slide '{args.target_slide}' not in LOO summary. "
            f"Available: {slide_names}"
        )

    print("Loading UMAP embedding (adata_full.h5ad)...")
    umap_coords = load_umap(args.adata_path)
    if len(umap_coords) != len(results_df):
        raise ValueError(
            f"UMAP row count ({len(umap_coords)}) != results.csv row count ({len(results_df)}). "
            "Ensure adata_full.h5ad and results.csv are from the same pipeline run."
        )

    print("Loading Phikon feature cache...")
    features_dict = load_slide_features(args.features_cache_dir, slide_names)
    print(f"  Loaded features for {len(features_dict)}/{len(slide_names)} slides")

    ann_dict = None
    if args.annotation_dir is not None:
        print("Loading annotations...")
        ann_dict = load_annotations(args.annotation_dir, slide_names)
    else:
        print("  --annotation-dir not provided; H5 will be skipped")

    metrics = {}
    patch_count_df = cluster_comp_df = dist_df = ann_df = None

    print("\nH1: Patch counts...")
    metrics["h1"], patch_count_df = investigate_patch_counts(
        results_df, loo_df, args.target_slide, args.output_dir
    )
    print(f"  {metrics['h1']['flag_msg']}")

    print("H2: Cluster composition...")
    metrics["h2"], cluster_comp_df = investigate_cluster_composition(
        results_df, loo_df, args.target_slide, args.output_dir
    )
    print(f"  {metrics['h2']['flag_msg']}")

    print("H3: Feature space distances...")
    metrics["h3"], dist_df = investigate_feature_distances(
        features_dict, loo_df, args.target_slide, args.output_dir
    )
    print(f"  {metrics['h3']['flag_msg']}")

    print("H4: UMAP position...")
    metrics["h4"], _ = investigate_umap_position(
        umap_coords, results_df, args.target_slide, args.output_dir
    )
    print(f"  {metrics['h4']['flag_msg']}")

    if ann_dict is not None:
        print("H5: Annotation pattern...")
        metrics["h5"], ann_df = investigate_annotations(
            ann_dict, loo_df, args.target_slide, args.output_dir
        )
        print(f"  {metrics['h5']['flag_msg']}")

    print("\nBuilding diagnostic table...")
    diag_tbl = build_diagnostic_table(loo_df, patch_count_df, cluster_comp_df, dist_df, ann_df)
    tbl_path = args.output_dir / "diagnostic_table.csv"
    diag_tbl.to_csv(tbl_path, index=False)
    print(f"  {tbl_path}")

    print("Writing report...")
    generate_report(args.output_dir, args.target_slide, loo_df, metrics)

    print(f"\nDone. Output in: {args.output_dir}")


if __name__ == "__main__":
    main()
