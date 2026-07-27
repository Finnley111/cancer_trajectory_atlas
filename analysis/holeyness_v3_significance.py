"""
Holeyness v3: significance test on the area-adjusted partial correlation.

v2 (`analysis/holeyness.py --v2`) found that rho(pseudotime, hole_pct) = 0.276 is
substantially confounded by duct area: the area-adjusted partial is 0.131 (area +
nuclear_density: 0.158), but v2 never ran a permutation test *on the partial* — its
permutation tests (global and within-slide) were both computed on the raw
correlation. v2's aggregation-sensitivity sweep (mean vs median, n_patches
thresholds) was likewise computed on the raw correlation only. This module closes
both gaps and investigates three slides with a near-zero/negative area-adjusted
partial (6028-4R-2M-1_x5, 6029-4L-2M-1_x5, 6031-4L-2M-1_x5).

This is a NEW module, not another flag on holeyness.py — v1/v2's already-validated
code path stays completely frozen; this module only *imports* from it.

Primary data source: v1's holeyness_per_duct.csv (median-aggregated, already
cross-checked against v2's own consistency check) plus v2's holeyness_validation_v2.json
(reused for reference numbers and per-slide raw-correlation figures, not recomputed).
The one exception is the mean-aggregation half of the patch-count sensitivity sweep:
the mean-aggregated per-duct table was never persisted, so it is re-derived here via
holeyness.py's existing, unmodified loader functions against the same raw inputs v1/v2
used (parse_measurement_export, load_duct_polygons, build_duct_table,
assign_patches_to_ducts, aggregate_per_duct) — CPU-only, deterministic, not an
atlas-pipeline rerun.

Writes to a NEW versioned directory; never touches v1 or v2 outputs.

CLI
---
  python -m cancer_trajectory_atlas.analysis.holeyness_v3_significance \\
      --section          2M-1 \\
      --v1-per-duct-csv  $SCRATCH/results/holeyness/2M-1/holeyness_per_duct.csv \\
      --v2-json          $SCRATCH/results/holeyness/2M-1/v2_area_adjusted/holeyness_validation_v2.json \\
      --export           $SCRATCH/data/holeyness/raw/combined_matched_measurements.txt \\
      --annotation-dir   ~/cancer_trajectory_atlas/data/annotations_ratio \\
      --slide-dimensions $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json \\
      --results          $SCRATCH/results/per_section/atlas_2M-1/results.csv \\
      --slide-list       ~/cancer_trajectory_atlas/jobs/slides_section1.txt \\
      --output-dir       $SCRATCH/results/holeyness/2M-1/v3_significance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .holeyness import (
    PATCH_SIZE_DEFAULT,
    REQUIRED_RESULTS_COLS,
    N_PATCH_THRESHOLDS,
    load_slide_list,
    load_slide_dimensions,
    parse_measurement_export,
    load_duct_polygons,
    build_duct_table,
    assign_patches_to_ducts,
    aggregate_per_duct,
    _safe_spearman,
    _partial_spearman,
    _partial_spearman_multi,
    _format_perm_p,
)

FLAGGED_SLIDES = ["6028-4R-2M-1_x5", "6029-4L-2M-1_x5", "6031-4L-2M-1_x5"]


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else str(v)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _partial_rho(pt: np.ndarray, hole: np.ndarray, controls: list) -> float:
    if len(controls) == 1:
        return _partial_spearman(pt, hole, controls[0])
    return _partial_spearman_multi(pt, hole, controls)


# ── Consistency checks ────────────────────────────────────────────────────────

def check_consistency_with_v2(per_duct: pd.DataFrame, v2_json: dict, tol: float = 1e-4) -> dict:
    """Recompute raw + area-adjusted partial from v1's CSV and compare against
    v2's saved reference numbers. Gates whether the rest of this report should
    be trusted."""
    pt   = per_duct["pseudotime"].values
    hole = per_duct["hole_pct"].values
    area = per_duct["area_um2"].values
    nd   = per_duct["nuclear_density"].values

    rho_raw, _ = _safe_spearman(pt, hole)
    partial_area    = _partial_spearman(pt, hole, area)
    partial_area_nd = _partial_spearman_multi(pt, hole, [area, nd])

    ref_raw          = v2_json["primary_correlation"]["rho_pt_hole_pct"]
    ref_partial_area = v2_json["area_covariate"]["partial_rho_pt_hole_given_area"]
    ref_partial_nd   = v2_json["area_covariate"]["partial_rho_pt_hole_given_area_and_nd"]

    diffs = {
        "raw": abs(rho_raw - ref_raw),
        "partial_given_area": abs(partial_area - ref_partial_area),
        "partial_given_area_and_nd": abs(partial_area_nd - ref_partial_nd),
    }
    return {
        "n_ducts": int(len(per_duct)),
        "v3_recomputed": {
            "rho_pt_hole_pct": rho_raw,
            "partial_rho_pt_hole_given_area": partial_area,
            "partial_rho_pt_hole_given_area_and_nd": partial_area_nd,
        },
        "v2_reference": {
            "rho_pt_hole_pct": ref_raw,
            "partial_rho_pt_hole_given_area": ref_partial_area,
            "partial_rho_pt_hole_given_area_and_nd": ref_partial_nd,
        },
        "max_abs_diff": max(diffs.values()),
        "all_match": bool(max(diffs.values()) < tol),
    }


def check_rederivation_consistency(per_duct_median_rederived: pd.DataFrame, per_duct_v1: pd.DataFrame) -> dict:
    """Sanity check: does re-deriving the median-aggregated table from raw
    inputs (the same code path used to get the mean-aggregated table) agree
    with v1's saved CSV? Guards the mean-aggregation numbers below."""
    rho_rederived, _ = _safe_spearman(
        per_duct_median_rederived["pseudotime"].values, per_duct_median_rederived["hole_pct"].values
    )
    rho_v1, _ = _safe_spearman(per_duct_v1["pseudotime"].values, per_duct_v1["hole_pct"].values)
    diff = abs(rho_rederived - rho_v1)
    same_n = len(per_duct_median_rederived) == len(per_duct_v1)
    return {
        "n_ducts_rederived": int(len(per_duct_median_rederived)),
        "n_ducts_v1_csv": int(len(per_duct_v1)),
        "rho_rederived_median_agg": rho_rederived,
        "rho_v1_csv": rho_v1,
        "max_abs_diff": diff,
        "all_match": bool(same_n and diff < 1e-3),
    }


# ── GAP 1: permutation test on the area-adjusted partial ─────────────────────

def run_partial_permutation_global(
    per_duct: pd.DataFrame, controls: list, n_permutations: int, rng: np.random.Generator
) -> dict:
    pt   = per_duct["pseudotime"].values
    hole = per_duct["hole_pct"].values
    obs = _partial_rho(pt, hole, controls)

    null = np.empty(n_permutations)
    for i in range(n_permutations):
        pt_perm = rng.permutation(pt)
        null[i] = abs(_partial_rho(pt_perm, hole, controls))

    perm_p = float(np.mean(null >= abs(obs)))
    null95 = float(np.percentile(null, 95))
    return {
        "observed_partial_rho": obs,
        "perm_p": perm_p,
        "perm_p_display": _format_perm_p(perm_p, n_permutations),
        "null95": null95,
        "exceeds_null95": bool(abs(obs) > null95),
        "n_permutations": n_permutations,
    }


def run_partial_permutation_within_slide(
    per_duct: pd.DataFrame, controls: list, n_permutations: int, rng: np.random.Generator
) -> dict:
    pt    = per_duct["pseudotime"].values
    hole  = per_duct["hole_pct"].values
    slide = per_duct["slide_name"].values
    obs = _partial_rho(pt, hole, controls)

    slide_idx = [np.where(slide == s)[0] for s in np.unique(slide)]
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        pt_perm = pt.copy()
        for idxs in slide_idx:
            pt_perm[idxs] = rng.permutation(pt_perm[idxs])
        null[i] = abs(_partial_rho(pt_perm, hole, controls))

    perm_p = float(np.mean(null >= abs(obs)))
    null95 = float(np.percentile(null, 95))
    return {
        "observed_partial_rho": obs,
        "perm_p": perm_p,
        "perm_p_display": _format_perm_p(perm_p, n_permutations),
        "null95": null95,
        "exceeds_null95": bool(abs(obs) > null95),
        "n_permutations": n_permutations,
    }


# ── GAP 2: aggregation & patch-count sensitivity on the adjusted partial ─────

def run_partial_aggregation_sensitivity(per_duct: pd.DataFrame) -> dict:
    """partial_rho_pt_hole_given_area at all-ducts and at each n_patches
    threshold, for whichever per_duct table (median- or mean-aggregated) is
    passed in."""
    pt_all, hole_all, area_all = (
        per_duct["pseudotime"].values, per_duct["hole_pct"].values, per_duct["area_um2"].values
    )
    result = {
        "all_ducts": {
            "partial_rho_pt_hole_given_area": _partial_spearman(pt_all, hole_all, area_all),
            "n_ducts": int(len(per_duct)),
        }
    }
    for thresh in N_PATCH_THRESHOLDS:
        subset = per_duct[per_duct["n_patches"] >= thresh]
        pt, hole, area = subset["pseudotime"].values, subset["hole_pct"].values, subset["area_um2"].values
        result[f"min_{thresh}_patches"] = {
            "partial_rho_pt_hole_given_area": _partial_spearman(pt, hole, area),
            "n_ducts": int(len(subset)),
        }
    return result


def rederive_median_and_mean_tables(args: argparse.Namespace):
    """Re-derive duct_table + patch assignment from the same raw inputs v1/v2
    used, via holeyness.py's existing unmodified loaders, to obtain the
    mean-aggregated per-duct table (never persisted by v1/v2). Returns
    (per_duct_median, per_duct_mean)."""
    print("\n=== Re-deriving duct table from raw inputs (for mean aggregation) ===")
    pipeline_slides = load_slide_list(args.slide_list)
    slide_dims = load_slide_dimensions(args.slide_dimensions)
    measurements = parse_measurement_export(args.export, pipeline_slides)
    polygons = load_duct_polygons(args.annotation_dir, pipeline_slides, slide_dims)
    duct_table = build_duct_table(measurements, polygons)
    if len(duct_table) == 0:
        sys.exit("ERROR: no ducts remain after UUID join during v3 raw-input re-derivation.")

    results_df = pd.read_csv(args.results, low_memory=False)
    missing_cols = [c for c in REQUIRED_RESULTS_COLS if c not in results_df.columns]
    if missing_cols:
        sys.exit(f"ERROR: results.csv missing columns: {missing_cols}")
    results_df = results_df[REQUIRED_RESULTS_COLS].drop_duplicates()
    results_df = results_df[results_df["slide_name"].isin(pipeline_slides)].copy()

    results_df = assign_patches_to_ducts(results_df, duct_table, args.patch_size)

    per_duct_median = aggregate_per_duct(results_df, duct_table, np.nanmedian, "median")
    per_duct_mean    = aggregate_per_duct(results_df, duct_table, np.nanmean,   "mean")
    return per_duct_median, per_duct_mean


# ── Per-slide investigation ───────────────────────────────────────────────────

def run_per_slide_investigation(per_duct: pd.DataFrame, v2_json: dict) -> dict:
    """Descriptive per-slide summary: is there something structurally
    different (duct size, patch coverage) about the flagged low-signal
    slides? Raw-correlation figures are pulled from v2's JSON, not
    recomputed."""
    v2_per_slide = {row["slide_name"]: row for row in v2_json["within_slide"]["per_slide"]}

    rows = []
    for slide_name, grp in per_duct.groupby("slide_name"):
        v2row = v2_per_slide.get(slide_name, {})
        rows.append({
            "slide_name": str(slide_name),
            "n_ducts": int(len(grp)),
            "median_area_um2": float(np.nanmedian(grp["area_um2"].values)),
            "median_n_patches": float(np.nanmedian(grp["n_patches"].values)),
            "frac_single_patch_ducts": float(np.mean(grp["n_patches"].values == 1)),
            "raw_rho_from_v2": v2row.get("rho_pt_hole_pct", float("nan")),
            "raw_p_from_v2": v2row.get("p_pt_hole_pct", float("nan")),
            "partial_rho_from_v2": v2row.get("partial_rho_pt_hole_given_area", float("nan")),
            "is_flagged": str(slide_name) in FLAGGED_SLIDES,
        })

    rows.sort(key=lambda r: r["partial_rho_from_v2"] if np.isfinite(r["partial_rho_from_v2"]) else 0.0)

    flagged = [r for r in rows if r["is_flagged"]]
    other = [r for r in rows if not r["is_flagged"]]

    def _avg(key, group):
        vals = [r[key] for r in group if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "flagged_mean_median_area_um2": _avg("median_area_um2", flagged),
        "other_mean_median_area_um2": _avg("median_area_um2", other),
        "flagged_mean_frac_single_patch": _avg("frac_single_patch_ducts", flagged),
        "other_mean_frac_single_patch": _avg("frac_single_patch_ducts", other),
    }

    return {"flagged_slides": FLAGGED_SLIDES, "per_slide": rows, "flagged_vs_other_summary": summary}


# ── Figure ────────────────────────────────────────────────────────────────────

def write_per_slide_partial_bar_chart(per_slide_investigation: dict, output_dir: Path, section: str) -> None:
    rows = [r for r in per_slide_investigation["per_slide"] if np.isfinite(r["partial_rho_from_v2"])]
    if not rows:
        print("  WARNING: no slides with a v2 partial rho — skipping bar chart")
        return
    rows = sorted(rows, key=lambda r: r["partial_rho_from_v2"])
    names = [r["slide_name"] for r in rows]
    vals = [r["partial_rho_from_v2"] for r in rows]
    colors = ["#D65F5F" if r["is_flagged"] else "#4878CF" for r in rows]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.5))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Area-adjusted partial rho(pseudotime, hole_pct)  [from v2]")
    ax.set_title(f"Section {section}: per-slide area-adjusted partial correlation\n"
                 f"(red = flagged low-signal slides; annotated: median duct area, %% single-patch ducts)")

    xmin, xmax = ax.get_xlim()
    span = xmax - xmin if xmax > xmin else 1.0
    for yi, r in zip(y, rows):
        label = f"area={r['median_area_um2']:.0f}µm², 1-patch={100 * r['frac_single_patch_ducts']:.0f}%"
        x = r["partial_rho_from_v2"]
        offset = 0.02 * span if x >= 0 else -0.02 * span
        ha = "left" if x >= 0 else "right"
        ax.text(x + offset, yi, label, va="center", ha=ha, fontsize=7)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"v3_per_slide_partial_bar.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── Output writers ────────────────────────────────────────────────────────────

def write_v3_report(
    output_dir: Path,
    section: str,
    consistency: dict,
    mean_table_consistency: dict,
    perm_area: dict,
    perm_area_nd: dict,
    agg_sens_median: dict,
    agg_sens_mean: dict,
    per_slide_investigation: dict,
) -> None:
    lines = [f"# Holeyness v3 — significance test on the area-adjusted partial — section {section}", ""]

    lines += ["## Consistency check (v3 recompute vs v2 saved output)", ""]
    lines.append(f"- n_ducts compared: {consistency['n_ducts']}")
    lines.append(f"- max abs diff: {_fmt(consistency['max_abs_diff'])}")
    verdict_c1 = "MATCH — v3 recompute agrees with v2." if consistency["all_match"] else \
        "MISMATCH — investigate before trusting the numbers below."
    lines.append(f"- **Verdict:** {verdict_c1}")
    lines.append("")

    lines += ["## Consistency check (mean-aggregation re-derivation vs v1 CSV)", ""]
    lines.append(
        f"- rho (rederived, median agg) = {_fmt(mean_table_consistency['rho_rederived_median_agg'])}, "
        f"rho (v1 CSV) = {_fmt(mean_table_consistency['rho_v1_csv'])}"
    )
    verdict_c2 = "MATCH — raw-input re-derivation path agrees with v1's saved table." \
        if mean_table_consistency["all_match"] else \
        "MISMATCH — the mean-aggregation numbers below should not be trusted until this is resolved."
    lines.append(f"- **Verdict:** {verdict_c2}")
    lines.append("")

    lines += ["## GAP 1: permutation test on the area-adjusted partial", ""]
    lines += ["### Controlling for area_um2", ""]
    lines.append(f"- observed partial rho = {_fmt(perm_area['global']['observed_partial_rho'])}")
    lines.append(f"- global shuffle: perm_p {perm_area['global']['perm_p_display']}, "
                 f"null95 = {_fmt(perm_area['global']['null95'])}")
    lines.append(f"- within-slide shuffle: perm_p {perm_area['within_slide']['perm_p_display']}, "
                 f"null95 = {_fmt(perm_area['within_slide']['null95'])}")
    lines.append("")
    lines += ["### Controlling for area_um2 + nuclear_density", ""]
    lines.append(f"- observed partial rho = {_fmt(perm_area_nd['global']['observed_partial_rho'])}")
    lines.append(f"- global shuffle: perm_p {perm_area_nd['global']['perm_p_display']}, "
                 f"null95 = {_fmt(perm_area_nd['global']['null95'])}")
    lines.append(f"- within-slide shuffle: perm_p {perm_area_nd['within_slide']['perm_p_display']}, "
                 f"null95 = {_fmt(perm_area_nd['within_slide']['null95'])}")
    sig_within = perm_area["within_slide"]["perm_p"] < 0.05
    verdict_gap1 = (
        f"SIGNIFICANT — the area-adjusted partial rho survives the (structurally correct) "
        f"within-slide permutation null, perm_p {perm_area['within_slide']['perm_p_display']}."
        if sig_within else
        f"NOT SIGNIFICANT — the area-adjusted partial rho does not survive the within-slide "
        f"permutation null, perm_p {perm_area['within_slide']['perm_p_display']}."
    )
    lines.append("")
    lines.append(f"- **Verdict:** {verdict_gap1}")
    lines.append("")

    lines += ["## GAP 2: aggregation & patch-count sensitivity on the adjusted partial", ""]
    lines += ["### Median aggregation", ""]
    for k, v in agg_sens_median.items():
        lines.append(f"- {k}: partial rho = {_fmt(v['partial_rho_pt_hole_given_area'])} (n={v['n_ducts']})")
    lines.append("")
    lines += ["### Mean aggregation", ""]
    for k, v in agg_sens_mean.items():
        lines.append(f"- {k}: partial rho = {_fmt(v['partial_rho_pt_hole_given_area'])} (n={v['n_ducts']})")
    lines.append("")
    lines.append(
        "- **Verdict:** compare the rows above across both tables — a large swing under stricter "
        "n_patches thresholds indicates the adjusted partial depends on poorly-sampled ducts; stable "
        "values across thresholds and both aggregation methods indicate the signal is not an artifact "
        "of poorly-sampled ducts."
    )
    lines.append("")

    lines += ["## Per-slide investigation", ""]
    lines.append(f"Flagged slides: {', '.join(per_slide_investigation['flagged_slides'])}")
    lines.append("")
    lines.append("| slide | n_ducts | median_area_um2 | median_n_patches | frac_1_patch | raw_rho (v2) | raw_p (v2) | partial_rho (v2) | flagged |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in per_slide_investigation["per_slide"]:
        lines.append(
            f"| {r['slide_name']} | {r['n_ducts']} | {_fmt(r['median_area_um2'])} | "
            f"{_fmt(r['median_n_patches'])} | {_fmt(r['frac_single_patch_ducts'])} | "
            f"{_fmt(r['raw_rho_from_v2'])} | {_fmt(r['raw_p_from_v2'])} | "
            f"{_fmt(r['partial_rho_from_v2'])} | {'yes' if r['is_flagged'] else ''} |"
        )
    lines.append("")
    s = per_slide_investigation["flagged_vs_other_summary"]
    lines.append(
        f"- Flagged slides: mean of median duct area = {_fmt(s['flagged_mean_median_area_um2'])} µm², "
        f"mean fraction single-patch ducts = {_fmt(s['flagged_mean_frac_single_patch'])}"
    )
    lines.append(
        f"- Other slides: mean of median duct area = {_fmt(s['other_mean_median_area_um2'])} µm², "
        f"mean fraction single-patch ducts = {_fmt(s['other_mean_frac_single_patch'])}"
    )
    area_smaller = (
        np.isfinite(s["flagged_mean_median_area_um2"]) and np.isfinite(s["other_mean_median_area_um2"])
        and s["flagged_mean_median_area_um2"] < 0.8 * s["other_mean_median_area_um2"]
    )
    coverage_worse = (
        np.isfinite(s["flagged_mean_frac_single_patch"]) and np.isfinite(s["other_mean_frac_single_patch"])
        and s["flagged_mean_frac_single_patch"] > 1.2 * s["other_mean_frac_single_patch"]
    )
    if area_smaller or coverage_worse:
        reasons = []
        if area_smaller:
            reasons.append("smaller median duct area")
        if coverage_worse:
            reasons.append("a higher fraction of single-patch ducts")
        verdict_slides = (
            f"The flagged slides show {' and '.join(reasons)} relative to the others — consistent "
            f"with a structural (sampling/duct-size) explanation for their weaker area-adjusted signal, "
            f"though this is descriptive, not a formal test."
        )
    else:
        verdict_slides = (
            "No clear structural difference in duct area or patch coverage between the flagged and "
            "other slides — the weaker area-adjusted signal in these three slides may reflect "
            "slide-to-slide noise rather than a systematic sampling artifact."
        )
    lines.append(f"- **Verdict:** {verdict_slides}")
    lines.append("")

    (output_dir / "holeyness_validation_v3.md").write_text("\n".join(lines), encoding="utf-8")


def write_v3_outputs(
    output_dir: Path,
    section: str,
    n_permutations: int,
    consistency: dict,
    reference_from_v2: dict,
    mean_table_consistency: dict,
    perm_area: dict,
    perm_area_nd: dict,
    agg_sens_median: dict,
    agg_sens_mean: dict,
    per_slide_investigation: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "section": section,
        "n_permutations": n_permutations,
        "consistency_check": consistency,
        "reference_from_v2": reference_from_v2,
        "partial_permutation": {
            "given_area": perm_area,
            "given_area_and_nd": perm_area_nd,
        },
        "aggregation_sensitivity_partial": {
            "median_aggregation": agg_sens_median,
            "mean_aggregation": agg_sens_mean,
            "mean_table_consistency_check": mean_table_consistency,
        },
        "per_slide_investigation": per_slide_investigation,
    }
    json_path = output_dir / "holeyness_validation_v3.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"  JSON: {json_path}")

    write_v3_report(
        output_dir, section, consistency, mean_table_consistency,
        perm_area, perm_area_nd, agg_sens_median, agg_sens_mean, per_slide_investigation,
    )
    print(f"  Markdown report: {output_dir / 'holeyness_validation_v3.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Holeyness v3: significance test on the area-adjusted partial correlation"
    )
    parser.add_argument("--section",          required=True, help="Section label, e.g. '2M-1'")
    parser.add_argument("--v1-per-duct-csv",  required=True, type=Path,
                        help="Path to v1's holeyness_per_duct.csv")
    parser.add_argument("--v2-json",          required=True, type=Path,
                        help="Path to v2's holeyness_validation_v2.json")
    parser.add_argument("--export",           required=True, type=Path,
                        help="Path to combined_matched_measurements.txt (for mean-aggregation re-derivation)")
    parser.add_argument("--annotation-dir",   required=True, type=Path,
                        help="Directory containing <slide>.json ratio annotation files")
    parser.add_argument("--slide-dimensions", required=True, type=Path,
                        help="Path to slide_dimensions.json")
    parser.add_argument("--results",          required=True, type=Path,
                        help="Path to this section's per-patch results.csv")
    parser.add_argument("--slide-list",       required=True, type=Path,
                        help="Text file with one pipeline slide_name per line")
    parser.add_argument("--output-dir",       required=True, type=Path,
                        help="Output directory (NEW versioned subdirectory)")
    parser.add_argument("--n-permutations",   default=1000, type=int)
    parser.add_argument("--patch-size",       default=PATCH_SIZE_DEFAULT, type=int)
    parser.add_argument("--seed",             default=42, type=int)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print(f"  Holeyness v3 significance — section {args.section}")
    print("=" * 60)

    per_duct = pd.read_csv(args.v1_per_duct_csv)
    with open(args.v2_json) as f:
        v2_json = json.load(f)

    print("\n=== Consistency check vs v2 ===")
    consistency = check_consistency_with_v2(per_duct, v2_json)
    print(f"  {consistency}")
    if not consistency["all_match"]:
        print("  WARNING: v3 recompute does not match v2's saved reference numbers.")

    reference_from_v2 = {
        "rho_pt_hole_pct": v2_json["primary_correlation"]["rho_pt_hole_pct"],
        "partial_rho_pt_hole_given_area": v2_json["area_covariate"]["partial_rho_pt_hole_given_area"],
        "partial_rho_pt_hole_given_area_and_nd": v2_json["area_covariate"]["partial_rho_pt_hole_given_area_and_nd"],
    }

    area = per_duct["area_um2"].values
    nd   = per_duct["nuclear_density"].values

    print("\n=== GAP 1: permutation test on the area-adjusted partial (controlling for area) ===")
    perm_area = {
        "global":       run_partial_permutation_global(per_duct, [area], args.n_permutations, rng),
        "within_slide": run_partial_permutation_within_slide(per_duct, [area], args.n_permutations, rng),
    }
    print(f"  global: {perm_area['global']}")
    print(f"  within_slide: {perm_area['within_slide']}")

    print("\n=== GAP 1: permutation test (controlling for area + nuclear_density) ===")
    perm_area_nd = {
        "global":       run_partial_permutation_global(per_duct, [area, nd], args.n_permutations, rng),
        "within_slide": run_partial_permutation_within_slide(per_duct, [area, nd], args.n_permutations, rng),
    }
    print(f"  global: {perm_area_nd['global']}")
    print(f"  within_slide: {perm_area_nd['within_slide']}")

    print("\n=== GAP 2: aggregation & patch-count sensitivity ===")
    agg_sens_median = run_partial_aggregation_sensitivity(per_duct)
    per_duct_median_rederived, per_duct_mean = rederive_median_and_mean_tables(args)
    mean_table_consistency = check_rederivation_consistency(per_duct_median_rederived, per_duct)
    print(f"  mean-table rederivation consistency: {mean_table_consistency}")
    if not mean_table_consistency["all_match"]:
        print("  WARNING: raw-input re-derivation does not match v1's saved table.")
    agg_sens_mean = run_partial_aggregation_sensitivity(per_duct_mean)

    print("\n=== Per-slide investigation ===")
    per_slide_investigation = run_per_slide_investigation(per_duct, v2_json)
    for r in per_slide_investigation["per_slide"]:
        print(f"  {r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_per_slide_partial_bar_chart(per_slide_investigation, args.output_dir, args.section)

    write_v3_outputs(
        args.output_dir, args.section, args.n_permutations,
        consistency, reference_from_v2, mean_table_consistency,
        perm_area, perm_area_nd, agg_sens_median, agg_sens_mean,
        per_slide_investigation,
    )

    print("\n" + "=" * 60)
    print(f"  HOLEYNESS V3 SIGNIFICANCE COMPLETE — section {args.section}")
    print("=" * 60)
    print(f"\n  partial rho | area              = {perm_area['global']['observed_partial_rho']:.4f}"
          f"  within-slide perm_p {perm_area['within_slide']['perm_p_display']}")
    print(f"  partial rho | area, nuc_density  = {perm_area_nd['global']['observed_partial_rho']:.4f}"
          f"  within-slide perm_p {perm_area_nd['within_slide']['perm_p_display']}")
    print(f"\n  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
