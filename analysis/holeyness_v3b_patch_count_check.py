"""
Holeyness v3b: patch-count-per-duct discriminant + within-slide undersampling check.

v3's per-slide investigation (`holeyness_v3_significance.py::run_per_slide_investigation`)
compared the 3 flagged low-signal slides (6028-4R-2M-1_x5, 6029-4L-2M-1_x5, 6031-4L-2M-1_x5)
against the other 5 on median DUCT AREA and found no clear difference. It never directly
tested PATCH COUNT PER DUCT, even though v3's own per-slide table already shows all 3 flagged
slides at median_n_patches = 2 (every other slide is 3 or 4), and two of the three flagged
slides have the two highest frac_single_patch_ducts values in the cohort.

This module, in order:
  0. Resolves an unexplained ~0.000118 mismatch between v3's own recomputed
     partial_rho_pt_hole_given_area (from v1's saved CSV) and v2's saved reference value
     (computed from fresh, full-precision data) BEFORE trusting anything downstream.
  1. Tests median_n_patches_per_duct and frac_single_patch_ducts (not area) as the
     discriminating variable between the 3 flagged and 5 other slides.
  2. Within each of the 3 flagged slides individually, tests whether restricting to ducts
     with >=3 patches changes that slide's area-adjusted partial correlation — the direct
     undersampling-hypothesis test.

This is a NEW module, not another flag on holeyness.py or holeyness_v3_significance.py — both
stay completely frozen; this module only *imports* from them. Writes to a NEW versioned
directory; never touches v1, v2, or v3 outputs.

CLI
---
  python -m cancer_trajectory_atlas.analysis.holeyness_v3b_patch_count_check \\
      --section          2M-1 \\
      --v1-per-duct-csv  $SCRATCH/results/holeyness/2M-1/holeyness_per_duct.csv \\
      --v2-json          $SCRATCH/results/holeyness/2M-1/v2_area_adjusted/holeyness_validation_v2.json \\
      --v3-json          $SCRATCH/results/holeyness/2M-1/v3_significance/holeyness_validation_v3.json \\
      --export           $SCRATCH/data/holeyness/raw/combined_matched_measurements.txt \\
      --annotation-dir   ~/cancer_trajectory_atlas/data/annotations_ratio \\
      --slide-dimensions $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json \\
      --results          $SCRATCH/results/per_section/atlas_2M-1/results.csv \\
      --slide-list       ~/cancer_trajectory_atlas/jobs/slides_section1.txt \\
      --output-dir       $SCRATCH/results/holeyness/2M-1/v3b_patch_count_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .holeyness import PATCH_SIZE_DEFAULT, _partial_spearman
from .holeyness_v3_significance import (
    FLAGGED_SLIDES,
    _fmt,
    _json_default,
    rederive_median_and_mean_tables,
)

N_PATCH_THRESHOLD_UNDERSAMPLING = 3

# Smallest attainable exact two-sided Mann-Whitney p-value for n1=3, n2=5
# (perfect separation): 2 / C(8,3) = 2/56.
MIN_ATTAINABLE_MWU_P_N3_N5 = 2.0 / 56.0


# ── Check 0: resolve the v3 consistency-check mismatch ───────────────────────

def resolve_consistency_discrepancy(
    per_duct_csv: pd.DataFrame,
    v2_json: dict,
    v3_json: dict,
    rederive_args: argparse.Namespace,
) -> dict:
    """
    v3's check_consistency_with_v2 recomputed partial_rho_pt_hole_given_area from
    v1's saved CSV (6-decimal-rounded via float_format="%.6f") and found a small
    mismatch against v2's saved reference (computed from fresh, full float64-precision
    data, never round-tripped through a file). Both use the exact same
    _partial_spearman function, so any mismatch must come from precision loss on the
    *input* values, not from a different formula/solver.

    This re-derives the full-precision duct table from the same raw inputs v1/v2/v3
    all used (cheap, deterministic, CPU-only — not an atlas-pipeline rerun) and checks
    whether recomputing from that full-precision table closes the gap.
    """
    ref = v2_json["area_covariate"]["partial_rho_pt_hole_given_area"]
    v3_reported_diff = v3_json["consistency_check"]["max_abs_diff"]

    pt_csv, hole_csv, area_csv = (
        per_duct_csv["pseudotime"].values,
        per_duct_csv["hole_pct"].values,
        per_duct_csv["area_um2"].values,
    )
    from_csv = _partial_spearman(pt_csv, hole_csv, area_csv)

    per_duct_full, _ = rederive_median_and_mean_tables(rederive_args)
    pt_full, hole_full, area_full = (
        per_duct_full["pseudotime"].values,
        per_duct_full["hole_pct"].values,
        per_duct_full["area_um2"].values,
    )
    from_full_precision = _partial_spearman(pt_full, hole_full, area_full)

    diff_csv = abs(from_csv - ref)
    diff_full = abs(from_full_precision - ref)

    round3_csv = round(from_csv, 3)
    round3_full = round(from_full_precision, 3)
    round3_ref = round(ref, 3)
    third_decimal_stable = (round3_csv == round3_full == round3_ref)

    # The full-precision recompute is judged to "resolve" the gap only if it lands
    # much closer to v2's reference than the CSV-rounded recompute did.
    resolved = diff_full < diff_csv / 10.0

    if resolved:
        source_verdict = (
            "RESOLVED — recomputing from full-precision data (bypassing v1's saved "
            "6-decimal-rounded CSV) reproduces v2's reference value far more closely "
            f"(diff {diff_full:.6g} vs {diff_csv:.6g} from the CSV path). The mismatch is "
            "floating-point precision loss from v1's holeyness_per_duct.csv being written "
            "with float_format='%.6f', not a solver or formula difference — both v2 and v3 "
            "call the identical _partial_spearman function."
        )
    elif diff_full < 1e-6:
        source_verdict = (
            "RESOLVED — the full-precision recompute matches v2's reference to within "
            "floating-point tolerance, confirming the CSV round-trip (float_format='%.6f' "
            "in v1's writer) as the source of the discrepancy."
        )
    else:
        source_verdict = (
            "UNRESOLVED — recomputing from full-precision data did NOT close the gap to "
            "v2's reference value (CSV path diff {:.6g}, full-precision path diff {:.6g}). "
            "The CSV-rounding hypothesis does not fully explain this discrepancy; the "
            "source is not confirmed. Do not assume the remaining checks in this report "
            "are unaffected by whatever the true cause is.".format(diff_csv, diff_full)
        )

    return {
        "v2_reference_partial_rho_pt_hole_given_area": ref,
        "v3_reported_max_abs_diff": v3_reported_diff,
        "recomputed_from_v1_csv_rounded": from_csv,
        "recomputed_from_full_precision_rederivation": from_full_precision,
        "abs_diff_csv_vs_v2_reference": diff_csv,
        "abs_diff_full_precision_vs_v2_reference": diff_full,
        "third_decimal_place_values": {
            "from_csv": round3_csv,
            "from_full_precision": round3_full,
            "v2_reference": round3_ref,
        },
        "third_decimal_place_stable": bool(third_decimal_stable),
        "source_resolved": bool(resolved or diff_full < 1e-6),
        "source_verdict": source_verdict,
    }


# ── Check 1: patch-count-per-duct as discriminant ─────────────────────────────

def run_patch_count_discriminant(v3_per_slide: list[dict]) -> dict:
    """Compare median_n_patches and frac_single_patch_ducts (from v3's already-computed
    per-slide table) between the 3 flagged and 5 other slides. Descriptive comparison is
    primary; Mann-Whitney U is reported for transparency but explicitly caveated given
    n=3 vs n=5."""
    flagged = [r for r in v3_per_slide if r["slide_name"] in FLAGGED_SLIDES]
    other = [r for r in v3_per_slide if r["slide_name"] not in FLAGGED_SLIDES]

    def _vals(rows, key):
        return np.array([r[key] for r in rows], dtype=float)

    result = {"flagged_slides": FLAGGED_SLIDES, "n_flagged": len(flagged), "n_other": len(other)}

    for key in ("median_n_patches", "frac_single_patch_ducts"):
        f_vals = _vals(flagged, key)
        o_vals = _vals(other, key)
        try:
            u_stat, p = mannwhitneyu(f_vals, o_vals, alternative="two-sided", method="exact")
            u_stat, p = float(u_stat), float(p)
        except ValueError:
            u_stat, p = float("nan"), float("nan")

        result[key] = {
            "flagged_values": f_vals.tolist(),
            "other_values": o_vals.tolist(),
            "flagged_mean": float(np.mean(f_vals)),
            "other_mean": float(np.mean(o_vals)),
            "flagged_median": float(np.median(f_vals)),
            "other_median": float(np.median(o_vals)),
            "mannwhitney_u": u_stat,
            "mannwhitney_p": p,
        }

    result["min_attainable_mwu_p_at_n3_n5"] = MIN_ATTAINABLE_MWU_P_N3_N5
    return result


# ── Check 2: per-slide undersampling test ─────────────────────────────────────

def run_within_slide_undersampling_check(per_duct_csv: pd.DataFrame) -> dict:
    """For each flagged slide individually: partial rho(pseudotime, hole_pct | area)
    at all ducts vs. ducts with >= N_PATCH_THRESHOLD_UNDERSAMPLING patches. Descriptive
    only — n per slide is far too small for a formal significance test."""
    per_slide = {}
    for slide_name in FLAGGED_SLIDES:
        grp = per_duct_csv[per_duct_csv["slide_name"] == slide_name]
        pt_all, hole_all, area_all = (
            grp["pseudotime"].values, grp["hole_pct"].values, grp["area_um2"].values,
        )
        partial_all = _partial_spearman(pt_all, hole_all, area_all)

        subset = grp[grp["n_patches"] >= N_PATCH_THRESHOLD_UNDERSAMPLING]
        pt_sub, hole_sub, area_sub = (
            subset["pseudotime"].values, subset["hole_pct"].values, subset["area_um2"].values,
        )
        partial_sub = _partial_spearman(pt_sub, hole_sub, area_sub)

        strengthened = (
            np.isfinite(partial_all) and np.isfinite(partial_sub)
            and abs(partial_sub) > abs(partial_all)
        )
        per_slide[slide_name] = {
            "n_ducts_all": int(len(grp)),
            "partial_rho_all_ducts": partial_all,
            "n_ducts_min_3_patches": int(len(subset)),
            "partial_rho_min_3_patches": partial_sub,
            "signal_strengthened_when_restricted": bool(strengthened),
        }

    n_strengthened = sum(1 for v in per_slide.values() if v["signal_strengthened_when_restricted"])
    return {
        "n_patch_threshold": N_PATCH_THRESHOLD_UNDERSAMPLING,
        "per_slide": per_slide,
        "n_slides_strengthened_of_3": n_strengthened,
    }


# ── Figure ────────────────────────────────────────────────────────────────────

def write_median_n_patches_bar_chart(v3_per_slide: list[dict], output_dir: Path, section: str) -> None:
    rows = sorted(v3_per_slide, key=lambda r: r["median_n_patches"])
    names = [r["slide_name"] for r in rows]
    vals = [r["median_n_patches"] for r in rows]
    colors = ["#D65F5F" if r["slide_name"] in FLAGGED_SLIDES else "#4878CF" for r in rows]

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(rows) + 1.5))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Median patches per duct")
    ax.set_title(f"Section {section}: median patches per duct, by slide\n(red = v3-flagged low-signal slides)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"v3b_median_n_patches_bar.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── Output writers ────────────────────────────────────────────────────────────

def write_v3b_report(
    output_dir: Path,
    section: str,
    consistency: dict,
    patch_count: dict,
    undersampling: dict,
) -> None:
    lines = [f"# Holeyness v3b — patch-count-per-duct discriminant — section {section}", ""]

    lines += ["## 0. v3 consistency-check mismatch — source resolution", ""]
    lines.append(f"- v2 reference partial rho (| area) = {_fmt(consistency['v2_reference_partial_rho_pt_hole_given_area'])}")
    lines.append(f"- v3-reported max_abs_diff = {_fmt(consistency['v3_reported_max_abs_diff'])}")
    lines.append(f"- Recomputed from v1's CSV (6-decimal rounded): {_fmt(consistency['recomputed_from_v1_csv_rounded'])}"
                 f"  (diff from v2 ref: {_fmt(consistency['abs_diff_csv_vs_v2_reference'])})")
    lines.append(f"- Recomputed from full-precision re-derivation: {_fmt(consistency['recomputed_from_full_precision_rederivation'])}"
                 f"  (diff from v2 ref: {_fmt(consistency['abs_diff_full_precision_vs_v2_reference'])})")
    d3 = consistency["third_decimal_place_values"]
    lines.append(
        f"- 3rd-decimal values — CSV: {d3['from_csv']}, full-precision: {d3['from_full_precision']}, "
        f"v2 reference: {d3['v2_reference']} — "
        + ("stable across all three." if consistency["third_decimal_place_stable"] else "NOT stable — differs at the 3rd decimal.")
    )
    lines.append(f"- **Verdict:** {consistency['source_verdict']}")
    lines.append("")

    lines += ["## 1. Patch-count-per-duct as discriminant (flagged vs other)", ""]
    lines.append(f"Flagged slides (n={patch_count['n_flagged']}): {', '.join(patch_count['flagged_slides'])}")
    lines.append(f"Other slides (n={patch_count['n_other']})")
    lines.append("")
    for key, label in (("median_n_patches", "median_n_patches_per_duct"), ("frac_single_patch_ducts", "frac_single_patch_ducts")):
        v = patch_count[key]
        lines.append(f"### {label}")
        lines.append(f"- Flagged: values={v['flagged_values']}, mean={_fmt(v['flagged_mean'])}, median={_fmt(v['flagged_median'])}")
        lines.append(f"- Other:   values={v['other_values']}, mean={_fmt(v['other_mean'])}, median={_fmt(v['other_median'])}")
        lines.append(
            f"- Mann-Whitney U = {_fmt(v['mannwhitney_u'])}, p = {_fmt(v['mannwhitney_p'])} "
            f"(exact test; smallest attainable p at n=3 vs n=5 is {patch_count['min_attainable_mwu_p_at_n3_n5']:.4g} "
            f"— reported for transparency only, NOT a claim of adequate statistical power at this n)"
        )
        lines.append("")

    patches_discriminate = (
        patch_count["median_n_patches"]["flagged_mean"] < patch_count["median_n_patches"]["other_mean"]
        and patch_count["frac_single_patch_ducts"]["flagged_mean"] > patch_count["frac_single_patch_ducts"]["other_mean"]
    )
    verdict1 = (
        "Patch-count-per-duct separates the flagged slides from the others more cleanly than duct "
        "area did in v3 (v3 found no clear area difference): flagged slides have lower "
        "median_n_patches and a higher frac_single_patch_ducts. This is a group-mean comparison "
        "at n=3 vs n=5 — directionally clear, but not a formally powered test."
        if patches_discriminate else
        "Patch-count-per-duct does NOT clearly separate the flagged slides from the others either — "
        "the group means do not show the expected direction (lower median_n_patches AND higher "
        "frac_single_patch_ducts for the flagged group)."
    )
    lines.append(f"- **Verdict:** {verdict1}")
    lines.append("")

    lines += ["## 2. Within-slide undersampling test (per flagged slide)", ""]
    lines.append(
        f"Partial rho(pseudotime, hole_pct | area) at all ducts vs. ducts with "
        f">= {undersampling['n_patch_threshold']} patches, computed independently per flagged slide. "
        f"Descriptive only — n per slide is far too small for a formal significance test."
    )
    lines.append("")
    lines.append("| slide | n_ducts (all) | partial rho (all) | n_ducts (>=3 patches) | partial rho (>=3 patches) | strengthened? |")
    lines.append("|---|---|---|---|---|---|")
    for slide_name, v in undersampling["per_slide"].items():
        lines.append(
            f"| {slide_name} | {v['n_ducts_all']} | {_fmt(v['partial_rho_all_ducts'])} | "
            f"{v['n_ducts_min_3_patches']} | {_fmt(v['partial_rho_min_3_patches'])} | "
            f"{'yes' if v['signal_strengthened_when_restricted'] else 'no'} |"
        )
    lines.append("")
    n_strengthened = undersampling["n_slides_strengthened_of_3"]
    if n_strengthened >= 2:
        verdict2 = (
            f"{n_strengthened}/3 flagged slides show a stronger (larger-magnitude) area-adjusted "
            f"partial correlation once ducts with fewer than {undersampling['n_patch_threshold']} "
            f"patches are excluded — this supports the undersampling explanation for why these "
            f"slides looked weak in the full-cohort v2/v3 numbers. Still descriptive given the "
            f"small per-slide n."
        )
    else:
        verdict2 = (
            f"Only {n_strengthened}/3 flagged slides show a stronger area-adjusted partial "
            f"correlation when restricted to ducts with >= {undersampling['n_patch_threshold']} "
            f"patches — the undersampling explanation is not clearly supported by this within-slide "
            f"test. Descriptive only given the small per-slide n."
        )
    lines.append(f"- **Verdict:** {verdict2}")
    lines.append("")

    (output_dir / "holeyness_validation_v3b.md").write_text("\n".join(lines), encoding="utf-8")


def write_v3b_outputs(
    output_dir: Path,
    section: str,
    consistency: dict,
    patch_count: dict,
    undersampling: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "section": section,
        "consistency_check_resolution": consistency,
        "patch_count_discriminant": patch_count,
        "within_slide_undersampling_check": undersampling,
    }
    json_path = output_dir / "holeyness_validation_v3b.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"  JSON: {json_path}")

    write_v3b_report(output_dir, section, consistency, patch_count, undersampling)
    print(f"  Markdown report: {output_dir / 'holeyness_validation_v3b.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Holeyness v3b: patch-count-per-duct discriminant + undersampling check"
    )
    parser.add_argument("--section",          required=True, help="Section label, e.g. '2M-1'")
    parser.add_argument("--v1-per-duct-csv",  required=True, type=Path,
                        help="Path to v1's holeyness_per_duct.csv")
    parser.add_argument("--v2-json",          required=True, type=Path,
                        help="Path to v2's holeyness_validation_v2.json")
    parser.add_argument("--v3-json",          required=True, type=Path,
                        help="Path to v3's holeyness_validation_v3.json")
    parser.add_argument("--export",           required=True, type=Path,
                        help="Path to combined_matched_measurements.txt (for full-precision re-derivation)")
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
    parser.add_argument("--patch-size",       default=PATCH_SIZE_DEFAULT, type=int)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Holeyness v3b patch-count check — section {args.section}")
    print("=" * 60)

    per_duct_csv = pd.read_csv(args.v1_per_duct_csv)
    with open(args.v2_json) as f:
        v2_json = json.load(f)
    with open(args.v3_json) as f:
        v3_json = json.load(f)

    print("\n=== Check 0: resolving v3 consistency-check mismatch ===")
    consistency = resolve_consistency_discrepancy(per_duct_csv, v2_json, v3_json, args)
    print(f"  {consistency}")

    print("\n=== Check 1: patch-count-per-duct as discriminant ===")
    v3_per_slide = v3_json["per_slide_investigation"]["per_slide"]
    patch_count = run_patch_count_discriminant(v3_per_slide)
    print(f"  {patch_count}")

    print("\n=== Check 2: within-slide undersampling test ===")
    undersampling = run_within_slide_undersampling_check(per_duct_csv)
    print(f"  {undersampling}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_median_n_patches_bar_chart(v3_per_slide, args.output_dir, args.section)

    write_v3b_outputs(args.output_dir, args.section, consistency, patch_count, undersampling)

    print("\n" + "=" * 60)
    print(f"  HOLEYNESS V3B PATCH-COUNT CHECK COMPLETE — section {args.section}")
    print("=" * 60)
    print(f"\n  Check 0 source resolved: {consistency['source_resolved']}")
    print(f"  Check 1 min_attainable_mwu_p (n=3 vs 5): {MIN_ATTAINABLE_MWU_P_N3_N5:.4g}")
    print(f"  Check 2 slides strengthened when restricted to >=3 patches: "
          f"{undersampling['n_slides_strengthened_of_3']}/3")
    print(f"\n  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
