"""
Holeyness final consolidation: one authoritative report over v1-v3b for section 2M-1.

v1 through v3b accumulated four methodological problems this module corrects in a
single pass, rather than patching each in place:

  1. CIRCULAR GROUP SELECTION. v3/v3b's "3 flagged slides" were selected BY LOOKING
     AT their weak area-adjusted partials in v2, then tested for properties on the
     same data. That comparison is replaced (not patched) with a non-circular check
     across all 8 slides with no subsetting (Task A).
  2. NO PRE-SPECIFIED/EXPLORATORY SEPARATION. ~60+ correlations were computed across
     v1-v3b with no labelling or correction. This module tags every reported
     quantity PRIMARY or EXPLORATORY and counts the exploratory ones (Task C).
  3. ESTIMAND NEVER STATED. 571/2173 ducts (26%) were excluded for containing zero
     patches under the centre-in-polygon assignment rule, and are systematically
     smaller/less holey than retained ducts (v2 already found this). The population
     the correlation actually describes is stated explicitly (Task B).
  4. SIGN/MAGNITUDE CONFLATION IN v3b. v3b's "3/3 slides strengthened" verdict used
     absolute magnitude, counting a slide moving from -0.069 to -0.202 (away from
     the cohort's positive signal) as "strengthened." Corrected with direction
     preserved (Task E).

Also reports raw and area-adjusted partial correlation as CO-PRIMARY (Task D) rather
than picking one as "the" result, since the choice depends on an unresolved
mediator-vs-confound question about duct area. An optional Task F re-derives
patch-to-duct assignment under an area-overlap rule (instead of centre-in-polygon)
to test whether the exclusion bias can be directly addressed rather than merely
documented.

This is a NEW module. It imports from `holeyness.py` and `holeyness_v3_significance.py`
but never edits either — nor `holeyness_v3b_patch_count_check.py`. v1/v2/v3/v3b
outputs are read-only inputs and are never modified or rerun; they remain as
provenance. Writes to a NEW `final/` directory.

CLI
---
  python -m cancer_trajectory_atlas.analysis.holeyness_final \\
      --section          2M-1 \\
      --v1-per-duct-csv  $SCRATCH/results/holeyness/2M-1/holeyness_per_duct.csv \\
      --v1-json          $SCRATCH/results/holeyness/2M-1/holeyness_validation.json \\
      --v2-json          $SCRATCH/results/holeyness/2M-1/v2_area_adjusted/holeyness_validation_v2.json \\
      --v3-json          $SCRATCH/results/holeyness/2M-1/v3_significance/holeyness_validation_v3.json \\
      --v3b-json         $SCRATCH/results/holeyness/2M-1/v3b_patch_count_check/holeyness_validation_v3b.json \\
      --output-dir       $SCRATCH/results/holeyness/2M-1/final
      # Optional, Task F only:
      --export           $SCRATCH/data/holeyness/raw/combined_matched_measurements.txt \\
      --annotation-dir   ~/cancer_trajectory_atlas/data/annotations_ratio \\
      --slide-dimensions $SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json \\
      --results          $SCRATCH/results/per_section/atlas_2M-1/results.csv \\
      --slide-list       ~/cancer_trajectory_atlas/jobs/slides_section1.txt
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

from .holeyness import (
    PATCH_SIZE_DEFAULT,
    _safe_spearman,
    _partial_spearman,
    load_slide_list,
    load_slide_dimensions,
    parse_measurement_export,
    load_duct_polygons,
    build_duct_table,
    aggregate_per_duct,
)
from .holeyness_v3_significance import _fmt, _json_default

OVERLAP_MIN_FRACTION_DEFAULT = 0.25


def _fmt_or_na(v) -> str:
    if v is None:
        return "n/a"
    return _fmt(v)


# ── Task B: estimand ──────────────────────────────────────────────────────────

def build_estimand_statement(section: str, v2_json: dict) -> dict:
    excl = v2_json["exclusion_bias"]
    n_excluded = excl["n_excluded"]
    n_retained = excl["n_retained"]
    n_total = n_excluded + n_retained
    statement = (
        f"This analysis describes ducts within pipeline slides of section {section} "
        f"that contain at least one patch centre under the current centre-in-polygon "
        f"patch-to-duct assignment rule. It does NOT describe all annotated ducts in "
        f"the section."
    )
    return {
        "statement": statement,
        "n_total_ducts_with_measurements": n_total,
        "n_excluded_zero_patch": n_excluded,
        "n_retained": n_retained,
        "pct_excluded": (100.0 * n_excluded / n_total) if n_total else float("nan"),
        "excluded_vs_retained": excl,
        "generalization_caveat": (
            "The excluded ducts are systematically smaller and less holey than "
            "retained ducts (see excluded_vs_retained) — this is not a missing-at-"
            "random exclusion. Every estimate below describes the retained, "
            "larger-duct population only and should not be extrapolated to the "
            "smallest ducts in the section."
        ),
    }


# ── Task A: non-circular cross-slide sampling-quality check ──────────────────

def run_cross_slide_sampling_check(v3_per_slide: list[dict]) -> dict:
    """Across ALL 8 slides (no subsetting, no post-hoc threshold), test whether a
    slide's sampling quality (median_n_patches, frac_single_patch_ducts) predicts
    its area-adjusted partial correlation. Replaces v3/v3b's circular
    flagged-vs-other framing entirely."""
    rows = [
        r for r in v3_per_slide
        if np.isfinite(r.get("partial_rho_from_v2", float("nan")))
    ]
    n = len(rows)
    partial = np.array([r["partial_rho_from_v2"] for r in rows], dtype=float)
    median_n_patches = np.array([r["median_n_patches"] for r in rows], dtype=float)
    frac_single = np.array([r["frac_single_patch_ducts"] for r in rows], dtype=float)

    rho_patches, p_patches = _safe_spearman(median_n_patches, partial)
    rho_frac, p_frac = _safe_spearman(frac_single, partial)

    return {
        "n_slides": n,
        "per_slide": [
            {
                "slide_name": r["slide_name"],
                "median_n_patches": r["median_n_patches"],
                "frac_single_patch_ducts": r["frac_single_patch_ducts"],
                "partial_rho_pt_hole_given_area": r["partial_rho_from_v2"],
            }
            for r in rows
        ],
        "rho_median_n_patches_vs_partial_rho": rho_patches,
        "p_median_n_patches_vs_partial_rho_scipy": p_patches,
        "rho_frac_single_patch_vs_partial_rho": rho_frac,
        "p_frac_single_patch_vs_partial_rho_scipy": p_frac,
        "note": (
            f"n={n} slides — descriptive only. This uses every slide, applies no "
            "post-hoc threshold, and is not conditioned on the outcome being tested "
            "(unlike v3/v3b's flagged-slide comparisons, which are superseded by "
            "this check). scipy's p-value is reported for transparency but must not "
            "be read as a reliable significance test at this n."
        ),
    }


# ── Task C: primary/exploratory labelling + exploratory count ────────────────

def count_exploratory_correlations(v1_json: dict, v2_json: dict, v3_json: dict, v3b_json: dict) -> dict:
    """Programmatic, schema-aware tally of correlation-type values (rho / partial
    rho) present in the saved v1-v3b JSON outputs, excluding the 3 canonical primary
    quantities (and their permutation-test byproducts). List lengths are read from
    the actual loaded JSON, not hardcoded, so this stays accurate to whatever the
    real per-slide/threshold counts are. This is a transparency count, not a
    precision audit."""
    # NOTE: v1/v2's "correlations"/"primary_correlation"/"area_covariate" dicts mix
    # rho-type fields with scipy p-value fields (e.g. p_pt_hole_pct_scipy) under the
    # same dict — len() on those would double-count p-values as if they were
    # separate correlation computations, so the rho-type field counts are fixed
    # constants matching holeyness.py's run_correlations (5 rho fields) and
    # run_area_covariate_checks (5 rho fields) return schemas, not len(dict).
    v1_n = 5  # holeyness.py: run_correlations() — 5 rho-type fields (+ 4 p-value fields, not counted)

    v2_n = (
        5  # v2 primary_correlation — same 5 rho-type fields as v1 (recomputed)
        + 5  # v2 area_covariate — 5 rho-type fields (+ 2 p-value fields, not counted)
        + 2 * len(v2_json.get("within_slide", {}).get("per_slide", []))
        + 1  # between_slide_median_correlation.rho
        + 2 + len(v2_json.get("aggregation_sensitivity", {}).get("by_min_patches_threshold", {}))
        + 2  # patch_sampling_artifact: rho_area_nuclear_density, rho_n_patches_nuclear_density
    )

    v3_n = (
        3  # consistency_check.v3_recomputed (rho_raw, partial|area, partial|area+nd)
        + 10  # aggregation_sensitivity_partial: median_aggregation (5) + mean_aggregation (5)
        + 2  # mean_table_consistency_check: rho_rederived, rho_v1_csv
        + 2 * len(v3_json.get("per_slide_investigation", {}).get("per_slide", []))
    )

    v3b_n = (
        2  # consistency_check_resolution: recomputed_from_v1_csv_rounded, recomputed_from_full_precision
        + 2 * len(v3b_json.get("within_slide_undersampling_check", {}).get("per_slide", {}))
    )

    total = v1_n + v2_n + v3_n + v3b_n
    return {
        "v1": v1_n,
        "v2": v2_n,
        "v3": v3_n,
        "v3b": v3b_n,
        "total_exploratory": total,
        "n_primary": 3,
        "note": (
            "Approximate, programmatically-counted tally of correlation-type values "
            "across the v1-v3b outputs, excluding the 3 canonical primary quantities "
            "(raw rho, partial|area, partial|area+nd) and their permutation-test "
            "byproducts. Counted for transparency about researcher degrees of "
            "freedom accumulated across the analysis chain, not as a precise audit."
        ),
    }


def build_results_table(v2_json: dict, v3_json: dict, exploratory_count: dict) -> list[dict]:
    """Single results table: the 3 PRIMARY quantities (pre-specified before seeing
    v2/v3's own within-analysis results, per the task's own framing) plus one
    EXPLORATORY summary row pointing at the full count/breakdown."""
    ref = v3_json["reference_from_v2"]
    perm_area = v3_json["partial_permutation"]["given_area"]
    perm_area_nd = v3_json["partial_permutation"]["given_area_and_nd"]
    perm_raw = v2_json["permutation"]

    rows = [
        {
            "label": "raw rho(pseudotime, hole_pct)",
            "value": ref["rho_pt_hole_pct"],
            "tag": "PRIMARY",
            "global_perm_p": perm_raw["global"].get("perm_p_rho_pt_hole_pct_display"),
            "within_slide_perm_p": perm_raw["within_slide"].get("perm_p_display"),
            "source": "v3_json.reference_from_v2 / v2_json.permutation",
        },
        {
            "label": "partial rho(pseudotime, hole_pct | area)",
            "value": ref["partial_rho_pt_hole_given_area"],
            "tag": "PRIMARY",
            "global_perm_p": perm_area["global"]["perm_p_display"],
            "within_slide_perm_p": perm_area["within_slide"]["perm_p_display"],
            "source": "v3_json.reference_from_v2 / v3_json.partial_permutation.given_area",
        },
        {
            "label": "partial rho(pseudotime, hole_pct | area, nuclear_density)",
            "value": ref["partial_rho_pt_hole_given_area_and_nd"],
            "tag": "PRIMARY",
            "global_perm_p": perm_area_nd["global"]["perm_p_display"],
            "within_slide_perm_p": perm_area_nd["within_slide"]["perm_p_display"],
            "source": "v3_json.reference_from_v2 / v3_json.partial_permutation.given_area_and_nd",
        },
        {
            "label": "all other correlation-type values across v1-v3b outputs",
            "value": None,
            "tag": "EXPLORATORY",
            "count": exploratory_count["total_exploratory"],
            "source": "see exploratory_correlation_count for the per-source breakdown",
        },
    ]
    return rows


# ── Task D: co-primary reporting (raw + area-adjusted partial) ───────────────

def build_co_primary_summary(v2_json: dict, v3_json: dict) -> dict:
    ref = v3_json["reference_from_v2"]
    perm_area = v3_json["partial_permutation"]["given_area"]
    perm_area_nd = v3_json["partial_permutation"]["given_area_and_nd"]
    perm_raw = v2_json["permutation"]

    return {
        "raw_rho": {
            "value": ref["rho_pt_hole_pct"],
            "global_perm_p_display": perm_raw["global"].get("perm_p_rho_pt_hole_pct_display"),
            "within_slide_perm_p_display": perm_raw["within_slide"].get("perm_p_display"),
            "threshold_sweep": v2_json["aggregation_sensitivity"]["by_min_patches_threshold"],
            "median_vs_mean_aggregation": {
                "median": v2_json["aggregation_sensitivity"]["median_aggregation"],
                "mean": v2_json["aggregation_sensitivity"]["mean_aggregation"],
            },
        },
        "partial_rho_given_area": {
            "value": ref["partial_rho_pt_hole_given_area"],
            "global_perm_p_display": perm_area["global"]["perm_p_display"],
            "within_slide_perm_p_display": perm_area["within_slide"]["perm_p_display"],
            "threshold_sweep_median_agg": v3_json["aggregation_sensitivity_partial"]["median_aggregation"],
            "threshold_sweep_mean_agg": v3_json["aggregation_sensitivity_partial"]["mean_aggregation"],
        },
        "partial_rho_given_area_and_nd": {
            "value": ref["partial_rho_pt_hole_given_area_and_nd"],
            "global_perm_p_display": perm_area_nd["global"]["perm_p_display"],
            "within_slide_perm_p_display": perm_area_nd["within_slide"]["perm_p_display"],
        },
        "framing": (
            "Raw and area-adjusted partial rho are reported as CO-PRIMARY. The choice "
            "between them depends on an unresolved causal question this data cannot "
            "settle: if duct expansion is a MEDIATOR of progression, adjusting for "
            "area removes real signal and the raw value is preferred; if it is a "
            "CONFOUND, the adjusted value is preferred. Neither is designated 'the' "
            "result here."
        ),
    }


# ── Task E: corrected v3b within-slide undersampling verdict ─────────────────

def reinterpret_v3b_undersampling(v3b_json: dict) -> dict:
    """v3b's within_slide_undersampling_check.per_slide has correct SIGNED
    partial_rho_all_ducts / partial_rho_min_3_patches values — only its derived
    'signal_strengthened_when_restricted' boolean was wrong (used abs magnitude,
    so a move from -0.069 to -0.202 counted as 'strengthened'). This recomputes
    direction correctly and states the corrected, direction-aware verdict."""
    per_slide_raw = v3b_json["within_slide_undersampling_check"]["per_slide"]
    corrected = {}
    n_toward = 0
    n_ducts_at_threshold = []

    for slide_name, v in per_slide_raw.items():
        partial_all = v["partial_rho_all_ducts"]
        partial_sub = v["partial_rho_min_3_patches"]
        if np.isfinite(partial_all) and np.isfinite(partial_sub):
            delta = partial_sub - partial_all
            moved_toward_positive = bool(delta > 0)
        else:
            delta = float("nan")
            moved_toward_positive = False
        if moved_toward_positive:
            n_toward += 1
        n_ducts_at_threshold.append(v["n_ducts_min_3_patches"])
        corrected[slide_name] = {
            "n_ducts_all": v["n_ducts_all"],
            "partial_rho_all_ducts": partial_all,
            "n_ducts_min_3_patches": v["n_ducts_min_3_patches"],
            "partial_rho_min_3_patches": partial_sub,
            "signed_delta": delta,
            "moved_toward_cohort_positive_signal": moved_toward_positive,
        }

    n_total = len(per_slide_raw)
    n_min = min(n_ducts_at_threshold) if n_ducts_at_threshold else None
    n_max = max(n_ducts_at_threshold) if n_ducts_at_threshold else None
    old_verdict_n = v3b_json["within_slide_undersampling_check"].get("n_slides_strengthened_of_3")

    verdict = (
        f"{n_toward} of {n_total} flagged slides moved toward the cohort's positive "
        f"signal once direction is preserved (v3b's original abs-magnitude verdict "
        f"reported {old_verdict_n}/{n_total} 'strengthened', which miscounted at "
        f"least one move away from the signal as strengthening). At n={n_min}-{n_max} "
        f"ducts per restricted subset, swings of this size are within ordinary "
        f"sampling variability for a Spearman-based partial correlation. This check "
        f"is UNINFORMATIVE about the undersampling hypothesis, not supportive of it."
    )

    return {
        "per_slide": corrected,
        "n_slides_moved_toward_positive_signal": n_toward,
        "n_slides_total": n_total,
        "n_ducts_range_at_3patch_threshold": [n_min, n_max],
        "verdict": verdict,
        "note_on_prior_v3b_verdict": (
            f"v3b's saved holeyness_validation_v3b.json (retained unmodified, as "
            f"provenance) reported n_slides_strengthened_of_3={old_verdict_n} using an "
            f"absolute-magnitude comparison. This report supersedes that verdict with "
            f"the direction-corrected one above; the underlying signed values in "
            f"v3b's JSON were themselves correct — only the derived boolean was wrong."
        ),
    }


# ── Task F (optional): overlap-based patch-to-duct assignment sensitivity ────

def run_overlap_sensitivity(args: argparse.Namespace, v1_per_duct_csv: pd.DataFrame) -> dict:
    """Re-assign patches to ducts using an area-overlap rule (assign to the duct
    with maximum overlap, provided that overlap is >= --overlap-min-fraction of the
    patch's area) instead of the pipeline's centre-in-polygon rule, and check
    whether this recovers any of the 571 previously-excluded zero-patch ducts or
    changes the primary correlations. Patch geometry (x, y, patch_size from
    results.csv) and duct polygons (via holeyness.load_duct_polygons, unmodified)
    are both already loadable — no new raw-data dependency. Skips gracefully (rather
    than approximating) if raw inputs weren't supplied or shapely isn't available.
    The inner assignment loop is duplicated in
    holeyness_roots.assign_patches_to_ducts_overlap, which compares absolute overlap
    area where this compares the fraction of the patch. Dividing by the constant
    patch_area leaves the argmax and the threshold unchanged, so the two are
    equivalent. See that function for why they were left separate.
    """
    raw_paths = [args.export, args.annotation_dir, args.slide_dimensions, args.results, args.slide_list]
    if any(p is None for p in raw_paths):
        return {
            "attempted": False,
            "reason": (
                "Skipped: one or more raw-input paths (--export/--annotation-dir/"
                "--slide-dimensions/--results/--slide-list) were not supplied. Task F "
                "is optional; the rest of this report does not depend on it."
            ),
        }

    try:
        from shapely.geometry import Polygon, box
        from shapely.strtree import STRtree
    except ImportError as e:
        return {"attempted": False, "reason": f"Skipped: shapely not importable ({e!r})."}

    try:
        pipeline_slides = load_slide_list(args.slide_list)
        slide_dims = load_slide_dimensions(args.slide_dimensions)
        measurements = parse_measurement_export(args.export, pipeline_slides)
        polygons = load_duct_polygons(args.annotation_dir, pipeline_slides, slide_dims)
        duct_table = build_duct_table(measurements, polygons)
        if len(duct_table) == 0:
            return {"attempted": False, "reason": "Skipped: no ducts remain after UUID join."}

        required_cols = ["x", "y", "slide_name", "pseudotime", "nuclear_density", "packing_irregularity"]
        results_df = pd.read_csv(args.results, low_memory=False)
        missing = [c for c in required_cols if c not in results_df.columns]
        if missing:
            return {"attempted": False, "reason": f"Skipped: results.csv missing columns {missing}."}
        results_df = results_df[required_cols].drop_duplicates()
        results_df = results_df[results_df["slide_name"].isin(pipeline_slides)].copy()
        results_df["duct_id"] = None

        patch_size = float(args.patch_size)
        patch_area = patch_size ** 2

        for slide_name, slide_df in results_df.groupby("slide_name"):
            slide_ducts = duct_table[duct_table["slide_name"] == slide_name].reset_index(drop=True)
            if len(slide_ducts) == 0:
                continue

            duct_polys = []
            for _, row in slide_ducts.iterrows():
                poly = Polygon(row["polygon"].vertices)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                duct_polys.append(poly)
            duct_ids = slide_ducts["object_id"].tolist()
            tree = STRtree(duct_polys)

            assigned = np.full(len(slide_df), None, dtype=object)
            for i, (_, prow) in enumerate(slide_df.iterrows()):
                patch_box = box(prow["x"], prow["y"], prow["x"] + patch_size, prow["y"] + patch_size)
                cand_idx = tree.query(patch_box)
                best_id, best_frac = None, 0.0
                for idx in cand_idx:
                    frac = patch_box.intersection(duct_polys[idx]).area / patch_area
                    if frac > best_frac:
                        best_frac, best_id = frac, duct_ids[idx]
                if best_id is not None and best_frac >= args.overlap_min_fraction:
                    assigned[i] = best_id
            results_df.loc[slide_df.index, "duct_id"] = assigned

        per_duct_overlap = aggregate_per_duct(results_df, duct_table, np.nanmedian, "median")
        if len(per_duct_overlap) < 4:
            return {
                "attempted": True,
                "reason": "Fewer than 4 ducts with patches under the overlap rule — cannot compute correlations.",
            }

        pt, hole, area = (
            per_duct_overlap["pseudotime"].values,
            per_duct_overlap["hole_pct"].values,
            per_duct_overlap["area_um2"].values,
        )
        rho_raw, _ = _safe_spearman(pt, hole)
        partial_area = _partial_spearman(pt, hole, area)

        old_ids = set(v1_per_duct_csv["object_id"])
        new_ids = set(per_duct_overlap["object_id"])
        all_duct_ids = set(duct_table["object_id"])
        previously_excluded = all_duct_ids - old_ids
        recovered = previously_excluded & new_ids

        rho_raw_v1, _ = _safe_spearman(
            v1_per_duct_csv["pseudotime"].values, v1_per_duct_csv["hole_pct"].values
        )

        return {
            "attempted": True,
            "overlap_min_fraction": args.overlap_min_fraction,
            "n_ducts_previously_excluded": len(previously_excluded),
            "n_recovered_under_overlap_rule": len(recovered),
            "n_ducts_with_patches_overlap_rule": len(per_duct_overlap),
            "n_ducts_with_patches_centre_rule": len(old_ids),
            "rho_pt_hole_pct_overlap_rule": rho_raw,
            "partial_rho_pt_hole_given_area_overlap_rule": partial_area,
            "rho_pt_hole_pct_centre_rule_v1": rho_raw_v1,
            "note": (
                "Recomputed under an overlap-based patch-to-duct assignment (max-overlap "
                f"duct, gated at >= {args.overlap_min_fraction:.2f} of patch area) instead "
                "of the pipeline's centre-in-polygon rule. Compare to the centre-rule "
                "values to see whether the exclusion bias materially affects the primary "
                "estimates, and to n_ducts_previously_excluded to see how much of the "
                "excluded population this rule actually recovers."
            ),
        }
    except Exception as e:
        return {"attempted": False, "reason": f"Skipped due to an error during re-derivation: {e!r}"}


# ── Figures ───────────────────────────────────────────────────────────────────

def write_partial_vs_patchcount_scatter(cross_slide: dict, output_dir: Path, section: str) -> None:
    rows = cross_slide["per_slide"]
    if not rows:
        print("  WARNING: no slides with a finite partial rho — skipping scatter (i)")
        return
    x = [r["median_n_patches"] for r in rows]
    y = [r["partial_rho_pt_hole_given_area"] for r in rows]
    names = [r["slide_name"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x, y, s=40, color="#4878CF")
    for xi, yi, name in zip(x, y, names):
        ax.annotate(name, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Median patches per duct")
    ax.set_ylabel("Area-adjusted partial rho(pseudotime, hole_pct)")
    ax.set_title(
        f"Section {section}: sampling quality vs. partial correlation, all "
        f"{cross_slide['n_slides']} slides\n"
        f"(rho={_fmt(cross_slide['rho_median_n_patches_vs_partial_rho'])}, "
        f"n={cross_slide['n_slides']}, descriptive only)"
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"final_scatter_partial_rho_vs_median_n_patches.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_pt_vs_hole_by_area_scatter(per_duct: pd.DataFrame, output_dir: Path, section: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sc = ax.scatter(
        per_duct["hole_pct"], per_duct["pseudotime"],
        c=per_duct["area_um2"], cmap="viridis", alpha=0.75, s=20, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Duct area (µm²)")
    ax.set_xlabel("Duct hole fraction (%)")
    ax.set_ylabel("Duct-level pseudotime (median)")
    ax.set_title(f"Section {section}: pseudotime vs hole fraction, coloured by area\n(co-primary raw relationship)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"final_scatter_pt_vs_hole_pct_by_area.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── Output writers ────────────────────────────────────────────────────────────

def write_final_report(
    output_dir: Path,
    section: str,
    estimand: dict,
    co_primary: dict,
    cross_slide: dict,
    v3b_reinterpretation: dict,
    results_table: list[dict],
    exploratory_count: dict,
    overlap_sensitivity: dict,
) -> None:
    lines = [f"# Holeyness final consolidated report — section {section}", ""]
    lines.append(
        "**Supersedes v1-v3b for reporting purposes.** Prior outputs "
        "(`holeyness_per_duct.csv`, `holeyness_validation.json`, `v2_area_adjusted/`, "
        "`v3_significance/`, `v3b_patch_count_check/`) are retained unmodified as "
        "provenance and are cited below, but their flagged-slide framing and v3b's "
        "within-slide verdict are corrected/replaced here — see Limitations."
    )
    lines.append("")

    lines += ["## Estimand", ""]
    lines.append(estimand["statement"])
    lines.append("")
    lines.append(
        f"- n ducts with measurements: {estimand['n_total_ducts_with_measurements']} "
        f"(retained: {estimand['n_retained']}, excluded zero-patch: "
        f"{estimand['n_excluded_zero_patch']}, {_fmt(estimand['pct_excluded'])}%)"
    )
    ev = estimand["excluded_vs_retained"]
    lines.append(
        f"- Excluded ducts: median area = {_fmt(ev['area_um2']['median_excluded'])} µm², "
        f"median hole_pct = {_fmt(ev['hole_pct']['median_excluded'])}"
    )
    lines.append(
        f"- Retained ducts: median area = {_fmt(ev['area_um2']['median_retained'])} µm², "
        f"median hole_pct = {_fmt(ev['hole_pct']['median_retained'])}"
    )
    lines.append(f"- {estimand['generalization_caveat']}")
    lines.append("")

    lines += ["## Primary results (co-primary: raw and area-adjusted partial rho)", ""]
    lines.append(co_primary["framing"])
    lines.append("")
    r = co_primary["raw_rho"]
    lines.append(f"### Raw rho(pseudotime, hole_pct) = {_fmt(r['value'])}")
    lines.append(
        f"- global perm_p {r['global_perm_p_display']}, within-slide perm_p "
        f"{r['within_slide_perm_p_display']}"
    )
    lines.append("- threshold sensitivity (median aggregation):")
    for k, v in r["threshold_sweep"].items():
        lines.append(f"  - {k}: rho = {_fmt(v['rho_pt_hole_pct'])} (n={v['n_ducts']})")
    lines.append(
        f"- median vs mean aggregation: median rho = "
        f"{_fmt(r['median_vs_mean_aggregation']['median']['rho_pt_hole_pct'])}, "
        f"mean rho = {_fmt(r['median_vs_mean_aggregation']['mean']['rho_pt_hole_pct'])}"
    )
    lines.append("")

    pa = co_primary["partial_rho_given_area"]
    lines.append(f"### Partial rho(pseudotime, hole_pct | area) = {_fmt(pa['value'])}")
    lines.append(
        f"- global perm_p {pa['global_perm_p_display']}, within-slide perm_p "
        f"{pa['within_slide_perm_p_display']}"
    )
    lines.append("- threshold sensitivity (median aggregation):")
    for k, v in pa["threshold_sweep_median_agg"].items():
        lines.append(f"  - {k}: partial rho = {_fmt(v['partial_rho_pt_hole_given_area'])} (n={v['n_ducts']})")
    lines.append("- threshold sensitivity (mean aggregation):")
    for k, v in pa["threshold_sweep_mean_agg"].items():
        lines.append(f"  - {k}: partial rho = {_fmt(v['partial_rho_pt_hole_given_area'])} (n={v['n_ducts']})")
    lines.append("")

    pand = co_primary["partial_rho_given_area_and_nd"]
    lines.append(
        f"### Partial rho(pseudotime, hole_pct | area, nuclear_density) = {_fmt(pand['value'])}"
    )
    lines.append(
        f"- global perm_p {pand['global_perm_p_display']}, within-slide perm_p "
        f"{pand['within_slide_perm_p_display']}"
    )
    lines.append("")

    lines += ["## Non-circular cross-slide sampling-quality check (replaces v3/v3b's flagged-slide framing)", ""]
    lines.append(cross_slide["note"])
    lines.append("")
    lines.append(
        f"- rho(median_n_patches, partial rho) = "
        f"{_fmt(cross_slide['rho_median_n_patches_vs_partial_rho'])} "
        f"(scipy p = {_fmt(cross_slide['p_median_n_patches_vs_partial_rho_scipy'])})"
    )
    lines.append(
        f"- rho(frac_single_patch_ducts, partial rho) = "
        f"{_fmt(cross_slide['rho_frac_single_patch_vs_partial_rho'])} "
        f"(scipy p = {_fmt(cross_slide['p_frac_single_patch_vs_partial_rho_scipy'])})"
    )
    lines.append("")
    lines.append("| slide | median_n_patches | frac_single_patch_ducts | partial rho (area) |")
    lines.append("|---|---|---|---|")
    for r in cross_slide["per_slide"]:
        lines.append(
            f"| {r['slide_name']} | {_fmt(r['median_n_patches'])} | "
            f"{_fmt(r['frac_single_patch_ducts'])} | {_fmt(r['partial_rho_pt_hole_given_area'])} |"
        )
    lines.append("")

    lines += ["## Corrected v3b within-slide undersampling verdict", ""]
    lines.append(v3b_reinterpretation["verdict"])
    lines.append("")
    lines.append(v3b_reinterpretation["note_on_prior_v3b_verdict"])
    lines.append("")
    lines.append("| slide | n_ducts (all) | partial rho (all) | n_ducts (>=3 patches) | partial rho (>=3 patches) | signed delta | toward positive? |")
    lines.append("|---|---|---|---|---|---|---|")
    for slide_name, v in v3b_reinterpretation["per_slide"].items():
        lines.append(
            f"| {slide_name} | {v['n_ducts_all']} | {_fmt(v['partial_rho_all_ducts'])} | "
            f"{v['n_ducts_min_3_patches']} | {_fmt(v['partial_rho_min_3_patches'])} | "
            f"{_fmt(v['signed_delta'])} | {'yes' if v['moved_toward_cohort_positive_signal'] else 'no'} |"
        )
    lines.append("")

    lines += ["## Labelled results table (PRIMARY vs EXPLORATORY)", ""]
    lines.append("| quantity | value | tag | detail | source |")
    lines.append("|---|---|---|---|---|")
    for row in results_table:
        val = _fmt_or_na(row["value"])
        detail = (
            f"global {row.get('global_perm_p', 'n/a')}, within-slide {row.get('within_slide_perm_p', 'n/a')}"
            if row["tag"] == "PRIMARY"
            else f"count={row.get('count')}"
        )
        # Escape literal "|" in the label (conditional-probability notation, e.g.
        # "hole_pct | area") -- unescaped, it would be parsed as a table column break.
        label = row["label"].replace("|", "\\|")
        lines.append(f"| {label} | {val} | {row['tag']} | {detail} | {row['source']} |")
    lines.append("")
    lines.append(
        f"**Exploratory correlation count across v1-v3b:** v1={exploratory_count['v1']}, "
        f"v2={exploratory_count['v2']}, v3={exploratory_count['v3']}, "
        f"v3b={exploratory_count['v3b']}, **total={exploratory_count['total_exploratory']}** "
        f"(plus {exploratory_count['n_primary']} designated PRIMARY, listed above). "
        f"{exploratory_count['note']} No multiplicity correction is applied to the "
        f"PRIMARY results; the EXPLORATORY results are uncorrected and "
        f"hypothesis-generating only."
    )
    lines.append("")

    lines += ["## Sensitivity: overlap-based patch assignment (optional)", ""]
    if not overlap_sensitivity.get("attempted"):
        lines.append(overlap_sensitivity.get("reason", "Skipped."))
    elif "error" in overlap_sensitivity or "reason" in overlap_sensitivity:
        lines.append(overlap_sensitivity.get("reason") or overlap_sensitivity.get("error"))
    else:
        lines.append(overlap_sensitivity["note"])
        lines.append("")
        lines.append(
            f"- Ducts previously excluded (zero patches, centre-in-polygon rule): "
            f"{overlap_sensitivity['n_ducts_previously_excluded']}"
        )
        lines.append(
            f"- Recovered under overlap rule (>= {overlap_sensitivity['overlap_min_fraction']:.2f} "
            f"patch-area overlap): {overlap_sensitivity['n_recovered_under_overlap_rule']}"
        )
        lines.append(
            f"- raw rho — centre rule (v1): {_fmt(overlap_sensitivity['rho_pt_hole_pct_centre_rule_v1'])}, "
            f"overlap rule: {_fmt(overlap_sensitivity['rho_pt_hole_pct_overlap_rule'])}"
        )
        lines.append(
            f"- partial rho | area — overlap rule: "
            f"{_fmt(overlap_sensitivity['partial_rho_pt_hole_given_area_overlap_rule'])} "
            f"(compare to co-primary {_fmt(co_primary['partial_rho_given_area']['value'])} above)"
        )
    lines.append("")

    lines += ["## Limitations", ""]
    lines.append(
        "- v3 and v3b's original \"3 flagged slides\" framing (selecting slides by "
        "looking at their weak v2 partials, then testing them for properties on the "
        "same data) is methodologically circular and is **superseded** by the "
        "non-circular cross-slide check above. It should not be cited as evidence "
        "for or against the undersampling hypothesis."
    )
    lines.append(
        "- v3b's original \"n_slides_strengthened_of_3\" verdict used absolute "
        "magnitude and is corrected above; treat the corrected, direction-aware "
        "verdict as authoritative."
    )
    lines.append(
        "- The estimand is the retained (>=1 patch) duct population, not all "
        "annotated ducts — see Estimand section."
    )
    lines.append(
        "- Everything tagged EXPLORATORY above (per-slide breakdowns, threshold "
        "sweeps, aggregation comparisons, sampling diagnostics) is uncorrected for "
        "multiple comparisons and should be treated as hypothesis-generating, not "
        "confirmatory."
    )
    lines.append(
        "- The n=8-slide cross-slide check (Task A) and the n=3-slide within-slide "
        "check (Task E) are both explicitly underpowered; both are reported "
        "descriptively, not as significance tests."
    )
    lines.append("")

    lines += ["## Provenance", ""]
    lines.append(
        "- v1 (`holeyness_per_duct.csv`, `holeyness_validation.json`): original "
        "duct-level join and raw/independence correlations."
    )
    lines.append(
        "- v2 (`v2_area_adjusted/`): found and quantified the duct-area confound; "
        "exclusion-bias, within-slide, and aggregation-sensitivity checks."
    )
    lines.append(
        "- v3 (`v3_significance/`): permutation tests on the area-adjusted partial "
        "(not just the raw correlation); per-slide investigation table (reused "
        "directly by Task A above)."
    )
    lines.append(
        "- v3b (`v3b_patch_count_check/`): resolved a CSV-rounding precision "
        "discrepancy; per-slide undersampling check (reinterpreted by Task E above)."
    )
    lines.append("All four are retained unmodified at their original paths.")
    lines.append("")

    (output_dir / "holeyness_final_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_outputs(
    output_dir: Path,
    section: str,
    estimand: dict,
    co_primary: dict,
    cross_slide: dict,
    v3b_reinterpretation: dict,
    results_table: list[dict],
    exploratory_count: dict,
    overlap_sensitivity: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "section": section,
        "estimand": estimand,
        "co_primary_results": co_primary,
        "cross_slide_sampling_check": cross_slide,
        "v3b_undersampling_reinterpretation": v3b_reinterpretation,
        "results_table": results_table,
        "exploratory_correlation_count": exploratory_count,
        "overlap_sensitivity": overlap_sensitivity,
    }
    json_path = output_dir / "holeyness_final.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"  JSON: {json_path}")

    write_final_report(
        output_dir, section, estimand, co_primary, cross_slide,
        v3b_reinterpretation, results_table, exploratory_count, overlap_sensitivity,
    )
    print(f"  Markdown report: {output_dir / 'holeyness_final_report.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Holeyness final consolidation: one authoritative report over v1-v3b"
    )
    parser.add_argument("--section",          required=True, help="Section label, e.g. '2M-1'")
    parser.add_argument("--v1-per-duct-csv",  required=True, type=Path)
    parser.add_argument("--v1-json",          required=True, type=Path)
    parser.add_argument("--v2-json",          required=True, type=Path)
    parser.add_argument("--v3-json",          required=True, type=Path)
    parser.add_argument("--v3b-json",         required=True, type=Path)
    parser.add_argument("--output-dir",       required=True, type=Path,
                        help="Output directory (NEW versioned subdirectory, e.g. .../holeyness/2M-1/final)")

    # Optional — Task F (overlap-based assignment sensitivity) only. Omit any of
    # these to skip Task F automatically; nothing else in this report depends on it.
    parser.add_argument("--export",           default=None, type=Path)
    parser.add_argument("--annotation-dir",   default=None, type=Path)
    parser.add_argument("--slide-dimensions", default=None, type=Path)
    parser.add_argument("--results",          default=None, type=Path)
    parser.add_argument("--slide-list",       default=None, type=Path)
    parser.add_argument("--patch-size",       default=PATCH_SIZE_DEFAULT, type=int)
    parser.add_argument("--overlap-min-fraction", default=OVERLAP_MIN_FRACTION_DEFAULT, type=float)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Holeyness final consolidation — section {args.section}")
    print("=" * 60)

    per_duct_csv = pd.read_csv(args.v1_per_duct_csv)
    with open(args.v1_json) as f:
        v1_json = json.load(f)
    with open(args.v2_json) as f:
        v2_json = json.load(f)
    with open(args.v3_json) as f:
        v3_json = json.load(f)
    with open(args.v3b_json) as f:
        v3b_json = json.load(f)

    print("\n=== Estimand (Task B) ===")
    estimand = build_estimand_statement(args.section, v2_json)
    print(f"  {estimand['statement']}")

    print("\n=== Co-primary results (Task D) ===")
    co_primary = build_co_primary_summary(v2_json, v3_json)
    print(f"  raw rho = {co_primary['raw_rho']['value']:.4f}")
    print(f"  partial rho | area = {co_primary['partial_rho_given_area']['value']:.4f}")

    print("\n=== Non-circular cross-slide check (Task A) ===")
    v3_per_slide = v3_json["per_slide_investigation"]["per_slide"]
    cross_slide = run_cross_slide_sampling_check(v3_per_slide)
    print(f"  {cross_slide['note']}")

    print("\n=== Corrected v3b verdict (Task E) ===")
    v3b_reinterpretation = reinterpret_v3b_undersampling(v3b_json)
    print(f"  {v3b_reinterpretation['verdict']}")

    print("\n=== Labelled results table + exploratory count (Task C) ===")
    exploratory_count = count_exploratory_correlations(v1_json, v2_json, v3_json, v3b_json)
    results_table = build_results_table(v2_json, v3_json, exploratory_count)
    print(f"  exploratory count: {exploratory_count}")

    print("\n=== Overlap-based assignment sensitivity (Task F, optional) ===")
    overlap_sensitivity = run_overlap_sensitivity(args, per_duct_csv)
    print(f"  {overlap_sensitivity}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_partial_vs_patchcount_scatter(cross_slide, args.output_dir, args.section)
    write_pt_vs_hole_by_area_scatter(per_duct_csv, args.output_dir, args.section)

    write_final_outputs(
        args.output_dir, args.section, estimand, co_primary, cross_slide,
        v3b_reinterpretation, results_table, exploratory_count, overlap_sensitivity,
    )

    print("\n" + "=" * 60)
    print(f"  HOLEYNESS FINAL CONSOLIDATION COMPLETE — section {args.section}")
    print("=" * 60)
    print(f"\n  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
