"""
Timepoint cohort: Stage B v2 -- within-cohort stain homogeneity gate, FULL
RESOLUTION (HARD GATE).

Stage B v1 (`timepoint_stain_homogeneity.py`, NOT modified here) computed its
gate measures via `compute_stain_features_from_ndpi_coarse` (the coarsest NDPI
pyramid level) for EVERY slide, and used a full-resolution PNG only for a
small validation check (n=7, since only 7 of ~32 slides had a converted PNG at
the time). That shortcut existed specifically so the project wouldn't have to
commit to converting ~30 slides at full resolution before this gate passed.

Stage C has since converted the remaining slides (22 of 24 succeeded; see
Stage A v2 for the corrected 29-slide usable cohort). Every usable slide now
has a full-resolution PNG in `timepoint_x5_full` -- the reason for the coarse
shortcut no longer applies. This module re-runs the SAME gate logic as Stage B
v1 (imported, not reimplemented) but computes slide-level stain features from
the full-resolution PNG for every slide, not the coarse NDPI proxy. This is a
FULL RE-VERIFICATION at full resolution and larger n, not the same coarse-
level run repeated -- see the "Comparison to Stage B v1" section of the
report, which is required output, not an afterthought.

This is a NEW, standalone module -- Stage B v1 is not modified, matching this
project's established convention (Stage B v1 itself reuses Stage 2's
primitives without editing Stage 2).

Reuses, unmodified:
  From timepoint_stain_homogeneity.py (Stage B v1): aggregate_to_mouse_level,
  pairwise_group_comparisons, spearman_weeks_trend, build_verdict,
  resolve_reference_rank_biserial, TIMEPOINT_GROUPS,
  compute_stain_features_from_ndpi_coarse, VALIDATION_RHO_THRESHOLD.
  From timepoint_stage2_stain_check.py: compute_slide_stain_features (the
  full-resolution-PNG feature function -- default downsample_factor=8,
  designed for exactly this whole-slide-image case), GATE_MEASURES,
  HEMATOXYLIN_GATE_MEASURES, MEASURES, _fmt, _normalize_stem, _png_path.
  From holeyness.py: _safe_spearman.

What changes vs Stage B v1:
  - The MAIN computation (the one that actually feeds pairwise_group_
    comparisons / spearman_weeks_trend / build_verdict) now calls
    compute_slide_stain_features(png_path) for every usable slide, not
    compute_stain_features_from_ndpi_coarse(ndpi_path).
  - An informational coarse-vs-full-res comparison is still reported (now
    over all ~29 usable slides instead of 7, since every one now has both a
    raw NDPI and a PNG) but does NOT gate this run -- Stage B v1 HALTed on
    validation disagreement because it needed to decide whether to trust the
    coarse proxy for slides that had no PNG yet; there are no such slides
    left in the corrected cohort, so nothing downstream depends on that
    agreement anymore. This is stated explicitly in the report rather than
    silently dropping the HALT behavior.
  - NOTE: this comparison reuses Stage B v1's `validate_coarse_proxy` LOGIC
    (rho + mean-abs-relative-difference formula, VALIDATION_RHO_THRESHOLD),
    but not the function itself -- `validate_coarse_proxy` reads each PNG
    itself, which would call the expensive `compute_slide_stain_features`
    (hematoxylin deconvolution) a SECOND time per slide, on top of the main
    computation's own call. `validate_coarse_vs_precomputed_fullres` below
    reuses the full-res features already computed by
    `compute_fullres_slide_features` and only reads the coarse NDPI side
    fresh -- same comparison, same primitives imported, half the full-res
    decode work. This mirrors the project's established pattern of
    reproducing a small (~20-line) piece of glue locally while importing the
    actual primitives (see Stage B v1's own docstring re: Stage 2).

Consumes Stage A v2's inventory (timepoint_cohort_inventory_v2.py) for the
corrected usable_slides list, and the existing stage2_reference_threshold.json
as the reference bar (loaded dynamically, no hardcoded fallback -- expected
value is reference_rank_biserial=0.3125 per the project's prior run, but this
module does not assume that number, it loads whatever is on disk).

Also loads Stage B v1's own output JSON (--prior-stageB-json) purely for the
required before/after comparison in the report -- never to recompute anything.

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_stain_homogeneity_v2 \\
      --stageA-inventory-json $SCRATCH/results/timepoint_cohort/stageA_inventory_v2/stageA_inventory_v2.json \\
      --converted-png-dir     $SCRATCH/data/timepoint_x5_full \\
      --reference-threshold-json \\
          $SCRATCH/results/timepoint_projection/stage2_reference_threshold/stage2_reference_threshold.json \\
      --prior-stageB-json \\
          $SCRATCH/results/timepoint_cohort/stageB_stain_homogeneity/stageB_stain_homogeneity.json \\
      --output-dir            $SCRATCH/results/timepoint_cohort/stageB_v2_fullres
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .holeyness import _safe_spearman
from .timepoint_stage2_stain_check import (
    GATE_MEASURES,
    _fmt,
    _png_path,
    compute_slide_stain_features,
)
from .timepoint_stain_homogeneity import (
    VALIDATION_RHO_THRESHOLD,
    aggregate_to_mouse_level,
    build_verdict,
    compute_stain_features_from_ndpi_coarse,
    pairwise_group_comparisons,
    resolve_reference_rank_biserial,
    spearman_weeks_trend,
)

# The specific measures the task brief calls out for direct before/after
# comparison against Stage B v1 -- reported with the most detail, though every
# gate measure is still in the full pairwise/trend tables below.
HEADLINE_COMPARISON_MEASURES = ["h_intensity_mean_masked", "h_intensity_median_masked"]
HEADLINE_COMPARISON_PAIR = "4Wvs12W"


# ── Main computation: full-res PNG features for every usable slide ───────────

def compute_fullres_slide_features(usable_rows: list[dict], converted_png_dir: Path) -> dict:
    """The substantive change vs Stage B v1: compute_slide_stain_features
    (full-resolution PNG) replaces compute_stain_features_from_ndpi_coarse for
    the numbers that actually drive the gate. Per-slide try/except mirrors
    Stage B v1's own robustness convention -- one bad PNG read must not crash
    an unattended, hours-long job."""
    slide_features = {}
    failed = {}
    for r in usable_rows:
        stem = r["raw_stem"]
        png_path = _png_path(converted_png_dir, stem)
        try:
            slide_features[stem] = compute_slide_stain_features(png_path)
            print(f"  {stem}: OK (full-res PNG)")
        except Exception as e:
            failed[stem] = repr(e)
            print(f"  {stem}: ERROR {e!r} -- excluded from comparison")
    return slide_features, failed


# ── Informational coarse-vs-full-res comparison (no redundant PNG read) ──────

def validate_coarse_vs_precomputed_fullres(
    usable_rows: list[dict], slide_features: dict[str, dict],
) -> dict:
    """Same comparison Stage B v1's validate_coarse_proxy makes (per gate
    measure: Spearman rho + mean absolute relative difference, agreement_ok
    at rho >= VALIDATION_RHO_THRESHOLD, both imported from Stage B v1 unmodified)
    -- but takes the full-res PNG features that compute_fullres_slide_features
    already computed instead of reading each PNG a second time. Only the
    coarse NDPI side is read fresh here. Per-slide try/except mirrors Stage B
    v1's own robustness convention -- one bad coarse read must not crash this
    (purely informational) comparison or the job around it."""
    coarse_vals: dict[str, list[float]] = {m: [] for m in GATE_MEASURES}
    png_vals: dict[str, list[float]] = {m: [] for m in GATE_MEASURES}
    per_slide = {}
    failed_slides = {}

    for r in usable_rows:
        stem = r["raw_stem"]
        if stem not in slide_features:
            continue  # already excluded upstream (full-res PNG read failed)
        ndpi_path = Path(r["source_dir"]) / f"{stem}.ndpi"
        try:
            coarse_feats = compute_stain_features_from_ndpi_coarse(ndpi_path)
        except Exception as e:
            failed_slides[stem] = repr(e)
            print(f"  {stem}: ERROR during coarse-proxy comparison {e!r} -- excluded")
            continue
        png_feats = slide_features[stem]
        per_slide[stem] = {"coarse": coarse_feats, "png": png_feats}
        for m in GATE_MEASURES:
            coarse_vals[m].append(coarse_feats[m])
            png_vals[m].append(png_feats[m])

    per_measure = {}
    all_ok = True
    for m in GATE_MEASURES:
        c = np.array(coarse_vals[m], dtype=float)
        p = np.array(png_vals[m], dtype=float)
        rho, pval = _safe_spearman(c, p)
        denom = np.where(np.abs(p) > 1e-9, np.abs(p), np.nan)
        mean_abs_rel_diff = float(np.nanmean(np.abs(c - p) / denom))
        agreement_ok = bool(np.isfinite(rho) and rho >= VALIDATION_RHO_THRESHOLD)
        all_ok = all_ok and agreement_ok
        per_measure[m] = {
            "rho": rho, "p": pval,
            "mean_abs_relative_difference": mean_abs_rel_diff,
            "agreement_ok": agreement_ok,
        }

    return {
        "n_slides": len(per_slide),
        "n_attempted": len(usable_rows),
        "failed_slides": failed_slides,
        "rho_threshold": VALIDATION_RHO_THRESHOLD,
        "per_measure": per_measure,
        "per_slide": per_slide,
        "all_gate_measures_agree": all_ok,
    }


# ── Comparison to Stage B v1 ──────────────────────────────────────────────────

def compare_to_v1(prior_result: dict, new_result: dict) -> dict:
    """Builds the required before/after comparison -- verdict, the two
    headline hematoxylin measures, and the 4Wvs12W pairwise comparison
    specifically (the best-powered, most consistent signal in the prior
    coarse-level run per the task brief). Defensive against missing keys
    (e.g. if v1 HALTed and never reached pairwise comparisons) -- reports
    'not available in prior run' rather than crashing."""
    comparison = {
        "prior_verdict": prior_result.get("verdict"),
        "new_verdict": new_result["verdict"],
        "verdict_changed": prior_result.get("verdict") != new_result["verdict"],
        "headline_pair": HEADLINE_COMPARISON_PAIR,
        "headline_measures": {},
    }

    prior_pairwise = prior_result.get("pairwise_group_comparisons", {}).get("excluding_suffix", {})
    new_pairwise = new_result["pairwise_group_comparisons"]["excluding_suffix"]
    prior_pair = prior_pairwise.get(HEADLINE_COMPARISON_PAIR)
    new_pair = new_pairwise.get(HEADLINE_COMPARISON_PAIR)

    for measure in HEADLINE_COMPARISON_MEASURES:
        prior_v = None
        if prior_pair and not prior_pair.get("skipped"):
            prior_v = prior_pair.get("per_measure", {}).get(measure)
        new_v = None
        if new_pair and not new_pair.get("skipped"):
            new_v = new_pair.get("per_measure", {}).get(measure)

        comparison["headline_measures"][measure] = {
            "prior_available": prior_v is not None,
            "new_available": new_v is not None,
            "prior_n1": prior_pair.get("n1") if prior_pair else None,
            "prior_n2": prior_pair.get("n2") if prior_pair else None,
            "new_n1": new_pair.get("n1") if new_pair else None,
            "new_n2": new_pair.get("n2") if new_pair else None,
            "prior_r_rb": prior_v.get("r_rb") if prior_v else None,
            "new_r_rb": new_v.get("r_rb") if new_v else None,
            "prior_confounded": prior_v.get("confounded_vs_reference_r") if prior_v else None,
            "new_confounded": new_v.get("confounded_vs_reference_r") if new_v else None,
        }

    return comparison


# ── Output writers ────────────────────────────────────────────────────────────

def write_report(result: dict, comparison: dict, output_dir: Path) -> None:
    lines = [
        "# Timepoint cohort — Stage B v2: within-cohort stain homogeneity gate, "
        "FULL RESOLUTION (HARD GATE)",
        "",
        "**This is a full re-verification at full resolution and larger n, not "
        "the same coarse-level run as Stage B v1.** Every usable slide's gate "
        "measures are computed from its full-resolution converted PNG "
        "(`compute_slide_stain_features`), not the coarse NDPI pyramid proxy "
        "Stage B v1 used for its main comparisons.",
        "",
    ]

    val = result["png_vs_coarse_validation_informational"]
    lines.append("## Coarse-NDPI-level vs. full-res-PNG agreement (INFORMATIONAL ONLY, not a gate)")
    lines.append("")
    lines.append(
        f"n={val['n_slides']} of {val['n_attempted']} usable slides successfully processed "
        f"both ways (all usable slides now have both a raw NDPI and a converted PNG, vs. "
        f"n=7 in Stage B v1). This is reported for documentation of how much the coarse "
        f"proxy would have differed -- it does NOT gate this run. Stage B v1 HALTed on "
        f"disagreement here because it needed to decide whether to trust the coarse proxy "
        f"for slides that had no PNG yet; every usable slide in this run already has a PNG, "
        f"so nothing downstream depends on this agreement."
    )
    if val["failed_slides"]:
        lines.append("")
        lines.append(f"**Failed during validation (excluded):** {val['failed_slides']}")
    lines.append("")
    lines.append("| measure | rho | mean abs relative diff | agreement OK (rho>=0.8) |")
    lines.append("|---|---|---|---|")
    for m, v in val["per_measure"].items():
        lines.append(
            f"| {m} | {_fmt(v['rho'])} | {_fmt(v['mean_abs_relative_difference'])} | "
            f"{v['agreement_ok']} |"
        )
    lines.append("")

    if result.get("feature_failures"):
        lines.append(
            f"**Slides that failed full-res feature computation (excluded from group "
            f"comparisons):** {result['feature_failures']}\n"
        )

    lines.append(
        f"**Reference rank-biserial threshold:** {result['reference_rank_biserial']:.4f} "
        "(loaded from stage2_reference_threshold.json — the project's own 2M-1 vs 2M-2 "
        "cross-section confound, recomputed at slide level; not hardcoded here)."
    )
    lines.append("")

    for label, key in [("excluding suffix slides (PRIMARY)", "excluding_suffix"),
                       ("including suffix slides", "including_suffix")]:
        lines.append(f"## Pairwise group comparisons — {label}")
        lines.append("")
        lines.append("| pair | n1 | n2 | underpowered | measure | r_rb | effect | confounded |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for pair_key, pair_result in result["pairwise_group_comparisons"][key].items():
            if pair_result.get("skipped"):
                lines.append(f"| {pair_key} | {pair_result['n1']} | {pair_result['n2']} | — | "
                              f"SKIPPED ({pair_result['reason']}) | | | |")
                continue
            for m, v in pair_result["per_measure"].items():
                lines.append(
                    f"| {pair_key} | {pair_result['n1']} | {pair_result['n2']} | "
                    f"{pair_result['underpowered']} | {m} | {_fmt(v['r_rb'])} | "
                    f"{v['effect_label']} | {v['confounded_vs_reference_r']} |"
                )
        lines.append("")

        lines.append(f"## Spearman weeks-trend — {label}")
        lines.append("")
        lines.append("| measure | rho | p | n | confounded |")
        lines.append("|---|---|---|---|---|")
        for m, v in result["spearman_weeks_trend"][key].items():
            lines.append(
                f"| {m} | {_fmt(v['rho'])} | {_fmt(v['p'])} | {v['n']} | "
                f"{v['confounded_vs_reference_r']} |"
            )
        lines.append("")

    lines.append(f"**Dual-timepoint mice:** {result['dual_timepoint_mice']}\n")
    lines.append(f"**Suffix slides (unknown provenance):** {result['suffix_slides']}\n")

    lines.append("## Comparison to Stage B v1 (coarse-NDPI proxy, n=7 validated)")
    lines.append("")
    lines.append(
        f"**Verdict:** v1 = `{comparison['prior_verdict']}` -> v2 (full-res) = "
        f"`{comparison['new_verdict']}`. "
        + ("**VERDICT CHANGED.**" if comparison["verdict_changed"] else "Verdict unchanged.")
    )
    lines.append("")
    lines.append(
        f"### {comparison['headline_pair']} pairwise comparison — headline hematoxylin measures"
    )
    lines.append("")
    lines.append(
        "This pair was the best-powered and most consistent signal in the prior "
        "coarse-level run per the task brief; reported here specifically, not just "
        "folded into the full table above."
    )
    lines.append("")
    lines.append("| measure | v1 n1/n2 | v1 r_rb | v1 confounded | v2 n1/n2 | v2 r_rb | v2 confounded |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in HEADLINE_COMPARISON_MEASURES:
        c = comparison["headline_measures"][m]
        v1_n = f"{c['prior_n1']}/{c['prior_n2']}" if c["prior_available"] else "n/a"
        v2_n = f"{c['new_n1']}/{c['new_n2']}" if c["new_available"] else "n/a"
        lines.append(
            f"| {m} | {v1_n} | {_fmt(c['prior_r_rb'])} | {c['prior_confounded']} | "
            f"{v2_n} | {_fmt(c['new_r_rb'])} | {c['new_confounded']} |"
        )
    lines.append("")
    lines.append(
        "**Group sizes are still small — all comparisons above are descriptive; do not "
        "lean on p-values.**\n"
    )

    lines.append(f"## Verdict: {result['verdict']}\n\n{result['verdict_rationale']}\n")

    (output_dir / "stageB_v2_fullres.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(result: dict, comparison: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_with_comparison = {**result, "comparison_to_v1": comparison}
    json_path = output_dir / "stageB_v2_fullres.json"
    with open(json_path, "w") as f:
        json.dump(result_with_comparison, f, indent=2, default=str)
    print(f"  JSON: {json_path}")
    write_report(result, comparison, output_dir)
    print(f"  Markdown report: {output_dir / 'stageB_v2_fullres.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timepoint cohort Stage B v2: within-cohort stain homogeneity hard gate, full resolution"
    )
    parser.add_argument("--stageA-inventory-json", required=True, type=Path,
                        help="Stage A v2 output: stageA_inventory_v2.json")
    parser.add_argument("--converted-png-dir", required=True, type=Path,
                        help="$SCRATCH/data/timepoint_x5_full")
    parser.add_argument("--reference-threshold-json", required=True, type=Path)
    parser.add_argument("--prior-stageB-json", required=True, type=Path,
                        help="Stage B v1 output, for the required before/after comparison")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print("  Timepoint cohort — Stage B v2: within-cohort stain homogeneity gate (FULL RES)")
    print("=" * 60)

    reference_rank_biserial = resolve_reference_rank_biserial(args.reference_threshold_json)

    with open(args.stageA_inventory_json) as f:
        stageA = json.load(f)
    all_slides = stageA["all_slides"]
    usable_slides = stageA["usable_slides"]
    print(f"\nLoaded Stage A v2 inventory: {len(all_slides)} total slides, "
          f"{len(usable_slides)} usable")

    # ── Main computation: full-res PNG features for every usable slide ──
    # Computed FIRST (not after a separate validation pass) so the informational
    # coarse-vs-full-res comparison below can reuse these instead of reading
    # each PNG a second time -- see validate_coarse_vs_precomputed_fullres's
    # docstring for why that redundant read mattered enough to avoid.
    print("\n=== Computing FULL-RESOLUTION stain features for all usable slides ===")
    slide_features, feature_failures = compute_fullres_slide_features(usable_slides, args.converted_png_dir)

    # ── Informational comparison: coarse-NDPI proxy vs already-computed full-res PNG ──
    print("\n=== Comparing coarse-NDPI proxy against full-res PNG (informational only) ===")
    validation = validate_coarse_vs_precomputed_fullres(usable_slides, slide_features)
    for m, v in validation["per_measure"].items():
        print(f"  {m}: rho={_fmt(v['rho'])} agreement_ok={v['agreement_ok']}")
    print("  (informational only -- does not gate this run)")

    mouse_to_weeks: dict[str, set] = {}
    for r in all_slides:
        if r["parse_ok"] and r["mouse_id"] is not None:
            mouse_to_weeks.setdefault(r["mouse_id"], set()).add(r["timepoint_weeks"])
    dual_timepoint_mice = sorted(m for m, weeks in mouse_to_weeks.items() if len(weeks) > 1)
    suffix_slides = [r["raw_stem"] for r in all_slides if r.get("suffix_flag")]

    print("\n=== Mouse-level aggregation ===")
    mouse_rows_excl = aggregate_to_mouse_level(all_slides, slide_features, exclude_suffix_slides=True)
    mouse_rows_incl = aggregate_to_mouse_level(all_slides, slide_features, exclude_suffix_slides=False)
    print(f"  excluding suffix slides: {len(mouse_rows_excl)} (mouse, timepoint) rows")
    print(f"  including suffix slides: {len(mouse_rows_incl)} (mouse, timepoint) rows")

    print("\n=== Group comparisons (full resolution) ===")
    pairwise_excl = pairwise_group_comparisons(mouse_rows_excl, reference_rank_biserial)
    pairwise_incl = pairwise_group_comparisons(mouse_rows_incl, reference_rank_biserial)
    trend_excl = spearman_weeks_trend(mouse_rows_excl, reference_rank_biserial)
    trend_incl = spearman_weeks_trend(mouse_rows_incl, reference_rank_biserial)

    verdict, rationale = build_verdict(pairwise_excl, trend_excl, pairwise_incl, trend_incl)
    print(f"\n  VERDICT: {verdict}")
    print(f"  {rationale}")

    result = {
        "png_vs_coarse_validation_informational": validation,
        "feature_failures": feature_failures,
        "mouse_level_features": {
            "excluding_suffix": mouse_rows_excl,
            "including_suffix": mouse_rows_incl,
        },
        "pairwise_group_comparisons": {
            "excluding_suffix": pairwise_excl,
            "including_suffix": pairwise_incl,
        },
        "spearman_weeks_trend": {
            "excluding_suffix": trend_excl,
            "including_suffix": trend_incl,
        },
        "reference_rank_biserial": reference_rank_biserial,
        "verdict": verdict,
        "verdict_rationale": rationale,
        "dual_timepoint_mice": dual_timepoint_mice,
        "suffix_slides": suffix_slides,
    }

    print("\n=== Loading Stage B v1 for required before/after comparison ===")
    with open(args.prior_stageB_json) as f:
        prior_result = json.load(f)
    comparison = compare_to_v1(prior_result, result)
    print(f"  v1 verdict: {comparison['prior_verdict']} -> v2 verdict: {comparison['new_verdict']} "
          f"(changed={comparison['verdict_changed']})")
    for m, c in comparison["headline_measures"].items():
        print(f"  {m}: v1 r_rb={_fmt(c['prior_r_rb'])} -> v2 r_rb={_fmt(c['new_r_rb'])}")

    write_outputs(result, comparison, args.output_dir)

    print("\n" + "=" * 60)
    print(f"  STAGE B v2 COMPLETE — VERDICT: {verdict}")
    print("=" * 60)
    print("\n  This is a full re-verification, not the coarse-level v1 run. Report both "
          "the new verdict and the comparison-to-v1 section. STOP HERE regardless of "
          "verdict -- no Stage D / projection work in this pass.")


if __name__ == "__main__":
    main()
