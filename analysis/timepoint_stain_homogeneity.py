"""
Timepoint cohort: Stage B -- within-cohort stain homogeneity gate (HARD GATE).

The cross-cohort experiment (timepoint slides projected onto 2M-1, compared
against 2M-1) is CANCELLED -- see timepoint_stage2_stain_check.py and
PROJECT_STATE.md. The replacement design compares timepoint groups (4W, 7W,
8W, 12W) WITHIN the timepoint cohort itself. This module is that replacement
design's own hard gate: it tests whether staining is homogeneous ACROSS
TIMEPOINT GROUPS. If sub-batches within the timepoint cohort align with
timepoint, this design fails for the same underlying reason the cross-cohort
one did, and the project stops here -- no correction is proposed, mirroring
the restraint of the original gate.

Efficiency: runs on the COARSEST NDPI pyramid level (slide.level_count - 1),
not full-resolution PNGs, since converting ~30 slides at full resolution is
expensive and must not be committed to before this gate passes. Before
trusting that shortcut, it is validated against the 7 slides that already
have a full-resolution PNG (timepoint_x5_full): masked stain stats are
computed both ways and compared. If they disagree, this module HALTS rather
than trusting an unvalidated proxy for the remaining, unconverted slides.

Reuses (does not reimplement) from timepoint_stage2_stain_check.py:
  _tissue_mask, RGB_CHANNEL_NAMES, MIN_PLAUSIBLE_TISSUE_FRACTION,
  GATE_MEASURES, HEMATOXYLIN_GATE_MEASURES, MEASURES, _rank_biserial_mwu,
  _fmt, _normalize_stem, _png_path, compute_slide_stain_features.
That existing, already-executed hard-gate module is NOT modified -- this
module's local `_stain_features_from_rgb` reproduces only the small
(~15-line) glue that loops over channels and calls the imported mask/
hematoxylin primitives; every threshold and formula is imported, not retyped.
Also reuses `_safe_spearman` from analysis/holeyness.py for the
monotonic-trend check, and validation.morphological_features's
_deconvolve_hematoxylin / compute_hematoxylin_intensity (the exact functions
that produce the pipeline's own h_intensity feature).

Consumes Stage A's inventory (analysis/timepoint_cohort_inventory.py) for
mouse_id / timepoint_weeks / suffix_flag per slide, and the existing
stage2_reference_threshold.json (the project's own 2M-1-vs-2M-2 cross-section
confound, recomputed at slide level) as the reference bar -- no new number is
hardcoded.

Known data handling, per the replacement design's brief:
  - the 3 " 2"-suffix slides (unknown provenance) are included in Stage A's
    inventory but held out of PRIMARY stats here; every comparison is also
    run WITH them, reported side by side, never silently merged.
  - mouse 6072 contributes to both a 7W and a 12W group (confirmed staggered
    harvest) -- flagged via dual_timepoint_mice in the output, not treated as
    two independent mice.
  - mouse (not slide) is the unit of inference -- aggregate_to_mouse_level
    medians slide-level features within (mouse_id, timepoint_weeks) before
    any statistical comparison.

STOP AFTER THIS MODULE. Report stageB_stain_homogeneity.md/.json and await
explicit confirmation before Stage C (conversion of the remaining slides) is
even written, let alone submitted.

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_stain_homogeneity \\
      --stageA-inventory-json $SCRATCH/results/timepoint_cohort/stageA_inventory/stageA_inventory.json \\
      --converted-png-dir     $SCRATCH/data/timepoint_x5_full \\
      --reference-threshold-json \\
          $SCRATCH/results/timepoint_projection/stage2_reference_threshold/stage2_reference_threshold.json \\
      --output-dir            $SCRATCH/results/timepoint_cohort/stageB_stain_homogeneity
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from ..validation.morphological_features import (
    _deconvolve_hematoxylin,
    compute_hematoxylin_intensity,
)
from .holeyness import _safe_spearman
from .timepoint_stage2_stain_check import (
    GATE_MEASURES,
    HEMATOXYLIN_GATE_MEASURES,
    MEASURES,
    MIN_PLAUSIBLE_TISSUE_FRACTION,
    RGB_CHANNEL_NAMES,
    _fmt,
    _normalize_stem,
    _png_path,
    _rank_biserial_mwu,
    _tissue_mask,
    compute_slide_stain_features,
)

# Chosen convention for this module, not a value from the task brief -- flagged
# explicitly here and in the report so it's visible for adjustment. Below this
# Spearman correlation (coarse-NDPI-level vs full-res-PNG masked stats, n=7),
# a gate measure's coarse-level proxy is not trusted for the unconverted slides.
VALIDATION_RHO_THRESHOLD = 0.8

TIMEPOINT_GROUPS = [4, 7, 8, 12]


# ── Shared stain-feature computation (mirrors, does not modify, ─────────────
#    timepoint_stage2_stain_check.py::compute_slide_stain_features) ──────────

def _stain_features_from_rgb(rgb: np.ndarray) -> dict:
    """Same computation as compute_slide_stain_features's body, taking an
    already-loaded RGB array directly (no PNG-specific open/resize step) --
    used both for the coarse-NDPI path and to keep the two feature sources
    directly comparable in the validation check below."""
    mask = _tissue_mask(rgb)
    tissue_fraction = float(mask.mean())

    features = {
        "tissue_fraction": tissue_fraction,
        "tissue_fraction_implausible": bool(tissue_fraction < MIN_PLAUSIBLE_TISSUE_FRACTION),
    }
    for i, ch in enumerate(RGB_CHANNEL_NAMES):
        channel_u8 = rgb[:, :, i]
        masked_vals = channel_u8[mask].astype(np.float64)
        features[f"rgb_mean_{ch}_masked"] = float(np.mean(masked_vals))
        features[f"rgb_median_{ch}_masked"] = float(np.median(masked_vals))
        features[f"rgb_mean_{ch}_whole_image_unmasked"] = float(np.mean(channel_u8))

    h_channel = _deconvolve_hematoxylin(rgb)
    h_masked = h_channel[mask]
    features["h_intensity_mean_masked"] = float(np.mean(h_masked))
    features["h_intensity_median_masked"] = float(np.median(h_masked))
    features["h_intensity_whole_image_unmasked"] = compute_hematoxylin_intensity(h_channel)

    return features


def compute_stain_features_from_ndpi_coarse(ndpi_path: Path) -> dict:
    """Reads the coarsest NDPI pyramid level (cheap -- no full-resolution
    decode). Mirrors the exact coarse-level read pattern already used in
    timepoint_inventory.py::ndpi_scale_and_crop_check."""
    import openslide  # lazy import -- only needed on Narval, where it's module-loaded

    slide = openslide.OpenSlide(str(ndpi_path))
    try:
        level = slide.level_count - 1
        region = slide.read_region((0, 0), level, slide.level_dimensions[level])
        rgb = np.array(region.convert("RGB"))
    finally:
        slide.close()
    return _stain_features_from_rgb(rgb)


# ── Validation: coarse-NDPI proxy vs. existing full-res PNG ──────────────────

def validate_coarse_proxy(
    stems_with_ndpi_and_png: list[tuple[str, Path, Path]],
) -> dict:
    """For the 7 slides that have BOTH a raw NDPI and an already-converted
    full-width PNG, computes masked stain stats both ways and compares. Per
    gate measure: Spearman rho (n=7) AND mean absolute relative difference
    (correlation alone can mislead at n=7). agreement_ok requires
    rho >= VALIDATION_RHO_THRESHOLD."""
    coarse_vals: dict[str, list[float]] = {m: [] for m in GATE_MEASURES}
    png_vals: dict[str, list[float]] = {m: [] for m in GATE_MEASURES}
    per_slide = {}

    for stem, ndpi_path, png_path in stems_with_ndpi_and_png:
        coarse_feats = compute_stain_features_from_ndpi_coarse(ndpi_path)
        png_feats = compute_slide_stain_features(png_path)
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
        "n_slides": len(stems_with_ndpi_and_png),
        "rho_threshold": VALIDATION_RHO_THRESHOLD,
        "per_measure": per_measure,
        "per_slide": per_slide,
        "all_gate_measures_agree": all_ok,
    }


# ── Mouse-level aggregation ───────────────────────────────────────────────────

def aggregate_to_mouse_level(
    inventory_rows: list[dict],
    slide_features: dict[str, dict],
    exclude_suffix_slides: bool,
) -> list[dict]:
    """Groups by (mouse_id, timepoint_weeks) -- NOT just mouse_id, since mouse
    6072 legitimately produces two rows (7W and 12W from a confirmed
    staggered harvest of the same animal). Medians slide-level features
    within each group. has_suffix_slide / dual_timepoint_mouse are explicit
    output columns, never silently resolved."""
    groups: dict[tuple, list[dict]] = {}
    for row in inventory_rows:
        if not row["parse_ok"] or row["mouse_id"] is None or row["timepoint_weeks"] is None:
            continue
        if row["raw_stem"] not in slide_features:
            continue
        if exclude_suffix_slides and row["suffix_flag"]:
            continue
        key = (row["mouse_id"], row["timepoint_weeks"])
        groups.setdefault(key, []).append(row)

    mouse_ids_present = {mouse_id for (mouse_id, _weeks) in groups}
    mouse_id_counts = {}
    for (mouse_id, _weeks) in groups:
        mouse_id_counts[mouse_id] = mouse_id_counts.get(mouse_id, 0) + 1

    result = []
    for (mouse_id, weeks), rows in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        entry = {
            "mouse_id": mouse_id,
            "timepoint_weeks": weeks,
            "n_slides": len(rows),
            "slide_stems": [r["raw_stem"] for r in rows],
            "has_suffix_slide": any(r["suffix_flag"] for r in rows),
            "dual_timepoint_mouse": mouse_id_counts[mouse_id] > 1,
        }
        for m in MEASURES:
            vals = [slide_features[r["raw_stem"]][m] for r in rows]
            entry[f"median_{m}"] = float(np.median(vals))
        result.append(entry)
    return result


# ── Group comparisons ──────────────────────────────────────────────────────────

def pairwise_group_comparisons(mouse_rows: list[dict], reference_rank_biserial: float) -> dict:
    results = {}
    for w1, w2 in itertools.combinations(TIMEPOINT_GROUPS, 2):
        group1 = [r for r in mouse_rows if r["timepoint_weeks"] == w1]
        group2 = [r for r in mouse_rows if r["timepoint_weeks"] == w2]
        pair_key = f"{w1}Wvs{w2}W"
        if not group1 or not group2:
            results[pair_key] = {"n1": len(group1), "n2": len(group2), "skipped": True,
                                  "reason": "one or both groups have zero mice"}
            continue
        per_measure = {}
        for m in GATE_MEASURES:
            v1 = np.array([r[f"median_{m}"] for r in group1], dtype=float)
            v2 = np.array([r[f"median_{m}"] for r in group2], dtype=float)
            comparison = _rank_biserial_mwu(v1, v2)
            comparison["confounded_vs_reference_r"] = bool(
                np.isfinite(comparison["r_rb"]) and abs(comparison["r_rb"]) >= reference_rank_biserial
            )
            per_measure[m] = comparison
        results[pair_key] = {
            "n1": len(group1), "n2": len(group2), "skipped": False,
            "underpowered": bool(len(group1) < 4 or len(group2) < 4),
            "per_measure": per_measure,
        }
    return results


def spearman_weeks_trend(mouse_rows: list[dict], reference_rank_biserial: float) -> dict:
    weeks = np.array([r["timepoint_weeks"] for r in mouse_rows], dtype=float)
    results = {}
    for m in GATE_MEASURES:
        vals = np.array([r[f"median_{m}"] for r in mouse_rows], dtype=float)
        rho, p = _safe_spearman(weeks, vals)
        results[m] = {
            "rho": rho, "p": p, "n": len(mouse_rows),
            "confounded_vs_reference_r": bool(np.isfinite(rho) and abs(rho) >= reference_rank_biserial),
        }
    return results


def _any_confounded(pairwise: dict, trend: dict) -> bool:
    for pair_result in pairwise.values():
        if pair_result.get("skipped"):
            continue
        if any(v["confounded_vs_reference_r"] for v in pair_result["per_measure"].values()):
            return True
    if any(v["confounded_vs_reference_r"] for v in trend.values()):
        return True
    return False


def build_verdict(
    pairwise_excluding: dict, trend_excluding: dict,
    pairwise_including: dict, trend_including: dict,
) -> tuple[str, str]:
    confounded_excluding = _any_confounded(pairwise_excluding, trend_excluding)
    confounded_including = _any_confounded(pairwise_including, trend_including)

    if confounded_excluding:
        return "FAIL", (
            "At least one tissue-masked gate measure separates timepoint groups (pairwise or "
            "monotonic trend) at or above the reference threshold, in the PRIMARY comparison "
            "(excluding the 3 unknown-provenance ' 2'-suffix slides). The within-cohort design "
            "is confounded and the project stops here -- no correction is proposed."
        )
    if not confounded_including:
        return "PASS", (
            "No tissue-masked gate measure separates timepoint groups (pairwise or monotonic "
            "trend) at or above the reference threshold, in either the primary (excluding "
            "suffix slides) or the full (including suffix slides) comparison. The within-cohort "
            "design is viable."
        )
    return "AMBIGUOUS", (
        "The PRIMARY comparison (excluding suffix slides) shows no confound, but including the "
        "3 unknown-provenance ' 2'-suffix slides DOES trigger the reference threshold on at "
        "least one measure. This inconsistency is reported as ambiguous, not rounded to PASS -- "
        "resolve what the suffix slides are before treating the design as fully viable."
    )


# ── Reference threshold loading ───────────────────────────────────────────────

def resolve_reference_rank_biserial(reference_json: Path) -> float:
    if not reference_json.exists():
        sys.exit(
            f"ERROR: --reference-threshold-json not found at:\n  {reference_json}\n"
            "Run analysis/stage2_reference_threshold.py "
            "(jobs/run_stage2_reference_threshold.sh) first. There is no hardcoded fallback."
        )
    with open(reference_json) as f:
        data = json.load(f)
    if "reference_rank_biserial" not in data:
        sys.exit(f"ERROR: {reference_json} does not contain a 'reference_rank_biserial' key.")
    value = float(data["reference_rank_biserial"])
    print(f"  Loaded reference_rank_biserial={value} from {reference_json}")
    return value


# ── Output writers ────────────────────────────────────────────────────────────

def write_report(result: dict, output_dir: Path) -> None:
    lines = ["# Timepoint cohort — Stage B: within-cohort stain homogeneity gate (HARD GATE)", ""]

    val = result["png_vs_coarse_validation"]
    lines.append("## Coarse-NDPI-level vs. full-res-PNG validation")
    lines.append("")
    lines.append(
        f"n={val['n_slides']} slides with both a raw NDPI and an existing full-width PNG. "
        f"Agreement bar: Spearman rho >= {val['rho_threshold']} (chosen convention, not from "
        f"the task brief -- adjust if needed)."
    )
    lines.append("")
    lines.append("| measure | rho | p | mean abs relative diff | agreement OK |")
    lines.append("|---|---|---|---|---|")
    for m, v in val["per_measure"].items():
        lines.append(
            f"| {m} | {_fmt(v['rho'])} | {_fmt(v['p'])} | "
            f"{_fmt(v['mean_abs_relative_difference'])} | {v['agreement_ok']} |"
        )
    lines.append("")

    if not val["all_gate_measures_agree"]:
        lines.append(
            "**HALT: at least one gate measure fails the coarse-vs-PNG agreement bar above.** "
            "The coarse-NDPI-level proxy is NOT validated for use on the remaining, unconverted "
            "slides. No group comparison was run. Resolve this before re-running Stage B.\n"
        )
        (output_dir / "stageB_stain_homogeneity.md").write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append(
        f"**Reference rank-biserial threshold:** {result['reference_rank_biserial']:.4f} "
        "(from stage2_reference_threshold.json — the project's own 2M-1 vs 2M-2 cross-section "
        "confound, recomputed at slide level)."
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

    lines.append(
        f"**Group sizes are small (n=5,2,6,7 mice or similar) — all comparisons above are "
        f"descriptive; do not lean on p-values.**\n"
    )
    lines.append(f"**Dual-timepoint mice:** {result['dual_timepoint_mice']}\n")
    lines.append(f"**Suffix slides (unknown provenance):** {result['suffix_slides']}\n")

    lines.append(f"## Verdict: {result['verdict']}\n\n{result['verdict_rationale']}\n")

    (output_dir / "stageB_stain_homogeneity.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stageB_stain_homogeneity.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  JSON: {json_path}")
    write_report(result, output_dir)
    print(f"  Markdown report: {output_dir / 'stageB_stain_homogeneity.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timepoint cohort Stage B: within-cohort stain homogeneity hard gate"
    )
    parser.add_argument("--stageA-inventory-json", required=True, type=Path)
    parser.add_argument("--converted-png-dir", required=True, type=Path,
                        help="e.g. $SCRATCH/data/timepoint_x5_full -- the 7 already-converted slides")
    parser.add_argument("--reference-threshold-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print("  Timepoint cohort — Stage B: within-cohort stain homogeneity gate")
    print("=" * 60)

    reference_rank_biserial = resolve_reference_rank_biserial(args.reference_threshold_json)

    with open(args.stageA_inventory_json) as f:
        stageA = json.load(f)
    inventory_rows = stageA["slides"]
    print(f"\nLoaded {len(inventory_rows)} slides from Stage A inventory")

    openslide_ok_rows = [r for r in inventory_rows if r["parse_ok"] and r["opens_in_openslide"]]
    print(f"  {len(openslide_ok_rows)} slides parse OK and open in OpenSlide")

    # ── Validation: coarse-NDPI vs existing full-res PNG, for the 7 already-converted slides ──
    print("\n=== Validating coarse-NDPI proxy against existing full-res PNGs ===")
    validation_triples = []
    for r in openslide_ok_rows:
        png_path = _png_path(args.converted_png_dir, r["raw_stem"])
        if png_path.exists():
            validation_triples.append((r["raw_stem"], Path(r["source_dir"]) / f"{r['raw_stem']}.ndpi", png_path))
    print(f"  {len(validation_triples)} slides have both NDPI and converted PNG")
    validation = validate_coarse_proxy(validation_triples)
    for m, v in validation["per_measure"].items():
        print(f"  {m}: rho={_fmt(v['rho'])} agreement_ok={v['agreement_ok']}")

    if not validation["all_gate_measures_agree"]:
        print("\n*** HALT: coarse-NDPI proxy does not agree with full-res PNG on all gate measures.")
        print("*** Not proceeding to group comparisons on the remaining, unconverted slides.")
        result = {"png_vs_coarse_validation": validation, "verdict": "HALT",
                  "verdict_rationale": "Coarse-NDPI proxy failed validation -- see above."}
        write_outputs(result, args.output_dir)
        sys.exit(1)

    # ── Compute coarse-NDPI stain features for every openslide-readable slide ──
    print("\n=== Computing coarse-NDPI stain features for all slides ===")
    slide_features = {}
    for r in openslide_ok_rows:
        ndpi_path = Path(r["source_dir"]) / f"{r['raw_stem']}.ndpi"
        try:
            slide_features[r["raw_stem"]] = compute_stain_features_from_ndpi_coarse(ndpi_path)
            print(f"  {r['raw_stem']}: OK")
        except Exception as e:
            print(f"  {r['raw_stem']}: ERROR {e!r} -- excluded from comparison")

    mouse_to_weeks: dict[str, set] = {}
    for r in inventory_rows:
        if r["parse_ok"] and r["mouse_id"] is not None:
            mouse_to_weeks.setdefault(r["mouse_id"], set()).add(r["timepoint_weeks"])
    dual_timepoint_mice = sorted(m for m, weeks in mouse_to_weeks.items() if len(weeks) > 1)
    suffix_slides = [r["raw_stem"] for r in inventory_rows if r.get("suffix_flag")]

    print("\n=== Mouse-level aggregation ===")
    mouse_rows_excl = aggregate_to_mouse_level(inventory_rows, slide_features, exclude_suffix_slides=True)
    mouse_rows_incl = aggregate_to_mouse_level(inventory_rows, slide_features, exclude_suffix_slides=False)
    print(f"  excluding suffix slides: {len(mouse_rows_excl)} (mouse, timepoint) rows")
    print(f"  including suffix slides: {len(mouse_rows_incl)} (mouse, timepoint) rows")

    print("\n=== Group comparisons ===")
    pairwise_excl = pairwise_group_comparisons(mouse_rows_excl, reference_rank_biserial)
    pairwise_incl = pairwise_group_comparisons(mouse_rows_incl, reference_rank_biserial)
    trend_excl = spearman_weeks_trend(mouse_rows_excl, reference_rank_biserial)
    trend_incl = spearman_weeks_trend(mouse_rows_incl, reference_rank_biserial)

    verdict, rationale = build_verdict(pairwise_excl, trend_excl, pairwise_incl, trend_incl)
    print(f"\n  VERDICT: {verdict}")
    print(f"  {rationale}")

    result = {
        "png_vs_coarse_validation": validation,
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
    write_outputs(result, args.output_dir)

    print("\n" + "=" * 60)
    print(f"  STAGE B COMPLETE — VERDICT: {verdict}")
    print("=" * 60)
    print("\n  STOP HERE. Report stageB_stain_homogeneity.md and await explicit confirmation")
    print("  before Stage C (conversion of remaining slides) is written or run.")


if __name__ == "__main__":
    main()
