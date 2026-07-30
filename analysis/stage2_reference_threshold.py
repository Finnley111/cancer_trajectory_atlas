"""
Timepoint projection, Stage 2 support: slide-level reference threshold (Task A).

Stage 2 (analysis/timepoint_stage2_stain_check.py) gates Stage 3 (GPU projection)
on whether timepoint slides are confounded with staining relative to the existing
2M-1 pipeline slides. That gate previously used `--known-confound-r 0.71`, a bar
taken from diagnostics/audit_feature_diagnostics.py's D3 check -- but D3 computed
rank-biserial on PER-PATCH h_intensity values (8244 2M-1 patches vs 10072 2M-2
patches), while Stage 2 compares PER-SLIDE summaries (n timepoint slides vs 8
2M-1 slides). Patch-level and slide-level rank-biserial are not the same
quantity: patch-level values carry within-slide plus between-slide variance,
slide-level medians carry only between-slide variance. Comparing a slide-level
effect size to a patch-level threshold was not a valid gate.

This module establishes the SLIDE-LEVEL reference threshold Stage 2 should
actually use, by recomputing the project's own known cross-section confound
(2M-1 vs 2M-2) at that same unit of analysis, using the identical tissue-masked
stain-feature method Stage 2 uses (imported from timepoint_stage2_stain_check.py,
not reimplemented, so the two are guaranteed comparable).

Reads ONLY the existing 16 original slides' already-converted, left-cropped PNGs
(jobs/slides_section1.txt = 8 2M-1 slides, jobs/slides_section2.txt = 8 2M-2
slides, $SCRATCH/data/MCF7_x5_cropped). Has NO dependency on the in-progress
no-crop timepoint conversion and can run before or independently of it. Writes
only to a new output directory.

Output: a single scalar `reference_rank_biserial` (the tissue-masked
h_intensity-mean rank-biserial |r| between the 8 2M-1 and 8 2M-2 slides, by
default), reported side-by-side with the old patch-level 0.71 and the explicit
magnitude of the difference, plus the full per-measure table (all of Stage 2's
MEASURES, masked and whole-image) for documentation. n=8 vs 8 is small --
this is reported descriptively, not as a high-confidence threshold.

CLI
---
  python -m cancer_trajectory_atlas.analysis.stage2_reference_threshold \\
      --section1-slide-list   ~/cancer_trajectory_atlas/jobs/slides_section1.txt \\
      --section2-slide-list   ~/cancer_trajectory_atlas/jobs/slides_section2.txt \\
      --png-dir               $SCRATCH/data/MCF7_x5_cropped \\
      --downsample-factor     8 \\
      --reference-measure     h_intensity_mean_masked \\
      --patch-level-reference-r 0.71 \\
      --output-dir            $SCRATCH/results/timepoint_projection/stage2_reference_threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .timepoint_stage2_stain_check import (
    D3_LARGE_EFFECT,
    D3_SMALL_EFFECT,
    GATE_MEASURES,
    HEMATOXYLIN_GATE_MEASURES,
    MEASURES,
    WHOLE_IMAGE_MEASURES,
    _fmt,
    _normalize_stem,
    _png_path,
    _rank_biserial_mwu,
    compute_slide_stain_features,
    load_slide_list,
)


def compute_all_slide_features(
    slide_stems: list[str], png_dir: Path, downsample_factor: int = 8,
) -> list[dict]:
    features = []
    for stem in slide_stems:
        png_path = _png_path(png_dir, stem)
        feats = compute_slide_stain_features(png_path, downsample_factor)
        print(f"  {stem}: tissue_fraction={feats['tissue_fraction']:.4f} "
              f"implausible={feats['tissue_fraction_implausible']}")
        features.append(feats)
    return features


def compare_sections(
    section1_features: list[dict], section2_features: list[dict],
) -> dict:
    """Slide-level 2M-1 (section1) vs 2M-2 (section2) comparison for every
    measure Stage 2 knows about -- reuses _rank_biserial_mwu directly, not a
    reimplementation."""
    results = {}
    for measure in MEASURES:
        v1 = np.array([f[measure] for f in section1_features], dtype=float)
        v2 = np.array([f[measure] for f in section2_features], dtype=float)
        results[measure] = _rank_biserial_mwu(v1, v2)
    return results


def build_reference_result(
    per_measure: dict,
    reference_measure: str,
    patch_level_reference_r: float,
    section1_features: list[dict],
    section2_features: list[dict],
) -> dict:
    ref = per_measure[reference_measure]
    reference_rank_biserial = abs(ref["r_rb"]) if np.isfinite(ref["r_rb"]) else None
    delta = (
        reference_rank_biserial - patch_level_reference_r
        if reference_rank_biserial is not None else None
    )

    implausible = [
        f["tissue_fraction"] for f in (section1_features + section2_features)
        if f["tissue_fraction_implausible"]
    ]

    return {
        "reference_measure": reference_measure,
        "reference_measure_signed_r": ref["r_rb"],
        "reference_rank_biserial": reference_rank_biserial,
        "patch_level_reference_r": patch_level_reference_r,
        "delta_slide_level_minus_patch_level": delta,
        "n_section1": ref["n1"],
        "n_section2": ref["n2"],
        "small_sample_caveat": (
            f"n={ref['n1']} vs {ref['n2']} slides. This reference is DESCRIPTIVE, not a "
            "high-confidence threshold -- at this sample size a single slide's stain "
            "characteristics can move the estimate substantially. Report it alongside "
            "the test statistic, not as a precise cutoff."
        ),
        "n_implausible_tissue_fraction_slides": len(implausible),
        "per_measure": per_measure,
    }


# ── Output writers ────────────────────────────────────────────────────────────

def write_report(
    result: dict,
    section1_stems: list[str],
    section2_stems: list[str],
    section1_features: list[dict],
    section2_features: list[dict],
    output_dir: Path,
) -> None:
    lines = [
        "# Stage 2 reference threshold — slide-level 2M-1 vs 2M-2 recomputation (Task A)",
        "",
    ]
    lines.append(
        f"Recomputes the project's known cross-section staining confound "
        f"(2M-1 vs 2M-2, n={len(section1_stems)} vs {len(section2_stems)} slides) at "
        f"SLIDE level, using the same tissue-masked stain-feature method Stage 2 uses, "
        f"so the gate threshold is measured at the same unit of analysis it is applied to."
    )
    lines.append("")

    r = result["reference_rank_biserial"]
    p = result["patch_level_reference_r"]
    lines.append("## Reference threshold — slide-level vs. the old patch-level number")
    lines.append("")
    lines.append(f"- **Reference measure:** `{result['reference_measure']}` "
                 f"(signed r = {_fmt(result['reference_measure_signed_r'])})")
    lines.append(f"- **New slide-level reference (|r|):** {_fmt(r) if r is not None else 'n/a'}")
    lines.append(f"- **Old patch-level reference (diagnostics/audit_feature_diagnostics.py D3):** {p:.4f}")
    if r is not None:
        lines.append(
            f"- **Delta (slide-level − patch-level):** {result['delta_slide_level_minus_patch_level']:+.4f} "
            f"— the two numbers are NOT interchangeable (different unit of analysis, "
            f"different variance components); this delta is reported so the substitution "
            f"is documented, not silent."
        )
    lines.append(f"- {result['small_sample_caveat']}")
    lines.append("")

    lines.append("## Gate measures (tissue-masked — what Stage 2's gate uses)")
    lines.append("")
    lines.append("| measure | median (2M-1) | median (2M-2) | n1/n2 | U | p | rank-biserial r | effect |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for measure in GATE_MEASURES:
        v = result["per_measure"][measure]
        marker = " **(reference)**" if measure == result["reference_measure"] else ""
        lines.append(
            f"| {measure}{marker} | {_fmt(v['median_group1'])} | {_fmt(v['median_group2'])} | "
            f"{v['n1']}/{v['n2']} | {_fmt(v['mwu_U'])} | {_fmt(v['mwu_p'])} | {_fmt(v['r_rb'])} | "
            f"{v['effect_label']} |"
        )
    lines.append("")

    lines.append("## Whole-image, unmasked measures (comparison only)")
    lines.append("")
    lines.append("| measure | median (2M-1) | median (2M-2) | n1/n2 | U | p | rank-biserial r | effect |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for measure in WHOLE_IMAGE_MEASURES:
        v = result["per_measure"][measure]
        lines.append(
            f"| {measure} | {_fmt(v['median_group1'])} | {_fmt(v['median_group2'])} | "
            f"{v['n1']}/{v['n2']} | {_fmt(v['mwu_U'])} | {_fmt(v['mwu_p'])} | {_fmt(v['r_rb'])} | "
            f"{v['effect_label']} |"
        )
    lines.append("")

    lines.append("## Tissue-fraction audit (masking sanity check)")
    lines.append("")
    lines.append("| slide | section | tissue fraction | implausible? |")
    lines.append("|---|---|---|---|")
    for stem, f in zip(section1_stems, section1_features):
        flag = "**YES**" if f["tissue_fraction_implausible"] else "no"
        lines.append(f"| {stem} | 2M-1 | {_fmt(f['tissue_fraction'])} | {flag} |")
    for stem, f in zip(section2_stems, section2_features):
        flag = "**YES**" if f["tissue_fraction_implausible"] else "no"
        lines.append(f"| {stem} | 2M-2 | {_fmt(f['tissue_fraction'])} | {flag} |")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if r is None:
        lines.append(
            "**UNDETERMINED.** The reference measure produced a non-finite rank-biserial "
            "value (insufficient finite per-slide values on one or both sides) — Stage 2 "
            "cannot get a usable default from this run. Investigate the tissue-fraction "
            "audit table above before retrying."
        )
    else:
        lines.append(
            f"Stage 2 will default to **reference_rank_biserial = {r:.4f}** "
            f"(measure: `{result['reference_measure']}`) via "
            f"`--reference-threshold-json` pointing at this run's JSON output, replacing "
            f"the previous patch-level 0.71. {result['small_sample_caveat']}"
        )
    lines.append("")

    (output_dir / "stage2_reference_threshold.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    result: dict,
    section1_stems: list[str],
    section2_stems: list[str],
    section1_features: list[dict],
    section2_features: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_result = {
        "section1_slides": section1_stems,
        "section2_slides": section2_stems,
        "per_slide_features": {
            "section1_2M-1": dict(zip(section1_stems, section1_features)),
            "section2_2M-2": dict(zip(section2_stems, section2_features)),
        },
        **result,
    }
    json_path = output_dir / "stage2_reference_threshold.json"
    with open(json_path, "w") as f:
        json.dump(full_result, f, indent=2)
    print(f"  JSON: {json_path}")
    write_report(result, section1_stems, section2_stems, section1_features, section2_features, output_dir)
    print(f"  Markdown report: {output_dir / 'stage2_reference_threshold.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 support (Task A): slide-level 2M-1 vs 2M-2 reference threshold, "
                     "recomputed at the same unit of analysis Stage 2's gate uses"
    )
    parser.add_argument("--section1-slide-list", required=True, type=Path,
                        help="e.g. jobs/slides_section1.txt (8 2M-1 slides)")
    parser.add_argument("--section2-slide-list", required=True, type=Path,
                        help="e.g. jobs/slides_section2.txt (8 2M-2 slides)")
    parser.add_argument("--png-dir", required=True, type=Path,
                        help="Existing MCF7_x5_cropped dir -- READ ONLY, never modified")
    parser.add_argument("--downsample-factor", default=8, type=int,
                        help="Integer downsample before stain-feature computation (default 8 -- "
                             "these are whole-slide images, not 112x112 patches; full resolution "
                             "OOM'd a 32G job by feeding rgb2hed a tens-of-thousands-of-pixels "
                             "array. Pass 1 only if you have a lot of memory to spare)")
    parser.add_argument("--reference-measure", default="h_intensity_mean_masked",
                        choices=HEMATOXYLIN_GATE_MEASURES,
                        help="Which gate measure's slide-level |r| becomes Stage 2's default "
                             "reference threshold (default: h_intensity_mean_masked, mirroring "
                             "D3's original mean-based h_intensity)")
    parser.add_argument("--patch-level-reference-r", default=0.71, type=float,
                        help="The old patch-level D3 number, reported for side-by-side "
                             "comparison only -- no longer used as a gate anywhere")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print("  Stage 2 reference threshold — slide-level 2M-1 vs 2M-2 (Task A)")
    print("=" * 60)

    section1_stems = [_normalize_stem(s) for s in load_slide_list(args.section1_slide_list)]
    section2_stems = [_normalize_stem(s) for s in load_slide_list(args.section2_slide_list)]
    print(f"\n2M-1 slides ({len(section1_stems)}): {section1_stems}")
    print(f"2M-2 slides ({len(section2_stems)}): {section2_stems}")

    print("\n=== Computing per-slide tissue-masked stain features ===")
    print("-- 2M-1 --")
    section1_features = compute_all_slide_features(section1_stems, args.png_dir, args.downsample_factor)
    print("-- 2M-2 --")
    section2_features = compute_all_slide_features(section2_stems, args.png_dir, args.downsample_factor)

    print("\n=== Slide-level comparison (2M-1 vs 2M-2) ===")
    per_measure = compare_sections(section1_features, section2_features)
    for measure, v in per_measure.items():
        gate_tag = "[GATE]" if measure in GATE_MEASURES else "[comparison only]"
        print(f"  {measure} {gate_tag}: r={_fmt(v['r_rb'])} ({v['effect_label']})")

    result = build_reference_result(
        per_measure, args.reference_measure, args.patch_level_reference_r,
        section1_features, section2_features,
    )

    write_outputs(result, section1_stems, section2_stems, section1_features, section2_features, args.output_dir)

    print("\n" + "=" * 60)
    print("  STAGE 2 REFERENCE THRESHOLD COMPLETE")
    print("=" * 60)
    r = result["reference_rank_biserial"]
    print(f"\n  reference_rank_biserial = {r}")
    print(f"  (was patch-level 0.71; delta = {result['delta_slide_level_minus_patch_level']})")
    print(f"  n_implausible_tissue_fraction_slides = {result['n_implausible_tissue_fraction_slides']}")
    print(f"\n  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
