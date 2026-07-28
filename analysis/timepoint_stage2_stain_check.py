"""
Timepoint projection, Stage 2: stain batch check (CPU) -- HARD GATE.

Before any GPU work, tests whether the 8 new timepoint slides (4W/8W) differ in
staining from the 8 existing pipeline 2M-1 slides. If they do, by an amount
comparable to the project's own known cross-section staining confound, projecting
these slides onto the 2M-1 manifold and comparing pseudotime by timepoint would be
uninterpretable -- timepoint would be confounded with staining, not just with mouse.

Reuses, rather than reimplements:
  - `validation.morphological_features._deconvolve_hematoxylin` +
    `compute_hematoxylin_intensity` -- the EXACT function that produces the
    pipeline's own `h_intensity` feature, so this check is directly comparable to
    the existing (not-yet-populated-in-this-checkout) cross-section h_intensity
    confound, not a similar-but-different reimplementation.
  - The tissue-masking convention (`L < 230` in LAB space, fall back to all pixels
    if fewer than 100 tissue pixels) `qc/stain_qc.py::_lab_stats` already uses,
    reimplemented locally as a small self-contained mask helper (that function
    returns LAB stats, not a reusable mask + RGB stats, so a few lines are
    duplicated here rather than refactoring an existing, unrelated QC script).
  - The rank-biserial effect-size formula and sign convention already established
    in `diagnostics/audit_feature_diagnostics.py::d3_cross_section`
    (`r = 2*U/(n1*n2) - 1`, positive = group 1 > group 2 more often), reimplemented
    locally as a 3-line helper -- that script is a standalone diagnostic, not a
    shared library, so a formula match (not a cross-import) is what keeps this
    comparable.

Interpretation rule (stated explicitly, per the task): `--known-confound-r`
(default 0.71) is the effect-size bar. This number does NOT come from anywhere in
this repo as of writing (confirmed by exploration) -- it is supplied externally for
this task as the previously-measured cross-section h_intensity rank-biserial shift,
and is treated here as a given premise, not re-derived. Any hematoxylin measure
with |r| >= that bar means timepoint is confounded with staining and the Stage 3
projection would be uninterpretable. All measures are also labelled against the
project's own general effect-size thresholds (D3_LARGE_EFFECT=0.3,
D3_SMALL_EFFECT=0.1) for context.

Reads existing pipeline PNGs READ-ONLY (never modified). Writes only to a new
output directory.

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_stage2_stain_check \\
      --new-slide-list      ~/cancer_trajectory_atlas/jobs/slides_timepoint.txt \\
      --new-png-dir         $SCRATCH/data/timepoint_x5_cropped \\
      --existing-slide-list ~/cancer_trajectory_atlas/jobs/slides_section1.txt \\
      --existing-png-dir    $SCRATCH/data/MCF7_x5_cropped \\
      --known-confound-r    0.71 \\
      --output-dir          $SCRATCH/results/timepoint_projection/stage2_stain_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import mannwhitneyu

from ..validation.morphological_features import (
    _deconvolve_hematoxylin,
    compute_hematoxylin_intensity,
)

D3_LARGE_EFFECT = 0.3   # matches diagnostics/audit_feature_diagnostics.py
D3_SMALL_EFFECT = 0.1   # matches diagnostics/audit_feature_diagnostics.py

RGB_CHANNEL_NAMES = ["R", "G", "B"]


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:+.4f}"
    return str(v)


# ── Per-slide feature computation ─────────────────────────────────────────────

def _tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Same convention as qc/stain_qc.py::_lab_stats: LAB L<230, fall back to all
    pixels if fewer than 100 tissue pixels survive."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    mask = lab[:, :, 0] < 230
    if mask.sum() < 100:
        mask = np.ones(lab.shape[:2], dtype=bool)
    return mask


def compute_slide_stain_features(png_path: Path, downsample_factor: int = 1) -> dict:
    img = Image.open(png_path).convert("RGB")
    if downsample_factor > 1:
        img = img.resize(
            (img.width // downsample_factor, img.height // downsample_factor),
            Image.Resampling.LANCZOS,
        )
    rgb = np.array(img)
    mask = _tissue_mask(rgb)

    features = {}
    for i, ch in enumerate(RGB_CHANNEL_NAMES):
        vals = rgb[:, :, i][mask].astype(np.float64)
        features[f"rgb_mean_{ch}"] = float(np.mean(vals))
        features[f"rgb_median_{ch}"] = float(np.median(vals))

    h_channel = _deconvolve_hematoxylin(rgb)
    h_masked = h_channel[mask]
    features["h_intensity_mean_masked"] = float(np.mean(h_masked))
    features["h_intensity_median_masked"] = float(np.median(h_masked))
    # Exact pipeline function, unmasked -- for direct comparability with the
    # manuscript's existing (whole-patch, unmasked) h_intensity feature values.
    features["h_intensity_whole_image_unmasked"] = compute_hematoxylin_intensity(h_channel)

    return features


MEASURES = [
    "rgb_mean_R", "rgb_mean_G", "rgb_mean_B",
    "rgb_median_R", "rgb_median_G", "rgb_median_B",
    "h_intensity_mean_masked", "h_intensity_median_masked",
    "h_intensity_whole_image_unmasked",
]
HEMATOXYLIN_MEASURES = [
    "h_intensity_mean_masked", "h_intensity_median_masked", "h_intensity_whole_image_unmasked",
]


# ── Group comparison ──────────────────────────────────────────────────────────

def _rank_biserial_mwu(group1: np.ndarray, group2: np.ndarray) -> dict:
    """Same formula/sign convention as diagnostics/audit_feature_diagnostics.py's
    D3 cross-section check: r positive = group1 > group2 more often."""
    n1, n2 = len(group1), len(group2)
    stat, pvalue = mannwhitneyu(group1, group2, alternative="two-sided")
    r_rb = 2.0 * float(stat) / (n1 * n2) - 1.0
    if abs(r_rb) >= D3_LARGE_EFFECT:
        effect_label = "large" if abs(r_rb) >= 0.5 else "medium"
    elif abs(r_rb) >= D3_SMALL_EFFECT:
        effect_label = "small"
    else:
        effect_label = "negligible"
    return {
        "n1": n1, "n2": n2,
        "median_group1": float(np.median(group1)),
        "median_group2": float(np.median(group2)),
        "mwu_U": float(stat),
        "mwu_p": float(pvalue),
        "r_rb": r_rb,
        "effect_label": effect_label,
    }


def run_stain_batch_check(
    new_features: list[dict], existing_features: list[dict], known_confound_r: float,
) -> dict:
    results = {}
    for measure in MEASURES:
        v_new = np.array([f[measure] for f in new_features], dtype=float)
        v_existing = np.array([f[measure] for f in existing_features], dtype=float)
        comparison = _rank_biserial_mwu(v_new, v_existing)
        comparison["confounded_vs_known_r"] = bool(abs(comparison["r_rb"]) >= known_confound_r)
        results[measure] = comparison

    hematoxylin_confounded = any(results[m]["confounded_vs_known_r"] for m in HEMATOXYLIN_MEASURES)
    any_confounded = any(results[m]["confounded_vs_known_r"] for m in MEASURES)

    return {
        "known_confound_r": known_confound_r,
        "d3_large_effect_threshold": D3_LARGE_EFFECT,
        "d3_small_effect_threshold": D3_SMALL_EFFECT,
        "per_measure": results,
        "hematoxylin_confounded": hematoxylin_confounded,
        "any_measure_confounded": any_confounded,
    }


# ── Output writers ────────────────────────────────────────────────────────────

def write_report(
    check: dict,
    new_slides: list[str],
    existing_slides: list[str],
    output_dir: Path,
) -> None:
    lines = ["# Timepoint projection — Stage 2: stain batch check (HARD GATE)", ""]
    lines.append(
        f"Comparing {len(new_slides)} new timepoint slides against "
        f"{len(existing_slides)} existing 2M-1 pipeline slides, slide-level values "
        f"only (n={len(new_slides)} vs n={len(existing_slides)})."
    )
    lines.append("")
    lines.append(
        f"**Interpretation rule:** |rank-biserial r| >= {check['known_confound_r']:.2f} "
        f"on a hematoxylin measure (the previously-measured cross-section h_intensity "
        f"shift magnitude, supplied for this task -- not independently re-derived "
        f"in this repo) means timepoint is confounded with staining and the Stage 3 "
        f"projection test would be uninterpretable. All measures are also labelled "
        f"against this project's general effect-size thresholds "
        f"(large >= {check['d3_large_effect_threshold']}, "
        f"small >= {check['d3_small_effect_threshold']}) for context."
    )
    lines.append("")

    lines.append("| measure | median (new) | median (2M-1) | U | p | rank-biserial r | effect | vs known confound |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for measure in MEASURES:
        v = check["per_measure"][measure]
        confound_flag = "CONFOUNDED" if v["confounded_vs_known_r"] else "below bar"
        lines.append(
            f"| {measure} | {_fmt(v['median_group1'])} | {_fmt(v['median_group2'])} | "
            f"{v['mwu_U']:.0f} | {v['mwu_p']:.3e} | {_fmt(v['r_rb'])} | "
            f"{v['effect_label']} | {confound_flag} |"
        )
    lines.append("")

    if check["hematoxylin_confounded"]:
        verdict = (
            "**STOP — DO NOT PROCEED TO STAGE 3.** At least one hematoxylin "
            f"(h_intensity-equivalent) measure shows |r| >= {check['known_confound_r']:.2f}, "
            "matching or exceeding the known cross-section staining confound. "
            "Timepoint is confounded with staining, not just with mouse — the "
            "projection-based timepoint comparison would be uninterpretable as "
            "currently designed."
        )
    else:
        verdict = (
            "**No hematoxylin measure reaches the known-confound bar "
            f"(|r| >= {check['known_confound_r']:.2f}).** Stage 3 may proceed "
            "PENDING EXPLICIT USER CONFIRMATION of these real numbers — this "
            "verdict does not by itself authorize proceeding; report these results "
            "and await confirmation, per the task's hard gate."
        )
    lines.append(f"## Verdict\n\n{verdict}\n")

    if check["any_measure_confounded"] and not check["hematoxylin_confounded"]:
        lines.append(
            "**Note:** a non-hematoxylin RGB measure (but no hematoxylin measure) "
            "reached the known-confound bar. Since the known reference confound "
            "(r=0.71) was specifically measured on h_intensity, a shift in raw RGB "
            "alone is reported for completeness but does not by itself trigger the "
            "STOP verdict above — raw RGB is more sensitive to illumination/scanner "
            "differences than to stain chemistry per se.\n"
        )

    (output_dir / "stage2_stain_check.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    check: dict,
    new_features: list[dict],
    existing_features: list[dict],
    new_slides: list[str],
    existing_slides: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "new_slides": new_slides,
        "existing_slides": existing_slides,
        "per_slide_features": {
            "new": dict(zip(new_slides, new_features)),
            "existing": dict(zip(existing_slides, existing_features)),
        },
        "group_comparison": check,
    }
    json_path = output_dir / "stage2_stain_check.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON: {json_path}")
    write_report(check, new_slides, existing_slides, output_dir)
    print(f"  Markdown report: {output_dir / 'stage2_stain_check.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_slide_list(path: Path) -> list[str]:
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def _normalize_stem(name: str) -> str:
    """Existing pipeline slide lists (e.g. jobs/slides_section1.txt) store the
    post-conversion slide_name WITH the '_x5' suffix; a new bare-NDPI-stem list
    (e.g. jobs/slides_timepoint.txt, matching Stage 1's convention) would not.
    Normalize both to the bare stem so PNG paths are constructed consistently
    regardless of which convention a given slide-list file uses."""
    return name[:-3] if name.endswith("_x5") else name


def _png_path(png_dir: Path, slide_name: str) -> Path:
    return png_dir / f"{_normalize_stem(slide_name)}_x5.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timepoint projection Stage 2: stain batch check (hard gate before GPU work)"
    )
    parser.add_argument("--new-slide-list", required=True, type=Path,
                        help="Text file, one bare NDPI stem per line (e.g. '6069-4R-4W')")
    parser.add_argument("--new-png-dir", required=True, type=Path)
    parser.add_argument("--existing-slide-list", required=True, type=Path,
                        help="e.g. jobs/slides_section1.txt (the 8 2M-1 pipeline slides)")
    parser.add_argument("--existing-png-dir", required=True, type=Path,
                        help="Existing MCF7_x5_cropped dir -- READ ONLY, never modified")
    parser.add_argument("--known-confound-r", default=0.71, type=float,
                        help="Effect-size bar from the previously-measured cross-section "
                             "h_intensity shift (default 0.71; supplied externally, not "
                             "derived in this repo)")
    parser.add_argument("--downsample-factor", default=1, type=int,
                        help="Optional integer downsample before stain-feature computation "
                             "(default 1 = full resolution, same as pipeline PNGs)")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print("  Timepoint projection — Stage 2: stain batch check")
    print("=" * 60)

    new_slides = load_slide_list(args.new_slide_list)
    existing_slides = load_slide_list(args.existing_slide_list)
    print(f"\nNew slides ({len(new_slides)}): {new_slides}")
    print(f"Existing 2M-1 slides ({len(existing_slides)}): {existing_slides}")

    print("\n=== Computing per-slide stain features ===")
    new_features = []
    for stem in new_slides:
        png_path = _png_path(args.new_png_dir, stem)
        feats = compute_slide_stain_features(png_path, args.downsample_factor)
        print(f"  {stem}: {feats}")
        new_features.append(feats)

    existing_features = []
    for stem in existing_slides:
        png_path = _png_path(args.existing_png_dir, stem)
        feats = compute_slide_stain_features(png_path, args.downsample_factor)
        print(f"  {stem}: {feats}")
        existing_features.append(feats)

    print("\n=== Group comparison (n={} vs n={}) ===".format(len(new_slides), len(existing_slides)))
    check = run_stain_batch_check(new_features, existing_features, args.known_confound_r)
    for measure, v in check["per_measure"].items():
        print(f"  {measure}: r={v['r_rb']:+.4f} ({v['effect_label']}), "
              f"confounded_vs_known_r={v['confounded_vs_known_r']}")

    write_outputs(check, new_features, existing_features, new_slides, existing_slides, args.output_dir)

    print("\n" + "=" * 60)
    print("  STAGE 2 STAIN BATCH CHECK COMPLETE -- HARD GATE")
    print("=" * 60)
    print(f"\n  hematoxylin_confounded = {check['hematoxylin_confounded']}")
    print("  STOP HERE. Report these results and await confirmation before Stage 3.")
    print(f"\n  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
