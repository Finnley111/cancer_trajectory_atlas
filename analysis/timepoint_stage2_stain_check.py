"""
Timepoint projection, Stage 2: stain batch check (CPU) -- HARD GATE.

Before any GPU work, tests whether the timepoint slides (4W/8W) differ in
staining from the existing pipeline 2M-1 slides. If they do, by an amount
comparable to the project's own known cross-section staining confound, projecting
these slides onto the 2M-1 manifold and comparing pseudotime by timepoint would be
uninterpretable -- timepoint would be confounded with staining, not just with mouse.

A prior run of this module was PROVISIONAL and is VOID. Two methodological
problems have been fixed since:

  PROBLEM 1 -- BACKGROUND DILUTION. The previous `_tissue_mask` used
  qc/stain_qc.py's LAB (L<230) convention, which is a different, looser rule
  than what the pipeline itself uses to decide "is this tissue" during real
  patch extraction. Full-width (no-crop) timepoint PNGs contain two mounted
  tissue pieces, the gap between them, and margins, while the left-cropped
  original 2M-1 PNGs contain one piece -- systematically different background
  fractions between the two batches would make a loosely-masked or whole-image
  stat compare background amount, not stain chemistry. Fixed by rebuilding the
  tissue mask from features/patching.py's ACTUAL patch-extraction criteria
  (`_has_tissue_hsv`'s HSV saturation/value rule, `_is_mostly_white`'s
  all-RGB-channel white rule), reimplemented at per-pixel/whole-image scale
  (those two functions return one bool per 112x112 patch via a fraction
  threshold; here the same per-pixel formulas build a continuous mask instead).
  Thresholds are pulled programmatically from those functions' own defaults
  (via `inspect.signature`), not re-hardcoded, so they can't silently drift.
  Unmasked whole-image stats are still computed and reported, clearly labelled,
  so the magnitude of the background effect stays documented rather than hidden
  -- but they no longer drive the STOP/proceed gate (see WHOLE_IMAGE_MEASURES
  below; the previous version had `h_intensity_whole_image_unmasked` inside the
  gate-driving measure set, which was exactly this bug).

  PROBLEM 2 -- WRONG UNIT OF ANALYSIS FOR THE GATE. The previous
  `--known-confound-r 0.71` came from diagnostics/audit_feature_diagnostics.py's
  D3 check, computed on PER-PATCH h_intensity values (8244 2M-1 patches vs 10072
  2M-2 patches). This module compares PER-SLIDE summaries (n timepoint slides vs
  8 2M-1 slides). Patch-level and slide-level rank-biserial are not the same
  quantity -- patch-level carries within-slide plus between-slide variance,
  slide-level medians carry only between-slide variance. Fixed by replacing the
  hardcoded 0.71 with a slide-level reference threshold, computed by the
  companion `analysis/stage2_reference_threshold.py` module (2M-1 vs 2M-2 at
  slide level, using the SAME masked-stats method as this module) and consumed
  here dynamically -- see --reference-rank-biserial / --reference-threshold-json
  below. No number is hardcoded as a fallback.

Reuses, rather than reimplements:
  - `validation.morphological_features._deconvolve_hematoxylin` +
    `compute_hematoxylin_intensity` -- the EXACT function that produces the
    pipeline's own `h_intensity` feature.
  - `features.patching._has_tissue_hsv` / `_is_mostly_white` -- the EXACT
    per-patch tissue-detection criteria the pipeline uses, reimplemented here at
    per-pixel/whole-image scale (see PROBLEM 1 above). Threshold VALUES are read
    off those functions' defaults programmatically, never re-typed as new
    literals.
  - The rank-biserial effect-size formula and sign convention already established
    in `diagnostics/audit_feature_diagnostics.py::d3_cross_section`
    (`r = 2*U/(n1*n2) - 1`, positive = group 1 > group 2 more often), reimplemented
    locally as a small helper -- that script is a standalone diagnostic, not a
    shared library, so a formula match (not a cross-import) is what keeps this
    comparable. Also mirrors its finite-value filtering (`v = v[np.isfinite(v)]`)
    so one slide with a degenerate mask can't crash a whole comparison.

Reads existing pipeline PNGs READ-ONLY (never modified). Writes only to a new
output directory.

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_stage2_stain_check \\
      --new-slide-list      ~/cancer_trajectory_atlas/jobs/slides_timepoint.txt \\
      --new-png-dir         $SCRATCH/data/timepoint_x5_full \\
      --existing-slide-list ~/cancer_trajectory_atlas/jobs/slides_section1.txt \\
      --existing-png-dir    $SCRATCH/data/MCF7_x5_cropped \\
      --reference-threshold-json \\
          $SCRATCH/results/timepoint_projection/stage2_reference_threshold/stage2_reference_threshold.json \\
      --output-dir          $SCRATCH/results/timepoint_projection/stage2_stain_check
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import mannwhitneyu

from ..validation.morphological_features import (
    _deconvolve_hematoxylin,
    compute_hematoxylin_intensity,
)
from ..features.patching import _has_tissue_hsv, _is_mostly_white

D3_LARGE_EFFECT = 0.3   # matches diagnostics/audit_feature_diagnostics.py
D3_SMALL_EFFECT = 0.1   # matches diagnostics/audit_feature_diagnostics.py

RGB_CHANNEL_NAMES = ["R", "G", "B"]

# Pulled programmatically from features/patching.py's own defaults, rather than
# re-typed as new literals, so this can never silently drift from what the
# pipeline actually uses to decide "is this tissue" during patch extraction.
_HSV_DEFAULTS = inspect.signature(_has_tissue_hsv).parameters
_WHITE_DEFAULTS = inspect.signature(_is_mostly_white).parameters
SAT_THRESH = _HSV_DEFAULTS["sat_thresh"].default      # 15
VAL_THRESH = _HSV_DEFAULTS["val_thresh"].default      # 230
WHITE_THRESH = _WHITE_DEFAULTS["white_thresh"].default  # 220

# Sanity floor only -- NOT a scientific threshold. A tissue mask below this
# fraction most likely means the mask failed (e.g. an all-background image)
# rather than that the slide genuinely has almost no tissue; flagged rather
# than silently used.
MIN_PLAUSIBLE_TISSUE_FRACTION = 0.02


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:+.4f}"
    return str(v)


# ── Per-slide feature computation ─────────────────────────────────────────────

def _tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixel-level tissue mask built from the SAME thresholds
    features/patching.py applies per-patch during real extraction
    (_has_tissue_hsv's sat/val thresholds, _is_mostly_white's white threshold).
    Those functions return one bool per 112x112 patch via a fraction rule; this
    reimplements their per-pixel formulas at whole-image scale (PIL's
    .convert("HSV"), the same conversion _has_tissue_hsv uses internally -- not
    a different colorspace library) to get a continuous boolean mask instead of
    a single yes/no. No all-pixels fallback for a small/empty mask -- that would
    silently reintroduce the whole-image background-dilution problem this mask
    exists to fix. A genuinely empty mask just leaves masked stats as `nan`
    (numpy's normal behavior for an empty slice); callers report
    tissue_fraction and flag it as implausible rather than papering over it."""
    hsv = np.array(Image.fromarray(rgb).convert("HSV"))
    has_color = hsv[:, :, 1] > SAT_THRESH
    is_dense = hsv[:, :, 2] < VAL_THRESH
    not_white = ~np.all(rgb > WHITE_THRESH, axis=-1)
    return has_color & is_dense & not_white


def compute_slide_stain_features(png_path: Path, downsample_factor: int = 8) -> dict:
    """downsample_factor default is 8, NOT 1 -- these are whole-slide images
    (tens of thousands of pixels per side), not the 112x112 patches
    _deconvolve_hematoxylin/rgb2hed were designed for. rgb2hed upcasts its
    entire input to float64 internally; at full resolution this OOM'd a 32G
    job on a single slide (confirmed on Narval). A slide-level mean/median
    tissue-masked stain statistic does not need per-pixel resolution --
    downsampling by 8 still leaves millions of pixels per slide, which is
    ample for a stable mean/median, while cutting peak memory by ~64x."""
    img = Image.open(png_path).convert("RGB")
    if downsample_factor > 1:
        img = img.resize(
            (img.width // downsample_factor, img.height // downsample_factor),
            Image.Resampling.LANCZOS,
        )
    rgb = np.array(img)
    mask = _tissue_mask(rgb)
    tissue_fraction = float(mask.mean())

    features = {
        "tissue_fraction": tissue_fraction,
        "tissue_fraction_implausible": bool(tissue_fraction < MIN_PLAUSIBLE_TISSUE_FRACTION),
    }
    for i, ch in enumerate(RGB_CHANNEL_NAMES):
        # Mask/mean on the uint8 view first, THEN cast only the (much smaller)
        # masked subset to float64 -- avoids ever materializing a full-image
        # float64 copy per channel.
        channel_u8 = rgb[:, :, i]
        masked_vals = channel_u8[mask].astype(np.float64)
        features[f"rgb_mean_{ch}_masked"] = float(np.mean(masked_vals))
        features[f"rgb_median_{ch}_masked"] = float(np.median(masked_vals))
        # Whole-image, unmasked -- secondary/comparison quantity only (see
        # WHOLE_IMAGE_MEASURES); documents the background-dilution magnitude,
        # does not drive any gate.
        features[f"rgb_mean_{ch}_whole_image_unmasked"] = float(np.mean(channel_u8))

    h_channel = _deconvolve_hematoxylin(rgb)
    h_masked = h_channel[mask]
    features["h_intensity_mean_masked"] = float(np.mean(h_masked))
    features["h_intensity_median_masked"] = float(np.median(h_masked))
    # Exact pipeline function, unmasked -- secondary/comparison quantity only,
    # for direct comparability with the manuscript's existing (whole-patch,
    # unmasked) h_intensity feature values. Does not drive the gate.
    features["h_intensity_whole_image_unmasked"] = compute_hematoxylin_intensity(h_channel)

    return features


# Tissue-masked measures -- these are what the pipeline's own tissue-detection
# criteria consider "tissue", so these are the scientifically meaningful ones.
RGB_MASKED_MEASURES = [f"rgb_mean_{ch}_masked" for ch in RGB_CHANNEL_NAMES] + \
                      [f"rgb_median_{ch}_masked" for ch in RGB_CHANNEL_NAMES]
HEMATOXYLIN_GATE_MEASURES = ["h_intensity_mean_masked", "h_intensity_median_masked"]
GATE_MEASURES = RGB_MASKED_MEASURES + HEMATOXYLIN_GATE_MEASURES

# Whole-image, unmasked measures -- reported for documentation (how much does
# background dilution matter) but NEVER used to drive any gate boolean.
WHOLE_IMAGE_MEASURES = [f"rgb_mean_{ch}_whole_image_unmasked" for ch in RGB_CHANNEL_NAMES] + \
                       ["h_intensity_whole_image_unmasked"]

# Full audit table -- everything computed, gate and non-gate alike.
MEASURES = GATE_MEASURES + WHOLE_IMAGE_MEASURES


# ── Group comparison ──────────────────────────────────────────────────────────

def _rank_biserial_mwu(group1: np.ndarray, group2: np.ndarray) -> dict:
    """Same formula/sign convention as diagnostics/audit_feature_diagnostics.py's
    D3 cross-section check: r positive = group1 > group2 more often. Also
    mirrors its finite-value filtering so one slide with a degenerate
    (all-nan) masked measure can't crash the whole comparison -- it's simply
    dropped from that measure's n, which is reported."""
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    group1 = group1[np.isfinite(group1)]
    group2 = group2[np.isfinite(group2)]
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return {
            "n1": n1, "n2": n2,
            "median_group1": float("nan"), "median_group2": float("nan"),
            "mwu_U": float("nan"), "mwu_p": float("nan"),
            "r_rb": float("nan"), "effect_label": "n/a (insufficient finite values)",
        }
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
    new_features: list[dict], existing_features: list[dict], reference_rank_biserial: float,
) -> dict:
    results = {}
    for measure in MEASURES:
        v_new = np.array([f[measure] for f in new_features], dtype=float)
        v_existing = np.array([f[measure] for f in existing_features], dtype=float)
        comparison = _rank_biserial_mwu(v_new, v_existing)
        is_gate_measure = measure in GATE_MEASURES
        comparison["is_gate_measure"] = is_gate_measure
        comparison["confounded_vs_reference_r"] = bool(
            is_gate_measure and np.isfinite(comparison["r_rb"])
            and abs(comparison["r_rb"]) >= reference_rank_biserial
        )
        results[measure] = comparison

    hematoxylin_confounded = any(
        results[m]["confounded_vs_reference_r"] for m in HEMATOXYLIN_GATE_MEASURES
    )
    any_measure_confounded = any(results[m]["confounded_vs_reference_r"] for m in GATE_MEASURES)

    implausible_new = [f.get("tissue_fraction_implausible", False) for f in new_features]
    implausible_existing = [f.get("tissue_fraction_implausible", False) for f in existing_features]

    return {
        "reference_rank_biserial": reference_rank_biserial,
        "d3_large_effect_threshold": D3_LARGE_EFFECT,
        "d3_small_effect_threshold": D3_SMALL_EFFECT,
        "per_measure": results,
        "hematoxylin_confounded": hematoxylin_confounded,
        "any_measure_confounded": any_measure_confounded,
        "any_implausible_tissue_fraction": bool(any(implausible_new) or any(implausible_existing)),
    }


# ── Output writers ────────────────────────────────────────────────────────────

def write_report(
    check: dict,
    new_features: list[dict],
    existing_features: list[dict],
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
        f"**Interpretation rule:** |rank-biserial r| >= {check['reference_rank_biserial']:.4f} "
        f"on a tissue-masked hematoxylin measure (the slide-level reference threshold "
        f"computed by analysis/stage2_reference_threshold.py from the project's own "
        f"2M-1 vs 2M-2 cross-section confound, at the SAME unit of analysis this gate "
        f"uses -- not the previous patch-level 0.71) means timepoint is confounded with "
        f"staining and the Stage 3 projection test would be uninterpretable. Only "
        f"TISSUE-MASKED measures drive this gate (see below); whole-image/unmasked "
        f"measures are reported for comparison only. All measures are also labelled "
        f"against this project's general effect-size thresholds "
        f"(large >= {check['d3_large_effect_threshold']}, "
        f"small >= {check['d3_small_effect_threshold']}) for context."
    )
    lines.append("")

    lines.append("## Tissue-masked measures (drive the gate)")
    lines.append("")
    lines.append("| measure | median (new) | median (2M-1) | n (new/existing) | U | p | rank-biserial r | effect | vs reference |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for measure in GATE_MEASURES:
        v = check["per_measure"][measure]
        confound_flag = "CONFOUNDED" if v["confounded_vs_reference_r"] else "below bar"
        lines.append(
            f"| {measure} | {_fmt(v['median_group1'])} | {_fmt(v['median_group2'])} | "
            f"{v['n1']}/{v['n2']} | {_fmt(v['mwu_U'])} | {_fmt(v['mwu_p'])} | {_fmt(v['r_rb'])} | "
            f"{v['effect_label']} | {confound_flag} |"
        )
    lines.append("")

    lines.append("## Whole-image, unmasked measures (comparison only -- do NOT drive the gate)")
    lines.append("")
    lines.append(
        "Reported so the magnitude of background dilution is documented, not hidden. "
        "A large shift here that is NOT mirrored in the masked table above is expected "
        "and consistent with a background-fraction difference between full-width and "
        "left-cropped PNGs, not a stain-chemistry difference."
    )
    lines.append("")
    lines.append("| measure | median (new) | median (2M-1) | n (new/existing) | U | p | rank-biserial r | effect |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for measure in WHOLE_IMAGE_MEASURES:
        v = check["per_measure"][measure]
        lines.append(
            f"| {measure} | {_fmt(v['median_group1'])} | {_fmt(v['median_group2'])} | "
            f"{v['n1']}/{v['n2']} | {_fmt(v['mwu_U'])} | {_fmt(v['mwu_p'])} | {_fmt(v['r_rb'])} | "
            f"{v['effect_label']} |"
        )
    lines.append("")

    lines.append("## Tissue-fraction audit (masking sanity check)")
    lines.append("")
    lines.append(
        f"Sanity floor: tissue_fraction < {MIN_PLAUSIBLE_TISSUE_FRACTION} is flagged as "
        f"IMPLAUSIBLE (likely a masking failure, not a real near-zero-tissue slide)."
    )
    lines.append("")
    lines.append("| slide | group | tissue fraction | implausible? |")
    lines.append("|---|---|---|---|")
    for stem, f in zip(new_slides, new_features):
        flag = "**YES**" if f["tissue_fraction_implausible"] else "no"
        lines.append(f"| {stem} | new (timepoint) | {_fmt(f['tissue_fraction'])} | {flag} |")
    for stem, f in zip(existing_slides, existing_features):
        flag = "**YES**" if f["tissue_fraction_implausible"] else "no"
        lines.append(f"| {stem} | existing (2M-1) | {_fmt(f['tissue_fraction'])} | {flag} |")
    lines.append("")

    if check["hematoxylin_confounded"]:
        verdict = (
            "**STOP — DO NOT PROCEED TO STAGE 3.** At least one tissue-masked "
            f"hematoxylin measure shows |r| >= {check['reference_rank_biserial']:.4f}, "
            "matching or exceeding the slide-level reference threshold (the project's "
            "own 2M-1 vs 2M-2 cross-section confound, recomputed at slide level). "
            "Timepoint is confounded with staining, not just with mouse — the "
            "projection-based timepoint comparison would be uninterpretable as "
            "currently designed."
        )
    else:
        verdict = (
            "**No tissue-masked hematoxylin measure reaches the reference bar "
            f"(|r| >= {check['reference_rank_biserial']:.4f}).** Stage 3 may proceed "
            "PENDING EXPLICIT USER CONFIRMATION of these real numbers — this "
            "verdict does not by itself authorize proceeding; report these results "
            "and await confirmation, per the task's hard gate."
        )
    lines.append(f"## Verdict\n\n{verdict}\n")

    if check["any_measure_confounded"] and not check["hematoxylin_confounded"]:
        lines.append(
            "**Note:** a tissue-masked raw RGB measure (but no hematoxylin measure) "
            "reached the reference bar. Since the reference threshold was specifically "
            "computed on h_intensity, a shift in raw RGB alone is reported for "
            "completeness but does not by itself trigger the STOP verdict above — raw "
            "RGB is more sensitive to illumination/scanner differences than to stain "
            "chemistry per se.\n"
        )

    if check["any_implausible_tissue_fraction"]:
        lines.append(
            "**Note:** at least one slide's tissue fraction is flagged IMPLAUSIBLE "
            "(see audit table above). Its masked statistics may be unreliable — "
            "inspect that slide directly before trusting the gate verdict.\n"
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
    write_report(check, new_features, existing_features, new_slides, existing_slides, output_dir)
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


def resolve_reference_rank_biserial(
    explicit: float | None, reference_json: Path,
) -> float:
    """Explicit --reference-rank-biserial wins if given. Otherwise load it from
    Task A's output JSON. No hardcoded numeric fallback -- if neither is
    available, this is a hard failure, not a silent default."""
    if explicit is not None:
        print(f"  Using explicit --reference-rank-biserial override: {explicit}")
        return explicit
    if not reference_json.exists():
        sys.exit(
            f"ERROR: no --reference-rank-biserial given, and --reference-threshold-json "
            f"not found at:\n  {reference_json}\n"
            "Run analysis/stage2_reference_threshold.py (jobs/run_stage2_reference_threshold.sh) "
            "first, or pass --reference-rank-biserial explicitly. There is no hardcoded "
            "fallback threshold."
        )
    with open(reference_json) as f:
        data = json.load(f)
    if "reference_rank_biserial" not in data:
        sys.exit(
            f"ERROR: {reference_json} does not contain a 'reference_rank_biserial' key. "
            "Is this really a stage2_reference_threshold.json output file?"
        )
    value = float(data["reference_rank_biserial"])
    print(f"  Loaded reference_rank_biserial={value} from {reference_json}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timepoint projection Stage 2: stain batch check (hard gate before GPU work)"
    )
    parser.add_argument("--new-slide-list", required=True, type=Path,
                        help="Text file, one bare NDPI stem per line (e.g. '6070-4L-4W')")
    parser.add_argument("--new-png-dir", required=True, type=Path,
                        help="e.g. $SCRATCH/data/timepoint_x5_full (corrected, no-crop) "
                             "or $SCRATCH/data/timepoint_x5_cropped (provisional/void)")
    parser.add_argument("--existing-slide-list", required=True, type=Path,
                        help="e.g. jobs/slides_section1.txt (the 8 2M-1 pipeline slides)")
    parser.add_argument("--existing-png-dir", required=True, type=Path,
                        help="Existing MCF7_x5_cropped dir -- READ ONLY, never modified")
    parser.add_argument("--reference-rank-biserial", default=None, type=float,
                        help="Explicit slide-level reference threshold override. If not "
                             "given, loaded from --reference-threshold-json. No hardcoded "
                             "numeric fallback.")
    parser.add_argument("--reference-threshold-json", required=True, type=Path,
                        help="Output of analysis/stage2_reference_threshold.py, e.g. "
                             "$SCRATCH/results/timepoint_projection/stage2_reference_threshold/"
                             "stage2_reference_threshold.json")
    parser.add_argument("--downsample-factor", default=8, type=int,
                        help="Integer downsample before stain-feature computation (default 8 -- "
                             "these are whole-slide images, not 112x112 patches; full resolution "
                             "OOM'd a 32G job by feeding rgb2hed a tens-of-thousands-of-pixels "
                             "array. Pass 1 only if you have a lot of memory to spare)")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print("  Timepoint projection — Stage 2: stain batch check")
    print("=" * 60)

    reference_rank_biserial = resolve_reference_rank_biserial(
        args.reference_rank_biserial, args.reference_threshold_json
    )

    new_slides = load_slide_list(args.new_slide_list)
    existing_slides = load_slide_list(args.existing_slide_list)
    print(f"\nNew slides ({len(new_slides)}): {new_slides}")
    print(f"Existing 2M-1 slides ({len(existing_slides)}): {existing_slides}")

    print("\n=== Computing per-slide stain features (tissue-masked) ===")
    new_features = []
    for stem in new_slides:
        png_path = _png_path(args.new_png_dir, stem)
        feats = compute_slide_stain_features(png_path, args.downsample_factor)
        print(f"  {stem}: tissue_fraction={feats['tissue_fraction']:.4f} "
              f"implausible={feats['tissue_fraction_implausible']}")
        new_features.append(feats)

    existing_features = []
    for stem in existing_slides:
        png_path = _png_path(args.existing_png_dir, stem)
        feats = compute_slide_stain_features(png_path, args.downsample_factor)
        print(f"  {stem}: tissue_fraction={feats['tissue_fraction']:.4f} "
              f"implausible={feats['tissue_fraction_implausible']}")
        existing_features.append(feats)

    print("\n=== Group comparison (n={} vs n={}) ===".format(len(new_slides), len(existing_slides)))
    check = run_stain_batch_check(new_features, existing_features, reference_rank_biserial)
    for measure, v in check["per_measure"].items():
        gate_tag = "[GATE]" if v["is_gate_measure"] else "[comparison only]"
        print(f"  {measure} {gate_tag}: r={_fmt(v['r_rb'])} ({v['effect_label']}), "
              f"confounded_vs_reference_r={v['confounded_vs_reference_r']}")

    write_outputs(check, new_features, existing_features, new_slides, existing_slides, args.output_dir)

    print("\n" + "=" * 60)
    print("  STAGE 2 STAIN BATCH CHECK COMPLETE -- HARD GATE")
    print("=" * 60)
    print(f"\n  hematoxylin_confounded = {check['hematoxylin_confounded']}")
    print(f"  any_implausible_tissue_fraction = {check['any_implausible_tissue_fraction']}")
    print("  STOP HERE. Report these results and await confirmation before Stage 3.")
    print(f"\n  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
