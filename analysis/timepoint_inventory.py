"""
Timepoint projection, Stage 1: convert + inventory.

8 additional annotated slides exist at timepoints (4W, 8W) different from the 16
single-timepoint pipeline slides, from different mice, never run through the
pipeline. This module is a pure read/report inventory step run AFTER converting
the 8 raw NDPI files to cropped PNGs (via the existing, unmodified
`run_all.py --convert` CLI — no new conversion code needed; pointing `--png-dir` at
a brand-new directory automatically gives that directory its own
`slide_dimensions.json` sidecar, so the existing `MCF7_x5_cropped` sidecar is never
touched).

Per slide, reports:
  - dimensions from the new (separate) slide_dimensions.json sidecar
  - an `ndpi_scale` cross-check: re-reads the raw NDPI's level-0 dimensions directly
    and compares `level0_width * ndpi_scale` against the sidecar's
    `original_full_width` (should match by construction; a mismatch means the
    conversion job used a different ndpi_scale than expected)
  - whether a matching raw annotation file exists in `data/annotations/` (checked
    computationally, not assumed — as of writing, none exist for these 8 slides,
    but the check must still run in case that changes)
  - a left-crop-assumption diagnostic: unlike the pipeline's original slides, there
    is no annotation-based cross-check available for these new slides, so this
    cannot be fully automatically verified. Instead, this computes an honest,
    partial proxy: an HSV tissue-fraction (same S>15, V<230 criteria
    `features/patching.py`'s `_has_tissue_hsv` uses for patches, reimplemented here
    at whole-image scale) for the left half vs. the discarded right half, read at
    the NDPI's coarsest pyramid level (cheap). A right-half tissue fraction that
    isn't small relative to the left is flagged for MANUAL VISUAL INSPECTION rather
    than asserted as fine either way.

Does NOT modify any existing pipeline output — writes only to a NEW output
directory, reads the raw NDPI files and the NEW sidecar this session's Stage-1
conversion produced.

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_inventory \\
      --slide-list       ~/cancer_trajectory_atlas/jobs/slides_timepoint.txt \\
      --ndpi-dir         $SCRATCH/data/timepoint_ndpi \\
      --png-dir          $SCRATCH/data/timepoint_x5_cropped \\
      --slide-dimensions $SCRATCH/data/timepoint_x5_cropped/slide_dimensions.json \\
      --annotation-dir   ~/cancer_trajectory_atlas/data/annotations \\
      --ndpi-scale       1.0 \\
      --output-dir       $SCRATCH/results/timepoint_projection/stage1_convert
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.color import rgb2hsv

RIGHT_OVER_LEFT_FLAG_THRESHOLD = 0.20  # right-half tissue frac > 20% of left's -> flag


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ── Tissue-fraction heuristic (pure numpy/skimage, no OpenSlide dependency) ───

def tissue_fraction_hsv(rgb: np.ndarray) -> float:
    """Fraction of pixels that are both saturated (S > 15/255) and non-bright
    (V < 230/255) -- the same HSV tissue criterion `features/patching.py`'s
    `_has_tissue_hsv` applies per-patch, applied here at whole-image scale."""
    hsv = rgb2hsv(rgb)  # skimage returns S, V in [0, 1]
    s_thresh = 15.0 / 255.0
    v_thresh = 230.0 / 255.0
    tissue_mask = (hsv[:, :, 1] > s_thresh) & (hsv[:, :, 2] < v_thresh)
    return float(np.mean(tissue_mask))


def left_right_tissue_check(rgb_full: np.ndarray) -> dict:
    """Given a full (uncropped, both-copies) slide image array, compute the HSV
    tissue fraction of the left half (kept by the pipeline's crop) vs. the right
    half (discarded) -- an honest, partial diagnostic for whether the "two
    side-by-side duplicate copies" assumption the pipeline's crop relies on
    actually holds for this slide. Does not assert the assumption holds either
    way; flags for manual review when the right half isn't clearly near-empty."""
    w = rgb_full.shape[1]
    left = rgb_full[:, : w // 2, :]
    right = rgb_full[:, w // 2 :, :]
    left_frac = tissue_fraction_hsv(left)
    right_frac = tissue_fraction_hsv(right)
    ratio = (right_frac / left_frac) if left_frac > 1e-9 else float("inf") if right_frac > 0 else 0.0
    return {
        "left_tissue_fraction": left_frac,
        "right_tissue_fraction": right_frac,
        "right_over_left_ratio": ratio,
        "needs_manual_review": bool(ratio > RIGHT_OVER_LEFT_FLAG_THRESHOLD),
    }


# ── OpenSlide-dependent per-slide checks (not exercised locally -- no OpenSlide
#    test fixture available off Narval; kept thin and isolated for that reason) ──

def ndpi_scale_and_crop_check(ndpi_path: Path, ndpi_scale: float, sidecar_dims: dict | None) -> dict:
    """Re-reads the raw NDPI's coarsest pyramid level directly (cheap -- no
    full-res read) to (a) cross-check ndpi_scale against the sidecar's recorded
    original_full_width, and (b) run left_right_tissue_check on that same coarse
    read. Returns both results together since they share one OpenSlide open()."""
    import openslide  # lazy import -- only needed on Narval, where it's module-loaded

    slide = openslide.OpenSlide(str(ndpi_path))
    try:
        raw_w0, raw_h0 = slide.level_dimensions[0]
        coarsest_level = slide.level_count - 1
        coarse_w, coarse_h = slide.level_dimensions[coarsest_level]
        region = slide.read_region((0, 0), coarsest_level, (coarse_w, coarse_h))
        rgb = np.array(region.convert("RGB"))
    finally:
        slide.close()

    expected_full_w = raw_w0 * ndpi_scale
    sidecar_full_w = sidecar_dims.get("original_full_width") if sidecar_dims else None
    # Sidecar width was computed from a possibly-int-truncated resize; allow a
    # small relative tolerance rather than requiring exact equality.
    scale_match = (
        sidecar_full_w is not None
        and abs(sidecar_full_w - expected_full_w) <= max(2, 0.01 * expected_full_w)
    )

    return {
        "raw_level0_width": raw_w0,
        "raw_level0_height": raw_h0,
        "ndpi_scale_used": ndpi_scale,
        "expected_full_width": expected_full_w,
        "sidecar_full_width": sidecar_full_w,
        "ndpi_scale_confirmed": bool(scale_match),
        "left_right_tissue_check": left_right_tissue_check(rgb),
    }


# ── Main per-slide inventory ──────────────────────────────────────────────────

def load_slide_list(path: Path) -> list[str]:
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def build_inventory(
    slide_stems: list[str],
    ndpi_dir: Path,
    png_dir: Path,
    slide_dimensions: dict,
    annotation_dir: Path,
    ndpi_scale: float,
) -> list[dict]:
    rows = []
    for stem in slide_stems:
        png_path = png_dir / f"{stem}_x5.png"
        sidecar_key = f"{stem}_x5.png"
        dims = slide_dimensions.get(sidecar_key)
        annotation_path = annotation_dir / f"{stem}.geojson"
        ndpi_path = ndpi_dir / f"{stem}.ndpi"

        row = {
            "slide_stem": stem,
            "png_exists": png_path.exists(),
            "dims": dims,
            "annotation_exists_geojson": annotation_path.exists(),
            "ndpi_path_exists": ndpi_path.exists(),
        }

        if ndpi_path.exists():
            try:
                row["ndpi_check"] = ndpi_scale_and_crop_check(ndpi_path, ndpi_scale, dims)
                row["ndpi_check_error"] = None
            except Exception as e:
                row["ndpi_check"] = None
                row["ndpi_check_error"] = repr(e)
        else:
            row["ndpi_check"] = None
            row["ndpi_check_error"] = "raw NDPI file not found -- cannot run left-crop/scale check"

        rows.append(row)
    return rows


# ── Output writers ────────────────────────────────────────────────────────────

def write_report(rows: list[dict], output_dir: Path) -> None:
    lines = ["# Timepoint projection — Stage 1: convert + inventory", ""]

    n_png_missing = sum(1 for r in rows if not r["png_exists"])
    n_annotation_present = sum(1 for r in rows if r["annotation_exists_geojson"])
    n_review_needed = sum(
        1 for r in rows
        if r["ndpi_check"] and r["ndpi_check"]["left_right_tissue_check"]["needs_manual_review"]
    )
    n_scale_mismatch = sum(
        1 for r in rows if r["ndpi_check"] and not r["ndpi_check"]["ndpi_scale_confirmed"]
    )

    lines.append(
        f"**Summary:** {len(rows)} slides inventoried. "
        f"{n_png_missing} missing converted PNG. "
        f"{n_annotation_present}/{len(rows)} have a matching raw annotation in "
        f"`data/annotations/` (none expected as of writing — this is a computed "
        f"check, not an assumption). "
        f"{n_scale_mismatch} ndpi_scale mismatches. "
        f"{n_review_needed} slides flagged for MANUAL VISUAL REVIEW of the "
        f"left-crop assumption."
    )
    lines.append("")
    lines.append(
        "**On the left-crop check:** unlike the pipeline's original slides, there is "
        "no annotation-based cross-check available for these new slides, so the "
        "left/right tissue-fraction heuristic below is a partial, honest proxy — a "
        "low `needs_manual_review` count does not *prove* the crop is correct, and "
        "a flagged slide should be inspected visually before trusting its converted "
        "PNG."
    )
    lines.append("")

    lines.append("| slide | PNG | dims (w×h) | annotation | ndpi_scale OK | left frac | right frac | right/left | review? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        dims = r["dims"] or {}
        w = dims.get("original_full_width")
        h = dims.get("original_full_height")
        nc = r["ndpi_check"]
        if nc:
            scale_ok = "yes" if nc["ndpi_scale_confirmed"] else "NO"
            lr = nc["left_right_tissue_check"]
            left_f, right_f, ratio = lr["left_tissue_fraction"], lr["right_tissue_fraction"], lr["right_over_left_ratio"]
            review = "YES" if lr["needs_manual_review"] else "no"
        else:
            scale_ok = "n/a"
            left_f = right_f = ratio = None
            review = f"error: {r['ndpi_check_error']}" if r["ndpi_check_error"] else "n/a"
        lines.append(
            f"| {r['slide_stem']} | {'yes' if r['png_exists'] else 'NO'} | "
            f"{_fmt(w)}×{_fmt(h)} | {'yes' if r['annotation_exists_geojson'] else 'no'} | "
            f"{scale_ok} | {_fmt(left_f)} | {_fmt(right_f)} | {_fmt(ratio)} | {review} |"
        )
    lines.append("")

    verdict = (
        "**Verdict:** all slides converted, no ndpi_scale mismatches, and no "
        "left-crop flags — proceed to Stage 2."
        if n_png_missing == 0 and n_scale_mismatch == 0 and n_review_needed == 0
        else "**Verdict:** one or more issues above need resolution/manual review "
             "before proceeding to Stage 2 — do not treat flagged slides' converted "
             "PNGs as validated."
    )
    lines.append(verdict)
    lines.append("")

    (output_dir / "stage1_inventory.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage1_inventory.json"
    with open(json_path, "w") as f:
        json.dump({"slides": rows}, f, indent=2, default=str)
    print(f"  JSON: {json_path}")
    write_report(rows, output_dir)
    print(f"  Markdown report: {output_dir / 'stage1_inventory.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timepoint projection Stage 1: convert + inventory (post-conversion report)"
    )
    parser.add_argument("--slide-list", required=True, type=Path,
                        help="Text file, one bare NDPI stem per line (e.g. '6069-4R-4W')")
    parser.add_argument("--ndpi-dir", required=True, type=Path)
    parser.add_argument("--png-dir", required=True, type=Path)
    parser.add_argument("--slide-dimensions", required=True, type=Path)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--ndpi-scale", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print("  Timepoint projection — Stage 1: convert + inventory")
    print("=" * 60)

    slide_stems = load_slide_list(args.slide_list)
    print(f"\nSlides ({len(slide_stems)}): {slide_stems}")

    with open(args.slide_dimensions) as f:
        slide_dimensions = json.load(f)

    rows = build_inventory(
        slide_stems, args.ndpi_dir, args.png_dir, slide_dimensions,
        args.annotation_dir, args.ndpi_scale,
    )
    for r in rows:
        print(f"  {r['slide_stem']}: png={r['png_exists']} "
              f"annotation={r['annotation_exists_geojson']} "
              f"ndpi_check_error={r['ndpi_check_error']}")

    write_outputs(rows, args.output_dir)

    print("\n" + "=" * 60)
    print("  STAGE 1 INVENTORY COMPLETE")
    print("=" * 60)
    print(f"\n  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
