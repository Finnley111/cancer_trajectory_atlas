"""
Timepoint projection: no-crop conversion (full-width PNG).

Stage 1 originally converted the timepoint slides via `run_all.py --convert`,
which hardcodes an unconditional crop to the left half of each NDPI -- correct
for the original 16 pipeline slides (documented, annotation-confirmed duplicate
right half), but NOT verified for these slides, which have no annotations at all.
Per-slide visual inspection (in QuPath) found the batch is NOT uniform: some
slides show duplicate content on both halves, some do not -- no single blanket
crop-or-don't-crop rule is safe across all of them without per-slide domain
knowledge this session doesn't have.

Resolution: convert WITHOUT cropping (keep full image width) and keep per-patch
x,y coordinates downstream (already how this pipeline's results.csv/projection
output works -- no new code needed there). This makes the crop decision
REVERSIBLE after the fact -- any specific slide can be filtered to
x < original_full_width/2 post-hoc later, once/if a slide is confirmed to need
that, rather than requiring a correct per-slide guess before ever converting it.
Cropping during conversion, by contrast, is irreversible: once discarded, the
right half's pixels can't be recovered without re-reading the raw NDPI.

This is a NEW, standalone script -- NOT a modification to `run_all.py`, which
stays completely frozen and is still used unmodified for the original 16
slides. Mirrors `run_all.py`'s own conversion logic (open NDPI, read the
requested pyramid level, apply `ndpi_scale` if set, save PNG) but omits the
left-half crop step, and improves on two things `run_all.py --convert` doesn't
do (learned the hard way earlier this session):
  - per-slide try/except (a corrupted NDPI must not abort the whole batch)
  - explicit slide-list scoping (only converts the slides asked for, not every
    .ndpi file that happens to sit in the directory)

Writes to a NEW output directory (`timepoint_x5_full`, distinct from the
existing `timepoint_x5_cropped`) with its own `slide_dimensions.json` sidecar,
so the already-existing (left-cropped) PNGs and their sidecar are never
touched -- they remain available for comparison if ever needed.

Slide list: exactly 7 slides (`jobs/slides_timepoint.txt`). `6069-4R-4W` is
permanently excluded -- confirmed corrupted at the OpenSlide level ("Restart
marker not found", not a transfer issue -- SHA256 matched the local copy).
There is no deferred/recovery directory or fallback path for it; if any of the
7 expected NDPIs is missing from `--ndpi-dir`, this script hard-fails before
doing any work rather than silently continuing with fewer slides.

Step 0 -- MPP pre-flight check (runs before any conversion): the original 16
pipeline slides were converted at `ndpi_scale=1.0` (confirmed against
`jobs/convert_ndpi.sh` and `pipeline_config.py`'s default, and cross-checked
against `run_all.py`'s sidecar math -- the documented 96000->48000 width change
for an original slide is explained by the left-half crop, not a 0.5 resize).
If these timepoint NDPIs were scanned at a different native resolution (MPP)
than the originals, converting them at the same `ndpi_scale=1.0` would make
patches cover a different physical tissue area than the manifold they're being
projected onto -- silently invalidating the projection. So before converting
anything, this script reads (metadata only, no pixel decode) the level-0
dimensions and `openslide.mpp-x`/`openslide.mpp-y` for every timepoint NDPI and
for a small fixed sample of the original NDPIs, and compares them:
  - If MPP values agree within tolerance, proceeds using `ndpi_scale=1.0` (this
    is now an OUTPUT of the check, not a CLI input that gets trusted blindly).
  - If MPP values disagree, or the property is missing for either batch, this
    is a HARD STOP -- no PNG is written. The scale that WOULD equalize physical
    resolution is computed and reported (derived from the MPP ratio, since the
    originals' scale is 1.0, that equalizing scale is exactly the MPP ratio),
    but never auto-applied.
The full comparison is always written to `--mpp-output-json` regardless of
outcome, so it's on record independent of the SLURM log.

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_convert_nocrop \\
      --slide-list          ~/cancer_trajectory_atlas/jobs/slides_timepoint.txt \\
      --ndpi-dir            $SCRATCH/data/timepoint_ndpi \\
      --png-dir             $SCRATCH/data/timepoint_x5_full \\
      --ndpi-level          0 \\
      --original-ndpi-dir   $SCRATCH/data/ndpi \\
      --original-sample-stems 6027-4L-2M-1_x5 6028-4L-2M-1_x5 6029-4L-2M-1_x5 \\
      --mpp-tolerance-pct   2.0 \\
      --mpp-output-json     $SCRATCH/results/timepoint_projection/mpp_verification.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .timepoint_inventory import load_slide_list
from .timepoint_stage2_stain_check import _normalize_stem

DEFAULT_ORIGINAL_SAMPLE_STEMS = [
    "6027-4L-2M-1_x5",
    "6028-4L-2M-1_x5",
    "6029-4L-2M-1_x5",
]

# Any property key containing one of these (case-insensitive) is logged as a
# fallback resolution signal, in case this scanner uses a vendor-specific key
# instead of the standard openslide.mpp-x/-y.
_FALLBACK_KEY_MARKERS = ("mpp", "resolution")


def resolve_ndpi_paths(stems: list[str], ndpi_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Looks up each stem in ndpi_dir. Returns (found, missing) -- no fallback
    directory; a missing stem is reported, not silently substituted."""
    found: dict[str, Path] = {}
    missing: list[str] = []
    for stem in stems:
        p = ndpi_dir / f"{stem}.ndpi"
        if p.exists():
            found[stem] = p
        else:
            missing.append(stem)
    return found, missing


# ── Step 0: MPP pre-flight check ──────────────────────────────────────────────

def read_ndpi_resolution_metadata(ndpi_path: Path) -> dict:
    """Reads level-0 dimensions and resolution metadata only -- no read_region,
    no pixel decode. Fast enough to run against every slide in seconds."""
    import openslide  # lazy import -- only needed on Narval, where it's module-loaded

    slide = openslide.OpenSlide(str(ndpi_path))
    try:
        level0_w, level0_h = slide.level_dimensions[0]
        props = dict(slide.properties)
    finally:
        slide.close()

    mpp_x = props.get("openslide.mpp-x")
    mpp_y = props.get("openslide.mpp-y")
    other_keys = {
        k: v for k, v in props.items()
        if k not in ("openslide.mpp-x", "openslide.mpp-y")
        and any(marker in k.lower() for marker in _FALLBACK_KEY_MARKERS)
    }
    return {
        "level0_width": int(level0_w),
        "level0_height": int(level0_h),
        "mpp_x": float(mpp_x) if mpp_x is not None else None,
        "mpp_y": float(mpp_y) if mpp_y is not None else None,
        "other_resolution_properties": other_keys,
    }


def _read_batch_metadata(paths: dict[str, Path]) -> dict:
    """Per-slide try/except -- one unreadable file must not abort the check."""
    per_slide: dict = {}
    for stem, path in paths.items():
        try:
            meta = read_ndpi_resolution_metadata(path)
            meta["error"] = None
        except Exception as e:
            meta = {
                "level0_width": None, "level0_height": None,
                "mpp_x": None, "mpp_y": None,
                "other_resolution_properties": {}, "error": repr(e),
            }
        per_slide[stem] = meta
    return per_slide


def _primary_mpp(meta: dict) -> float | None:
    """mpp-x, falling back to mpp-y if mpp-x is absent."""
    if meta.get("mpp_x") is not None:
        return meta["mpp_x"]
    return meta.get("mpp_y")


def mpp_preflight_check(
    timepoint_paths: dict[str, Path],
    original_paths: dict[str, Path],
    tolerance_pct: float = 2.0,
) -> dict:
    """Compares native resolution (MPP) between the timepoint batch and a
    sample of the original 16 slides. Returns a full record, including a
    'decision' of 'PROCEED' or 'HALT' and (when computable) a recommended
    ndpi_scale. Never guesses a scale when metadata is missing."""
    import statistics

    tp_meta = _read_batch_metadata(timepoint_paths)
    orig_meta = _read_batch_metadata(original_paths)

    tp_valid = {s: _primary_mpp(m) for s, m in tp_meta.items() if _primary_mpp(m) is not None}
    orig_valid = {s: _primary_mpp(m) for s, m in orig_meta.items() if _primary_mpp(m) is not None}

    result = {
        "tolerance_pct": tolerance_pct,
        "timepoint_metadata": tp_meta,
        "original_metadata": orig_meta,
        "timepoint_valid_mpp": tp_valid,
        "original_valid_mpp": orig_valid,
    }

    if not tp_valid or not orig_valid:
        result.update({
            "timepoint_median_mpp": None,
            "original_median_mpp": None,
            "ratio": None,
            "decision": "HALT",
            "reason": "missing_mpp_metadata",
            "recommended_ndpi_scale": None,
            "explanation": (
                f"Cannot verify physical resolution: {len(tp_valid)}/{len(tp_meta)} "
                f"timepoint slides and {len(orig_valid)}/{len(orig_meta)} original "
                f"slides had a readable openslide.mpp-x/mpp-y. Refusing to guess a "
                f"scale factor -- fix the missing metadata (check "
                f"other_resolution_properties for a vendor-specific key) before "
                f"converting anything."
            ),
        })
        return result

    tp_median = statistics.median(tp_valid.values())
    orig_median = statistics.median(orig_valid.values())
    ratio = tp_median / orig_median
    result["timepoint_median_mpp"] = tp_median
    result["original_median_mpp"] = orig_median
    result["ratio"] = ratio

    if abs(ratio - 1.0) <= tolerance_pct / 100.0:
        result.update({
            "decision": "PROCEED",
            "reason": "mpp_match",
            "recommended_ndpi_scale": 1.0,
            "explanation": (
                f"Timepoint median MPP ({tp_median:.5f}) matches original median MPP "
                f"({orig_median:.5f}) within {tolerance_pct}% (ratio={ratio:.4f}). "
                f"Proceeding with ndpi_scale=1.0, matching the original pipeline's "
                f"actual conversion scale (jobs/convert_ndpi.sh, "
                f"pipeline_config.py default)."
            ),
        })
    else:
        result.update({
            "decision": "HALT",
            "reason": "mpp_mismatch",
            "recommended_ndpi_scale": ratio,
            "explanation": (
                f"Timepoint median MPP ({tp_median:.5f}) does NOT match original "
                f"median MPP ({orig_median:.5f}) within {tolerance_pct}% "
                f"(ratio={ratio:.4f}). Converting at ndpi_scale=1.0 would cover a "
                f"different physical tissue area per patch than the manifold "
                f"expects. Since the originals were converted at ndpi_scale=1.0, "
                f"the scale that would equalize physical output resolution for the "
                f"timepoint batch is exactly this MPP ratio: recommended_ndpi_scale="
                f"{ratio:.4f}. This is a RECOMMENDATION ONLY -- not applied "
                f"automatically. Halting before any PNG is written."
            ),
        })
    return result


def write_mpp_verification(result: dict, output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  MPP verification JSON: {output_json}")


# ── Conversion ─────────────────────────────────────────────────────────────────

def _png_is_valid(png_path: Path) -> bool:
    """Cheap integrity check (PIL's verify(), not a full pixel decode) --
    catches truncated/corrupted files left behind by an interrupted prior run
    (e.g. a walltime kill mid-write). `Path.exists()` alone can't distinguish
    a complete PNG from a 0-byte or partial one; this closed the gap that let
    a truncated timepoint PNG sit silently "converted" until Stage 2 actually
    tried to read it. verify() leaves the file object unusable afterward, so
    this always opens a fresh handle rather than reusing one for real work."""
    from PIL import Image

    try:
        with Image.open(png_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def convert_ndpi_full_width(
    ndpi_paths: dict[str, Path],
    png_dir: Path,
    ndpi_level: int,
    ndpi_scale: float,
) -> dict:
    """Converts each NDPI to a full-width (uncropped) PNG. Idempotent: if the
    target PNG already exists AND passes an integrity check, skips
    re-encoding but still (re-)records its dimensions in the sidecar,
    matching run_all.py's own incremental-rerun convention. If a target PNG
    exists but fails the integrity check (truncated/corrupted), it is
    re-converted rather than silently trusted. Catches per-slide exceptions
    so one corrupted NDPI can't abort the batch (all 7 expected slides here
    are already known-readable -- this is a safety net, not a workaround for
    a known-bad file)."""
    from PIL import Image
    import openslide  # lazy import -- only needed on Narval, where it's module-loaded

    png_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = png_dir / "slide_dimensions.json"
    sidecar: dict = {}
    if sidecar_path.exists():
        with open(sidecar_path) as f:
            sidecar = json.load(f)

    results: dict = {}
    for stem, ndpi_path in ndpi_paths.items():
        out_name = f"{stem}_x5.png"
        out_path = png_dir / out_name
        try:
            slide = openslide.OpenSlide(str(ndpi_path))
            try:
                raw_w, raw_h = slide.level_dimensions[ndpi_level]
                full_w = int(raw_w * ndpi_scale)
                full_h = int(raw_h * ndpi_scale)

                if out_path.exists() and _png_is_valid(out_path):
                    print(f"  SKIP (exists, verified): {out_name}")
                else:
                    if out_path.exists():
                        print(
                            f"  WARNING: {out_name} exists but failed integrity check "
                            f"(truncated/corrupted) -- re-converting"
                        )
                    region = slide.read_region((0, 0), ndpi_level, (raw_w, raw_h))
                    img = region.convert("RGB")
                    if ndpi_scale != 1.0:
                        img = img.resize((full_w, full_h), Image.Resampling.LANCZOS)
                    # NO CROP -- full width/height kept, unlike run_all.py's
                    # unconditional left-half crop.
                    img.save(out_path, "PNG", compress_level=6)
                    print(
                        f"  Converted (full width, no crop): {stem} -> "
                        f"{out_name} ({full_w}x{full_h})"
                    )

                sidecar[out_name] = {
                    "original_full_width": full_w,
                    "original_full_height": full_h,
                    # No crop applied -- cropped_width/height are the full
                    # dimensions, not full_width // 2. Kept as the same two
                    # keys the pipeline's sidecar schema already uses so
                    # downstream readers don't need special-casing.
                    "cropped_width": full_w,
                    "cropped_height": full_h,
                }
                results[stem] = {"status": "ok", "error": None}
            finally:
                slide.close()
        except Exception as e:
            results[stem] = {"status": "error", "error": repr(e)}
            print(f"  ERROR converting {stem}: {e!r} -- skipping, continuing to next slide")

    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"\n  Dimensions sidecar: {sidecar_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert timepoint NDPI slides to full-width PNG (no crop) -- "
                     "new, standalone script; does not modify run_all.py"
    )
    parser.add_argument("--slide-list", required=True, type=Path)
    parser.add_argument("--ndpi-dir", required=True, type=Path)
    parser.add_argument("--png-dir", required=True, type=Path,
                        help="NEW output directory (e.g. $SCRATCH/data/timepoint_x5_full) "
                             "-- separate from the existing left-cropped "
                             "timepoint_x5_cropped, which is never touched")
    parser.add_argument("--ndpi-level", default=0, type=int)
    parser.add_argument("--original-ndpi-dir", required=True, type=Path,
                        help="Raw NDPI directory for the original 16 pipeline slides "
                             "(e.g. $SCRATCH/data/ndpi), used only for the MPP "
                             "pre-flight check")
    parser.add_argument("--original-sample-stems", nargs="+",
                        default=DEFAULT_ORIGINAL_SAMPLE_STEMS,
                        help="2-3 original slide stems (one per mouse) to read MPP "
                             "metadata from for comparison")
    parser.add_argument("--mpp-tolerance-pct", default=2.0, type=float,
                        help="Max %% difference between timepoint and original median "
                             "MPP to treat as a match (default 2.0)")
    parser.add_argument("--mpp-output-json", required=True, type=Path,
                        help="Where to write the MPP comparison record, independent "
                             "of the SLURM log (e.g. "
                             "$SCRATCH/results/timepoint_projection/mpp_verification.json)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Timepoint projection — no-crop conversion (full-width PNG)")
    print("=" * 60)

    stems = [_normalize_stem(s) for s in load_slide_list(args.slide_list)]
    print(f"\nSlides requested ({len(stems)}): {stems}")

    ndpi_paths, missing = resolve_ndpi_paths(stems, args.ndpi_dir)
    if missing:
        print(f"\n*** HARD FAIL: {len(missing)} expected slide(s) not found in {args.ndpi_dir}: {missing}")
        print("*** This script expects exactly 7 slides, all in --ndpi-dir. No fallback "
              "directory exists. Place the missing NDPI(s) there and re-run.")
        sys.exit(1)
    print(f"Found all {len(ndpi_paths)} expected slides in {args.ndpi_dir}")

    print("\n=== Step 0: MPP pre-flight check ===")
    original_stems = [_normalize_stem(s) for s in args.original_sample_stems]
    original_paths, original_missing = resolve_ndpi_paths(original_stems, args.original_ndpi_dir)
    if original_missing:
        print(f"  WARNING: original sample slide(s) not found in {args.original_ndpi_dir}: {original_missing}")
    print(f"  Timepoint batch: {len(ndpi_paths)} slides from {args.ndpi_dir}")
    print(f"  Original sample: {len(original_paths)} slides from {args.original_ndpi_dir} ({original_stems})")

    mpp_result = mpp_preflight_check(ndpi_paths, original_paths, args.mpp_tolerance_pct)
    write_mpp_verification(mpp_result, args.mpp_output_json)

    print(f"\n  Timepoint median MPP : {mpp_result['timepoint_median_mpp']}")
    print(f"  Original median MPP  : {mpp_result['original_median_mpp']}")
    print(f"  Ratio (tp/orig)      : {mpp_result['ratio']}")
    print(f"  DECISION             : {mpp_result['decision']}")
    print(f"  {mpp_result['explanation']}")

    if mpp_result["decision"] == "HALT":
        print("\n*** HARD STOP: MPP pre-flight check failed. No PNG will be written. ***")
        sys.exit(1)

    ndpi_scale = mpp_result["recommended_ndpi_scale"]
    print(f"\n  Proceeding with ndpi_scale={ndpi_scale} (determined by MPP pre-flight check)")

    print("\n=== Converting (no crop) ===")
    results = convert_ndpi_full_width(ndpi_paths, args.png_dir, args.ndpi_level, ndpi_scale)

    summary_path = args.png_dir / "conversion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    print("\n" + "=" * 60)
    print(f"  DONE — {n_ok}/{len(stems)} slides converted successfully (full width, no crop)")
    print("=" * 60)
    for stem, r in results.items():
        if r["status"] != "ok":
            print(f"  NOT converted: {stem} -- {r['error']}")
    print(f"\n  PNG dir: {args.png_dir}")
    print(f"  Summary: {summary_path}")
    print(
        "\n  Next: re-run analysis.timepoint_inventory against this new PNG dir/sidecar "
        "to refresh the per-slide report (dims will now show full width; the "
        "left/right tissue-fraction check re-reads the raw NDPI directly, so its "
        "numbers are unaffected by the crop-vs-no-crop choice -- only now purely "
        "informational, not a blocker, since both halves are kept either way)."
    )


if __name__ == "__main__":
    main()
