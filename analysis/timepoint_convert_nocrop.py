"""
Timepoint projection: no-crop conversion (full-width PNG).

Stage 1 originally converted the 8 timepoint slides via `run_all.py --convert`,
which hardcodes an unconditional crop to the left half of each NDPI -- correct
for the original 16 pipeline slides (documented, annotation-confirmed duplicate
right half), but NOT verified for these 8, which have no annotations at all.
Per-slide visual inspection (in QuPath) found the batch is NOT uniform: some
slides show duplicate content on both halves, some do not -- no single blanket
crop-or-don't-crop rule is safe across all 8 without per-slide domain knowledge
this session doesn't have.

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

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_convert_nocrop \\
      --slide-list        ~/cancer_trajectory_atlas/jobs/slides_timepoint.txt \\
      --ndpi-dir          $SCRATCH/data/timepoint_ndpi \\
      --ndpi-deferred-dir $SCRATCH/data/timepoint_ndpi_deferred \\
      --png-dir           $SCRATCH/data/timepoint_x5_full \\
      --ndpi-level        0 \\
      --ndpi-scale        1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .timepoint_inventory import load_slide_list
from .timepoint_stage2_stain_check import _normalize_stem


def resolve_ndpi_paths(
    stems: list[str], ndpi_dir: Path, ndpi_deferred_dir: Path | None,
) -> tuple[dict[str, Path], list[str]]:
    """Looks up each stem in ndpi_dir first, then ndpi_deferred_dir (e.g. for a
    slide like 6069-4R-4W that's currently set aside). Returns (found, missing)."""
    found: dict[str, Path] = {}
    missing: list[str] = []
    for stem in stems:
        p = ndpi_dir / f"{stem}.ndpi"
        if not p.exists() and ndpi_deferred_dir is not None:
            p2 = ndpi_deferred_dir / f"{stem}.ndpi"
            if p2.exists():
                p = p2
        if p.exists():
            found[stem] = p
        else:
            missing.append(stem)
    return found, missing


def convert_ndpi_full_width(
    ndpi_paths: dict[str, Path],
    png_dir: Path,
    ndpi_level: int,
    ndpi_scale: float,
) -> dict:
    """Converts each NDPI to a full-width (uncropped) PNG. Idempotent: if the
    target PNG already exists, skips re-encoding but still (re-)records its
    dimensions in the sidecar, matching run_all.py's own incremental-rerun
    convention. Catches per-slide exceptions so one corrupted NDPI (e.g. the
    known 6069-4R-4W "Restart marker not found" failure) can't abort the batch."""
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

                if out_path.exists():
                    print(f"  SKIP (exists): {out_name}")
                else:
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
    parser.add_argument("--ndpi-deferred-dir", default=None, type=Path,
                        help="Secondary directory to check for a slide not found in "
                             "--ndpi-dir (e.g. 6069-4R-4W, currently set aside there)")
    parser.add_argument("--png-dir", required=True, type=Path,
                        help="NEW output directory (e.g. $SCRATCH/data/timepoint_x5_full) "
                             "-- separate from the existing left-cropped "
                             "timepoint_x5_cropped, which is never touched")
    parser.add_argument("--ndpi-level", default=0, type=int)
    parser.add_argument("--ndpi-scale", default=1.0, type=float)
    args = parser.parse_args()

    print("=" * 60)
    print("  Timepoint projection — no-crop conversion (full-width PNG)")
    print("=" * 60)

    stems = [_normalize_stem(s) for s in load_slide_list(args.slide_list)]
    print(f"\nSlides requested ({len(stems)}): {stems}")

    ndpi_paths, missing = resolve_ndpi_paths(stems, args.ndpi_dir, args.ndpi_deferred_dir)
    print(f"Found: {len(ndpi_paths)}  Missing: {missing}")

    print("\n=== Converting (no crop) ===")
    results = convert_ndpi_full_width(ndpi_paths, args.png_dir, args.ndpi_level, args.ndpi_scale)

    for stem in missing:
        results[stem] = {"status": "error", "error": "raw NDPI file not found in --ndpi-dir or --ndpi-deferred-dir"}

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
