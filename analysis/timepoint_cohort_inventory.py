"""
Timepoint cohort: Stage A -- cohort inventory and design table.

The cross-cohort experiment (project 4W/8W timepoint slides onto the 2M-1
manifold and compare pseudotime against 2M-1) is CANCELLED -- its own hard gate
(timepoint_stage2_stain_check.py) found the timepoint batch confounded with
staining relative to 2M-1 by an amount comparable to the project's own known
cross-section confound, and that contrast is not identifiable by any
correction. See PROJECT_STATE.md for the full writeup.

Replacement design: a WITHIN-timepoint-cohort comparison across four
timepoints (4W, 7W, 8W, 12W) drawn from ~30 slides in
$SCRATCH/data/timepoint_ndpi and $SCRATCH/data/timepoint_ndpi_deferred. This
module is Stage A of that replacement design: it builds the FIRST real
inventory of this cohort (no manifest for it exists anywhere in this repo --
the raw NDPI files live only on the cluster). It is read-only and descriptive;
it does not gate PASS/FAIL. Stage B (timepoint_stain_homogeneity.py) is the
hard gate that consumes this inventory's output.

This is a NEW, standalone module -- distinct from and does not modify
`analysis/timepoint_inventory.py`, which is a different, already-superseded
module belonging to the cancelled cross-cohort experiment's own Stage 1 (fixed
7/8-slide list, crop/scale diagnostics specific to that design).

Known data issues this inventory surfaces, never silently resolves:
  - `6069-4R-4W` fails OpenSlide ("Restart marker not found") -- recorded via
    opens_in_openslide=False / openslide_error, not dropped from the table.
  - Three slides carry a literal " 2" suffix (`6054-4r-8W 2`, `6055-4L-8W 2`,
    `6056-4L-8W 2`; note lowercase "r" on the first) of unknown meaning --
    parsed and flagged (suffix_flag=True), left in the inventory for Stage B
    to decide how to handle.
  - Mouse 6072 contributes both `6072-4L-7W` and `6072-4R-12W` (confirmed
    staggered harvest, same animal) -- surfaced explicitly via the
    mouse_6072_check summary so this cross-timepoint dependency is visible,
    not discovered later by accident.
  - Multiple slides per mouse are common -- mouse_id is parsed and reported
    for every slide so downstream analyses can use mouse (not slide) as the
    unit of inference.

Reuses (does not reimplement):
  - the metadata-only NDPI-reading pattern from
    timepoint_convert_nocrop.py::read_ndpi_resolution_metadata (level0
    dimensions, mpp_x/mpp_y, other_resolution_properties) -- extended here
    with scan-date extraction.
  - analysis/holeyness.py::_safe_spearman for the scan-date-vs-timepoint
    confound check.

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_cohort_inventory \\
      --ndpi-dir       $SCRATCH/data/timepoint_ndpi \\
      --ndpi-dir       $SCRATCH/data/timepoint_ndpi_deferred \\
      --output-dir     $SCRATCH/results/timepoint_cohort/stageA_inventory
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .holeyness import _safe_spearman

# Any property key containing one of these (case-insensitive) is treated as a
# resolution signal worth recording, in case a scanner uses a vendor-specific
# key instead of the standard openslide.mpp-x/-y.
_FALLBACK_KEY_MARKERS = ("mpp", "resolution")

# Any property key containing this (case-insensitive) is treated as a
# candidate embedded scan date/time. The exact key OpenSlide/NDPI exposes is
# not assumed -- every matching key is read and reported, not just one guess.
_DATE_KEY_MARKER = "date"

# Ordered candidate datetime formats to try against a raw scan-date string.
# TIFF/EXIF's "%Y:%m:%d %H:%M:%S" is the most likely for NDPI; ISO variants
# are tried as fallbacks. An unparseable value is reported raw, never dropped.
_DATE_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")

# Confirmed real stem variants this must parse:
#   6070-4L-4W, 6072-4L-7W, 6072-4R-12W          (standard)
#   6054-4r-8W 2, 6055-4L-8W 2, 6056-4L-8W 2     (lowercase side / " N" suffix)
#   6069-4R-4W                                    (excluded elsewhere, but must
#                                                   still parse cleanly here --
#                                                   this module doesn't exclude
#                                                   anything, it only reports)
# `gland` is captured defensively (all confirmed examples are gland "4",
# matching the original pipeline's own 4L/4R convention) but not assumed fixed
# -- a different gland number would still parse and be visible in the table.
_STEM_RE = re.compile(
    r"^(?P<mouse>\d{3,5})-(?P<gland>\d)(?P<side>[LR])-(?P<weeks>\d+)W(?P<suffix>\s+\d+)?$",
    re.IGNORECASE,
)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ── Filename parsing ───────────────────────────────────────────────────────────

def parse_stem(raw_stem: str) -> dict:
    """Never raises. On no match, all fields are None except raw_stem,
    parse_ok=False, parse_error -- every slide's parse result (success or
    failure) is meant to be visible in the inventory, not just failures."""
    m = _STEM_RE.match(raw_stem.strip())
    if not m:
        return {
            "raw_stem": raw_stem,
            "mouse_id": None,
            "side": None,
            "side_raw": None,
            "gland": None,
            "timepoint_weeks": None,
            "suffix_flag": False,
            "suffix_raw": None,
            "parse_ok": False,
            "parse_error": "stem did not match expected <mouse>-<gland><side>-<weeks>W[ N] pattern",
        }
    suffix = m.group("suffix")
    return {
        "raw_stem": raw_stem,
        "mouse_id": m.group("mouse"),
        "side": m.group("side").upper(),
        "side_raw": m.group("side"),
        "gland": m.group("gland"),
        "timepoint_weeks": int(m.group("weeks")),
        "suffix_flag": suffix is not None,
        "suffix_raw": suffix,
        "parse_ok": True,
        "parse_error": None,
    }


# ── Directory scan ────────────────────────────────────────────────────────────

def iter_ndpi_files(ndpi_dirs: list[Path]) -> list[dict]:
    """Globs *.ndpi (case-insensitive) in every given directory. Returns one
    row per file found, with source_dir recorded and a duplicate_in_both_dirs
    flag if the same stem shows up under more than one directory (should not
    happen for a 'main' + 'deferred' split, but checked rather than assumed)."""
    seen: dict[str, list[Path]] = {}
    rows = []
    for ndpi_dir in ndpi_dirs:
        if not ndpi_dir.exists():
            continue
        for path in sorted(ndpi_dir.iterdir()):
            if path.suffix.lower() != ".ndpi":
                continue
            stem = path.stem
            seen.setdefault(stem, []).append(path)
            rows.append({"raw_stem": stem, "path": path, "source_dir": str(ndpi_dir)})
    for row in rows:
        row["duplicate_in_both_dirs"] = len(seen[row["raw_stem"]]) > 1
    return rows


# ── Per-slide metadata (OpenSlide, metadata-only, no pixel decode) ────────────

def _extract_scan_dates(props: dict) -> list[dict]:
    """Scans every property key containing 'date' (case-insensitive) -- the
    exact key NDPI/OpenSlide exposes is not assumed. Never raises on an
    unparseable value; reports it raw instead."""
    dates = []
    for key, raw_value in props.items():
        if _DATE_KEY_MARKER not in key.lower():
            continue
        parsed_iso = None
        parse_error = None
        for fmt in _DATE_FORMATS:
            try:
                parsed_iso = datetime.strptime(raw_value, fmt).isoformat()
                break
            except (ValueError, TypeError) as e:
                parse_error = repr(e)
        if parsed_iso is not None:
            parse_error = None
        dates.append({
            "date_key": key,
            "raw_value": raw_value,
            "parsed_date_iso": parsed_iso,
            "parse_error": parse_error,
        })
    return dates


def read_slide_metadata(ndpi_path: Path) -> dict:
    """Metadata-only read (level-0 dimensions + properties), no read_region --
    mirrors timepoint_convert_nocrop.py::read_ndpi_resolution_metadata, plus
    scan-date extraction. Wrapped by the caller in try/except so one
    unreadable NDPI cannot abort the whole run."""
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
        "scan_dates": _extract_scan_dates(props),
    }


# ── Build inventory ────────────────────────────────────────────────────────────

def build_inventory(ndpi_dirs: list[Path]) -> list[dict]:
    rows = []
    for f in iter_ndpi_files(ndpi_dirs):
        parsed = parse_stem(f["raw_stem"])
        row = {
            **parsed,
            "source_dir": f["source_dir"],
            "duplicate_in_both_dirs": f["duplicate_in_both_dirs"],
        }
        try:
            row["file_size_bytes"] = f["path"].stat().st_size
        except OSError as e:
            row["file_size_bytes"] = None
            row["file_size_error"] = repr(e)
        try:
            meta = read_slide_metadata(f["path"])
            row.update(meta)
            row["opens_in_openslide"] = True
            row["openslide_error"] = None
        except Exception as e:
            row.update({
                "level0_width": None, "level0_height": None,
                "mpp_x": None, "mpp_y": None,
                "other_resolution_properties": {}, "scan_dates": [],
            })
            row["opens_in_openslide"] = False
            row["openslide_error"] = repr(e)
        rows.append(row)
    return rows


def counts_by_timepoint(rows: list[dict]) -> dict:
    result: dict = {}
    for row in rows:
        weeks = row["timepoint_weeks"]
        key = str(weeks) if weeks is not None else "unparsed"
        bucket = result.setdefault(key, {"n_slides": 0, "mouse_ids": set()})
        bucket["n_slides"] += 1
        if row["mouse_id"] is not None:
            bucket["mouse_ids"].add(row["mouse_id"])
    return {
        k: {"n_slides": v["n_slides"], "n_distinct_mice": len(v["mouse_ids"]),
            "mouse_ids": sorted(v["mouse_ids"])}
        for k, v in result.items()
    }


def mouse_6072_check(rows: list[dict]) -> dict:
    mouse_rows = [r for r in rows if r["mouse_id"] == "6072"]
    slides_7w = [r["raw_stem"] for r in mouse_rows if r["timepoint_weeks"] == 7]
    slides_12w = [r["raw_stem"] for r in mouse_rows if r["timepoint_weeks"] == 12]
    return {
        "slides_7w": slides_7w,
        "slides_12w": slides_12w,
        "both_present": bool(slides_7w) and bool(slides_12w),
    }


def scan_date_vs_timepoint_confound(rows: list[dict]) -> dict:
    """Per-mouse median parsed scan date (as an ordinal) vs. per-mouse median
    timepoint_weeks, correlated via _safe_spearman (reused from holeyness.py,
    not reimplemented). Reports plainly if no date parses at all rather than
    crashing."""
    by_mouse: dict[str, list[tuple[float, int]]] = {}
    for row in rows:
        if row["mouse_id"] is None or row["timepoint_weeks"] is None:
            continue
        parsed_dates = [d["parsed_date_iso"] for d in row.get("scan_dates", []) if d["parsed_date_iso"]]
        if not parsed_dates:
            continue
        ordinal = datetime.fromisoformat(parsed_dates[0]).toordinal()
        by_mouse.setdefault(row["mouse_id"], []).append((ordinal, row["timepoint_weeks"]))

    if not by_mouse:
        return {
            "rho": None, "p": None, "n_mice": 0,
            "interpretation": "No parseable scan date found across the cohort -- "
                               "cannot check for a scan-date/timepoint confound.",
        }

    mouse_dates = np.array([np.median([d for d, _ in v]) for v in by_mouse.values()], dtype=float)
    mouse_weeks = np.array([np.median([w for _, w in v]) for v in by_mouse.values()], dtype=float)
    rho, p = _safe_spearman(mouse_dates, mouse_weeks)
    n_mice = len(by_mouse)
    if np.isnan(rho):
        interpretation = f"Only {n_mice} mice had a parseable scan date -- too few to assess (need >= 4)."
    elif abs(rho) >= 0.5:
        interpretation = (
            f"STRONG scan-date/timepoint association (rho={rho:.3f}, n={n_mice} mice) -- "
            "slides may have been scanned in batches that align with timepoint. This is "
            "a confound candidate and must be checked against Stage B's stain results."
        )
    else:
        interpretation = f"No strong scan-date/timepoint association (rho={rho:.3f}, n={n_mice} mice)."
    return {"rho": rho, "p": p, "n_mice": n_mice, "interpretation": interpretation}


# ── Output writers ────────────────────────────────────────────────────────────

def write_report(rows: list[dict], counts: dict, mouse_6072: dict, confound: dict, output_dir: Path) -> None:
    lines = ["# Timepoint cohort -- Stage A: cohort inventory", ""]

    n_parse_fail = sum(1 for r in rows if not r["parse_ok"])
    n_openslide_fail = sum(1 for r in rows if not r["opens_in_openslide"])
    n_dup = sum(1 for r in rows if r["duplicate_in_both_dirs"])

    lines.append(
        f"**Summary:** {len(rows)} .ndpi files found across both directories. "
        f"{n_parse_fail} failed filename parsing. {n_openslide_fail} failed to open in "
        f"OpenSlide. {n_dup} stem(s) present in both directories (unexpected)."
    )
    lines.append("")

    lines.append("## Counts by timepoint")
    lines.append("")
    lines.append("| timepoint (weeks) | n slides | n distinct mice | mouse IDs |")
    lines.append("|---|---|---|---|")
    for key in sorted(counts.keys(), key=lambda k: (k == "unparsed", k)):
        c = counts[key]
        lines.append(f"| {key} | {c['n_slides']} | {c['n_distinct_mice']} | {', '.join(c['mouse_ids'])} |")
    lines.append("")

    lines.append("## Mouse 6072 dual-timepoint check")
    lines.append("")
    lines.append(
        f"7W slides: {mouse_6072['slides_7w'] or 'none found'}. "
        f"12W slides: {mouse_6072['slides_12w'] or 'none found'}. "
        f"Both present: **{mouse_6072['both_present']}**."
    )
    lines.append("")

    lines.append("## Scan-date vs. timepoint confound check")
    lines.append("")
    lines.append(confound["interpretation"])
    if confound["rho"] is not None:
        lines.append(f"\nrho={_fmt(confound['rho'])}, p={_fmt(confound['p'])}, n_mice={confound['n_mice']}")
    lines.append("")

    lines.append("## Per-slide inventory")
    lines.append("")
    lines.append(
        "| raw stem | source dir | mouse | side | weeks | suffix | parse ok | "
        "opens? | level0 w×h | mpp-x | mpp-y | file size (MB) | dup? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        parse_flag = "yes" if r["parse_ok"] else f"**NO** ({r['parse_error']})"
        opens_flag = "yes" if r["opens_in_openslide"] else f"**NO** ({r['openslide_error']})"
        dims = f"{_fmt(r['level0_width'])}×{_fmt(r['level0_height'])}"
        size_mb = _fmt(r["file_size_bytes"] / (1024 * 1024)) if r["file_size_bytes"] is not None else "n/a"
        dup_flag = "**YES**" if r["duplicate_in_both_dirs"] else "no"
        lines.append(
            f"| {r['raw_stem']} | {Path(r['source_dir']).name} | {_fmt(r['mouse_id'])} | "
            f"{_fmt(r['side'])} | {_fmt(r['timepoint_weeks'])} | "
            f"{r['suffix_raw'] or 'no'} | {parse_flag} | {opens_flag} | {dims} | "
            f"{_fmt(r['mpp_x'])} | {_fmt(r['mpp_y'])} | {size_mb} | {dup_flag} |"
        )
    lines.append("")

    lines.append(
        "## Verdict\n\n"
        "This stage is descriptive only -- it does not gate PASS/FAIL. Its output "
        "(mouse/timepoint assignment, per-slide metadata) feeds Stage B "
        "(timepoint_stain_homogeneity.py), which is the hard gate.\n"
    )

    (output_dir / "stageA_inventory.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(rows: list[dict], counts: dict, mouse_6072: dict, confound: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "slides": rows,
        "counts_by_timepoint": counts,
        "mouse_6072_check": mouse_6072,
        "scan_date_confound": confound,
    }
    json_path = output_dir / "stageA_inventory.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  JSON: {json_path}")
    write_report(rows, counts, mouse_6072, confound, output_dir)
    print(f"  Markdown report: {output_dir / 'stageA_inventory.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timepoint cohort Stage A: cohort inventory (descriptive, not a gate)"
    )
    parser.add_argument("--ndpi-dir", required=True, type=Path, action="append",
                        help="Pass twice: once for $SCRATCH/data/timepoint_ndpi, once for "
                             "$SCRATCH/data/timepoint_ndpi_deferred")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print("  Timepoint cohort -- Stage A: cohort inventory")
    print("=" * 60)
    print(f"\nScanning directories: {[str(d) for d in args.ndpi_dir]}")

    rows = build_inventory(args.ndpi_dir)
    print(f"\nFound {len(rows)} .ndpi files")
    for r in rows:
        print(
            f"  {r['raw_stem']}: parse_ok={r['parse_ok']} mouse={r['mouse_id']} "
            f"weeks={r['timepoint_weeks']} opens={r['opens_in_openslide']}"
        )

    counts = counts_by_timepoint(rows)
    mouse_6072 = mouse_6072_check(rows)
    confound = scan_date_vs_timepoint_confound(rows)

    print("\n=== Counts by timepoint ===")
    for k, v in sorted(counts.items()):
        print(f"  {k}W: {v['n_slides']} slides, {v['n_distinct_mice']} mice")

    print(f"\n=== Mouse 6072 check ===\n  {mouse_6072}")
    print(f"\n=== Scan-date confound ===\n  {confound['interpretation']}")

    write_outputs(rows, counts, mouse_6072, confound, args.output_dir)

    print("\n" + "=" * 60)
    print("  STAGE A COHORT INVENTORY COMPLETE")
    print("=" * 60)
    print(f"\n  Output dir: {args.output_dir}")
    print("\n  Next: analysis/timepoint_stain_homogeneity.py (Stage B, HARD GATE) "
          "consumes this inventory.")


if __name__ == "__main__":
    main()
