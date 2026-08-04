"""
Timepoint cohort: Stage A v2 -- corrected cohort inventory (full-scale re-verification).

Stage A v1 (`timepoint_cohort_inventory.py`, NOT modified here -- see below)
built the first inventory of this cohort against the raw NDPI files only,
before Stage C conversion had run. Stage C has since completed: 22 of the 24
"remaining" slides converted successfully into `timepoint_x5_full`, joining
the 7 already-converted slides for 29 total. Stage C also surfaced two data
issues Stage A v1 could not have known about:

  1. `6041-4L-12W` fails at `read_region` with OpenSlideError("Restart marker
     not found") -- the same failure mode already known for `6069-4R-4W`.
     Both are corrupted and permanently excluded, not retried.
  2. `60997-4L-4W-2` -- Stage A v1's filename-parse failure -- has level0
     dimensions (86400x49280) and file size (602.7 MB) identical to
     `6097-4L-4W` from the original 7-slide batch. This is a near-certain
     duplicate/rescan, not a distinct 20th slide. Never silently dropped:
     `check_duplicate_60997` below verifies and reports this match explicitly.

This is a NEW, standalone module -- Stage A v1 is not modified, matching this
project's established convention (Stage B reuses Stage 2's primitives without
editing Stage 2; `timepoint_convert_nocrop.py` is a new sibling to `run_all.py`
rather than an edit to it). `parse_stem`, `iter_ndpi_files`,
`read_slide_metadata`, `build_inventory`, `counts_by_timepoint`,
`mouse_6072_check`, `scan_date_vs_timepoint_confound`, and `_fmt` are imported
directly from Stage A v1, not reimplemented.

What's new here, on top of Stage A v1's per-slide metadata:
  - `png_exists`: whether `{raw_stem}_x5.png` exists in `--converted-png-dir`
    (`timepoint_x5_full`), using the exact path convention Stage B and Stage 2
    already use (`_normalize_stem`/`_png_path`, imported from
    `timepoint_stage2_stain_check.py`, not reimplemented).
  - The "usable cohort" = parse_ok AND opens_in_openslide AND png_exists AND
    not the confirmed 60997 duplicate. This naturally excludes 6041-4L-12W and
    6069-4R-4W (Stage C never produced a PNG for either) without hardcoding
    those stems -- but any row that opens in OpenSlide, isn't the confirmed
    duplicate, and STILL has no PNG is flagged as an unexplained gap (expected
    membership: exactly {6041-4L-12W, 6069-4R-4W}) so a genuinely new,
    unaccounted-for conversion failure can't hide inside the same filter.
  - A per-(mouse, timepoint_weeks) coverage check: every group present in the
    FULL (pre-exclusion) inventory is checked against the usable cohort: does
    it still have >= 1 usable slide? Reported for every group that drops to
    zero, not just the specific 6041/12W case named in the task brief -- this
    generalizes the check so any other mouse/timepoint silently losing its
    only slide would also be caught.
  - counts_by_timepoint and scan_date_vs_timepoint_confound (both reused
    verbatim from Stage A v1) are recomputed on the usable-cohort rows only,
    and the report states the v1 comparison numbers (n=19 mice,
    rho=-0.126, p=0.6065) alongside the new ones for a direct before/after.

Descriptive only -- does not gate PASS/FAIL. Feeds Stage B v2
(`timepoint_stain_homogeneity_v2.py`), the hard gate.

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_cohort_inventory_v2 \\
      --ndpi-dir           $SCRATCH/data/timepoint_ndpi \\
      --ndpi-dir           $SCRATCH/data/timepoint_ndpi_deferred \\
      --converted-png-dir  $SCRATCH/data/timepoint_x5_full \\
      --output-dir         $SCRATCH/results/timepoint_cohort/stageA_inventory_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .timepoint_cohort_inventory import (
    build_inventory,
    counts_by_timepoint,
    mouse_6072_check,
    scan_date_vs_timepoint_confound,
    _fmt,
)
from .timepoint_stage2_stain_check import _png_path

# The confirmed prior (v1, coarse-NDPI-inventory, pre-Stage-C) scan-date/
# timepoint confound result -- from stageA_inventory.md -- kept here ONLY as a
# fixed comparison point for the report, never as a computed value.
V1_SCAN_DATE_CONFOUND = {"rho": -0.1263, "p": 0.6065, "n_mice": 19}

# The two stems Stage C found corrupted at the read_region level. Not used to
# drive any exclusion decision (that's done by png_exists, computed fresh
# against the real converted-PNG directory) -- used only to label the
# unexplained-gap check so a THIRD, previously-unknown failure is visibly
# different from these two expected ones.
KNOWN_CORRUPTED_STEMS = {"6041-4L-12W", "6069-4R-4W"}

DUPLICATE_STEM = "60997-4L-4W-2"
DUPLICATE_OF_STEM = "6097-4L-4W"


# ── PNG-existence check ────────────────────────────────────────────────────────

def annotate_png_existence(rows: list[dict], converted_png_dir: Path) -> None:
    """Mutates each row in place, adding png_exists. Uses the exact same
    stem-normalization/path convention as Stage B / Stage 2
    (_normalize_stem/_png_path) so "does this slide have a converted PNG" is
    answered identically everywhere in this project."""
    for row in rows:
        png_path = _png_path(converted_png_dir, row["raw_stem"])
        row["png_exists"] = png_path.exists()
        row["png_path"] = str(png_path)


# ── Duplicate determination ────────────────────────────────────────────────────

def check_duplicate_60997(rows: list[dict]) -> dict:
    """Never silently resolved: explicitly locates 60997-4L-4W-2 and
    6097-4L-4W and reports whether their level0 dims / file size actually
    match, rather than assuming the determination made from the earlier,
    partial inventory still holds."""
    dup_row = next((r for r in rows if r["raw_stem"] == DUPLICATE_STEM), None)
    orig_row = next((r for r in rows if r["raw_stem"] == DUPLICATE_OF_STEM), None)

    if dup_row is None:
        return {
            "checked": False,
            "reason": f"{DUPLICATE_STEM} not found in this inventory run -- "
                       "nothing to determine (it may have been removed from "
                       "the source directory).",
        }
    if orig_row is None:
        return {
            "checked": False,
            "reason": f"{DUPLICATE_OF_STEM} not found in this inventory run -- "
                       "cannot compare against it.",
            "dup_row_summary": {
                "level0_width": dup_row["level0_width"],
                "level0_height": dup_row["level0_height"],
                "file_size_bytes": dup_row["file_size_bytes"],
            },
        }

    dims_match = (
        dup_row["level0_width"] == orig_row["level0_width"]
        and dup_row["level0_height"] == orig_row["level0_height"]
    )
    size_match = (
        dup_row["file_size_bytes"] is not None
        and orig_row["file_size_bytes"] is not None
        and dup_row["file_size_bytes"] == orig_row["file_size_bytes"]
    )
    is_duplicate = dims_match and size_match

    return {
        "checked": True,
        "is_duplicate": is_duplicate,
        "dims_match": dims_match,
        "size_match": size_match,
        f"{DUPLICATE_STEM}_dims": f"{dup_row['level0_width']}x{dup_row['level0_height']}",
        f"{DUPLICATE_OF_STEM}_dims": f"{orig_row['level0_width']}x{orig_row['level0_height']}",
        f"{DUPLICATE_STEM}_size_bytes": dup_row["file_size_bytes"],
        f"{DUPLICATE_OF_STEM}_size_bytes": orig_row["file_size_bytes"],
        "determination": (
            f"CONFIRMED duplicate/rescan of {DUPLICATE_OF_STEM} -- level0 dims and "
            "file size both match exactly. Excluded from the usable cohort as "
            "duplicate data, not counted as a 20th 4W slide."
            if is_duplicate else
            "NOT a confirmed duplicate on this run's real data -- dims and/or file "
            "size do NOT match. The earlier duplicate determination does not hold; "
            "treat 60997-4L-4W-2 as a genuinely separate slide pending manual review, "
            "do not silently exclude it."
        ),
    }


# ── Usable-cohort construction ─────────────────────────────────────────────────

def build_usable_cohort(rows: list[dict], duplicate_check: dict) -> tuple[list[dict], list[dict]]:
    """Returns (usable_rows, unexplained_gap_rows). A row is usable iff it
    parses, opens in OpenSlide, has a converted PNG, and is not the confirmed
    60997 duplicate. unexplained_gap_rows are rows that parse + open in
    OpenSlide + are NOT the confirmed duplicate, yet still have no PNG and are
    NOT one of the two known-corrupted stems -- i.e. a gap this run cannot
    explain and must not silently swallow."""
    is_confirmed_dup = duplicate_check.get("checked") and duplicate_check.get("is_duplicate")

    usable, unexplained = [], []
    for r in rows:
        is_dup_row = is_confirmed_dup and r["raw_stem"] == DUPLICATE_STEM
        r["excluded_as_duplicate"] = bool(is_dup_row)

        if not (r["parse_ok"] and r["opens_in_openslide"]):
            r["usable"] = False
            r["exclusion_reason"] = (
                "parse_fail" if not r["parse_ok"] else "openslide_fail"
            )
            continue
        if is_dup_row:
            r["usable"] = False
            r["exclusion_reason"] = f"duplicate_of_{DUPLICATE_OF_STEM}"
            continue
        if not r["png_exists"]:
            r["usable"] = False
            if r["raw_stem"] in KNOWN_CORRUPTED_STEMS:
                r["exclusion_reason"] = "corrupted_stage_c_conversion_failure"
            else:
                r["exclusion_reason"] = "UNEXPLAINED_no_png_and_not_known_corrupted"
                unexplained.append(r)
            continue

        r["usable"] = True
        r["exclusion_reason"] = None
        usable.append(r)

    return usable, unexplained


# ── Per-(mouse, timepoint) coverage check ─────────────────────────────────────

def coverage_check(all_rows: list[dict], usable_rows: list[dict]) -> dict:
    """For every (mouse_id, timepoint_weeks) group present anywhere in the
    FULL pre-exclusion inventory, checks whether it still has >= 1 usable
    slide. Generalizes the task's specific "does excluding 6041-4L-12W remove
    mouse 6041's only 12W slide" question to every mouse/timepoint, so any
    other group silently losing its only slide is also caught."""
    full_groups: dict[tuple, list[str]] = {}
    for r in all_rows:
        if r["mouse_id"] is None or r["timepoint_weeks"] is None:
            continue
        full_groups.setdefault((r["mouse_id"], r["timepoint_weeks"]), []).append(r["raw_stem"])

    usable_groups: dict[tuple, list[str]] = {}
    for r in usable_rows:
        usable_groups.setdefault((r["mouse_id"], r["timepoint_weeks"]), []).append(r["raw_stem"])

    zeroed_out = []
    all_group_status = []
    for key in sorted(full_groups, key=lambda k: (k[1], k[0])):
        mouse_id, weeks = key
        before = full_groups[key]
        after = usable_groups.get(key, [])
        entry = {
            "mouse_id": mouse_id, "timepoint_weeks": weeks,
            "slides_before": before, "slides_after": after,
            "n_before": len(before), "n_after": len(after),
            "dropped_to_zero": len(after) == 0,
        }
        all_group_status.append(entry)
        if entry["dropped_to_zero"]:
            zeroed_out.append(entry)

    return {
        "n_groups_checked": len(full_groups),
        "n_groups_dropped_to_zero": len(zeroed_out),
        "zeroed_out_groups": zeroed_out,
        "all_group_status": all_group_status,
    }


# ── Output writers ────────────────────────────────────────────────────────────

def write_report(
    all_rows: list[dict], usable_rows: list[dict], unexplained_gap_rows: list[dict],
    duplicate_check: dict, coverage: dict,
    counts_usable: dict, confound_usable: dict,
    output_dir: Path,
) -> None:
    lines = ["# Timepoint cohort -- Stage A v2: corrected cohort inventory", ""]

    lines.append(
        "**This is a full re-verification pass, not the coarse-level v1 run** -- "
        "Stage C conversion has since completed (22 of 24 remaining slides "
        "converted, joining the 7 already converted for 29 total). This report "
        "rebuilds the cohort inventory against the corrected, `timepoint_x5_full`-"
        "converted set."
    )
    lines.append("")

    lines.append(
        f"**Summary:** {len(all_rows)} .ndpi files in the full raw inventory. "
        f"{len(usable_rows)} form the corrected usable cohort. "
        f"{len(unexplained_gap_rows)} unexplained gap(s) (opens in OpenSlide, not "
        f"the confirmed duplicate, still has no converted PNG, and is not one of "
        f"the two known-corrupted stems)."
    )
    lines.append("")

    lines.append("## 60997-4L-4W-2 duplicate determination")
    lines.append("")
    lines.append(duplicate_check.get("determination", duplicate_check.get("reason", "n/a")))
    if duplicate_check.get("checked"):
        lines.append("")
        lines.append(
            f"- {DUPLICATE_STEM}: {duplicate_check[f'{DUPLICATE_STEM}_dims']}, "
            f"{_fmt(duplicate_check[f'{DUPLICATE_STEM}_size_bytes'] / (1024*1024) if duplicate_check[f'{DUPLICATE_STEM}_size_bytes'] else None)} MB"
        )
        lines.append(
            f"- {DUPLICATE_OF_STEM}: {duplicate_check[f'{DUPLICATE_OF_STEM}_dims']}, "
            f"{_fmt(duplicate_check[f'{DUPLICATE_OF_STEM}_size_bytes'] / (1024*1024) if duplicate_check[f'{DUPLICATE_OF_STEM}_size_bytes'] else None)} MB"
        )
    lines.append("")

    if unexplained_gap_rows:
        lines.append("## ⚠ UNEXPLAINED conversion gaps -- review before trusting the usable cohort")
        lines.append("")
        lines.append(
            "The following slide(s) parse, open in OpenSlide, are not the confirmed "
            "60997 duplicate, yet have no converted PNG in `timepoint_x5_full` -- and "
            "are NOT one of the two known-corrupted stems (6041-4L-12W, 6069-4R-4W). "
            "This is unexpected; check `conversion_summary.json` from Stage C for "
            "these stems before trusting counts below."
        )
        lines.append("")
        for r in unexplained_gap_rows:
            lines.append(f"- `{r['raw_stem']}` (mouse {r['mouse_id']}, {r['timepoint_weeks']}W)")
        lines.append("")

    lines.append("## Per-(mouse, timepoint) coverage check")
    lines.append("")
    lines.append(
        f"{coverage['n_groups_checked']} (mouse, timepoint) groups existed in the "
        f"full pre-exclusion inventory. **{coverage['n_groups_dropped_to_zero']} "
        f"dropped to zero usable slides** after applying the corrected exclusions."
    )
    lines.append("")
    if coverage["zeroed_out_groups"]:
        lines.append("**Groups with ZERO usable slides (flagged):**")
        lines.append("")
        for g in coverage["zeroed_out_groups"]:
            lines.append(
                f"- mouse {g['mouse_id']}, {g['timepoint_weeks']}W: had "
                f"{g['slides_before']}, now has none."
            )
        lines.append("")
    else:
        lines.append(
            "No group dropped to zero -- every mouse/timepoint combination present "
            "in the raw inventory retains at least one usable slide in the corrected "
            "cohort (e.g. mouse 6041 loses `6041-4L-12W` but keeps `6041-4R-12W` at "
            "12W; mouse 6069 loses `6069-4R-4W` but keeps `6069-4L-4W` at 4W)."
        )
        lines.append("")

    lines.append("### Full per-group detail")
    lines.append("")
    lines.append("| mouse | weeks | n before | n after | slides after | dropped to zero |")
    lines.append("|---|---|---|---|---|---|")
    for g in coverage["all_group_status"]:
        flag = "**YES**" if g["dropped_to_zero"] else "no"
        lines.append(
            f"| {g['mouse_id']} | {g['timepoint_weeks']} | {g['n_before']} | "
            f"{g['n_after']} | {', '.join(g['slides_after']) or '(none)'} | {flag} |"
        )
    lines.append("")

    lines.append("## Counts by timepoint -- corrected usable cohort")
    lines.append("")
    lines.append("| timepoint (weeks) | n slides | n distinct mice | mouse IDs |")
    lines.append("|---|---|---|---|")
    for key in sorted(counts_usable.keys(), key=lambda k: (k == "unparsed", k)):
        c = counts_usable[key]
        lines.append(f"| {key} | {c['n_slides']} | {c['n_distinct_mice']} | {', '.join(c['mouse_ids'])} |")
    lines.append("")

    lines.append("## Scan-date vs. timepoint confound -- corrected usable cohort")
    lines.append("")
    lines.append(confound_usable["interpretation"])
    if confound_usable["rho"] is not None:
        lines.append(
            f"\nrho={_fmt(confound_usable['rho'])}, p={_fmt(confound_usable['p'])}, "
            f"n_mice={confound_usable['n_mice']}"
        )
    lines.append("")
    lines.append(
        f"**v1 comparison (coarse, pre-Stage-C inventory):** "
        f"rho={V1_SCAN_DATE_CONFOUND['rho']:.4f}, p={V1_SCAN_DATE_CONFOUND['p']:.4f}, "
        f"n_mice={V1_SCAN_DATE_CONFOUND['n_mice']} -- not significant. "
    )
    if confound_usable["rho"] is not None:
        still_ns = abs(confound_usable["rho"]) < 0.5
        lines.append(
            "The corrected, larger cohort " +
            ("still shows no strong association (consistent with v1)."
             if still_ns else
             "now shows a STRONG association -- this is a CHANGE from v1 and "
             "should be treated as a new confound candidate, not dismissed by "
             "analogy to the earlier null result.")
        )
    lines.append("")

    lines.append("## Per-slide inventory (full raw set, with usability)")
    lines.append("")
    lines.append(
        "| raw stem | mouse | weeks | parse ok | opens? | png exists? | usable? | "
        "exclusion reason |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in all_rows:
        parse_flag = "yes" if r["parse_ok"] else "**NO**"
        opens_flag = "yes" if r["opens_in_openslide"] else "**NO**"
        png_flag = "yes" if r.get("png_exists") else "**NO**"
        usable_flag = "**yes**" if r.get("usable") else "no"
        lines.append(
            f"| {r['raw_stem']} | {_fmt(r['mouse_id'])} | {_fmt(r['timepoint_weeks'])} | "
            f"{parse_flag} | {opens_flag} | {png_flag} | {usable_flag} | "
            f"{r.get('exclusion_reason') or '—'} |"
        )
    lines.append("")

    lines.append(
        "## Verdict\n\n"
        "This stage is descriptive only -- it does not gate PASS/FAIL. Its usable-"
        "cohort output feeds Stage B v2 (timepoint_stain_homogeneity_v2.py), the "
        "hard gate.\n"
    )

    (output_dir / "stageA_inventory_v2.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    all_rows: list[dict], usable_rows: list[dict], unexplained_gap_rows: list[dict],
    duplicate_check: dict, coverage: dict,
    counts_usable: dict, mouse_6072: dict, confound_usable: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "all_slides": all_rows,
        "usable_slides": usable_rows,
        "unexplained_gap_slides": unexplained_gap_rows,
        "duplicate_60997_check": duplicate_check,
        "coverage_check": coverage,
        "counts_by_timepoint_usable": counts_usable,
        "mouse_6072_check": mouse_6072,
        "scan_date_confound_usable": confound_usable,
        "v1_scan_date_confound_comparison": V1_SCAN_DATE_CONFOUND,
    }
    json_path = output_dir / "stageA_inventory_v2.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  JSON: {json_path}")
    write_report(
        all_rows, usable_rows, unexplained_gap_rows, duplicate_check, coverage,
        counts_usable, confound_usable, output_dir,
    )
    print(f"  Markdown report: {output_dir / 'stageA_inventory_v2.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timepoint cohort Stage A v2: corrected cohort inventory (descriptive, not a gate)"
    )
    parser.add_argument("--ndpi-dir", required=True, type=Path, action="append",
                        help="Pass twice: once for $SCRATCH/data/timepoint_ndpi, once for "
                             "$SCRATCH/data/timepoint_ndpi_deferred")
    parser.add_argument("--converted-png-dir", required=True, type=Path,
                        help="$SCRATCH/data/timepoint_x5_full -- the Stage C output directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print("  Timepoint cohort -- Stage A v2: corrected cohort inventory")
    print("=" * 60)
    print(f"\nScanning directories: {[str(d) for d in args.ndpi_dir]}")
    print(f"Converted PNG dir: {args.converted_png_dir}")

    rows = build_inventory(args.ndpi_dir)
    print(f"\nFound {len(rows)} .ndpi files")

    annotate_png_existence(rows, args.converted_png_dir)

    duplicate_check = check_duplicate_60997(rows)
    print(f"\n=== 60997-4L-4W-2 duplicate determination ===\n  {duplicate_check.get('determination', duplicate_check.get('reason'))}")

    usable_rows, unexplained_gap_rows = build_usable_cohort(rows, duplicate_check)
    print(f"\n=== Usable cohort ===\n  {len(usable_rows)} usable slides "
          f"(of {len(rows)} total in raw inventory)")
    if unexplained_gap_rows:
        print(f"  *** {len(unexplained_gap_rows)} UNEXPLAINED gap(s): "
              f"{[r['raw_stem'] for r in unexplained_gap_rows]}")

    coverage = coverage_check(rows, usable_rows)
    print(f"\n=== Coverage check ===\n  {coverage['n_groups_dropped_to_zero']} of "
          f"{coverage['n_groups_checked']} (mouse, timepoint) groups dropped to zero usable slides")

    counts_usable = counts_by_timepoint(usable_rows)
    mouse_6072 = mouse_6072_check(usable_rows)
    confound_usable = scan_date_vs_timepoint_confound(usable_rows)

    print("\n=== Counts by timepoint (usable cohort) ===")
    for k, v in sorted(counts_usable.items()):
        print(f"  {k}W: {v['n_slides']} slides, {v['n_distinct_mice']} mice")

    print(f"\n=== Scan-date confound (usable cohort) ===\n  {confound_usable['interpretation']}")
    print(f"  v1 comparison: rho={V1_SCAN_DATE_CONFOUND['rho']}, n_mice={V1_SCAN_DATE_CONFOUND['n_mice']}")

    write_outputs(
        rows, usable_rows, unexplained_gap_rows, duplicate_check, coverage,
        counts_usable, mouse_6072, confound_usable, args.output_dir,
    )

    print("\n" + "=" * 60)
    print("  STAGE A v2 COHORT INVENTORY COMPLETE")
    print("=" * 60)
    print(f"\n  Output dir: {args.output_dir}")
    print("\n  Next: analysis/timepoint_stain_homogeneity_v2.py (Stage B v2, HARD GATE) "
          "consumes this inventory's usable_slides.")


if __name__ == "__main__":
    main()
