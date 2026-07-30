"""
Crop calibration: read-only diagnostic phase, hard gate before any re-conversion.

Stage 1's left/right HSV tissue-fraction diagnostic (analysis/timepoint_inventory.py)
flagged all 7 converted timepoint slides for manual review -- right/left ratios
0.30-1.05, not the near-zero expected if the discarded right half were blank
margin. But that diagnostic has never been run on the ORIGINAL 16 pipeline
slides, which are documented as using the exact same "two copies side by side"
NDPI layout. Without that baseline, the 0.30-1.05 range on the new slides is an
uncalibrated measurement -- it could mean the new batch genuinely differs, or it
could just be what this diagnostic always reports on the known-safe layout.

This module does two things, both strictly read-only:
  Task A: runs the EXISTING, UNMODIFIED left/right tissue-fraction diagnostic
    (imported from analysis/timepoint_inventory.py, not reimplemented) on all 16
    original slides' raw NDPI files, and compares the resulting range/median
    directly against the 7 new timepoint slides' ALREADY-RECORDED values from
    Stage 1 (reused, not recomputed).
  Task B: extracts every embedded associated image (Hamamatsu NDPI files
    typically carry 'macro'/'label' overview images) for both batches, saves
    each as PNG, and builds one labelled contact sheet per batch -- a single
    scriptable view of physical slide layout without opening each file
    individually in QuPath.

Both tasks read ONLY the NDPI's coarsest pyramid level or small embedded
associated images -- never full resolution (a prior Stage 1 attempt OOM-killed
reading full resolution during actual conversion). Both tasks catch per-slide
exceptions, log the slide name and error, and continue -- a single unreadable
file (e.g. 6069-4R-4W, known to fail with a JPEG "Restart marker not found"
error) must not abort the whole run.

Does NOT modify run_all.py, analysis/timepoint_inventory.py, or
analysis/timepoint_stage2_stain_check.py -- only imports from them. Does NOT
convert any NDPI, write any PNG into an existing pipeline directory, or touch
any existing sidecar. Writes only to a NEW output directory.

CLI
---
  python -m cancer_trajectory_atlas.analysis.crop_calibration \\
      --original-ndpi-dir         $SCRATCH/data/MCF7_x5 \\
      --original-slide-lists      ~/cancer_trajectory_atlas/jobs/slides_section1.txt \\
                                   ~/cancer_trajectory_atlas/jobs/slides_section2.txt \\
      --timepoint-ndpi-dir        $SCRATCH/data/timepoint_ndpi \\
      --timepoint-ndpi-deferred-dir $SCRATCH/data/timepoint_ndpi_deferred \\
      --timepoint-slide-list      ~/cancer_trajectory_atlas/jobs/slides_timepoint.txt \\
      --stage1-inventory-json     $SCRATCH/results/timepoint_projection/stage1_convert/stage1_inventory.json \\
      --output-dir                $SCRATCH/results/crop_calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .timepoint_inventory import ndpi_scale_and_crop_check, load_slide_list
from .timepoint_stage2_stain_check import _normalize_stem

NEAR_ZERO_RATIO_THRESHOLD = 0.10
PREFERRED_ASSOCIATED_IMAGE_KEYS = ("macro", "label")


def _md_escape(s) -> str:
    """Escape literal '|' so arbitrary error-message content can't break a
    markdown table row (same lesson/fix as holeyness_final.py's results table)."""
    return str(s).replace("|", "\\|") if s else ""


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# ── Task A: calibrate the existing diagnostic on the 16 original slides ──────

def run_calibration_on_originals(original_ndpi_dir: Path, original_stems: list[str]) -> dict:
    """Runs the EXISTING, UNMODIFIED ndpi_scale_and_crop_check (which internally
    calls left_right_tissue_check) on each original slide's raw NDPI, reading
    only the coarsest pyramid level. sidecar_dims=None since there's no
    timepoint-style sidecar for the originals to cross-check scale against --
    only the nested left_right_tissue_check result is used."""
    per_slide: dict = {}
    errors: dict = {}
    for stem in original_stems:
        path = original_ndpi_dir / f"{stem}.ndpi"
        if not path.exists():
            errors[stem] = "raw NDPI file not found"
            continue
        try:
            check = ndpi_scale_and_crop_check(path, 1.0, None)
            per_slide[stem] = check["left_right_tissue_check"]
        except Exception as e:
            errors[stem] = repr(e)
    return {"per_slide": per_slide, "errors": errors}


def summarize_ratios(per_slide: dict) -> dict:
    ratios = [
        v["right_over_left_ratio"] for v in per_slide.values()
        if np.isfinite(v["right_over_left_ratio"])
    ]
    if not ratios:
        return {"n": 0, "min": None, "max": None, "median": None}
    return {
        "n": len(ratios),
        "min": float(np.min(ratios)),
        "max": float(np.max(ratios)),
        "median": float(np.median(ratios)),
    }


def load_timepoint_ratios_from_stage1(stage1_inventory_json: Path) -> dict:
    """Reuses Stage 1's already-recorded left_right_tissue_check values for the
    successfully-converted timepoint slides -- does NOT recompute them."""
    with open(stage1_inventory_json) as f:
        data = json.load(f)
    per_slide = {}
    for row in data["slides"]:
        nc = row.get("ndpi_check")
        if nc and "left_right_tissue_check" in nc:
            per_slide[row["slide_stem"]] = nc["left_right_tissue_check"]
    return per_slide


def build_verdict(orig_summary: dict, tp_summary: dict) -> str:
    """States plainly which of the two possible outcomes holds, or reports an
    overlap/ambiguity explicitly rather than forcing a side."""
    if orig_summary["n"] == 0 or tp_summary["n"] == 0:
        return (
            f"UNDETERMINED -- too few successfully-read slides in one or both "
            f"groups to compare (originals n={orig_summary['n']}, "
            f"new slides n={tp_summary['n']})."
        )

    o_min, o_max, o_med = orig_summary["min"], orig_summary["max"], orig_summary["median"]
    t_min, t_max, t_med = tp_summary["min"], tp_summary["max"], tp_summary["median"]
    ranges_overlap = max(o_min, t_min) <= min(o_max, t_max)

    if ranges_overlap:
        return (
            f"(i) OVERLAPPING -- originals range [{o_min:.3f}, {o_max:.3f}] "
            f"(median {o_med:.3f}, n={orig_summary['n']}) and new slides range "
            f"[{t_min:.3f}, {t_max:.3f}] (median {t_med:.3f}, n={tp_summary['n']}) "
            f"overlap. The diagnostic does not cleanly discriminate between the "
            f"two batches by this measure alone -- the new slides' ratios are "
            f"NOT clearly anomalous relative to the originals' own known-safe "
            f"two-copies layout."
        )
    if o_max < t_min:
        return (
            f"(ii) SEPARATED -- originals range [{o_min:.3f}, {o_max:.3f}] "
            f"(median {o_med:.3f}, n={orig_summary['n']}) sits entirely below "
            f"the new slides' range [{t_min:.3f}, {t_max:.3f}] "
            f"(median {t_med:.3f}, n={tp_summary['n']}). The new batch "
            f"genuinely differs in physical slide layout from the originals by "
            f"this measure."
            + (
                " Originals are near-zero as expected for a blank duplicate half."
                if o_max < NEAR_ZERO_RATIO_THRESHOLD else
                " Note: originals are NOT near-zero either (median "
                f"{o_med:.3f}) -- both batches show non-trivial right-half "
                "tissue content, just at different magnitudes; interpret with "
                "that in mind rather than assuming the originals' right half "
                "is blank."
            )
        )
    return (
        f"SEPARATED, OPPOSITE DIRECTION FROM EXPECTED -- originals range "
        f"[{o_min:.3f}, {o_max:.3f}] (median {o_med:.3f}, n={orig_summary['n']}) "
        f"sits entirely ABOVE the new slides' range [{t_min:.3f}, {t_max:.3f}] "
        f"(median {t_med:.3f}, n={tp_summary['n']}). This is a genuine "
        f"separation but not in the direction the 'originals are near-zero' "
        f"hypothesis predicted -- worth a closer look before drawing any "
        f"conclusion, not forced into either (i) or (ii)."
    )


# ── Task B: extract embedded associated (macro/label) images ─────────────────

def _resolve_timepoint_path(stem: str, primary_dir: Path, deferred_dir: Path | None) -> Path | None:
    p = primary_dir / f"{stem}.ndpi"
    if p.exists():
        return p
    if deferred_dir is not None:
        p2 = deferred_dir / f"{stem}.ndpi"
        if p2.exists():
            return p2
    return None


def run_macro_extraction(
    stems_and_paths: list[tuple[str, Path | None]],
    batch_label: str,
    output_dir: Path,
) -> dict:
    import openslide  # lazy import -- only needed on Narval, where it's module-loaded

    macro_dir = output_dir / "macro_images" / batch_label
    macro_dir.mkdir(parents=True, exist_ok=True)

    per_slide: dict = {}
    panel_images: dict = {}

    for stem, path in stems_and_paths:
        if path is None:
            per_slide[stem] = {"available_keys": [], "error": "raw NDPI file not found"}
            panel_images[stem] = None
            continue
        try:
            slide = openslide.OpenSlide(str(path))
            try:
                keys = list(slide.associated_images.keys())
                saved = []
                for key in keys:
                    img = slide.associated_images[key].convert("RGB")
                    img.save(macro_dir / f"{stem}__{key}.png")
                    saved.append(key)
                per_slide[stem] = {"available_keys": saved, "error": None}

                preferred = None
                for pref in PREFERRED_ASSOCIATED_IMAGE_KEYS:
                    if pref in slide.associated_images:
                        preferred = slide.associated_images[pref].convert("RGB")
                        break
                if preferred is None and keys:
                    preferred = slide.associated_images[keys[0]].convert("RGB")
                panel_images[stem] = preferred
            finally:
                slide.close()
        except Exception as e:
            per_slide[stem] = {"available_keys": [], "error": repr(e)}
            panel_images[stem] = None

    return {"per_slide": per_slide, "panel_images": panel_images, "macro_dir": str(macro_dir)}


def write_contact_sheet(panel_images: dict, output_path: Path, batch_label: str) -> None:
    stems = list(panel_images.keys())
    n = len(stems)
    ncols = min(4, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)

    for i, stem in enumerate(stems):
        ax = axes[i // ncols][i % ncols]
        img = panel_images[stem]
        if img is not None:
            ax.imshow(np.array(img))
        else:
            ax.text(
                0.5, 0.5, "no associated\nimage available",
                ha="center", va="center", fontsize=8, transform=ax.transAxes,
            )
            ax.set_facecolor("#eeeeee")
        ax.set_title(stem, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Crop calibration — associated images, {batch_label} batch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Output writers ────────────────────────────────────────────────────────────

def write_report(
    output_dir: Path,
    step0_notes: dict,
    task_a: dict,
    task_b_original: dict,
    task_b_timepoint: dict,
) -> None:
    lines = ["# Crop calibration — read-only diagnostic, hard gate", ""]

    lines += ["## Step 0 — paths and function provenance", ""]
    for note in step0_notes["notes"]:
        lines.append(f"- {note}")
    lines.append("")

    lines += ["## Task A — calibration on the 16 original pipeline slides", ""]
    lines.append("| slide | left frac | right frac | ratio | error |")
    lines.append("|---|---|---|---|---|")
    for stem in task_a["original_stems"]:
        v = task_a["originals"]["per_slide"].get(stem)
        err = task_a["originals"]["errors"].get(stem)
        if v:
            lines.append(
                f"| {stem} | {_fmt(v['left_tissue_fraction'])} | "
                f"{_fmt(v['right_tissue_fraction'])} | {_fmt(v['right_over_left_ratio'])} | |"
            )
        else:
            lines.append(f"| {stem} | | | | {_md_escape(err) or 'unknown error'} |")
    lines.append("")

    o_sum = task_a["originals_summary"]
    t_sum = task_a["timepoint_summary"]
    lines.append(
        f"**Originals:** n={o_sum['n']}, range [{_fmt(o_sum['min'])}, {_fmt(o_sum['max'])}], "
        f"median {_fmt(o_sum['median'])}"
    )
    lines.append(
        f"**New timepoint slides (reused from Stage 1, not recomputed):** "
        f"n={t_sum['n']}, range [{_fmt(t_sum['min'])}, {_fmt(t_sum['max'])}], "
        f"median {_fmt(t_sum['median'])}"
    )
    lines.append("")
    lines.append(f"**Verdict:** {task_a['verdict']}")
    lines.append("")

    lines += ["## Task B — embedded associated image (macro/label) extraction", ""]
    lines.append("### Original batch")
    lines.append("| slide | available keys | error |")
    lines.append("|---|---|---|")
    for stem, v in task_b_original["per_slide"].items():
        lines.append(f"| {stem} | {', '.join(v['available_keys']) or 'none'} | {_md_escape(v['error'])} |")
    lines.append("")
    lines.append("### Timepoint batch")
    lines.append("| slide | available keys | error |")
    lines.append("|---|---|---|")
    for stem, v in task_b_timepoint["per_slide"].items():
        lines.append(f"| {stem} | {', '.join(v['available_keys']) or 'none'} | {_md_escape(v['error'])} |")
    lines.append("")
    lines.append(
        "Contact sheets: `contact_sheet_original.png`, `contact_sheet_timepoint.png` "
        "(one panel per slide, blank-labelled panels for slides with no associated "
        "image or a read error)."
    )
    lines.append("")

    lines += ["## Errors encountered", ""]
    any_errors = False
    for stem, err in task_a["originals"]["errors"].items():
        lines.append(f"- Task A, {stem}: {err}")
        any_errors = True
    for stem, v in task_b_original["per_slide"].items():
        if v["error"]:
            lines.append(f"- Task B (original), {stem}: {v['error']}")
            any_errors = True
    for stem, v in task_b_timepoint["per_slide"].items():
        if v["error"]:
            lines.append(f"- Task B (timepoint), {stem}: {v['error']}")
            any_errors = True
    if not any_errors:
        lines.append("None.")
    lines.append("")

    lines += ["## Hard gate", ""]
    lines.append(
        "**STOP HERE.** This phase is read-only diagnostics only. No conversion "
        "script has been written, no slide has been re-converted, and no "
        "existing PNG or `slide_dimensions.json` has been modified. The "
        "decision about whether and how to re-convert any slide depends on "
        "Task A's verdict above and should be made after reviewing this "
        "report, not automatically from this job."
    )
    lines.append("")

    (output_dir / "crop_calibration_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    step0_notes: dict,
    task_a: dict,
    task_b_original: dict,
    task_b_timepoint: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # panel_images (PIL Image objects) aren't JSON-serializable and aren't
    # meaningful in the JSON output -- strip them before writing.
    task_b_original_json = {"per_slide": task_b_original["per_slide"], "macro_dir": task_b_original["macro_dir"]}
    task_b_timepoint_json = {"per_slide": task_b_timepoint["per_slide"], "macro_dir": task_b_timepoint["macro_dir"]}

    result = {
        "step0_notes": step0_notes,
        "task_a": task_a,
        "task_b_original": task_b_original_json,
        "task_b_timepoint": task_b_timepoint_json,
    }
    json_path = output_dir / "crop_calibration.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"  JSON: {json_path}")

    write_report(output_dir, step0_notes, task_a, task_b_original_json, task_b_timepoint_json)
    print(f"  Markdown report: {output_dir / 'crop_calibration_report.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop calibration: read-only diagnostic phase, hard gate before re-conversion"
    )
    parser.add_argument("--original-ndpi-dir", required=True, type=Path,
                        help="Raw NDPI directory for the 16 original pipeline slides "
                             "(jobs/convert_ndpi.sh's proven convention, e.g. $SCRATCH/data/MCF7_x5 "
                             "-- NOTE this differs from paths.json's 'raw_ndpi' value)")
    parser.add_argument("--original-slide-lists", required=True, type=Path, nargs="+",
                        help="e.g. jobs/slides_section1.txt jobs/slides_section2.txt (16 total, _x5-suffixed)")
    parser.add_argument("--timepoint-ndpi-dir", required=True, type=Path)
    parser.add_argument("--timepoint-ndpi-deferred-dir", default=None, type=Path,
                        help="Where slides set aside (e.g. 6069-4R-4W) currently sit, if not in --timepoint-ndpi-dir")
    parser.add_argument("--timepoint-slide-list", required=True, type=Path)
    parser.add_argument("--stage1-inventory-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print("  Crop calibration — read-only diagnostic, hard gate")
    print("=" * 60)

    original_stems_raw: list[str] = []
    for list_path in args.original_slide_lists:
        original_stems_raw.extend(load_slide_list(list_path))
    original_stems = [_normalize_stem(s) for s in original_stems_raw]
    print(f"\nOriginal slides ({len(original_stems)}): {original_stems}")

    timepoint_stems = [_normalize_stem(s) for s in load_slide_list(args.timepoint_slide_list)]
    print(f"Timepoint slides ({len(timepoint_stems)}): {timepoint_stems}")

    step0_notes = {
        "notes": [
            "paths.json's 'raw_ndpi' = '~/scratch/data/ndpi', but the proven-working "
            "job script (jobs/convert_ndpi.sh, which actually produced the existing "
            "16 slides' PNGs) hardcodes a DIFFERENT path. This job uses the "
            f"job-script convention as authoritative: --original-ndpi-dir="
            f"{args.original_ndpi_dir} (matches the documented Issue 5 discrepancy "
            "pattern already noted in PROJECT_STATE.md for annotations).",
            "The existing HSV tissue-fraction diagnostic is "
            "left_right_tissue_check() in analysis/timepoint_inventory.py:81-99, "
            "called here via the existing, unmodified ndpi_scale_and_crop_check() "
            "wrapper (timepoint_inventory.py:105-139) with sidecar_dims=None -- "
            "identical code path to Stage 1's, not a reimplementation.",
            "That wrapper reads slide.level_count - 1 (timepoint_inventory.py:115), "
            "which is by construction OpenSlide's coarsest available pyramid "
            "level. Confirmed: both tasks here read only that level or small "
            "embedded associated images, never full resolution.",
        ]
    }
    print("\n=== Step 0 ===")
    for n in step0_notes["notes"]:
        print(f"  - {n}")

    print("\n=== Task A: calibration on 16 original slides ===")
    originals = run_calibration_on_originals(args.original_ndpi_dir, original_stems)
    for stem, err in originals["errors"].items():
        print(f"  ERROR {stem}: {err}")
    originals_summary = summarize_ratios(originals["per_slide"])
    timepoint_ratios = load_timepoint_ratios_from_stage1(args.stage1_inventory_json)
    timepoint_summary = summarize_ratios(timepoint_ratios)
    verdict = build_verdict(originals_summary, timepoint_summary)
    print(f"  originals summary: {originals_summary}")
    print(f"  timepoint summary (reused from Stage 1): {timepoint_summary}")
    print(f"  verdict: {verdict}")

    task_a = {
        "original_stems": original_stems,
        "originals": originals,
        "originals_summary": originals_summary,
        "timepoint_ratios_reused_from_stage1": timepoint_ratios,
        "timepoint_summary": timepoint_summary,
        "verdict": verdict,
    }

    print("\n=== Task B: associated (macro/label) image extraction ===")
    original_paths = [(s, args.original_ndpi_dir / f"{s}.ndpi") for s in original_stems]
    original_paths = [(s, p if p.exists() else None) for s, p in original_paths]
    task_b_original = run_macro_extraction(original_paths, "original", args.output_dir)

    timepoint_paths = [
        (s, _resolve_timepoint_path(s, args.timepoint_ndpi_dir, args.timepoint_ndpi_deferred_dir))
        for s in timepoint_stems
    ]
    task_b_timepoint = run_macro_extraction(timepoint_paths, "timepoint", args.output_dir)

    for stem, v in {**task_b_original["per_slide"], **task_b_timepoint["per_slide"]}.items():
        print(f"  {stem}: keys={v['available_keys']} error={v['error']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_contact_sheet(
        task_b_original["panel_images"], args.output_dir / "contact_sheet_original.png", "original"
    )
    write_contact_sheet(
        task_b_timepoint["panel_images"], args.output_dir / "contact_sheet_timepoint.png", "timepoint"
    )

    write_outputs(args.output_dir, step0_notes, task_a, task_b_original, task_b_timepoint)

    print("\n" + "=" * 60)
    print("  CROP CALIBRATION COMPLETE — HARD GATE, STOP HERE")
    print("=" * 60)
    print(f"\n  Task A verdict: {verdict}")
    print(f"  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
