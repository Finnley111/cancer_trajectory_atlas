"""
Timepoint cohort: Stage F -- ROI / patch-composition mismatch check.

WHY THIS EXISTS
---------------
Stage D projected all 29 timepoint slides onto the manifold and EVERY slide
returned frac_beyond_training_p99 = 1.0 -- total extrapolation, with no
discrimination between slides (training's own median first-neighbour distance
19.3, p99 25.5; every projected slide sits uniformly at 32-40). Stage E's
pseudotime-vs-weeks correlation was therefore computed entirely outside the
training manifold's support, which makes it uninterpretable as it stands.

The hypothesis under test: the 29 timepoint slides have no annotations, so
Stage D patched them WHOLE-SLIDE, while the training manifold was built
EXCLUSIVELY from annotated tumor ROI patches. Stroma, necrosis and other
non-tumor tissue are therefore present in the projection set but were never in
training. That is a patch-COMPOSITION mismatch, distinct from (and possibly
larger than) the hematoxylin-intensity confound Stage B found.

This module exists to test whether Stage D/E's correlation is an artifact of
comparing tumor-only training data against whole-slide projection data. It is
NOT an attempt to validate the timepoint hypothesis, and it deliberately does
NOT conclude whether timepoint projection is valid -- it reports which of three
pre-specified outcomes the evidence matches and stops.

WHAT IS RE-USED RATHER THAN RECOMPUTED (this is mostly re-analysis)
------------------------------------------------------------------
  - Stage D already saved PER-PATCH arrays, index-aligned with its Phikon
    cache: per_slide/projected_pt_{stem}.npy and per_slide/nn_mean_knn_{stem}.npy.
    So Task B's filtered extrapolation rate and Task C's filtered pseudotime
    are pure MASKING of saved arrays -- no re-projection, no GPU, no Phikon.
  - The training manifold's per-patch morphology already exists in its
    results.csv (run_all.py:624-635 writes all six morphological features
    alongside x/y/slide/cluster/pseudotime). That is the reference
    distribution for Task A and the Task B threshold -- read, never
    recomputed, which is what "do not re-extract features for the training
    cohort" requires.
  - validation/morphological_features.py::compute_morphological_features --
    the exact six-feature function used everywhere else in this pipeline.
  - analysis/timepoint_diagnostic.py::compute_block -- Task C's statistics are
    produced by the SAME code that produced the Stage E numbers they are
    compared against, so any difference is the filtering, not the method.
  - analysis/timepoint_projection.py::aggregate_to_mouse_level -- identical
    (mouse_id, timepoint_weeks) aggregation as every prior stage.

The only genuinely new computation is regenerating timepoint patch PIXELS
(Stage D cached 768-dim vectors, not images) and running the six features on a
per-slide sample of them.

THE INDEX-ALIGNMENT ASSUMPTION, AND HOW IT IS CHECKED
-----------------------------------------------------
get_patches_from_array is deterministic, so re-running it with Stage D's exact
settings reproduces patches in the same order as Stage D's cached features and
saved per-patch arrays. Everything here depends on that. It is therefore
CHECKED, not assumed: each slide's regenerated patch count must equal the
length of its saved projected_pt array, and a mismatch hard-fails that slide
rather than silently mis-indexing every downstream number.

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_roi_mismatch \\
      --stageA-inventory-json $SCRATCH/results/timepoint_cohort/stageA_inventory_v2/stageA_inventory_v2.json \\
      --stageD-json           $SCRATCH/results/timepoint_cohort/stageD_projection/stageD_projection.json \\
      --stageD-per-slide-dir  $SCRATCH/results/timepoint_cohort/stageD_projection/per_slide \\
      --stageE-json           $SCRATCH/results/timepoint_cohort/stageE_diagnostic/stageE_diagnostic.json \\
      --training-results-csv  $SCRATCH/results/baseline/atlas_none_harmony_median/results.csv \\
      --png-dir               $SCRATCH/data/timepoint_x5_full \\
      --output-dir            $SCRATCH/results/timepoint_cohort/stageF_roi_mismatch_check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from ..features.patching import get_patches_from_array
from .holeyness import _safe_spearman
from .timepoint_diagnostic import H_MEASURES, PARTIAL_MIN_N, compute_block
from .timepoint_projection import aggregate_to_mouse_level
from .timepoint_stage2_stain_check import _fmt, _png_path

# The six features computed by validation/morphological_features.py. Named here
# only so the report iterates them in a stable order.
MORPH_FEATURES = [
    "nuclear_density", "mean_nuclear_area", "nc_ratio",
    "texture_entropy", "h_intensity", "packing_irregularity",
]

# Primary tumor-likeness criterion. nuclear_density is chosen because it is the
# same feature this pipeline already uses for multi-root DPT root selection, so
# its behaviour on this data is partially characterised rather than novel.
FILTER_FEATURE = "nuclear_density"

# Threshold variants, all expressed as quantile pairs of the TRAINING
# distribution. IQR is the primary (as specified in the task brief); the wider
# bands are reported as a sensitivity ladder because the IQR by construction
# excludes half of the training distribution ITSELF, which would make the
# filter look artificially restrictive if it were reported alone.
THRESHOLD_VARIANTS = {
    "IQR (p25-p75, PRIMARY)": (25, 75),
    "p5-p95": (5, 95),
    "p1-p99": (1, 99),
}
PRIMARY_THRESHOLD = "IQR (p25-p75, PRIMARY)"

# Minimum surviving filtered patches for a slide to contribute a median.
# STATED AND APPLIED, never silent; slides below it are reported as excluded
# with a reason. The ladder mirrors holeyness.py's own N_PATCH_THRESHOLDS
# convention of reporting sensitivity across counts rather than asserting one
# (that constant is per-DUCT, so it is precedent for the reporting style, not a
# number reused here).
MIN_FILTERED_PATCHES = 50
PATCH_COUNT_LADDER = (10, 20, 50, 100)

# CHOSEN CONVENTIONS for the three-outcome verdict, not values from the task
# brief -- flagged here and in the report so they are visible and adjustable.
# A cohort-median frac_beyond_p99 must fall by at least this much (absolute) to
# count as "meaningfully reduced".
EXTRAPOLATION_MEANINGFUL_DROP = 0.25
# Below this, extrapolation is additionally called "largely resolved" (the
# training manifold's own rate beyond its p99 is ~0.01 by definition).
EXTRAPOLATION_LARGELY_RESOLVED = 0.10
# The filtered raw correlation must retain at least this fraction of the
# original |rho|, with the same sign, to count as "surviving".
CORRELATION_SURVIVES_FRACTION = 0.50


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- correct near p=0, unlike the normal
    approximation, which matters precisely in the 'very few patches survive'
    case that is itself a finding here."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ── Training reference (read, never recomputed) ──────────────────────────────

def load_training_reference(csv_path: Path) -> dict:
    """Per-patch morphology of the manifold's OWN patches, straight from the
    training run's results.csv."""
    import pandas as pd

    if not csv_path.exists():
        sys.exit(
            f"ERROR: training results.csv not found:\n  {csv_path}\n"
            "This is the reference distribution for both Task A and the Task B "
            "threshold. It is written by run_all.py alongside the manifold; confirm "
            "--training-results-csv points at the SAME run as the projector Stage D "
            "used, or the reference will not describe the manifold being projected onto."
        )
    df = pd.read_csv(csv_path)
    missing = [f for f in MORPH_FEATURES if f not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: {csv_path} is missing morphological columns {missing}.\n"
            f"Present columns: {list(df.columns)}\n"
            "Stage F cannot build a reference distribution without them."
        )
    ref = {f: df[f].to_numpy(dtype=float) for f in MORPH_FEATURES}
    ref = {f: v[np.isfinite(v)] for f, v in ref.items()}
    print(f"  Training reference: {len(df)} patches from {csv_path}")
    return ref


# ── Per-slide morphology on a sample (the only new computation) ──────────────

def compute_slide_morphology(
    png_path: Path, stem: str, n_expected: int, sample_size: int, seed: int,
    patch_size: int, stride: int, cache_dir: Path,
) -> dict:
    """Regenerates patches, verifies index alignment against Stage D, samples,
    and computes the six features on the sample. Cached per slide so a walltime
    timeout resumes rather than restarts."""
    from PIL import Image
    from ..validation.morphological_features import compute_morphological_features

    cache_file = cache_dir / f"{stem}_morph.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        print(f"    Morph cache hit: {cache_file.name}")
        return {"sample_idx": z["sample_idx"], "n_patches_total": int(z["n_patches_total"]),
                "features": {f: z[f] for f in MORPH_FEATURES}}

    Image.MAX_IMAGE_PIXELS = None
    img_arr = np.array(Image.open(png_path).convert("RGB"))
    print(f"    Image: {img_arr.shape[1]} x {img_arr.shape[0]}")

    # Stage D's exact settings: whole-slide, no ROI (these slides have no
    # annotations). Deterministic -> same order as Stage D's cached arrays.
    patches, _coords = get_patches_from_array(
        img_arr, patch_size=patch_size, stride=stride, image_name=stem,
        roi_polygons=None, exclude_polygons=None, min_roi_coverage=None,
    )
    del img_arr

    n_total = len(patches)
    if n_total != n_expected:
        raise ValueError(
            f"INDEX ALIGNMENT FAILURE for {stem}: regenerated {n_total} patches but "
            f"Stage D saved {n_expected} per-patch values. Patch extraction is not "
            "reproducing Stage D's order, so every filtered index would silently point "
            "at the wrong patch. Refusing to use this slide. (Check that --patch-size/"
            "--stride/--png-dir match Stage D's extraction_settings.)"
        )

    rng = np.random.default_rng(abs(hash(stem)) % (2**32) ^ seed)
    n_sample = int(min(sample_size, n_total)) if sample_size else n_total
    sample_idx = np.sort(rng.choice(n_total, size=n_sample, replace=False))

    print(f"    Computing 6 morphological features on {n_sample} of {n_total} patches")
    feats = compute_morphological_features(patches[sample_idx], use_stardist=False)
    del patches

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, sample_idx=sample_idx,
                        n_patches_total=np.array(n_total), **feats)
    return {"sample_idx": sample_idx, "n_patches_total": n_total, "features": feats}


# ── Task A: characterize the extrapolating tissue ────────────────────────────

def task_a_distributions(timepoint_feats: dict, training_ref: dict) -> dict:
    """Side-by-side distribution comparison on all six features. Reported
    BEFORE any filtering so the reader learns what the extrapolated tissue
    actually looks like before seeing whether filtering changes anything."""
    out = {}
    for f in MORPH_FEATURES:
        tp = timepoint_feats[f][np.isfinite(timepoint_feats[f])]
        tr = training_ref[f]
        if len(tp) == 0 or len(tr) == 0:
            out[f] = {"insufficient_data": True}
            continue
        ks_stat, ks_p = ks_2samp(tp, tr)
        out[f] = {
            "timepoint": {
                "n": int(len(tp)), "median": float(np.median(tp)),
                "q25": float(np.percentile(tp, 25)), "q75": float(np.percentile(tp, 75)),
                "mean": float(np.mean(tp)), "sd": float(np.std(tp)),
            },
            "training": {
                "n": int(len(tr)), "median": float(np.median(tr)),
                "q25": float(np.percentile(tr, 25)), "q75": float(np.percentile(tr, 75)),
                "mean": float(np.mean(tr)), "sd": float(np.std(tr)),
            },
            "ks_stat": float(ks_stat), "ks_p": float(ks_p),
            "wasserstein": float(wasserstein_distance(tp, tr)),
            "median_ratio_timepoint_over_training": (
                float(np.median(tp) / np.median(tr)) if np.median(tr) != 0 else None),
        }
    return out


# ── Task B: filter to tumor-like, re-test extrapolation ──────────────────────

def build_thresholds(training_ref: dict) -> dict:
    v = training_ref[FILTER_FEATURE]
    return {
        name: {"lo": float(np.percentile(v, lo)), "hi": float(np.percentile(v, hi)),
               "quantiles": [lo, hi]}
        for name, (lo, hi) in THRESHOLD_VARIANTS.items()
    }


def task_b_filter(
    per_slide_morph: dict, slide_meta: dict, thresholds: dict,
    stageD_per_slide: dict, training_p99: float,
) -> dict:
    """For each threshold variant and each slide: which sampled patches pass,
    and what the extrapolation rate is among them. The extrapolation rate is
    recomputed by masking Stage D's OWN saved nn_mean_knn array against the
    SAME training p99 Stage D used -- so filtered and unfiltered numbers are
    directly comparable by construction, not by re-derivation."""
    results = {}
    for tname, t in thresholds.items():
        per_slide = []
        for stem, m in per_slide_morph.items():
            nd = m["features"][FILTER_FEATURE]
            idx = m["sample_idx"]
            nn = stageD_per_slide[stem]["nn_mean_knn"][idx]  # sample-aligned
            keep = (nd >= t["lo"]) & (nd <= t["hi"]) & np.isfinite(nd)
            n_keep = int(keep.sum())
            n_samp = int(len(idx))
            lo_ci, hi_ci = _wilson_ci(n_keep, n_samp)
            frac_unfiltered = float(np.mean(nn > training_p99))
            frac_filtered = float(np.mean(nn[keep] > training_p99)) if n_keep else None
            per_slide.append({
                "raw_stem": stem,
                "mouse_id": slide_meta[stem]["mouse_id"],
                "timepoint_weeks": slide_meta[stem]["timepoint_weeks"],
                "has_suffix_slide": slide_meta[stem]["has_suffix_slide"],
                "n_sampled": n_samp,
                "n_surviving": n_keep,
                "frac_surviving": n_keep / n_samp if n_samp else float("nan"),
                "frac_surviving_ci95": [lo_ci, hi_ci],
                "implied_count_of_all_patches": (
                    int(round(n_keep / n_samp * m["n_patches_total"])) if n_samp else None),
                "n_patches_total": m["n_patches_total"],
                "frac_beyond_p99_unfiltered_sample": frac_unfiltered,
                "frac_beyond_p99_filtered": frac_filtered,
                "stageD_frac_beyond_p99_allpatches": stageD_per_slide[stem]["stageD_frac"],
            })
        filt = [s["frac_beyond_p99_filtered"] for s in per_slide
                if s["frac_beyond_p99_filtered"] is not None]
        unfilt = [s["frac_beyond_p99_unfiltered_sample"] for s in per_slide]
        results[tname] = {
            "threshold": thresholds[tname],
            "per_slide": per_slide,
            "cohort_median_frac_beyond_p99_unfiltered": float(np.median(unfilt)) if unfilt else None,
            "cohort_median_frac_beyond_p99_filtered": float(np.median(filt)) if filt else None,
            "cohort_median_frac_surviving": float(np.median([s["frac_surviving"] for s in per_slide])),
            "n_slides_with_zero_survivors": sum(1 for s in per_slide if s["n_surviving"] == 0),
        }
    return results


def assess_extrapolation_change(task_b: dict, threshold_name: str) -> dict:
    r = task_b[threshold_name]
    before = r["cohort_median_frac_beyond_p99_unfiltered"]
    after = r["cohort_median_frac_beyond_p99_filtered"]
    if before is None or after is None:
        return {"assessable": False,
                "reason": "no surviving patches at this threshold -- extrapolation "
                          "change cannot be assessed."}
    drop = before - after
    return {
        "assessable": True, "before": before, "after": after, "absolute_drop": drop,
        "meaningfully_reduced": bool(drop >= EXTRAPOLATION_MEANINGFUL_DROP),
        "largely_resolved": bool(after < EXTRAPOLATION_LARGELY_RESOLVED),
        "drop_threshold_used": EXTRAPOLATION_MEANINGFUL_DROP,
        "resolved_threshold_used": EXTRAPOLATION_LARGELY_RESOLVED,
    }


# ── Task C: re-run the Stage E diagnostic on the filtered cohort ─────────────

def task_c_correlations(
    task_b_primary: dict, per_slide_morph: dict, stageD_per_slide: dict,
    stageE: dict, thresholds: dict, min_patches: int, n_perm: int, seed: int,
) -> dict:
    """Mouse-level median pseudotime over ONLY the filtered patches, then the
    same raw/partial correlations Stage E computed -- via Stage E's own
    compute_block, so the comparison is method-identical."""
    t = thresholds[PRIMARY_THRESHOLD]

    slide_entries, excluded = [], []
    for s in task_b_primary["per_slide"]:
        stem = s["raw_stem"]
        m = per_slide_morph[stem]
        nd = m["features"][FILTER_FEATURE]
        idx = m["sample_idx"]
        keep = (nd >= t["lo"]) & (nd <= t["hi"]) & np.isfinite(nd)
        n_keep = int(keep.sum())
        if n_keep < min_patches:
            excluded.append({"raw_stem": stem, "n_surviving": n_keep,
                             "reason": f"fewer than {min_patches} surviving filtered "
                                       f"patches in the {len(idx)}-patch sample"})
            continue
        pt = stageD_per_slide[stem]["projected_pt"][idx][keep]
        slide_entries.append({
            "raw_stem": stem,
            "mouse_id": s["mouse_id"], "timepoint_weeks": s["timepoint_weeks"],
            "has_suffix_slide": s["has_suffix_slide"],
            "pseudotime_median": float(np.median(pt)),
            "frac_beyond_training_p99": s["frac_beyond_p99_filtered"],
            "n_filtered_patches": n_keep,
        })

    mouse_incl = aggregate_to_mouse_level(slide_entries)
    mouse_excl = aggregate_to_mouse_level(
        [e for e in slide_entries if not e["has_suffix_slide"]])

    # Hematoxylin comes from Stage E's already-joined rows, so the values and
    # the mouse membership are identical to the originals by construction --
    # only the pseudotime differs. Never recomputed here.
    blocks, blocks_drop, dropped_rows = {}, {}, {}
    rows_used, rows_used_drop = {}, {}
    for variant, mouse_rows in (("excluding_suffix", mouse_excl),
                                ("including_suffix", mouse_incl)):
        pt_by_key = {(m["mouse_id"], m["timepoint_weeks"]): m["median_projected_pseudotime"]
                     for m in mouse_rows}
        rows = []
        missing = []
        for r in stageE["joined_mouse_rows"][variant]:
            key = (r["mouse_id"], r["timepoint_weeks"])
            if key not in pt_by_key:
                missing.append(f"{key[0]}@{key[1]}W")
                continue
            rows.append({**r, "pseudotime": pt_by_key[key]})
        dropped_rows[variant] = missing
        if len(rows) < 4:
            blocks[variant] = {"insufficient_n": True, "n_mice_rows": len(rows),
                               "mouse_rows": [f"{r['mouse_id']}@{r['timepoint_weeks']}W"
                                              for r in rows]}
            blocks_drop[variant] = {"insufficient_n": True, "n_mice_rows": 0}
            continue
        blocks[variant] = compute_block(rows, n_perm, np.random.default_rng(seed))
        rows_used[variant] = rows
        no6072 = [r for r in rows if r["mouse_id"] != "6072"]
        rows_used_drop[variant] = no6072
        blocks_drop[variant] = (
            compute_block(no6072, n_perm, np.random.default_rng(seed))
            if len(no6072) >= 4 else {"insufficient_n": True, "n_mice_rows": len(no6072)})

    # Explain any non-finite partial rather than emitting a bare 'n/a'.
    annotate_nan_partials(blocks, rows_used)
    annotate_nan_partials(blocks_drop, rows_used_drop)

    ladder = {}
    for thr in PATCH_COUNT_LADDER:
        kept = [s for s in task_b_primary["per_slide"] if s["n_surviving"] >= thr]
        ladder[str(thr)] = {
            "n_slides_retained": len(kept),
            "n_mouse_timepoint_rows": len({(s["mouse_id"], s["timepoint_weeks"]) for s in kept}),
        }

    return {
        "min_filtered_patches": min_patches,
        "excluded_slides": excluded,
        "mouse_rows_dropped_vs_stageE": dropped_rows,
        "patch_count_ladder": ladder,
        "slide_entries": slide_entries,
        "mouse_level": {"excluding_suffix": mouse_excl, "including_suffix": mouse_incl},
        "blocks": blocks,
        "blocks_drop_6072": blocks_drop,
    }


def annotate_nan_partials(blocks: dict, rows_by_variant: dict) -> None:
    """A non-finite partial correlation otherwise renders as a bare 'n/a',
    which tells the reader nothing about WHY. `_partial_spearman` returns nan
    for two very different reasons, and the distinction matters:
      - fewer than 10 valid observations (its own guard), i.e. too little data;
      - a near-zero denominator, which happens when the control is nearly
        collinear with the predictor -- the partial is then ill-conditioned
        rather than merely absent, and a reader must not mistake it for
        'no residual association'.
    Annotates each block in place with `nan_reason` so the report can say which."""
    for variant, block in blocks.items():
        if block.get("insufficient_n"):
            continue
        rows = rows_by_variant.get(variant, [])
        n = len(rows)
        weeks = np.array([r["timepoint_weeks"] for r in rows], dtype=float)
        for m, entry in block.get("partial", {}).items():
            if np.isfinite(entry.get("rho", float("nan"))):
                continue
            ctrl = np.array([r[m] for r in rows], dtype=float)
            rho_cw, _ = _safe_spearman(ctrl, weeks)
            if n < PARTIAL_MIN_N:
                entry["nan_reason"] = f"n={n} < {PARTIAL_MIN_N} (guard in _partial_spearman)"
            elif np.isfinite(rho_cw) and abs(rho_cw) > 0.999:
                entry["nan_reason"] = (
                    f"ill-conditioned: control is collinear with weeks "
                    f"(rho={rho_cw:+.3f}), so the partial denominator collapses. "
                    f"NOT evidence of no residual association.")
            else:
                entry["nan_reason"] = "undefined (degenerate input)"


def assess_correlation_change(task_c: dict, stageE: dict) -> dict:
    """Did the raw correlation survive filtering? Compared on the PRIMARY
    (excluding-suffix) cut, matching Stage E's own primary."""
    new = task_c["blocks"].get("excluding_suffix", {})
    if new.get("insufficient_n"):
        return {"assessable": False,
                "reason": f"only {new.get('n_mice_rows', 0)} mouse rows survived filtering "
                          "-- too few to recompute the correlation."}
    old_rho = stageE["blocks"]["excluding_suffix"]["raw"]["rho"]
    new_rho = new["raw"]["rho"]
    if not (np.isfinite(old_rho) and np.isfinite(new_rho)):
        return {"assessable": False, "reason": "non-finite correlation."}
    same_sign = (old_rho >= 0) == (new_rho >= 0)
    retained = abs(new_rho) / abs(old_rho) if old_rho != 0 else float("nan")
    return {
        "assessable": True, "original_rho": float(old_rho), "filtered_rho": float(new_rho),
        "fraction_retained": float(retained), "same_sign": bool(same_sign),
        "survives": bool(same_sign and retained >= CORRELATION_SURVIVES_FRACTION),
        "survives_threshold_used": CORRELATION_SURVIVES_FRACTION,
    }


# ── Three-outcome framework ──────────────────────────────────────────────────

OUTCOMES = {
    "i": "(i) Filtering RESOLVES the extrapolation AND the correlation SURVIVES — "
         "evidence the original correlation may reflect real signal, though the "
         "hematoxylin confound question from Stage B remains separately open.",
    "ii": "(ii) Filtering RESOLVES the extrapolation AND the correlation WEAKENS or "
          "DISAPPEARS — evidence the original correlation was substantially driven by "
          "tissue-composition differences rather than progression.",
    "iii": "(iii) Filtering does NOT resolve the extrapolation — the ROI/composition "
           "mismatch is not the (or not the only) explanation, and the extrapolation "
           "problem remains unresolved regardless of what Task C shows.",
}


def select_outcome(extrap: dict, corr: dict) -> dict:
    if not extrap.get("assessable"):
        return {"outcome": "iii", "text": OUTCOMES["iii"],
                "why": f"Extrapolation change could not be assessed: {extrap.get('reason')}"}
    if not extrap["meaningfully_reduced"]:
        return {"outcome": "iii", "text": OUTCOMES["iii"],
                "why": (f"Cohort-median frac_beyond_p99 moved {extrap['before']:.3f} -> "
                        f"{extrap['after']:.3f} (drop {extrap['absolute_drop']:.3f}), below "
                        f"the {EXTRAPOLATION_MEANINGFUL_DROP} absolute-drop bar.")}
    if not corr.get("assessable"):
        return {"outcome": "iii", "text": OUTCOMES["iii"],
                "why": (f"Extrapolation dropped, but the correlation could not be "
                        f"recomputed: {corr.get('reason')}")}
    base = (f"Cohort-median frac_beyond_p99 moved {extrap['before']:.3f} -> "
            f"{extrap['after']:.3f}; filtered raw rho {corr['filtered_rho']:+.3f} vs "
            f"original {corr['original_rho']:+.3f} "
            f"({corr['fraction_retained']:.0%} retained, same sign={corr['same_sign']}).")
    key = "i" if corr["survives"] else "ii"
    return {"outcome": key, "text": OUTCOMES[key], "why": base}


# ── Output writers ───────────────────────────────────────────────────────────

def _md_escape(s) -> str:
    return str(s).replace("|", "\\|")


def write_report(result: dict, output_dir: Path) -> None:
    L = ["# Timepoint cohort — Stage F: ROI / patch-composition mismatch check", ""]

    L.append(
        "**What this analysis is.** It exists to test whether Stage D/E's "
        "pseudotime-vs-timepoint correlation is an artifact of comparing **tumor-only "
        "training data** against **whole-slide projection data** — the 29 timepoint "
        "slides have no annotations and were patched whole-slide, while the manifold "
        "was built exclusively from annotated tumor ROI patches. It is **not** an "
        "attempt to validate the timepoint hypothesis, and it does **not** conclude "
        "whether timepoint projection is valid or invalid. It reports which of three "
        "pre-specified outcomes the evidence matches and stops there."
    )
    L.append("")
    L.append("The three possible outcomes, stated in advance:")
    L.append("")
    for k in ("i", "ii", "iii"):
        L.append(f"- {OUTCOMES[k]}")
    L.append("")
    sel = result["outcome"]
    L.append(f"### The data match outcome {sel['outcome'].upper()}")
    L.append("")
    L.append(sel["text"])
    L.append("")
    L.append(f"*Basis:* {sel['why']}")
    L.append("")

    # ---- Task A ----
    L.append("## Task A — what kind of tissue is extrapolating")
    L.append("")
    L.append(
        f"Six morphological features (the pipeline's own "
        f"`compute_morphological_features`) on {result['n_timepoint_patches_sampled']} "
        f"timepoint patches sampled across {result['n_slides']} slides, against the "
        f"training manifold's own per-patch values read from its `results.csv`. "
        f"Read this before Task B: it says what the extrapolated tissue *is*."
    )
    L.append("")
    L.append("| feature | timepoint median [IQR] | training median [IQR] | median ratio | KS | Wasserstein |")
    L.append("|---|---|---|---|---|---|")
    for f in MORPH_FEATURES:
        a = result["task_a"][f]
        if a.get("insufficient_data"):
            L.append(f"| {f} | insufficient data | | | | |")
            continue
        tp, tr = a["timepoint"], a["training"]
        ratio = a["median_ratio_timepoint_over_training"]
        L.append(
            f"| {f} | {tp['median']:.4g} [{tp['q25']:.4g}–{tp['q75']:.4g}] | "
            f"{tr['median']:.4g} [{tr['q25']:.4g}–{tr['q75']:.4g}] | "
            f"{'n/a' if ratio is None else f'{ratio:.2f}x'} | "
            f"{a['ks_stat']:.3f} | {a['wasserstein']:.4g} |")
    L.append("")
    L.append(
        "A timepoint median markedly BELOW training on `nuclear_density` (ratio < 1) "
        "indicates the extra patches are sparse — stroma, fat, necrosis or background — "
        "consistent with the whole-slide-vs-tumor-ROI hypothesis. A ratio near 1 with "
        "large KS on other features would point elsewhere."
    )
    L.append("")

    # ---- Task B ----
    L.append("## Task B — does restricting to tumor-like patches fix the extrapolation?")
    L.append("")
    L.append(
        f"Filter feature: `{FILTER_FEATURE}` (the same feature this pipeline uses for "
        f"multi-root DPT root selection). Bands are quantiles of the TRAINING "
        f"distribution. Extrapolation is recomputed by masking Stage D's own saved "
        f"per-patch nearest-neighbour distances against the SAME training p99 Stage D "
        f"used ({result['training_p99']:.4f}) — so filtered and unfiltered numbers are "
        f"comparable by construction."
    )
    L.append("")
    L.append("| threshold | band | median % patches surviving | slides w/ 0 survivors | cohort median frac>p99 unfiltered | filtered |")
    L.append("|---|---|---|---|---|---|")
    for tname, r in result["task_b"].items():
        t = r["threshold"]
        unf = r["cohort_median_frac_beyond_p99_unfiltered"]
        fil = r["cohort_median_frac_beyond_p99_filtered"]
        L.append(
            f"| {tname} | [{t['lo']:.4g}, {t['hi']:.4g}] | "
            f"{r['cohort_median_frac_surviving']:.1%} | {r['n_slides_with_zero_survivors']} | "
            f"{'n/a' if unf is None else f'{unf:.3f}'} | "
            f"{'n/a' if fil is None else f'{fil:.3f}'} |")
    L.append("")
    ex = result["extrapolation_assessment"]
    if ex.get("assessable"):
        L.append(
            f"**Primary ({PRIMARY_THRESHOLD}):** cohort-median frac_beyond_p99 "
            f"{ex['before']:.3f} → {ex['after']:.3f} (absolute drop {ex['absolute_drop']:.3f}). "
            f"Meaningfully reduced: **{ex['meaningfully_reduced']}** "
            f"(bar: ≥{EXTRAPOLATION_MEANINGFUL_DROP} absolute). Largely resolved: "
            f"**{ex['largely_resolved']}** (bar: <{EXTRAPOLATION_LARGELY_RESOLVED}). "
            f"Both bars are CHOSEN CONVENTIONS for this module, not values from the task "
            f"brief. For scale, the training manifold's own rate beyond its p99 is ~0.01 "
            f"by definition."
        )
    else:
        L.append(f"**Primary ({PRIMARY_THRESHOLD}):** not assessable — {ex.get('reason')}")
    L.append("")
    L.append(f"### Per-slide survival — {PRIMARY_THRESHOLD}")
    L.append("")
    L.append("| slide | mouse | weeks | sampled | surviving | % [95% CI] | implied count of all patches | frac>p99 filtered | (Stage D, all patches) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in result["task_b"][PRIMARY_THRESHOLD]["per_slide"]:
        lo, hi = s["frac_surviving_ci95"]
        ff = s["frac_beyond_p99_filtered"]
        L.append(
            f"| {_md_escape(s['raw_stem'])} | {s['mouse_id']} | {s['timepoint_weeks']} | "
            f"{s['n_sampled']} | {s['n_surviving']} | "
            f"{s['frac_surviving']:.1%} [{lo:.1%}–{hi:.1%}] | "
            f"{s['implied_count_of_all_patches']} of {s['n_patches_total']} | "
            f"{'n/a' if ff is None else f'{ff:.3f}'} | "
            f"{s['stageD_frac_beyond_p99_allpatches']} |")
    L.append("")

    # ---- Task C ----
    L.append("## Task C — does the correlation survive filtering?")
    L.append("")
    if not (ex.get("assessable") and ex.get("meaningfully_reduced")):
        L.append(
            "> **⚠ READ THIS FIRST: Task B's filtering did NOT meaningfully reduce the "
            "extrapolation.** The numbers below are still computed on projections that "
            "sit outside the training manifold's support, so they **do not resolve "
            "anything new** about whether the correlation reflects progression. They are "
            "reported for completeness only, as required, and must not be read as a "
            "corrected result."
        )
        L.append("")
    tc = result["task_c"]
    L.append(
        f"Mouse-level median pseudotime over ONLY filtered patches, aggregated on "
        f"(mouse_id, timepoint_weeks) as in every prior stage. Minimum "
        f"{tc['min_filtered_patches']} surviving patches per slide (stated and applied; "
        f"excluded slides listed below). Statistics come from Stage E's own "
        f"`compute_block`, so any difference from the original is the filtering, not "
        f"the method."
    )
    L.append("")
    nan_notes: list[str] = []
    L.append("| cut | quantity | ORIGINAL (Stage E) | FILTERED (Stage F) |")
    L.append("|---|---|---|---|")
    for variant, vlabel in (("excluding_suffix", "excl. suffix (PRIMARY)"),
                            ("including_suffix", "incl. suffix")):
        for drop6072, dlabel in ((False, ""), (True, ", drop 6072")):
            okey = "blocks_drop_6072" if drop6072 else "blocks"
            old = result["stageE_blocks"][okey].get(variant, {})
            new = tc["blocks_drop_6072" if drop6072 else "blocks"].get(variant, {})
            label = f"{vlabel}{dlabel}"
            if new.get("insufficient_n") or old.get("insufficient_n"):
                L.append(f"| {label} | raw rho | "
                         f"{_fmt(old.get('raw', {}).get('rho')) if not old.get('insufficient_n') else 'n/a'} | "
                         f"insufficient n ({new.get('n_mice_rows', 0)} rows) |")
                continue
            L.append(f"| {label} | raw rho (n={old['n_mice_rows']}→{new['n_mice_rows']}) | "
                     f"{_fmt(old['raw']['rho'])} | {_fmt(new['raw']['rho'])} |")
            for m in H_MEASURES:
                short = m.replace("median_h_intensity_", "h_").replace("_masked", "")
                new_p = new["partial"][m]
                cell = _fmt(new_p["rho"])
                if new_p.get("nan_reason"):
                    cell = f"n/a — {new_p['nan_reason']}"
                    nan_notes.append(f"`{label}` / `{short}`: {new_p['nan_reason']}")
                L.append(f"| {label} | partial \\| {short} | "
                         f"{_fmt(old['partial'][m]['rho'])} | {cell} |")
    L.append("")
    if nan_notes:
        L.append(
            "**Why some filtered partials are `n/a`** — a non-finite partial "
            "correlation is not the same as 'no residual association', so the reason "
            "is stated rather than left blank:")
        L.append("")
        for n in dict.fromkeys(nan_notes):
            L.append(f"- {n}")
        L.append("")
    if tc["excluded_slides"]:
        L.append("**Slides excluded for too few surviving patches:**")
        L.append("")
        for e in tc["excluded_slides"]:
            L.append(f"- `{_md_escape(e['raw_stem'])}` — {e['n_surviving']} surviving; {e['reason']}")
        L.append("")
    for variant, miss in tc["mouse_rows_dropped_vs_stageE"].items():
        if miss:
            L.append(f"**Mouse rows present in Stage E but absent after filtering "
                     f"({variant}):** {', '.join(miss)}")
            L.append("")
    L.append("**Surviving-slide count at other minimum-patch thresholds:**")
    L.append("")
    L.append("| min patches | slides retained | mouse-timepoint rows |")
    L.append("|---|---|---|")
    for thr, v in tc["patch_count_ladder"].items():
        L.append(f"| {thr} | {v['n_slides_retained']} | {v['n_mouse_timepoint_rows']} |")
    L.append("")

    L.append("## Verdict")
    L.append("")
    L.append(f"**Outcome {sel['outcome'].upper()}** — {sel['text']}")
    L.append("")
    L.append(f"*Basis:* {sel['why']}")
    L.append("")
    L.append(
        "This module deliberately does not state whether timepoint projection is valid "
        "or invalid; that interpretation, and what it means for the paper, is left to "
        "the reader."
    )
    L.append("")

    (output_dir / "stageF_roi_mismatch.md").write_text("\n".join(L), encoding="utf-8")


def write_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "stageF_roi_mismatch.json"
    with open(p, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  JSON: {p}")
    write_report(result, output_dir)
    print(f"  Markdown report: {output_dir / 'stageF_roi_mismatch.md'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Timepoint cohort Stage F: ROI/patch-composition mismatch check")
    ap.add_argument("--stageA-inventory-json", required=True, type=Path)
    ap.add_argument("--stageD-json", required=True, type=Path)
    ap.add_argument("--stageD-per-slide-dir", required=True, type=Path)
    ap.add_argument("--stageE-json", required=True, type=Path)
    ap.add_argument("--training-results-csv", required=True, type=Path)
    ap.add_argument("--png-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--morph-sample-per-slide", type=int, default=3000,
                    help="Patches per slide for morphology (0 = all patches, exact "
                         "counts but far slower). Default 3000 resolves a 1%% survival "
                         "rate to ~30+-5 patches.")
    ap.add_argument("--min-filtered-patches", type=int, default=MIN_FILTERED_PATCHES)
    ap.add_argument("--patch-size", type=int, default=112)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 60)
    print("  Timepoint cohort — Stage F: ROI/composition mismatch check")
    print("=" * 60)
    print("\n  Tests whether Stage D/E's correlation is an artifact of tumor-only")
    print("  training vs whole-slide projection. Does NOT validate the timepoint")
    print("  hypothesis and does NOT conclude projection is valid or invalid.\n")

    for p in (args.stageA_inventory_json, args.stageD_json, args.stageE_json):
        if not p.exists():
            sys.exit(f"ERROR: required input not found:\n  {p}")
    if not args.stageD_per_slide_dir.exists():
        sys.exit(f"ERROR: Stage D per-slide dir not found:\n  {args.stageD_per_slide_dir}\n"
                 "Stage F re-analyses the per-patch arrays Stage D saved there.")

    stageA = json.loads(args.stageA_inventory_json.read_text())
    stageD = json.loads(args.stageD_json.read_text())
    stageE = json.loads(args.stageE_json.read_text())

    base = stageD.get("training_support_baseline", {})
    if not base.get("available"):
        sys.exit("ERROR: Stage D's training_support_baseline is unavailable, so there "
                 "is no p99 to test filtered patches against.")
    training_p99 = float(base["mean_knn_distance"]["p99"])
    print(f"  Training p99 (mean-kNN distance), from Stage D: {training_p99:.4f}")

    print("\n=== Training morphological reference ===")
    training_ref = load_training_reference(args.training_results_csv)

    stageD_by_stem = {e["raw_stem"]: e for e in stageD["per_slide"]}
    slide_meta, stageD_per_slide = {}, {}
    for row in stageA["usable_slides"]:
        stem = row["raw_stem"]
        if stem not in stageD_by_stem:
            continue
        pt_p = args.stageD_per_slide_dir / f"projected_pt_{stem}.npy"
        nn_p = args.stageD_per_slide_dir / f"nn_mean_knn_{stem}.npy"
        if not (pt_p.exists() and nn_p.exists()):
            print(f"  WARNING: missing per-slide arrays for {stem} -- skipped")
            continue
        stageD_per_slide[stem] = {
            "projected_pt": np.load(pt_p), "nn_mean_knn": np.load(nn_p),
            "stageD_frac": stageD_by_stem[stem].get("frac_beyond_training_p99"),
        }
        slide_meta[stem] = {
            "mouse_id": row["mouse_id"], "timepoint_weeks": row["timepoint_weeks"],
            "has_suffix_slide": bool(row.get("suffix_flag", False)),
        }
    print(f"  {len(stageD_per_slide)} slides with Stage D per-patch arrays")

    morph_cache = args.output_dir / "per_slide_morph"
    per_slide_morph, failed = {}, {}
    print(f"\n=== Morphology on {args.morph_sample_per_slide or 'ALL'} patches/slide ===")
    for i, stem in enumerate(sorted(stageD_per_slide), 1):
        print(f"\n  [{i}/{len(stageD_per_slide)}] {stem}")
        try:
            per_slide_morph[stem] = compute_slide_morphology(
                _png_path(args.png_dir, stem), stem,
                n_expected=len(stageD_per_slide[stem]["projected_pt"]),
                sample_size=args.morph_sample_per_slide, seed=args.seed,
                patch_size=args.patch_size, stride=args.stride, cache_dir=morph_cache)
        except Exception as e:
            failed[stem] = repr(e)
            print(f"    ERROR: {e!r} -- excluded")
    if not per_slide_morph:
        sys.exit("ERROR: no slide produced morphological features; nothing to analyse.")

    print("\n=== Task A: distribution comparison ===")
    pooled = {f: np.concatenate([m["features"][f] for m in per_slide_morph.values()])
              for f in MORPH_FEATURES}
    task_a = task_a_distributions(pooled, training_ref)
    for f in MORPH_FEATURES:
        a = task_a[f]
        if not a.get("insufficient_data"):
            print(f"  {f}: timepoint {a['timepoint']['median']:.4g} vs training "
                  f"{a['training']['median']:.4g}  KS={a['ks_stat']:.3f}")

    print("\n=== Task B: filter + extrapolation re-test ===")
    thresholds = build_thresholds(training_ref)
    task_b = task_b_filter(per_slide_morph, slide_meta, thresholds,
                           stageD_per_slide, training_p99)
    for tname, r in task_b.items():
        print(f"  {tname}: median surviving {r['cohort_median_frac_surviving']:.1%}; "
              f"frac>p99 {r['cohort_median_frac_beyond_p99_unfiltered']} -> "
              f"{r['cohort_median_frac_beyond_p99_filtered']}")
    extrap = assess_extrapolation_change(task_b, PRIMARY_THRESHOLD)

    print("\n=== Task C: correlations on the filtered cohort ===")
    task_c = task_c_correlations(
        task_b[PRIMARY_THRESHOLD], per_slide_morph, stageD_per_slide, stageE,
        thresholds, args.min_filtered_patches, args.n_permutations, args.seed)
    corr = assess_correlation_change(task_c, stageE)
    if corr.get("assessable"):
        print(f"  raw rho {corr['original_rho']:+.4f} -> {corr['filtered_rho']:+.4f} "
              f"({corr['fraction_retained']:.0%} retained)")
    else:
        print(f"  not assessable: {corr.get('reason')}")

    outcome = select_outcome(extrap, corr)
    print(f"\n  OUTCOME {outcome['outcome'].upper()}: {outcome['text']}")

    result = {
        "what_this_is": (
            "Tests whether Stage D/E's correlation is an artifact of tumor-only training "
            "data vs whole-slide projection data. NOT a validation of the timepoint "
            "hypothesis; deliberately does not conclude whether projection is valid."),
        "n_slides": len(per_slide_morph),
        "n_timepoint_patches_sampled": int(sum(len(m["sample_idx"])
                                               for m in per_slide_morph.values())),
        "morph_sample_per_slide": args.morph_sample_per_slide,
        "training_p99": training_p99,
        "training_results_csv": str(args.training_results_csv),
        "failed_slides": failed,
        "task_a": task_a,
        "thresholds": thresholds,
        "task_b": task_b,
        "extrapolation_assessment": extrap,
        "task_c": task_c,
        "correlation_assessment": corr,
        "stageE_blocks": {"blocks": stageE["blocks"],
                          "blocks_drop_6072": stageE["blocks_drop_6072"]},
        "outcome": outcome,
        "chosen_conventions": {
            "EXTRAPOLATION_MEANINGFUL_DROP": EXTRAPOLATION_MEANINGFUL_DROP,
            "EXTRAPOLATION_LARGELY_RESOLVED": EXTRAPOLATION_LARGELY_RESOLVED,
            "CORRELATION_SURVIVES_FRACTION": CORRELATION_SURVIVES_FRACTION,
            "MIN_FILTERED_PATCHES": args.min_filtered_patches,
            "note": "Chosen for this module, not from the task brief.",
        },
    }
    write_outputs(result, args.output_dir)

    print("\n" + "=" * 60)
    print(f"  STAGE F COMPLETE — OUTCOME {outcome['outcome'].upper()}")
    print("=" * 60)
    print("\n  Report Task A and Task B alongside Task C's numbers -- never Task C alone.")
    print("  Interpretation of what this outcome means is left to the user. STOP HERE.")


if __name__ == "__main__":
    main()
