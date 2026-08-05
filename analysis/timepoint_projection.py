"""
Timepoint cohort: Stage D -- feature extraction + projection onto the existing
manifold.

CONTEXT / STATUS -- READ BEFORE USING ANY NUMBER THIS PRODUCES
--------------------------------------------------------------
Stage B v2 (`timepoint_stain_homogeneity_v2.py`) returned FAIL against the
project's stain gate. The precise finding, which must be carried forward
accurately: the RGB channels are mostly NOT separating by timepoint
(negligible-to-small in most comparisons). The measure driving FAIL is
specifically HEMATOXYLIN INTENSITY, which separates in every adequately-powered
pairwise comparison and shows a monotonic trend with weeks. That is a narrow,
hematoxylin-specific effect, consistent with EITHER a reagent-side confound OR
genuine cellularity change with tumor age. Do not describe it as "broad
staining differences" -- it is not one.

The gate stands and NO correction is applied here. This stage runs anyway as a
background, non-blocking DIAGNOSTIC -- explicitly not a step toward a claimed
positive result. Its purpose is to feed Stage E, which asks whether projected
pseudotime carries information beyond that one already-confounded channel.

What this module does NOT do: retrain, refit, or modify anything. It loads the
SAVED `AtlasProjector` and calls it. `run_all.py`, `projector.py`, the manifold,
its Harmony correction, and every projector artifact are untouched and read-only.

Reuses (does not reimplement):
  - features/patching.py::get_patches_from_array -- called with
    roi_polygons=None/exclude_polygons=None, which is EXACTLY how run_all.py
    (:345-353) handles a slide with annotation=None. These 29 slides have no
    annotations, so whole-slide patching with the standard tissue filters
    applies, matching the pipeline's own behaviour for that case.
  - features/extractors.py::load_model_components / extract_features_from_model
    -- the same pre-loaded-model pattern run_all.py uses for cache population.
  - features/patching.py::sample_patches -- only if --max-patches-per-slide is
    given (default: no cap, matching run_all.py's always-uncapped cache).
  - analysis/projector.py::AtlasProjector.load + .project(method="knn") -- the
    identical call analysis/loo_project.py:90 makes.

STAIN NORMALIZATION IS FIXED AT "none" and there is no flag to change it. The
manifold was built with `--stain-method none` (canonical config), so matching it
is required; and applying a normalizer here would be exactly the "correction"
this stage is explicitly not allowed to apply. `run_all.py`'s
build_normalizer("none")/normalize_slide(arr, None) path is a documented no-op
(data/stain_normalization.py:70-71, :106-107), so it is simply not invoked.

ON THE UMAP STEP: `AtlasProjector._project_knn` ends with an optional
`umap_reducer_.transform(X_pca)`. That is cosmetic -- it produces coordinates
used only for plotting, and this pass is forbidden from producing any figure.
It is also very expensive at this scale (~60k patches/slide x 29 slides).
`--skip-umap` (DEFAULT TRUE) sets `projector.umap_reducer_ = None` on the
in-memory copy before projecting, so that block is skipped. Every number that
matters -- scaler -> PCA -> nearest-centroid cluster -> KNN pseudotime -- comes
from the identical, untouched code path and is bit-for-bit what LOO projection
would produce. Pass --no-skip-umap to restore the full call. Nothing is written
back to the projector on disk either way.

CLI
---
  python -m cancer_trajectory_atlas.analysis.timepoint_projection \\
      --stageA-inventory-json $SCRATCH/results/timepoint_cohort/stageA_inventory_v2/stageA_inventory_v2.json \\
      --png-dir               $SCRATCH/data/timepoint_x5_full \\
      --projector-dir         $SCRATCH/results/baseline/atlas_none_harmony_median/projector \\
      --features-cache-dir    $SCRATCH/data/timepoint_features_cache \\
      --output-dir            $SCRATCH/results/timepoint_cohort/stageD_projection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..features.patching import get_patches_from_array, sample_patches
from .timepoint_stage2_stain_check import _fmt, _png_path

# Fraction of a slide's patches lying beyond the TRAINING manifold's p99
# mean-k-NN distance, above which the slide is flagged as substantially
# extrapolated. This is a CHOSEN CONVENTION for this module, not a value from
# the task brief -- stated here and in the report so it is visible and
# adjustable, mirroring how Stage B v1 disclosed VALIDATION_RHO_THRESHOLD.
EXTRAPOLATION_FRACTION_WARN = 0.20

# Training points sampled to build the in-support reference distribution. The
# full training set is ~480k points; an all-pairs self-query is not tractable,
# and is not needed for a percentile estimate.
DEFAULT_BASELINE_SAMPLE = 20000

TIMEPOINT_GROUPS = [4, 7, 8, 12]


# ── Projector provenance (hard gate, runs before any GPU work) ───────────────

def _matches(a: np.ndarray | None, b: np.ndarray | None, rng: np.random.Generator,
             n_probe: int = 2000) -> bool:
    """Value-equality probe on a random row subsample. Shape alone cannot
    distinguish X_pca_original from X_pca_harmony -- Harmony preserves
    dimensionality, so the corrected and uncorrected matrices are the SAME
    shape. Only the values differ, so the values are what gets compared."""
    if a is None or b is None:
        return False
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    n = a.shape[0]
    idx = rng.choice(n, size=min(n_probe, n), replace=False)
    return bool(np.allclose(np.asarray(a[idx], dtype=np.float64),
                            np.asarray(b[idx], dtype=np.float64),
                            rtol=1e-4, atol=1e-6))


def check_projector_provenance(projector, seed: int = 0) -> dict:
    """Reports -- with evidence, not assumption -- which representation the
    saved KNN pseudotime regressor was actually fitted on.

    run_all.py:539-544 stores, when Harmony ran:
        adata.X                      = X_embed  = Harmony-CORRECTED
        adata.obsm["X_pca_original"] = X_pca    = pre-Harmony PCA
        adata.obsm["X_pca_harmony"]  = X_embed  = Harmony-CORRECTED
    and AtlasProjector.from_training (projector.py:49-52) fits the KNN on
    X_pca_original when it exists. This verifies that actually happened in the
    artifact being loaded, rather than trusting it. Hard-fails if the KNN turns
    out to be fitted on the corrected matrix -- that would silently violate the
    'never touch the Harmony-corrected representation' constraint."""
    rng = np.random.default_rng(seed)
    knn = projector.knn_pseudotime_
    if knn is None:
        sys.exit("ERROR: loaded projector has no knn_pseudotime_ regressor.")

    fit_X = getattr(knn, "_fit_X", None)
    adata = projector.adata_train_

    info = {
        "knn_n_neighbors": int(knn.n_neighbors),
        "knn_weights": str(getattr(knn, "weights", "?")),
        "knn_fit_shape": list(np.asarray(fit_X).shape) if fit_X is not None else None,
        "adata_train_available": adata is not None,
        "has_X_pca_original": False,
        "has_X_pca_harmony": False,
        "fitted_on": "unknown",
    }

    if adata is None:
        info["note"] = (
            "adata_train.h5ad not loaded (missing, or scanpy unavailable) -- cannot "
            "verify which space the KNN was fitted in. The KNN itself is already "
            "fitted and pickled, so nothing is refitted regardless; but this check "
            "could not be completed."
        )
        return info

    x_orig = adata.obsm.get("X_pca_original") if hasattr(adata, "obsm") else None
    x_harm = adata.obsm.get("X_pca_harmony") if hasattr(adata, "obsm") else None
    x_main = getattr(adata, "X", None)
    info["has_X_pca_original"] = x_orig is not None
    info["has_X_pca_harmony"] = x_harm is not None
    info["n_training_samples"] = int(adata.n_obs) if hasattr(adata, "n_obs") else None

    if fit_X is None:
        info["note"] = ("sklearn did not expose _fit_X on this estimator -- cannot "
                        "verify the fitted space. Nothing is refitted regardless.")
        return info

    matched_original = _matches(fit_X, x_orig, rng)
    matched_harmony = _matches(fit_X, x_harm, rng)
    matched_main = _matches(fit_X, x_main, rng)

    if matched_original:
        info["fitted_on"] = "X_pca_original (pre-Harmony PCA)"
    elif matched_harmony or (matched_main and x_orig is not None):
        info["fitted_on"] = "HARMONY-CORRECTED matrix"
    elif matched_main:
        info["fitted_on"] = "adata.X (no batch correction present -- plain PCA)"

    if x_orig is not None and not matched_original:
        sys.exit(
            "ERROR: this manifold HAS a pre-Harmony representation "
            "(obsm['X_pca_original']), but the saved KNN pseudotime regressor was "
            f"NOT fitted on it (detected: {info['fitted_on']}).\n"
            "Projecting through it would use the Harmony-corrected representation, "
            "violating this task's explicit constraint. Refusing to proceed.\n"
            "Check that --projector-dir points at the intended run."
        )
    return info


# ── Training-manifold support baseline ────────────────────────────────────────

def compute_training_support_baseline(projector, sample_size: int, seed: int) -> dict:
    """Mean-k-NN-distance distribution for the TRAINING points themselves --
    the reference that makes a projected patch's distance interpretable.

    Without this, an 'extrapolation distance' is an uncalibrated number. (This
    project has already been burned once by reading a diagnostic as anomalous
    before establishing what it reports on known-good data.)

    The self-match MUST be dropped: querying the fitted index with a point that
    is IN the index returns itself at ~zero distance. Asking for k+1 and
    dropping the self leaves the same k genuine neighbours a projected
    (out-of-index) patch would get. Skipping this would deflate the baseline
    and make every projected slide look extrapolated by construction.

    The self is removed BY INDEX IDENTITY rather than by assuming it sorts
    first. Both facts below were verified against real sklearn behaviour, not
    assumed:
      - the self-distance is ~1e-7, NOT exactly 0 (sklearn's
        ||a||^2 + ||b||^2 - 2ab formulation loses precision near zero), so a
        zero-test would not reliably find it either;
      - "self is at column 0" is simply not true when the index contains
        duplicate points: in a duplicate-heavy synthetic check, 300 of 600
        query rows returned the twin first and the self second.
    Honest note on impact: when that swap happens both candidates sit at ~0
    distance, so dropping the wrong one changes the retained mean only
    negligibly. This is exactness/hygiene, not a fix for a result-changing
    bug -- but it costs nothing and removes an ordering assumption that
    demonstrably does not hold. `n_rows_self_match_not_found` is reported so a
    genuine failure of the assumption would be visible rather than silent."""
    knn = projector.knn_pseudotime_
    fit_X = np.asarray(getattr(knn, "_fit_X", None))
    if fit_X is None or fit_X.ndim != 2:
        return {"available": False,
                "reason": "sklearn did not expose _fit_X; cannot build the baseline."}

    k = int(knn.n_neighbors)
    n_train = fit_X.shape[0]
    rng = np.random.default_rng(seed)
    n_sample = int(min(sample_size, n_train))
    idx = rng.choice(n_train, size=n_sample, replace=False)

    print(f"  Training points: {n_train}; sampling {n_sample} for the support baseline")
    dists, idxs = knn.kneighbors(fit_X[idx], n_neighbors=k + 1)

    # Drop the self-match by INDEX IDENTITY (see docstring -- not by position,
    # not by a zero-distance test).
    self_mask = idxs == idx[:, None]
    has_self = self_mask.any(axis=1)
    n_missing_self = int((~has_self).sum())
    if n_missing_self:
        # Defensive: should not happen when querying with points that are in the
        # index. Drop the farthest neighbour instead so every row still yields
        # exactly k distances, and report the count rather than hiding it.
        self_mask[~has_self, -1] = True
    first_self = self_mask.argmax(axis=1)          # first True per row
    keep = np.ones(self_mask.shape, dtype=bool)
    keep[np.arange(n_sample), first_self] = False
    dists = dists[keep].reshape(n_sample, k)

    mean_k = dists.mean(axis=1)
    first_nn = dists[:, 0]

    return {
        "available": True,
        "n_training_points": int(n_train),
        "n_sampled": n_sample,
        "k": k,
        "self_match_dropped": True,
        "self_match_dropped_by": "index identity",
        "n_rows_self_match_not_found": n_missing_self,
        "mean_knn_distance": {
            "median": float(np.median(mean_k)),
            "p95": float(np.percentile(mean_k, 95)),
            "p99": float(np.percentile(mean_k, 99)),
            "max": float(np.max(mean_k)),
        },
        "first_nn_distance": {
            "median": float(np.median(first_nn)),
            "p95": float(np.percentile(first_nn, 95)),
            "p99": float(np.percentile(first_nn, 99)),
        },
    }


# ── Per-slide extraction + projection ─────────────────────────────────────────

def extract_or_load_features(
    png_path: Path, stem: str, cache_dir: Path, args, model_state: dict,
) -> tuple[np.ndarray | None, int, str | None]:
    """Returns (features, n_patches_extracted, error). Cache hit short-circuits
    all image decoding and GPU work -- which is what makes this job resumable
    after a walltime timeout."""
    from PIL import Image

    cache_file = cache_dir / f"{stem}_features.npy"
    if cache_file.exists():
        feats = np.load(cache_file)
        print(f"    Cache hit: {cache_file.name} ({len(feats)} patches)")
        return feats, len(feats), None

    if not png_path.exists():
        return None, 0, f"PNG not found: {png_path}"

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(png_path).convert("RGB")
    img_arr = np.array(img)
    del img
    print(f"    Image size: {img_arr.shape[1]} x {img_arr.shape[0]}")

    # No annotations for these slides -> whole-slide patching, exactly as
    # run_all.py:345-353 does when slide_entry["annotation"] is None.
    # Stain normalization is fixed at "none" (see module docstring), so the
    # normalize_slide() no-op is simply not called.
    patches, coords = get_patches_from_array(
        img_arr,
        patch_size=args.patch_size,
        stride=args.stride,
        image_name=stem,
        roi_polygons=None,
        exclude_polygons=None,
        min_roi_coverage=None,
    )
    del img_arr

    if len(patches) == 0:
        return None, 0, "no patches passed the tissue filters"

    n_extracted = len(patches)
    if args.max_patches_per_slide:
        patches, coords, _ = sample_patches(
            patches, coords, args.max_patches_per_slide, args.patch_sample_seed, stem,
        )
        print(f"    Capped to {len(patches)} patches (from {n_extracted})")

    from ..features.extractors import load_model_components, extract_features_from_model
    if not model_state["loaded"]:
        print(f"    Pre-loading {args.model} model...")
        m, p, d = load_model_components(args.model)
        model_state.update({"model": m, "processor": p, "device": d, "loaded": True})
    feats = extract_features_from_model(
        patches, model_state["model"], model_state["processor"],
        model_state["device"], batch_size=args.batch_size,
    )
    del patches
    np.save(cache_file, feats)
    print(f"    Saved to cache: {cache_file.name}")
    return feats, n_extracted, None


def project_slide(projector, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pseudotime, mean_knn_distance, first_nn_distance).

    `.project(method="knn")` is the identical call loo_project.py:90 makes; the
    returned AnnData's .X is the PCA-space representation, reused directly for
    the neighbour query so the scaler/PCA transform is not repeated."""
    adata_proj = projector.project(feats, method="knn")
    pt = adata_proj.obs["pseudotime"].values.astype(float)
    X_pca = np.asarray(adata_proj.X)

    knn = projector.knn_pseudotime_
    # Second kneighbors pass: .predict() inside project() already did one
    # internally, but sklearn does not expose those distances. Recomputing is
    # ~2x the neighbour cost and is preferred over reimplementing sklearn's
    # distance-weighting (which has a zero-distance special case that is easy
    # to get subtly wrong).
    dists, _ = knn.kneighbors(X_pca, n_neighbors=int(knn.n_neighbors))
    return pt, dists.mean(axis=1), dists[:, 0]


def summarize_slide(stem: str, row: dict, pt: np.ndarray, mean_knn: np.ndarray,
                    first_nn: np.ndarray, n_extracted: int, baseline: dict) -> dict:
    entry = {
        "raw_stem": stem,
        "mouse_id": row["mouse_id"],
        "timepoint_weeks": row["timepoint_weeks"],
        "has_suffix_slide": bool(row.get("suffix_flag", False)),
        "n_patches_extracted": int(n_extracted),
        "n_patches_projected": int(len(pt)),
        "pseudotime_min": float(np.min(pt)),
        "pseudotime_median": float(np.median(pt)),
        "pseudotime_max": float(np.max(pt)),
        "pseudotime_std": float(np.std(pt)),
        "nn_mean_knn_distance_median": float(np.median(mean_knn)),
        "nn_first_distance_median": float(np.median(first_nn)),
    }
    if baseline.get("available"):
        p95 = baseline["mean_knn_distance"]["p95"]
        p99 = baseline["mean_knn_distance"]["p99"]
        frac95 = float(np.mean(mean_knn > p95))
        frac99 = float(np.mean(mean_knn > p99))
        entry.update({
            "frac_beyond_training_p95": frac95,
            "frac_beyond_training_p99": frac99,
            "substantially_extrapolated": bool(frac99 > EXTRAPOLATION_FRACTION_WARN),
        })
    else:
        entry.update({
            "frac_beyond_training_p95": None,
            "frac_beyond_training_p99": None,
            "substantially_extrapolated": None,
        })
    return entry


# ── Mouse-level aggregation ───────────────────────────────────────────────────

def aggregate_to_mouse_level(slide_entries: list[dict]) -> list[dict]:
    """Groups by (mouse_id, timepoint_weeks) -- NOT mouse_id alone. Mouse 6072
    legitimately contributes both a 7W and a 12W row (confirmed staggered
    harvest of the same animal); collapsing on mouse_id would silently merge
    two different timepoints. Matches Stage B v2's key exactly so Stage E can
    join the two tables."""
    groups: dict[tuple, list[dict]] = {}
    for e in slide_entries:
        if e["mouse_id"] is None or e["timepoint_weeks"] is None:
            continue
        groups.setdefault((e["mouse_id"], e["timepoint_weeks"]), []).append(e)

    mouse_counts: dict[str, int] = {}
    for (mouse_id, _w) in groups:
        mouse_counts[mouse_id] = mouse_counts.get(mouse_id, 0) + 1

    out = []
    for (mouse_id, weeks), rows in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        out.append({
            "mouse_id": mouse_id,
            "timepoint_weeks": weeks,
            "n_slides": len(rows),
            "slide_stems": [r["raw_stem"] for r in rows],
            "has_suffix_slide": any(r["has_suffix_slide"] for r in rows),
            "dual_timepoint_mouse": mouse_counts[mouse_id] > 1,
            "median_projected_pseudotime": float(np.median(
                [r["pseudotime_median"] for r in rows])),
            "median_frac_beyond_training_p99": (
                float(np.median([r["frac_beyond_training_p99"] for r in rows]))
                if all(r["frac_beyond_training_p99"] is not None for r in rows) else None
            ),
        })
    return out


def summarize_by_timepoint(slide_entries: list[dict]) -> dict:
    result = {}
    for w in TIMEPOINT_GROUPS:
        rows = [e for e in slide_entries if e["timepoint_weeks"] == w]
        if not rows:
            continue
        fr = [r["frac_beyond_training_p99"] for r in rows
              if r["frac_beyond_training_p99"] is not None]
        result[str(w)] = {
            "n_slides": len(rows),
            "median_pseudotime": float(np.median([r["pseudotime_median"] for r in rows])),
            "median_frac_beyond_training_p99": float(np.median(fr)) if fr else None,
            "n_slides_substantially_extrapolated": sum(
                1 for r in rows if r.get("substantially_extrapolated")),
        }
    return result


# ── Output writers ────────────────────────────────────────────────────────────

def write_report(result: dict, output_dir: Path) -> None:
    lines = ["# Timepoint cohort — Stage D: projection onto the existing manifold", ""]

    lines.append(
        "**This is a background diagnostic run, executed despite a FAILED stain "
        "gate. It is not a step toward a claimed positive result.** Stage B v2 "
        "found a specifically HEMATOXYLIN-SPECIFIC separation by timepoint (the RGB "
        "channels are mostly negligible-to-small) — consistent with either a "
        "reagent-side confound or genuine cellularity change with tumor age. No "
        "correction has been applied. Nothing here is a validated timepoint result."
    )
    lines.append("")

    prov = result["projector_provenance"]
    lines.append("## Projector provenance (verified, not assumed)")
    lines.append("")
    lines.append(f"- Projector dir: `{result['projector_dir']}`")
    lines.append(f"- Training samples: {prov.get('n_training_samples', 'n/a')}")
    lines.append(f"- KNN fitted on: **{prov['fitted_on']}**")
    lines.append(f"- KNN k / weights: {prov['knn_n_neighbors']} / {prov['knn_weights']}")
    lines.append(f"- `X_pca_original` present: {prov['has_X_pca_original']}; "
                 f"`X_pca_harmony` present: {prov['has_X_pca_harmony']}")
    if prov.get("note"):
        lines.append(f"- NOTE: {prov['note']}")
    lines.append("")
    lines.append(
        "Nothing was retrained or refitted: the KNN regressor is loaded already-fitted "
        "from the saved artifact. The Harmony-corrected representation is never used."
    )
    lines.append("")

    base = result["training_support_baseline"]
    lines.append("## PROJECTION VALIDITY — read before any pseudotime number below")
    lines.append("")
    if not base.get("available"):
        lines.append(f"**Baseline unavailable:** {base.get('reason')} — extrapolation "
                     "cannot be assessed, so every pseudotime value below is of unknown "
                     "validity.")
    else:
        mk = base["mean_knn_distance"]
        lines.append(
            f"Reference distribution built from {base['n_sampled']} of "
            f"{base['n_training_points']} training points (self-match at distance 0 "
            f"dropped), k={base['k']}: median mean-k-NN distance {mk['median']:.4f}, "
            f"p95 {mk['p95']:.4f}, p99 {mk['p99']:.4f}."
        )
        lines.append("")
        lines.append(
            f"A slide is flagged **substantially extrapolated** when more than "
            f"{EXTRAPOLATION_FRACTION_WARN:.0%} of its patches exceed the training p99. "
            f"That threshold is a CHOSEN CONVENTION for this module, not a value from "
            f"the task brief — adjust if needed."
        )
        lines.append("")
        n_extrap = sum(1 for e in result["per_slide"] if e.get("substantially_extrapolated"))
        if n_extrap:
            lines.append(
                f"> **{n_extrap} of {len(result['per_slide'])} slides are substantially "
                f"extrapolated.** Their projected pseudotime describes regions of feature "
                f"space the manifold does not actually cover; the KNN still returns a "
                f"number for every patch regardless. Treat those slides' values — and any "
                f"timepoint group containing them — as a limitation on interpretation, "
                f"not as a measurement."
            )
        else:
            lines.append(
                "> No slide exceeds the extrapolation threshold: projected patches sit "
                "within the training manifold's support on this measure."
            )
    lines.append("")

    lines.append("### Extrapolation by timepoint group")
    lines.append("")
    lines.append("| weeks | n slides | median pseudotime | median % patches beyond training p99 | slides flagged |")
    lines.append("|---|---|---|---|---|")
    for w, v in result["by_timepoint"].items():
        frac = v["median_frac_beyond_training_p99"]
        lines.append(
            f"| {w} | {v['n_slides']} | {_fmt(v['median_pseudotime'])} | "
            f"{'n/a' if frac is None else f'{frac:.1%}'} | "
            f"{v['n_slides_substantially_extrapolated']} |"
        )
    lines.append("")

    lines.append("## Per-slide results")
    lines.append("")
    lines.append("| slide | mouse | weeks | extracted | projected | PT min | PT median | PT max | PT std | med mean-kNN dist | % > p95 | % > p99 | extrapolated |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in result["per_slide"]:
        f95 = e["frac_beyond_training_p95"]
        f99 = e["frac_beyond_training_p99"]
        lines.append(
            f"| {e['raw_stem']} | {e['mouse_id']} | {e['timepoint_weeks']} | "
            f"{e['n_patches_extracted']} | {e['n_patches_projected']} | "
            f"{_fmt(e['pseudotime_min'])} | {_fmt(e['pseudotime_median'])} | "
            f"{_fmt(e['pseudotime_max'])} | {_fmt(e['pseudotime_std'])} | "
            f"{_fmt(e['nn_mean_knn_distance_median'])} | "
            f"{'n/a' if f95 is None else f'{f95:.1%}'} | "
            f"{'n/a' if f99 is None else f'{f99:.1%}'} | "
            f"{'**YES**' if e.get('substantially_extrapolated') else 'no'} |"
        )
    lines.append("")

    if result["failed_slides"]:
        lines.append(f"**Slides that failed (excluded):** {result['failed_slides']}")
        lines.append("")

    lines.append("## Mouse-level aggregation (unit of inference)")
    lines.append("")
    lines.append("| mouse | weeks | n slides | median projected pseudotime | has suffix slide | dual-timepoint mouse |")
    lines.append("|---|---|---|---|---|---|")
    for m in result["mouse_level"]:
        lines.append(
            f"| {m['mouse_id']} | {m['timepoint_weeks']} | {m['n_slides']} | "
            f"{_fmt(m['median_projected_pseudotime'])} | {m['has_suffix_slide']} | "
            f"{m['dual_timepoint_mouse']} |"
        )
    lines.append("")
    lines.append(
        "Aggregation key is (mouse_id, timepoint_weeks), matching Stage B v2 — mouse "
        "6072 contributes both a 7W and a 12W row from a confirmed staggered harvest "
        "of the same animal, and is flagged `dual_timepoint_mouse`."
    )
    lines.append("")
    lines.append("Next: Stage E (`timepoint_diagnostic.py`) consumes this mouse-level table.")
    lines.append("")

    (output_dir / "stageD_projection.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stageD_projection.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  JSON: {json_path}")
    write_report(result, output_dir)
    print(f"  Markdown report: {output_dir / 'stageD_projection.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timepoint cohort Stage D: extract Phikon features and project "
                    "onto the existing manifold (diagnostic; no retraining)"
    )
    parser.add_argument("--stageA-inventory-json", required=True, type=Path)
    parser.add_argument("--png-dir", required=True, type=Path)
    parser.add_argument("--projector-dir", required=True, type=Path,
                        help="Saved AtlasProjector dir, e.g. "
                             "$SCRATCH/results/baseline/atlas_none_harmony_median/projector")
    parser.add_argument("--features-cache-dir", required=True, type=Path,
                        help="SEPARATE cache dir -- never the existing features_cache")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--patch-size", type=int, default=112)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--model", default="phikon")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-patches-per-slide", type=int, default=None,
                        help="Default None = no cap (matches run_all.py's uncapped "
                             "cache). Set this only if the job cannot finish in "
                             "walltime; no paired comparison here requires a cap.")
    parser.add_argument("--patch-sample-seed", type=int, default=42)
    parser.add_argument("--baseline-sample-size", type=int, default=DEFAULT_BASELINE_SAMPLE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-umap", action="store_true", default=True,
                        help="Skip the cosmetic UMAP transform (default). Pseudotime "
                             "is unaffected; see module docstring.")
    parser.add_argument("--no-skip-umap", dest="skip_umap", action="store_false")
    args = parser.parse_args()

    print("=" * 60)
    print("  Timepoint cohort — Stage D: feature extraction + projection")
    print("=" * 60)
    print("\n  DIAGNOSTIC RUN despite a FAILED stain gate (hematoxylin-specific).")
    print("  Not a validated timepoint result. No correction applied.\n")

    if not args.projector_dir.exists():
        sys.exit(
            f"ERROR: --projector-dir not found:\n  {args.projector_dir}\n"
            "PROJECT_STATE.md marks baseline/atlas_none_harmony_median as 'pending' — "
            "confirm that run completed and wrote its projector/ subdirectory. "
            "There is no fallback to another manifold: projecting onto a different "
            "run would silently change every pseudotime value."
        )
    for required in ("scaler.pkl", "pca.pkl", "knn_pseudotime.pkl"):
        if not (args.projector_dir / required).exists():
            sys.exit(f"ERROR: {args.projector_dir} is missing {required} — not a "
                     "complete AtlasProjector directory.")

    with open(args.stageA_inventory_json) as f:
        stageA = json.load(f)
    usable = stageA["usable_slides"]
    print(f"Loaded Stage A v2 inventory: {len(usable)} usable slides")

    from .projector import AtlasProjector
    print(f"\n=== Loading projector: {args.projector_dir} ===")
    projector = AtlasProjector.load(str(args.projector_dir))

    print("\n=== Projector provenance check (hard gate) ===")
    provenance = check_projector_provenance(projector, seed=args.seed)
    for k, v in provenance.items():
        print(f"  {k}: {v}")

    if args.skip_umap:
        # In-memory only; the saved artifact on disk is never written to.
        projector.umap_reducer_ = None
        print("\n  UMAP transform disabled (cosmetic only; pseudotime unaffected).")

    print("\n=== Training-manifold support baseline ===")
    baseline = compute_training_support_baseline(
        projector, args.baseline_sample_size, args.seed)
    if baseline.get("available"):
        mk = baseline["mean_knn_distance"]
        print(f"  median={mk['median']:.4f}  p95={mk['p95']:.4f}  p99={mk['p99']:.4f}")
    else:
        print(f"  UNAVAILABLE: {baseline.get('reason')}")

    args.features_cache_dir.mkdir(parents=True, exist_ok=True)
    per_slide_dir = args.output_dir / "per_slide"
    per_slide_dir.mkdir(parents=True, exist_ok=True)

    model_state = {"loaded": False, "model": None, "processor": None, "device": None}
    slide_entries = []
    failed_slides = {}

    print(f"\n=== Processing {len(usable)} slides ===")
    for i, row in enumerate(usable, 1):
        stem = row["raw_stem"]
        print(f"\n  [{i}/{len(usable)}] {stem}")
        png_path = _png_path(args.png_dir, stem)
        try:
            feats, n_extracted, err = extract_or_load_features(
                png_path, stem, args.features_cache_dir, args, model_state)
            if err is not None:
                failed_slides[stem] = err
                print(f"    SKIPPED: {err}")
                continue
            pt, mean_knn, first_nn = project_slide(projector, feats)
            del feats
            np.save(per_slide_dir / f"projected_pt_{stem}.npy", pt)
            np.save(per_slide_dir / f"nn_mean_knn_{stem}.npy", mean_knn)
            entry = summarize_slide(stem, row, pt, mean_knn, first_nn, n_extracted, baseline)
            slide_entries.append(entry)
            print(f"    PT median={entry['pseudotime_median']:.4f}  "
                  f"beyond p99={entry['frac_beyond_training_p99']}")
        except Exception as e:  # one bad slide must not abort an overnight job
            failed_slides[stem] = repr(e)
            print(f"    ERROR: {e!r} -- excluded")

    if not slide_entries:
        sys.exit("ERROR: no slide projected successfully; nothing to aggregate.")

    # Both variants emitted, matching Stage B v2's sensitivity convention, so
    # Stage E can consume either directly rather than re-deriving them.
    mouse_level = aggregate_to_mouse_level(slide_entries)
    mouse_level_excl = aggregate_to_mouse_level(
        [e for e in slide_entries if not e["has_suffix_slide"]])
    by_timepoint = summarize_by_timepoint(slide_entries)

    result = {
        "projector_dir": str(args.projector_dir),
        "projector_provenance": provenance,
        "training_support_baseline": baseline,
        "extrapolation_fraction_warn": EXTRAPOLATION_FRACTION_WARN,
        "extraction_settings": {
            "patch_size": args.patch_size, "stride": args.stride,
            "model": args.model, "stain_method": "none (fixed)",
            "max_patches_per_slide": args.max_patches_per_slide,
            "roi": "whole-slide (no annotations for this cohort)",
            "umap_skipped": bool(args.skip_umap),
        },
        "per_slide": slide_entries,
        "mouse_level": mouse_level,
        "mouse_level_by_suffix": {
            "excluding_suffix": mouse_level_excl,
            "including_suffix": mouse_level,
        },
        "by_timepoint": by_timepoint,
        "failed_slides": failed_slides,
    }
    write_outputs(result, args.output_dir)

    print("\n" + "=" * 60)
    print("  STAGE D COMPLETE")
    print("=" * 60)
    n_extrap = sum(1 for e in slide_entries if e.get("substantially_extrapolated"))
    print(f"\n  {len(slide_entries)} slides projected, {len(failed_slides)} failed")
    print(f"  {n_extrap} slide(s) flagged substantially extrapolated")
    print(f"  Output dir: {args.output_dir}")
    print("\n  Next: Stage E (analysis/timepoint_diagnostic.py).")


if __name__ == "__main__":
    main()
