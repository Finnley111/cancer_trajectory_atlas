"""
Root-selection sensitivity: is the pseudotime axis an artifact of its own root rule?

WHY THIS EXISTS
---------------
`analysis/diffusion.py:compute_dpt_multi_root()` selects its 20 DPT roots as
`np.argsort(nuclear_density)[:20]` — the 20 patches with the LOWEST nuclear
density, computed by `validation/morphological_features.py:
compute_nuclear_density_quick()`. But `nuclear_density` is ALSO (a) one of the
six morphological features used to validate pseudotime in
`validation/correlations.py`, and (b) the covariate partialled out in
`analysis/cellularity_confound.py`. The pseudotime axis is therefore partly
DEFINED by a quantity it is later VALIDATED against, and partly by the quantity
used to ADJUST that validation. Neither the reported nuclear_density correlation
nor the "collapses under cellularity adjustment" finding can be read as
independent until that circularity is quantified.

This module quantifies it with two checks:

  CHECK A (primary) — geometry-seeded roots. Re-runs multi-root DPT per section
    changing ONLY the root selection rule: roots become the 20 patches at an
    extreme of the first non-trivial diffusion component (DC1), i.e. derived
    from manifold geometry rather than from any image-derived feature.
    Everything else is byte-identical to the production run because it is
    LITERALLY REUSED, not recomputed: the same cached-feature-derived PCA
    embedding (adata.X), the same stored neighbour graph (adata.obsp), the same
    stored diffusion map (adata.obsm['X_diffmap']) and its eigenvalues
    (adata.uns['diffmap_evals']). Only `sc.tl.dpt` re-runs, with the same
    20 roots, the same per-root inf-clamping, the same median-across-roots
    aggregation, and the same min-max normalisation as
    compute_dpt_multi_root(). The headline number is the Spearman correlation
    between the ORIGINAL and GEOMETRY-SEEDED pseudotime vectors.

  CHECK C (robustness) — alternative confound covariate. Re-runs the cellularity
    confound analysis on the EXISTING, UNCHANGED per-section pseudotime,
    partialling out `nc_ratio` instead of `nuclear_density`. nc_ratio is a
    density-related measure that is NOT used in root selection, so it separates
    "controlling for cellularity" from "controlling for the axis's own origin".

TAIL CHOICE (Check A)
---------------------
DC1 has two tails and the task's direction requirement ("seed at the sparse /
low-density end, so the axis runs comparably to the current one") cannot be
resolved from geometry alone. Rather than pick silently, BOTH tails are run and
both are reported in full. The tail whose 20 root patches have the lower mean
nuclear_density is labelled `direction_matched` — nuclear_density is used here
ONLY to orient/label an axis whose root SET was already fully determined by DC1
geometry, which is a strictly weaker dependence than using it to choose the
roots. The other tail is reported alongside, not discarded.

REPRODUCIBILITY GUARDS
----------------------
Every value this module compares against is first re-derived from the stored
artifacts and cross-checked against the saved JSON:
  - original feature correlations, recomputed from adata.obs, are checked
    against validation.json's `feature_correlations`;
  - the existing nuclear_density-controlled partials, recomputed with this
    module's parameterised loop, are checked against the saved
    cellularity_confound.json.
Any value that cannot be reproduced is REPORTED AS UNREPRODUCIBLE rather than
silently replaced with a recomputation presented as comparable.

CONSTRAINTS HONOURED
--------------------
NEW module. Imports from `analysis/diffusion.py`, `analysis/cellularity_confound.py`,
and `validation/correlations.py` but modifies none of them. Never re-extracts
features. Never writes into an existing results directory — all output goes to
the NEW --output-dir.

CLI
---
  python -m cancer_trajectory_atlas.analysis.root_sensitivity \\
      --sections   2M-1 2M-2 \\
      --run-dirs   $SCRATCH/results/per_section/atlas_2M-1 \\
                   $SCRATCH/results/per_section/atlas_2M-2 \\
      --output-dir $SCRATCH/results/root_sensitivity \\
      --n-roots    20 \\
      --n-permutations 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# Imported, never modified.
from .cellularity_confound import partial_spearman
from ..validation.correlations import permutation_test

MORPH_FEATURES = [
    "nuclear_density",
    "mean_nuclear_area",
    "nc_ratio",
    "texture_entropy",
    "h_intensity",
    "packing_irregularity",
]

# Production threshold, from cellularity_confound.analyze_run_nuclear_density.
SURVIVE_THRESHOLD = 0.1

# Tolerance for declaring a saved value reproduced from the stored artifacts.
REPRO_TOL = 5e-3

# |delta rho| below which a correlation is called robust to root choice.
ROBUST_DELTA_TOL = 0.10


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else str(v)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _safe_rho(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_run(run_dir: Path):
    """Load a per-section run's AnnData plus its saved validation/confound JSON.

    Returns (adata, validation_json_or_None, confound_json_or_None).
    """
    import anndata as ad

    h5ad = run_dir / "adata_full.h5ad"
    if not h5ad.exists():
        raise FileNotFoundError(
            f"{h5ad} not found — this analysis reuses an existing per-section run "
            "and cannot regenerate it."
        )
    adata = ad.read_h5ad(h5ad)

    validation = None
    vpath = run_dir / "validation.json"
    if vpath.exists():
        with open(vpath) as f:
            validation = json.load(f)

    confound = None
    cpath = run_dir / "cellularity_confound" / "cellularity_confound.json"
    if cpath.exists():
        with open(cpath) as f:
            confound = json.load(f)

    return adata, validation, confound


def check_required_columns(adata) -> list:
    """Return the list of required obs columns that are missing."""
    required = MORPH_FEATURES + ["pseudotime"]
    return [c for c in required if c not in adata.obs.columns]


# ── Check A: diffusion-component root selection ───────────────────────────────

def pick_diffusion_component(adata, dc_index_override: int = None) -> dict:
    """Identify the first NON-TRIVIAL diffusion component in adata.obsm['X_diffmap'].

    scanpy's `sc.tl.diffmap` returns the steady-state eigenvector as column 0;
    the first component carrying trajectory structure is normally column 1.

    The steady-state vector is NOT numerically constant — it is unit-norm and
    near-constant-positive (it tracks the stationary distribution / local graph
    degree), so its std is small but nonzero. Detecting it by "std ~= 0" fails,
    which is exactly the bug that made the first run of this analysis seed roots
    from graph density instead of from DC1. Detection therefore keys on the two
    decisive signatures: eigenvalue == 1, and |mean|/std >> 0 (a non-trivial
    component is mean-zero, so its std equals 1/sqrt(n_patches) for a unit-norm
    eigenvector, and its |mean|/std is ~0).
    """
    if "X_diffmap" not in adata.obsm:
        raise KeyError(
            "adata.obsm['X_diffmap'] not found — the stored run has no diffusion "
            "map to seed from. Check A cannot run without re-running the pipeline, "
            "which this analysis is forbidden from doing."
        )
    dm = np.asarray(adata.obsm["X_diffmap"], dtype=float)
    stds = dm.std(axis=0)
    max_std = float(stds.max()) if stds.size else 0.0

    means = dm.mean(axis=0)
    # |mean| / std. A non-trivial diffusion component is mean-zero, so this is
    # ~0; the steady-state eigenvector is near-constant and positive, so this is
    # large (order sqrt(N) in the limit). This is the decisive discriminator —
    # NOT std alone, which merely looks "small" for the trivial vector rather
    # than zero, and which an earlier version of this function wrongly relied on.
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_over_std = np.abs(means) / np.where(stds > 0, stds, np.nan)

    evals = adata.uns.get("diffmap_evals")
    evals_arr = np.asarray(evals).ravel().astype(float) if evals is not None else None
    evals_list = [float(v) for v in evals_arr] if evals_arr is not None else None

    trivial = []
    reasons = {}
    for i in range(dm.shape[1]):
        why = []
        # Signature 1: eigenvalue exactly 1 — the stationary eigenvector of a
        # diffusion (row-stochastic) operator.
        if evals_arr is not None and i < len(evals_arr) and abs(evals_arr[i] - 1.0) <= 1e-6:
            why.append(f"eigenvalue {evals_arr[i]:.8f} ~= 1")
        # Signature 2: near-constant single-sign vector.
        if np.isfinite(mean_over_std[i]) and mean_over_std[i] > 1.0:
            why.append(f"|mean|/std = {mean_over_std[i]:.2f} > 1 (near-constant)")
        # Signature 3: numerically constant.
        if stds[i] <= 1e-8 * max(max_std, 1e-12):
            why.append("numerically constant")
        if why:
            trivial.append(i)
            reasons[str(i)] = "; ".join(why)

    nontrivial = [int(i) for i in range(dm.shape[1]) if i not in trivial]
    if not nontrivial:
        raise ValueError("Every diffusion-map column looks trivial — cannot seed from DC1.")

    if dc_index_override is not None:
        if not (0 <= dc_index_override < dm.shape[1]):
            raise ValueError(
                f"--dc-index {dc_index_override} out of range for X_diffmap with "
                f"{dm.shape[1]} columns."
            )
        dc_index = int(dc_index_override)
        selection = f"forced via --dc-index {dc_index}"
    else:
        dc_index = nontrivial[0]
        selection = "first non-trivial column"

    # Hard guard: refuse to seed from a column that looks like the steady state.
    if dc_index in trivial:
        raise ValueError(
            f"X_diffmap column {dc_index} was selected but is TRIVIAL "
            f"({reasons[str(dc_index)]}). Seeding roots from the steady-state "
            "eigenvector ranks patches by stationary graph density, not by "
            "position along a diffusion component, which is not the intended "
            "test. First non-trivial column is "
            f"{nontrivial[0]}; re-run with --dc-index {nontrivial[0]} to override."
        )

    if dc_index != 1:
        print(f"  NOTE: DC1 resolved to X_diffmap column {dc_index}, not the usual "
              f"column 1. Trivial columns detected: {trivial}.")

    return {
        "n_diffmap_columns": int(dm.shape[1]),
        "column_stds": [float(s) for s in stds],
        "column_means": [float(m) for m in means],
        "column_abs_mean_over_std": [float(r) for r in mean_over_std],
        "expected_std_if_unit_norm_mean_zero": float(1.0 / np.sqrt(dm.shape[0])),
        "trivial_columns_detected": trivial,
        "trivial_detection_reasons": reasons,
        "dc_index_used": dc_index,
        "dc_index_selection": selection,
        "dc_index_note": (
            f"Using X_diffmap column {dc_index} as DC1 ({selection}). Columns "
            f"{trivial} were detected as trivial and skipped — see "
            "trivial_detection_reasons. A non-trivial diffusion component is "
            "unit-norm and mean-zero, so its std should be ~1/sqrt(n_patches) = "
            f"{1.0 / np.sqrt(dm.shape[0]):.6f}; compare against column_stds."
        ),
        "diffmap_evals": evals_list,
    }


def select_roots_from_dc(dc_values: np.ndarray, n_roots: int, tail: str) -> list:
    """The n_roots patches at one extreme of DC1.

    tail='low'  → most negative DC1 (argsort ascending, first n)
    tail='high' → most positive DC1 (argsort descending, first n)

    This mirrors compute_dpt_multi_root's `np.argsort(x)[:n_roots]` selection
    shape exactly; only the ranked quantity changes.
    """
    order = np.argsort(dc_values)
    if tail == "low":
        return [int(i) for i in order[:n_roots]]
    if tail == "high":
        return [int(i) for i in order[::-1][:n_roots]]
    raise ValueError(f"tail must be 'low' or 'high', got {tail!r}")


def build_dpt_adata(adata):
    """A minimal AnnData carrying exactly what sc.tl.dpt reads.

    sc.tl.dpt needs: uns['neighbors'], obsp['distances'], obsp['connectivities'],
    obsm['X_diffmap'], uns['diffmap_evals'], uns['iroot']. Copying that subset
    instead of the full object keeps the 20 per-root copies cheap without
    changing any input to the DPT computation — the morphological feature
    columns, UMAP coordinates, and PAGA results the full object also carries are
    never read by dpt.
    """
    import anndata as ad

    missing = [k for k in ("distances", "connectivities") if k not in adata.obsp]
    if missing:
        raise KeyError(
            f"adata.obsp missing {missing} — the stored neighbour graph is "
            "incomplete, so Check A cannot reuse it. Refusing to rebuild the "
            "graph, which would no longer be a root-only change."
        )
    if "neighbors" not in adata.uns:
        raise KeyError("adata.uns['neighbors'] not found — stored graph metadata is incomplete.")
    if "diffmap_evals" not in adata.uns:
        raise KeyError("adata.uns['diffmap_evals'] not found — cannot reuse the stored diffusion map.")

    small = ad.AnnData(X=np.asarray(adata.X, dtype=np.float32))
    small.obsm["X_diffmap"] = np.asarray(adata.obsm["X_diffmap"], dtype=np.float32)
    # Copy EVERY obsp entry, not just the two required ones: uns['neighbors'] may
    # reference non-default graph keys (connectivities_key/distances_key), and
    # dropping them here would make sc.tl.dpt fall back to a different graph.
    for key in adata.obsp.keys():
        small.obsp[key] = adata.obsp[key]
    small.uns["neighbors"] = adata.uns["neighbors"]
    small.uns["diffmap_evals"] = adata.uns["diffmap_evals"]
    return small


def verify_fast_dpt_equivalence(adata, root_idx: int) -> dict:
    """Prove that mutating one AnnData in place gives byte-identical DPT to the
    production path's per-root `adata.copy()`.

    compute_dpt_multi_root copies the object for every root. That copy is for
    safety, not for the numbers: sc.tl.dpt reads uns['iroot'] and writes
    obs['dpt_pseudotime']. Reusing one object is therefore expected to be
    identical and is many times cheaper, which is what makes a null of useful
    size affordable. Expected is not verified, so this checks it on real data
    before the fast path is used, and raises rather than silently proceeding.
    """
    import scanpy as sc

    a = build_dpt_adata(adata)
    a.uns["iroot"] = int(root_idx)
    sc.tl.dpt(a)
    via_copy = a.obs["dpt_pseudotime"].values.copy()

    b = build_dpt_adata(adata)
    b.uns["iroot"] = int(root_idx)
    sc.tl.dpt(b)
    _ = b.obs["dpt_pseudotime"].values.copy()      # first use of the reused object
    b.uns["iroot"] = int(root_idx)
    sc.tl.dpt(b)                                    # second use, same object
    via_reuse = b.obs["dpt_pseudotime"].values.copy()

    finite = np.isfinite(via_copy) & np.isfinite(via_reuse)
    max_abs_diff = float(np.abs(via_copy[finite] - via_reuse[finite]).max()) if finite.any() else 0.0
    identical = bool(max_abs_diff == 0.0
                     and np.array_equal(np.isfinite(via_copy), np.isfinite(via_reuse)))
    if not identical:
        raise RuntimeError(
            f"Fast DPT path is NOT equivalent to the per-root-copy path "
            f"(max abs diff {max_abs_diff:.3e}). Refusing to use it — the null "
            "draws would not be comparable to the primary results. Re-run with "
            "--no-fast-null to force the copy path."
        )
    return {"root_index": int(root_idx), "max_abs_diff": max_abs_diff,
            "identical": identical,
            "note": ("Verified on real data: reusing one AnnData across roots gives "
                     "identical DPT to the production per-root copy. The fast path "
                     "is used ONLY for null draws; the DC1 tails and every primary "
                     "result still use the copy path.")}


def run_multi_root_dpt(adata, root_indices: list, base=None, reuse: bool = False) -> tuple:
    """Re-implementation of compute_dpt_multi_root's AGGREGATION, with the root
    set supplied rather than derived from nuclear density.

    Mirrors analysis/diffusion.py:compute_dpt_multi_root lines 171-197 exactly:
      per root  → adata.copy(), set uns['iroot'], sc.tl.dpt
      non-finite → clamped to the max finite value of that root's vector
      aggregate → median across roots, then min-max to [0, 1]
      also      → std across roots (un-normalised)

    Returns (pseudotime, pseudotime_std, notes).
    """
    import scanpy as sc

    if base is None:
        base = build_dpt_adata(adata)
    n_patches = base.n_obs
    n_roots = len(root_indices)
    pt_matrix = np.zeros((n_roots, n_patches), dtype=np.float64)
    n_nonfinite_total = 0

    for r_i, root_idx in enumerate(root_indices):
        tmp = base if reuse else base.copy()
        tmp.uns["iroot"] = int(root_idx)
        sc.tl.dpt(tmp)
        pt = tmp.obs["dpt_pseudotime"].values.copy()

        finite_mask = np.isfinite(pt)
        if not finite_mask.all():
            n_nonfinite_total += int((~finite_mask).sum())
            pt[~finite_mask] = pt[finite_mask].max() if finite_mask.any() else 0.0

        pt_matrix[r_i] = pt

    pseudotime_median = np.median(pt_matrix, axis=0)
    pseudotime_std = np.std(pt_matrix, axis=0)

    pt_min, pt_max = float(pseudotime_median.min()), float(pseudotime_median.max())
    if pt_max - pt_min < 1e-10:
        pseudotime = np.zeros(n_patches)
        degenerate = True
    else:
        pseudotime = (pseudotime_median - pt_min) / (pt_max - pt_min)
        degenerate = False

    notes = {
        "n_roots": n_roots,
        "n_patches": int(n_patches),
        "n_nonfinite_clamped_across_all_roots": n_nonfinite_total,
        "median_pt_range_before_normalisation": [pt_min, pt_max],
        "degenerate_zero_range": degenerate,
    }
    return pseudotime, pseudotime_std, notes


def root_provenance(adata, roots: list) -> dict:
    """How concentrated are the 20 roots across slides?

    A root set drawn from one small contiguous region of one slide is a local
    outlier lobe, not a trajectory endpoint, and anchoring DPT there produces an
    axis dominated by that slide's idiosyncrasies. This is a far more reliable
    validity signal than the mean nuclear_density of 20 patches.
    """
    key = "slide_id" if "slide_id" in adata.obs.columns else None
    out = {"index_min": int(min(roots)), "index_max": int(max(roots)),
           "index_span": int(max(roots) - min(roots)), "n_patches": int(adata.n_obs)}
    out["index_span_as_frac_of_dataset"] = out["index_span"] / max(adata.n_obs - 1, 1)
    if key is None:
        out["slide_breakdown"] = None
        out["note"] = "adata.obs has no 'slide_id' column — provenance is index-based only."
        return out
    labels = adata.obs[key].astype(str).values[np.asarray(roots)]
    uniq, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    out["n_distinct_slides"] = int(len(uniq))
    out["slide_breakdown"] = {str(uniq[i]): int(counts[i]) for i in order}
    out["max_share_from_one_slide"] = float(counts.max() / len(roots))
    out["single_slide_dominated"] = bool(counts.max() / len(roots) >= 0.8)
    out["note"] = (
        f"{out['n_distinct_slides']} distinct slide(s) among {len(roots)} roots; "
        f"largest single-slide share {out['max_share_from_one_slide']:.0%}; "
        f"index span {out['index_span']} of {adata.n_obs}. A set concentrated in one "
        "slide (>=80%) and a narrow index span indicate a local outlier lobe rather "
        "than a manifold endpoint — treat that tail's axis as unreliable."
    )
    return out


def _draw_roots(adata, dc_values, n_roots, mode, rng):
    """Uniform or locally-clustered root draw.

    The production rule takes the 20 LOWEST-nuclear_density patches, which is a
    tightly clustered set, not a spread one. A uniform null therefore differs
    from it in two ways at once (which region, and clustered vs spread), and
    cannot separate them. The 'clustered' mode draws a random seed patch and
    takes its n_roots nearest neighbours in diffusion space, matching the
    production set's local structure while randomising its location — so
    comparing the two nulls isolates location from clustering.
    """
    n = adata.n_obs
    if mode == "uniform":
        return [int(i) for i in rng.choice(n, size=n_roots, replace=False)]
    if mode == "clustered":
        dm = np.asarray(adata.obsm["X_diffmap"], dtype=float)
        seed_idx = int(rng.integers(n))
        d = np.linalg.norm(dm - dm[seed_idx], axis=1)
        return [int(i) for i in np.argsort(d)[:n_roots]]
    raise ValueError(f"unknown null mode {mode!r}")


def run_root_null(adata, original_pt, obs, dc_values, n_roots, n_draws,
                  mode="uniform", seed=0, base=None, reuse=False) -> dict:
    """Null baseline for root choice, with FULL per-feature correlations.

    Without this there is no scale for the DC1 numbers, and no way to ask the
    question that actually matters: is the ORIGINAL feature correlation inside
    or outside the range you would get from an arbitrary root set? A correlation
    that sits inside the null range is not attributable to the root rule; one
    that sits outside it is.

    Only pseudotime and the six correlations are computed per draw (no
    permutation suite), so the cost is n_draws * n_roots DPT runs.
    """
    rng = np.random.default_rng(seed)
    rhos, draws = [], []
    per_feature_rhos = {f: [] for f in MORPH_FEATURES}

    for d in range(n_draws):
        roots = _draw_roots(adata, dc_values, n_roots, mode, rng)
        pt, _, _ = run_multi_root_dpt(adata, roots, base=base, reuse=reuse)
        r = float(spearmanr(original_pt, pt).statistic)
        corrs = correlate_all_features(obs, pt)
        for f in MORPH_FEATURES:
            per_feature_rhos[f].append(corrs[f]["rho"])
        rhos.append(r)
        draws.append({
            "draw": d, "rho_vs_original": r,
            "feature_rho": {f: corrs[f]["rho"] for f in MORPH_FEATURES},
            "root_provenance": root_provenance(adata, roots),
        })
        print(f"    {mode} draw {d + 1}/{n_draws}: rho_vs_original = {r:+.4f}")

    arr = np.abs(np.array(rhos, dtype=float))
    feature_summary = {}
    for f in MORPH_FEATURES:
        v = np.array([x for x in per_feature_rhos[f] if np.isfinite(x)], dtype=float)
        orig = _safe_rho(original_pt, obs[f].values.astype(float))
        if v.size == 0:
            feature_summary[f] = {"original_rho": orig, "null_min": None,
                                  "null_max": None, "original_inside_null_range": None}
            continue
        lo, hi = float(v.min()), float(v.max())
        inside = bool(lo <= orig <= hi)
        feature_summary[f] = {
            "original_rho": orig,
            "null_min": lo, "null_median": float(np.median(v)), "null_max": hi,
            "null_per_draw": [float(x) for x in v],
            "original_inside_null_range": inside,
            "distance_outside_null_range": 0.0 if inside else float(min(abs(orig - lo), abs(orig - hi))),
            "attribution": (
                "inside null range — NOT attributable to the root rule"
                if inside else
                "outside null range — the root rule contributes to this value"
            ),
        }

    return {
        "mode": mode, "n_draws": n_draws, "n_roots": n_roots, "seed": seed,
        "rho_vs_original_per_draw": rhos,
        "mean_abs_rho": float(arr.mean()), "median_abs_rho": float(np.median(arr)),
        "min_abs_rho": float(arr.min()), "max_abs_rho": float(arr.max()),
        "feature_rho_null": feature_summary,
        "draws": draws,
        "note": (
            f"{n_draws} {mode} root sets, same aggregation as every other run here. "
            "TWO readings: (1) compare |rho_vs_original| against the DC1 tails — a "
            "DC1 tail below the null floor is an ATYPICAL root set, not evidence the "
            "axis is fragile; (2) for each feature, check original_inside_null_range "
            "— an original correlation inside the null range is not attributable to "
            "the root-selection rule."
        ),
    }


def axis_identity_diagnostic(adata, dc_values, original_pt) -> dict:
    """What IS the production pseudotime, geometrically?

    If root choice barely matters (see the nulls), the axis is fixed by the
    manifold, and it is worth asking which manifold quantity it is:
      - monotone in DC1        -> a genuine 1-D trajectory coordinate
      - tracking |DC1 - median| or distance from the diffusion-map centroid
                               -> an ECCENTRICITY measure ("how far from the
                                  middle"), which is directionless and would
                                  mean 'early' vs 'late' is set only by the root
                                  rule's choice of end, not by the geometry.
    Cheap: correlations on stored arrays, no DPT.
    """
    dm = np.asarray(adata.obsm["X_diffmap"], dtype=float)
    nontrivial = [i for i in range(dm.shape[1])
                  if not (abs(dm[:, i].mean()) / max(dm[:, i].std(), 1e-30) > 1.0)]
    sub = dm[:, nontrivial] if nontrivial else dm
    centroid_dist = np.linalg.norm(sub - sub.mean(axis=0), axis=1)
    dc_ecc = np.abs(dc_values - np.median(dc_values))

    r_dc = _safe_rho(original_pt, dc_values)
    r_ecc_dc = _safe_rho(original_pt, dc_ecc)
    r_ecc_cen = _safe_rho(original_pt, centroid_dist)

    if abs(r_dc) >= 0.7:
        verdict = ("MONOTONE IN DC1 — the pseudotime is essentially the first "
                   "diffusion coordinate, i.e. a genuine 1-D manifold axis.")
    elif max(abs(r_ecc_dc), abs(r_ecc_cen)) >= 0.7 and abs(r_dc) < 0.4:
        verdict = ("ECCENTRICITY — the pseudotime tracks distance from the middle "
                   "of the manifold rather than position along an axis. It is "
                   "directionless by construction, so which end counts as 'early' "
                   "is set entirely by the root rule, NOT by the geometry. Feature "
                   "correlation SIGNS inherit that choice.")
    else:
        verdict = ("NEITHER cleanly — the pseudotime is not well described by DC1 "
                   "nor by a simple eccentricity measure; interpret with care.")

    return {
        "rho_pt_vs_dc1": r_dc,
        "rho_pt_vs_abs_dc1_from_median": r_ecc_dc,
        "rho_pt_vs_diffmap_centroid_distance": r_ecc_cen,
        "n_nontrivial_components_used": len(nontrivial),
        "verdict": verdict,
    }


def correlate_all_features(obs, pseudotime: np.ndarray) -> dict:
    """Spearman rho + p for each of the six morphological features."""
    out = {}
    for feat in MORPH_FEATURES:
        vals = obs[feat].values.astype(float)
        valid = np.isfinite(vals) & np.isfinite(pseudotime)
        if valid.sum() < 10:
            out[feat] = {"rho": float("nan"), "p_value": float("nan")}
            continue
        res = spearmanr(pseudotime[valid], vals[valid])
        out[feat] = {"rho": float(res.statistic), "p_value": float(res.pvalue)}
    return out


def reconcile_original_correlations(recomputed: dict, validation: dict) -> dict:
    """Cross-check rho recomputed from adata.obs against validation.json.

    A mismatch means the saved number cannot be reproduced from the stored
    artifacts; it is reported as such rather than being quietly replaced.
    """
    if validation is None or "feature_correlations" not in validation:
        return {
            "status": "UNAVAILABLE",
            "note": (
                "validation.json absent or missing 'feature_correlations' — the "
                "original rho values in this report were recomputed from "
                "adata.obs and could NOT be cross-checked against the saved "
                "validation output."
            ),
            "per_feature": {},
        }

    saved = validation["feature_correlations"]
    per_feature = {}
    mismatches = []
    for feat in MORPH_FEATURES:
        s = saved.get(feat, {}).get("rho")
        r = recomputed.get(feat, {}).get("rho")
        if s is None or r is None or not np.isfinite(r):
            per_feature[feat] = {"saved": s, "recomputed": r, "match": None}
            continue
        delta = abs(float(s) - float(r))
        ok = delta <= REPRO_TOL
        per_feature[feat] = {
            "saved": float(s),
            "recomputed": float(r),
            "abs_delta": float(delta),
            "match": bool(ok),
        }
        if not ok:
            mismatches.append(feat)

    return {
        "status": "REPRODUCED" if not mismatches else "MISMATCH",
        "tolerance": REPRO_TOL,
        "mismatched_features": mismatches,
        "note": (
            "All original feature correlations were reproduced from "
            "adata_full.h5ad within tolerance; the side-by-side comparison below "
            "is internally consistent."
            if not mismatches else
            f"Could NOT reproduce {mismatches} from adata_full.h5ad within "
            f"{REPRO_TOL}. The saved validation.json value and the value "
            "recomputed from stored obs disagree for these features. Treat the "
            "corresponding side-by-side rows as unverified — do not read the "
            "delta column for them as a root-choice effect, since part of it is "
            "an unexplained discrepancy in the baseline."
        ),
        "per_feature": per_feature,
    }


def run_check_a_for_section(
    section: str,
    adata,
    validation: dict,
    n_roots: int,
    n_permutations: int,
    dc_index_override: int = None,
    n_random_draws: int = 0,
    use_fast_null: bool = True,
) -> dict:
    """Check A for one section: both DC1 tails, full side-by-side."""
    obs = adata.obs
    original_pt = obs["pseudotime"].values.astype(float)
    nuclear_density = obs["nuclear_density"].values.astype(float)

    dc_info = pick_diffusion_component(adata, dc_index_override)
    dc_values = np.asarray(adata.obsm["X_diffmap"], dtype=float)[:, dc_info["dc_index_used"]]

    # Original correlations, recomputed from stored obs and reconciled.
    original_corrs = correlate_all_features(obs, original_pt)
    reconciliation = reconcile_original_correlations(original_corrs, validation)

    morph_dict = {f: obs[f].values.astype(float) for f in MORPH_FEATURES}

    tails = {}
    for tail in ("low", "high"):
        roots = select_roots_from_dc(dc_values, n_roots, tail)
        root_nd = nuclear_density[roots]
        print(f"\n  [{section}] DC1 '{tail}' tail: {n_roots} roots, "
              f"mean nuclear_density = {root_nd.mean():.6f}")

        pt_new, pt_std_new, dpt_notes = run_multi_root_dpt(adata, roots)

        rho_pt, p_pt = spearmanr(original_pt, pt_new)
        print(f"  [{section}] rho(original PT, geometry-seeded PT) = {rho_pt:+.4f}")

        new_corrs = correlate_all_features(obs, pt_new)

        print(f"  [{section}] {n_permutations}-shuffle permutation null "
              f"(production validation.correlations.permutation_test) ...")
        perm = permutation_test(pt_new, morph_dict, n_permutations=n_permutations)

        per_feature = {}
        for feat in MORPH_FEATURES:
            o = original_corrs[feat]["rho"]
            n = new_corrs[feat]["rho"]
            delta = abs(n - o) if np.isfinite(o) and np.isfinite(n) else float("nan")
            sign_pres = (
                bool(np.sign(o) == np.sign(n))
                if np.isfinite(o) and np.isfinite(n) and o != 0 and n != 0
                else None
            )
            robust = (
                bool(np.isfinite(delta) and delta <= ROBUST_DELTA_TOL and sign_pres)
                if sign_pres is not None else False
            )
            per_feature[feat] = {
                "original_rho": o,
                "geometry_seeded_rho": n,
                "abs_delta": delta,
                "sign_preserved": sign_pres,
                "geometry_seeded_p": new_corrs[feat]["p_value"],
                "geometry_seeded_perm_p": perm.get(feat, {}).get("perm_p_value"),
                "geometry_seeded_perm_significant": perm.get(feat, {}).get("significant"),
                "robust_to_root_choice": robust,
                "baseline_reproduced": reconciliation["per_feature"].get(feat, {}).get("match"),
            }

        tails[tail] = {
            "tail": tail,
            "root_indices": roots,
            "root_provenance": root_provenance(adata, roots),
            "root_dc1_values": [float(v) for v in dc_values[roots]],
            "root_nuclear_density_mean": float(root_nd.mean()),
            "root_nuclear_density_min": float(root_nd.min()),
            "root_nuclear_density_max": float(root_nd.max()),
            "dpt_notes": dpt_notes,
            "rho_original_vs_reseeded_pseudotime": float(rho_pt),
            "p_original_vs_reseeded_pseudotime": float(p_pt),
            "per_feature": per_feature,
            "pseudotime": pt_new,          # popped before JSON serialisation
            "pseudotime_std": pt_std_new,  # popped before JSON serialisation
        }

    # Direction labelling. The 20-root mean nuclear_density is a noisy basis
    # (20 patches, heavily overlapping ranges), so the primary criterion is now
    # rho(DC1, nuclear_density) over ALL patches: the sparse end is whichever DC1
    # tail runs opposite to nuclear density. The 20-root means are retained as a
    # secondary signal and disagreement between the two is reported, not hidden.
    nd_low = tails["low"]["root_nuclear_density_mean"]
    nd_high = tails["high"]["root_nuclear_density_mean"]
    by_root_means = "low" if nd_low <= nd_high else "high"

    axis_identity = axis_identity_diagnostic(adata, dc_values, original_pt)
    print(f"  [{section}] axis identity: {axis_identity['verdict']}")

    nulls = {}
    fast_check = None
    if n_random_draws > 0:
        fast_base = None
        if use_fast_null:
            print(f"  [{section}] verifying fast DPT path against the copy path ...")
            fast_check = verify_fast_dpt_equivalence(adata, int(np.argmin(dc_values)))
            print(f"  [{section}] fast path verified identical "
                  f"(max abs diff {fast_check['max_abs_diff']:.1e})")
            fast_base = build_dpt_adata(adata)
        for mode in ("uniform", "clustered"):
            print(f"  [{section}] {mode} root null ({n_random_draws} draws) ...")
            nulls[mode] = run_root_null(
                adata, original_pt, obs, dc_values, n_roots, n_random_draws, mode=mode,
                base=fast_base, reuse=use_fast_null,
            )
            print(f"  [{section}] {mode} |rho| range "
                  f"[{nulls[mode]['min_abs_rho']:.4f}, {nulls[mode]['max_abs_rho']:.4f}], "
                  f"median {nulls[mode]['median_abs_rho']:.4f}")
    random_null = nulls.get("uniform")

    rho_dc_nd = _safe_rho(dc_values, nuclear_density)
    by_full_n = "low" if rho_dc_nd >= 0 else "high"
    direction_matched = by_full_n if np.isfinite(rho_dc_nd) else by_root_means

    root_mean_gap = abs(nd_low - nd_high) / max(abs(nd_low), abs(nd_high), 1e-12)
    prov_low = tails["low"]["root_provenance"]
    prov_high = tails["high"]["root_provenance"]

    return {
        "section": section,
        "n_patches": int(adata.n_obs),
        "n_roots": n_roots,
        "n_permutations": n_permutations,
        "diffusion_component": dc_info,
        "original_correlation_reconciliation": reconciliation,
        "direction_matched_tail": direction_matched,
        "direction_agreement": {
            "by_full_n_rho_dc1_vs_nuclear_density": by_full_n,
            "rho_dc1_vs_nuclear_density": rho_dc_nd,
            "by_20_root_means": by_root_means,
            "root_mean_relative_gap": float(root_mean_gap),
            "criteria_agree": bool(by_full_n == by_root_means),
        },
        "direction_basis": (
            f"Primary criterion: rho(DC1, nuclear_density) over all {int(adata.n_obs)} "
            f"patches = {rho_dc_nd:+.4f}, so the sparse/low-density end is the "
            f"'{by_full_n}' tail. Secondary criterion (mean nuclear_density of the "
            f"{n_roots} roots: low {nd_low:.6f} vs high {nd_high:.6f}, relative gap "
            f"{root_mean_gap:.1%}) points to '{by_root_means}' — these "
            f"{'AGREE' if by_full_n == by_root_means else 'DISAGREE, so the label is unreliable and BOTH tails must be read'}. "
            "The 20-root mean is noisy at this n; the full-N criterion is preferred. "
            "nuclear_density is used ONLY to orient the axis — each tail's root SET "
            "was fixed by DC1 geometry alone, a strictly weaker dependence than the "
            "production rule, which uses nuclear_density to pick the roots. Both "
            "tails are reported in full; neither is discarded. Root provenance — "
            f"low tail: {prov_low['note']} high tail: {prov_high['note']}"
        ),
        "tails": tails,
        "random_root_null": random_null,
        "root_nulls": nulls,
        "axis_identity": axis_identity,
        "fast_dpt_equivalence_check": fast_check,
    }


# ── Check C: alternative confound covariate ───────────────────────────────────

def partial_with_permutation(
    pt: np.ndarray,
    obs,
    control_feature: str,
    test_features: list,
    n_permutations: int,
) -> dict:
    """Partial Spearman of pseudotime vs each test feature, controlling for one
    covariate, with a permutation null on the partial rho.

    This mirrors analysis/cellularity_confound.py:analyze_run_nuclear_density's
    loop exactly — same algebraic 3-variable partial formula (via the imported
    `partial_spearman`), same `np.random.default_rng(42)` seed, same
    precomputation of rho(feature, control) outside the permutation loop, same
    two-sided |null| >= |partial| p-value, same |partial_rho| >= 0.1 survival
    threshold. It is re-implemented here ONLY because the production function
    hardcodes `nuclear_density` as the covariate and is not parameterised; the
    production module is not modified. Parity is verified separately by
    reproducing the saved nuclear_density results with this same loop.
    """
    control = obs[control_feature].values.astype(float)

    rho_pt_control = _safe_rho(pt, control)
    rho_feat_control = {
        f: _safe_rho(obs[f].values.astype(float), control) for f in test_features
    }

    rng = np.random.default_rng(42)
    perm_nulls = {f: [] for f in test_features}
    for _ in range(n_permutations):
        pt_shuf = rng.permutation(pt)
        for feat in test_features:
            fvals = obs[feat].values.astype(float)
            valid = np.isfinite(pt_shuf) & np.isfinite(fvals) & np.isfinite(control)
            if valid.sum() < 10:
                perm_nulls[feat].append(float("nan"))
                continue
            rho_xy = float(spearmanr(pt_shuf[valid], fvals[valid]).statistic)
            rho_xz = float(spearmanr(pt_shuf[valid], control[valid]).statistic)
            rho_yz = rho_feat_control[feat]
            denom = np.sqrt((1 - rho_xz ** 2) * (1 - rho_yz ** 2))
            perm_nulls[feat].append(
                float((rho_xy - rho_xz * rho_yz) / denom) if denom >= 1e-10 else float("nan")
            )

    features = {}
    survivors, collapses = [], []
    for feat in test_features:
        fvals = obs[feat].values.astype(float)
        raw_rho = _safe_rho(pt, fvals)
        prho = partial_spearman(pt, fvals, control)
        delta = prho - raw_rho if np.isfinite(prho) and np.isfinite(raw_rho) else float("nan")
        nulls = np.array([v for v in perm_nulls[feat] if np.isfinite(v)])
        perm_p = float(np.mean(np.abs(nulls) >= abs(prho))) if nulls.size and np.isfinite(prho) else float("nan")

        status = "SURVIVES" if np.isfinite(prho) and abs(prho) >= SURVIVE_THRESHOLD else "collapses"
        (survivors if status == "SURVIVES" else collapses).append(feat)

        features[feat] = {
            "raw_rho": raw_rho,
            "partial_rho": prho,
            "delta": delta,
            "partial_perm_p": perm_p,
            "status": status,
        }

    return {
        "control_feature": control_feature,
        "rho_pt_vs_control": rho_pt_control,
        "n_permutations": n_permutations,
        "features": features,
        "summary": {"survivors": survivors, "collapses": collapses},
    }


def verify_against_saved_confound(recomputed: dict, saved: dict) -> dict:
    """Parity check: does this module's parameterised loop reproduce the saved
    nuclear_density-controlled partials?"""
    if saved is None or "features" not in saved:
        return {
            "status": "UNAVAILABLE",
            "note": (
                "cellularity_confound.json not found in the run directory — the "
                "existing nuclear_density-controlled partials shown below were "
                "recomputed by this module and could NOT be cross-checked against "
                "the original saved output."
            ),
            "per_feature": {},
        }

    per_feature = {}
    mismatches = []
    for feat, sv in saved["features"].items():
        rv = recomputed["features"].get(feat)
        if rv is None:
            continue
        s, r = sv.get("partial_rho"), rv.get("partial_rho")
        if s is None or r is None or not (np.isfinite(float(s)) and np.isfinite(float(r))):
            per_feature[feat] = {"saved": s, "recomputed": r, "match": None}
            continue
        delta = abs(float(s) - float(r))
        ok = delta <= REPRO_TOL
        per_feature[feat] = {
            "saved": float(s), "recomputed": float(r),
            "abs_delta": float(delta), "match": bool(ok),
        }
        if not ok:
            mismatches.append(feat)

    return {
        "status": "REPRODUCED" if not mismatches else "MISMATCH",
        "tolerance": REPRO_TOL,
        "mismatched_features": mismatches,
        "note": (
            "This module's parameterised partial-correlation loop reproduces the "
            "saved nuclear_density-controlled partials within tolerance, so the "
            "nc_ratio-controlled numbers it produces are computed by a verified-"
            "equivalent code path."
            if not mismatches else
            f"This module's loop did NOT reproduce the saved partials for "
            f"{mismatches} within {REPRO_TOL}. The nc_ratio comparison for those "
            "features is therefore not established as methodologically identical "
            "to the saved analysis and must not be read as directly comparable."
        ),
        "per_feature": per_feature,
    }


def run_check_c_for_section(
    section: str,
    adata,
    saved_confound: dict,
    n_permutations: int,
) -> dict:
    """Check C for one section: nuclear_density vs nc_ratio as covariate."""
    obs = adata.obs
    pt = obs["pseudotime"].values.astype(float)

    # Each covariate must be excluded from its own test set.
    features_ctrl_nd = [f for f in MORPH_FEATURES if f != "nuclear_density"]
    features_ctrl_nc = [f for f in MORPH_FEATURES if f != "nc_ratio"]
    comparable = [f for f in features_ctrl_nd if f in features_ctrl_nc]

    print(f"\n  [{section}] Check C — control: nuclear_density (reproduction of existing) ...")
    res_nd = partial_with_permutation(pt, obs, "nuclear_density", features_ctrl_nd, n_permutations)
    parity = verify_against_saved_confound(res_nd, saved_confound)

    print(f"  [{section}] Check C — control: nc_ratio (new) ...")
    res_nc = partial_with_permutation(pt, obs, "nc_ratio", features_ctrl_nc, n_permutations)

    nd_vals = obs["nuclear_density"].values.astype(float)
    nc_vals = obs["nc_ratio"].values.astype(float)
    rho_nd_nc = _safe_rho(nd_vals, nc_vals)

    comparison = {}
    verdict_changes = []
    for feat in comparable:
        a = res_nd["features"][feat]
        b = res_nc["features"][feat]
        changed = a["status"] != b["status"]
        if changed:
            verdict_changes.append(feat)
        comparison[feat] = {
            "raw_rho": a["raw_rho"],
            "partial_rho_ctrl_nuclear_density": a["partial_rho"],
            "status_ctrl_nuclear_density": a["status"],
            "partial_rho_ctrl_nc_ratio": b["partial_rho"],
            "status_ctrl_nc_ratio": b["status"],
            "verdict_changed": changed,
        }

    return {
        "section": section,
        "n_patches": int(adata.n_obs),
        "n_permutations": n_permutations,
        "rho_nuclear_density_vs_nc_ratio": rho_nd_nc,
        "covariate_independence_caveat": (
            f"nuclear_density and nc_ratio are themselves correlated at Spearman "
            f"rho = {rho_nd_nc:+.4f} in this section. Substituting nc_ratio is "
            "therefore a ROBUSTNESS check on covariate choice, NOT a fully "
            "independent test of the confound. The higher this correlation, the "
            "less the two adjustments can differ, and the weaker the evidence "
            "either way; read the comparison below with that number in hand."
        ),
        "control_nuclear_density": res_nd,
        "control_nc_ratio": res_nc,
        "reproduction_of_saved_confound": parity,
        "comparable_features": comparable,
        "asymmetry_note": (
            "Each covariate is excluded from its own test set, so the two analyses "
            "do not cover identical features: nc_ratio has a partial-|nuclear_density "
            "value but no partial-|nc_ratio value, and nuclear_density the reverse. "
            f"Only {comparable} appear under both covariates and are directly "
            "comparable; the other two rows are reported for completeness but have "
            "no counterpart."
        ),
        "comparison": comparison,
        "verdict_changes": verdict_changes,
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def write_check_a_figure(section: str, section_result: dict, output_dir: Path) -> None:
    tail = section_result["direction_matched_tail"]
    t = section_result["tails"][tail]
    feats = MORPH_FEATURES
    x = np.arange(len(feats))
    orig = [t["per_feature"][f]["original_rho"] for f in feats]
    new = [t["per_feature"][f]["geometry_seeded_rho"] for f in feats]
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.bar(x - w / 2, orig, w, label="original (density-seeded)", color="#4878CF", alpha=0.85)
    ax.bar(x + w / 2, new, w, label=f"geometry-seeded (DC1 {tail} tail)", color="#6ACC65", alpha=0.85)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in feats], fontsize=8)
    ax.set_ylabel("Spearman ρ with pseudotime")
    ax.set_title(f"{section}: feature correlations by root rule")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.hexbin(
        section_result["_original_pt"], t["pseudotime"],
        gridsize=45, cmap="Blues", mincnt=1,
    )
    ax2.plot([0, 1], [0, 1], "r--", lw=1)
    ax2.set_xlabel("Original pseudotime (density-seeded roots)")
    ax2.set_ylabel(f"Geometry-seeded pseudotime (DC1 {tail} tail)")
    ax2.set_title(
        f"{section}: ρ = {t['rho_original_vs_reseeded_pseudotime']:+.4f}\n"
        "(the headline root-sensitivity number)"
    )

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"check_a_root_sensitivity_{section}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_check_c_figure(section: str, section_result: dict, output_dir: Path) -> None:
    comp = section_result["comparison"]
    feats = section_result["comparable_features"]
    if not feats:
        return
    x = np.arange(len(feats))
    raw = [comp[f]["raw_rho"] for f in feats]
    p_nd = [comp[f]["partial_rho_ctrl_nuclear_density"] for f in feats]
    p_nc = [comp[f]["partial_rho_ctrl_nc_ratio"] for f in feats]
    w = 0.27

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, raw, w, label="raw ρ", color="#4878CF", alpha=0.85)
    ax.bar(x, p_nd, w, label="partial | nuclear_density", color="#D65F5F", alpha=0.85)
    ax.bar(x + w, p_nc, w, label="partial | nc_ratio (new)", color="#EE854A", alpha=0.85)
    for yl in (SURVIVE_THRESHOLD, -SURVIVE_THRESHOLD):
        ax.axhline(yl, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in feats], fontsize=8)
    ax.set_ylabel("Spearman ρ with pseudotime")
    ax.set_title(
        f"{section}: confound adjustment by covariate\n"
        f"ρ(nuclear_density, nc_ratio) = {section_result['rho_nuclear_density_vs_nc_ratio']:+.3f}; "
        "dashed = ±0.1 survival threshold"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"check_c_covariate_{section}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Verdicts ──────────────────────────────────────────────────────────────────

def build_verdicts(check_a: dict, check_c: dict) -> dict:
    """Three plain verdicts. Verdicts 1 and 2 are NULL-REFERENCED: the question is
    not "did the correlation move" (any reseeding moves it) but "did it move
    outside the range an arbitrary root set produces"."""

    def _null_rows(feature):
        rows = []
        for section, res in check_a.items():
            nul = (res.get("root_nulls") or {}).get("uniform")
            if not nul:
                continue
            fs = nul["feature_rho_null"].get(feature)
            if fs and fs.get("original_inside_null_range") is not None:
                rows.append((section, fs))
        return rows

    # ── 1. nuclear_density ────────────────────────────────────────────────────
    nd_rows = _null_rows("nuclear_density")
    if nd_rows:
        outside = [s for s, fs in nd_rows if not fs["original_inside_null_range"]]
        detail = "; ".join(
            f"{s}: original {_fmt(fs['original_rho'])} vs random-root null "
            f"[{_fmt(fs['null_min'])}, {_fmt(fs['null_max'])}] → "
            f"{'OUTSIDE' if not fs['original_inside_null_range'] else 'inside'}"
            for s, fs in nd_rows
        )
        v1 = (
            f"{'NOT ROBUST' if outside else 'ROBUST'} — {detail}. "
            + ("The nuclear_density correlation lies within the range produced by "
               "arbitrary root sets, so it is NOT attributable to the "
               "lowest-nuclear-density root rule and the circularity concern does "
               "not undermine it."
               if not outside else
               f"In {outside} the original value falls outside what arbitrary root "
               "sets produce, so the root rule does contribute to it and it must "
               "not be presented as independent validation.")
        )
    else:
        v1 = ("UNDETERMINED — no random-root null was computed (--n-random-draws 0), "
              "so there is no reference range and the DC1 comparison alone cannot "
              "separate a fragile correlation from an atypical root set.")

    # ── 2. the other five features ────────────────────────────────────────────
    others = [f for f in MORPH_FEATURES if f != "nuclear_density"]
    inside_n, total_n, outside_list = 0, 0, []
    for feat in others:
        for section, fs in _null_rows(feat):
            total_n += 1
            if fs["original_inside_null_range"]:
                inside_n += 1
            else:
                outside_list.append(
                    f"{section}/{feat} (orig {_fmt(fs['original_rho'])}, null "
                    f"[{_fmt(fs['null_min'])}, {_fmt(fs['null_max'])}])"
                )
    if total_n:
        v2 = (
            f"{inside_n} of {total_n} (section, feature) pairs sit INSIDE the "
            "random-root null range and are therefore not attributable to the root "
            "rule. "
            + ("All five features in both sections are root-insensitive."
               if not outside_list else
               f"Outside the null range: {'; '.join(outside_list)} — the root rule "
               "contributes to these and they need the caveat.")
        )
    else:
        v2 = "UNDETERMINED — no random-root null was computed."

    # DC1 tails, reported separately with their provenance caveat.
    dc_bits = []
    for section, res in check_a.items():
        nul = (res.get("root_nulls") or {}).get("uniform")
        floor = _fmt(nul["min_abs_rho"]) if nul else "n/a"
        for tail in ("low", "high"):
            t = res["tails"][tail]
            pv = t.get("root_provenance", {})
            dom = pv.get("single_slide_dominated")
            dc_bits.append(
                f"{section}/DC1-{tail}: rho {_fmt(t['rho_original_vs_reseeded_pseudotime'])}"
                + (f", {pv.get('max_share_from_one_slide', 0):.0%} from one slide"
                   if dom else "")
            )
    v2 += (
        " DC1-seeded tails, for reference: " + "; ".join(dc_bits)
        + ". A DC1 tail falling BELOW the random-root floor is an atypical, "
        "single-slide-dominated root set rather than evidence of a fragile axis — "
        "check root_provenance before reading any DC1 tail as a robustness result."
    )

    # ── 3. confound covariate (unchanged logic) ───────────────────────────────
    c_bits, any_change = [], False
    for section, res in check_c.items():
        nd_surv = res["control_nuclear_density"]["summary"]["survivors"]
        nc_surv = res["control_nc_ratio"]["summary"]["survivors"]
        changes = res["verdict_changes"]
        if changes:
            any_change = True
        c_bits.append(
            f"{section}: survivors under nuclear_density = {nd_surv or 'none'}; "
            f"under nc_ratio = {nc_surv or 'none'}; verdict changes on "
            f"{changes or 'no feature'} "
            f"(rho(nd, nc_ratio) = {_fmt(res['rho_nuclear_density_vs_nc_ratio'])})"
        )
    v3 = (
        f"{'DEPENDS ON COVARIATE CHOICE' if any_change else 'HOLDS under the alternative covariate'} — "
        + " | ".join(c_bits)
        + ". Because nuclear_density and nc_ratio are themselves correlated (see "
        "per-section value above), this is a robustness check, not an independent test."
    )

    # ── 4. what the axis actually is ──────────────────────────────────────────
    v4 = " | ".join(
        f"{section}: rho(PT, DC1) = {_fmt(res['axis_identity']['rho_pt_vs_dc1'])}, "
        f"rho(PT, centroid distance) = "
        f"{_fmt(res['axis_identity']['rho_pt_vs_diffmap_centroid_distance'])} — "
        f"{res['axis_identity']['verdict']}"
        for section, res in check_a.items() if "axis_identity" in res
    ) or "UNDETERMINED — axis identity diagnostic not computed."

    return {
        "1_nuclear_density_robust_to_root_choice": v1,
        "2_other_five_features_robust_to_root_choice": v2,
        "3_cellularity_confound_holds_under_alternative_covariate": v3,
        "4_what_the_pseudotime_axis_geometrically_is": v4,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(output_dir: Path, check_a: dict, check_c: dict, verdicts: dict) -> None:
    L = ["# Root-selection sensitivity — Checks A and C", ""]

    L.append(
        "**Why this analysis exists.** `analysis/diffusion.py:compute_dpt_multi_root()` "
        "picks its 20 DPT roots as `np.argsort(nuclear_density)[:20]` — the 20 patches "
        "with the lowest nuclear density, from `compute_nuclear_density_quick()`. "
        "`nuclear_density` is simultaneously one of the six morphological features the "
        "resulting pseudotime is validated against, and the covariate partialled out in "
        "the cellularity confound analysis. The pseudotime axis is therefore partly "
        "defined by a quantity it is later validated against, and partly by the quantity "
        "used to adjust that validation, so neither the reported nuclear_density "
        "correlation nor the \"collapses under cellularity adjustment\" finding can be "
        "read as independent evidence until the circularity is quantified. Check A "
        "re-seeds the trajectory from manifold geometry instead of image content, "
        "changing the root rule and nothing else, and asks how much the pseudotime "
        "ordering and its six feature correlations actually move. Check C leaves the "
        "existing pseudotime completely untouched and swaps the confound covariate for "
        "one not entangled with root selection. Existing per-section runs are read-only "
        "inputs throughout; no features were re-extracted and no existing results "
        "directory was written to."
    )
    L.append("")

    # ── Check A ───────────────────────────────────────────────────────────────
    L += ["## Check A — geometry-seeded roots (primary)", ""]
    L.append(
        "Roots are the 20 patches at an extreme of the first non-trivial diffusion "
        "component (DC1) of the **already-computed** diffusion map. The PCA embedding "
        "(`adata.X`), the neighbour graph (`adata.obsp`, Leiden k=15 cosine / diffusion "
        "k=30 euclidean as built by the production run), the diffusion map "
        "(`adata.obsm['X_diffmap']`) and its eigenvalues are all reused verbatim from "
        "`adata_full.h5ad` — only `sc.tl.dpt` re-runs. Root count (20), per-root "
        "inf-clamping, median-across-roots aggregation and min-max normalisation are "
        "identical to the production path."
    )
    L.append("")

    L += ["### Headline: original vs geometry-seeded pseudotime", ""]
    L.append(
        "This is the single most informative number. Near 1.0 means root choice barely "
        "matters and the circularity concern is closed; divergence bounds how much root "
        "selection drives the reported results."
    )
    L.append("")
    L.append("| section | tail | direction-matched? | ρ(original PT, geometry-seeded PT) | roots: distinct slides | max single-slide share |")
    L.append("|---|---|---|---|---|---|")
    for section, res in check_a.items():
        for tail in ("low", "high"):
            t = res["tails"][tail]
            mark = "**yes**" if tail == res["direction_matched_tail"] else "no"
            pv = t.get("root_provenance", {})
            nsl = pv.get("n_distinct_slides")
            shr = pv.get("max_share_from_one_slide")
            warn = " ⚠︎" if pv.get("single_slide_dominated") else ""
            L.append(
                f"| {section} | DC1 {tail} | {mark} | "
                f"**{_fmt(t['rho_original_vs_reseeded_pseudotime'])}** | "
                f"{nsl if nsl is not None else 'n/a'}{warn} | "
                f"{f'{shr:.0%}' if shr is not None else 'n/a'} |"
            )
    L.append("")
    L.append(
        "⚠︎ marks a tail whose roots are >=80% from a single slide. Such a tail is a "
        "local outlier lobe, not a manifold endpoint; anchoring DPT there produces an "
        "axis dominated by one slide and its correlations should not be read as a "
        "root-choice effect."
    )
    L.append("")
    L += ["### What the pseudotime axis geometrically is", ""]
    for section, res in check_a.items():
        ai = res.get("axis_identity")
        if not ai:
            continue
        L.append(
            f"- **{section}** — rho(PT, DC1) = {_fmt(ai['rho_pt_vs_dc1'])}, "
            f"rho(PT, |DC1 - median|) = {_fmt(ai['rho_pt_vs_abs_dc1_from_median'])}, "
            f"rho(PT, diffusion-map centroid distance) = "
            f"{_fmt(ai['rho_pt_vs_diffmap_centroid_distance'])}. {ai['verdict']}"
        )
    L.append("")

    L += ["### Feature correlations vs the random-root null (the attribution test)", ""]
    L.append(
        "The question is not whether a correlation moves under reseeding — any "
        "reseeding moves it — but whether the original value falls **outside** the "
        "range an arbitrary root set produces. Inside the range means the root rule "
        "does not account for it."
    )
    L.append("")
    L.append("| section | feature | original rho | random-root null range | inside? | attribution |")
    L.append("|---|---|---|---|---|---|")
    for section, res in check_a.items():
        nul = (res.get("root_nulls") or {}).get("uniform")
        if not nul:
            continue
        for feat in MORPH_FEATURES:
            fs = nul["feature_rho_null"].get(feat, {})
            if fs.get("original_inside_null_range") is None:
                continue
            L.append(
                f"| {section} | {feat} | {_fmt(fs['original_rho'])} | "
                f"[{_fmt(fs['null_min'])}, {_fmt(fs['null_max'])}] | "
                f"{'**yes**' if fs['original_inside_null_range'] else '**NO**'} | "
                f"{fs['attribution']} |"
            )
    L.append("")

    nulls = {sec: (r.get("root_nulls") or {}).get("uniform") for sec, r in check_a.items()}
    if any(nulls.values()):
        L.append("**Random-root null baseline.** Without a null there is no scale for the "
                 "numbers above: 0.52 means one thing if random roots give 0.1 and quite "
                 "another if they give 0.5.")
        L.append("")
        L.append("| section | draws | random |ρ| median | random |ρ| max | best DC1 tail |ρ| | DC1 exceeds random ceiling? |")
        L.append("|---|---|---|---|---|---|")
        for section, res in check_a.items():
            best = max(abs(res["tails"][t]["rho_original_vs_reseeded_pseudotime"])
                       for t in ("low", "high"))
            for mode, nl in (res.get("root_nulls") or {}).items():
                L.append(
                    f"| {section} ({mode}) | {nl['n_draws']} | "
                    f"{_fmt(nl['median_abs_rho'])} | {_fmt(nl['max_abs_rho'])} | "
                    f"{_fmt(best)} | "
                    f"{'**yes**' if best > nl['max_abs_rho'] else '**NO** — the DC1 tails are ATYPICAL root sets, not a fragile axis'} |"
                )
        L.append("")
        L.append(
            "Two null modes are reported. **uniform** spreads 20 roots at random; "
            "**clustered** takes the 20 nearest neighbours of a random patch in "
            "diffusion space, matching the tight local structure of the production "
            "root set (the 20 lowest-nuclear-density patches are a clustered set, "
            "not a spread one) while randomising its location. Comparing the two "
            "separates *where* the roots sit from *how clustered* they are."
        )
        L.append("")

    for section, res in check_a.items():
        L += [f"### {section}", ""]
        dc = res["diffusion_component"]
        L.append(f"- n patches: {res['n_patches']}; roots per run: {res['n_roots']}; "
                 f"permutations: {res['n_permutations']}")
        L.append(f"- {dc['dc_index_note']}")
        da = res.get("direction_agreement", {})
        L.append(f"- Tail chosen as direction-matched: **DC1 {res['direction_matched_tail']}**"
                 + ("" if da.get("criteria_agree", True) else
                    " — **the two orientation criteria DISAGREE; read both tails as co-primary**")
                 + f". {res['direction_basis']}")
        rec = res["original_correlation_reconciliation"]
        L.append(f"- Baseline reproducibility: **{rec['status']}** — {rec['note']}")
        L.append("")

        for tail in ("low", "high"):
            t = res["tails"][tail]
            primary = " (direction-matched)" if tail == res["direction_matched_tail"] else ""
            L.append(f"#### DC1 {tail} tail{primary}")
            L.append("")
            L.append(
                f"ρ(original PT, geometry-seeded PT) = **{_fmt(t['rho_original_vs_reseeded_pseudotime'])}** "
                f"(p = {_fmt(t['p_original_vs_reseeded_pseudotime'])}). "
                f"Non-finite DPT values clamped across all roots: "
                f"{t['dpt_notes']['n_nonfinite_clamped_across_all_roots']}."
            )
            L.append("")
            L.append("| feature | original ρ | geometry-seeded ρ | abs Δ | sign preserved | perm p (new) | robust? |")
            L.append("|---|---|---|---|---|---|---|")
            for feat in MORPH_FEATURES:
                r = t["per_feature"][feat]
                sign = "yes" if r["sign_preserved"] else ("no" if r["sign_preserved"] is False else "n/a")
                flag = " ⚠︎" if r["baseline_reproduced"] is False else ""
                L.append(
                    f"| {feat}{flag} | {_fmt(r['original_rho'])} | {_fmt(r['geometry_seeded_rho'])} | "
                    f"{_fmt(r['abs_delta'])} | {sign} | {_fmt(r['geometry_seeded_perm_p'])} | "
                    f"{'**ROBUST**' if r['robust_to_root_choice'] else 'not robust'} |"
                )
            L.append("")
            L.append(
                f"Robust = |Δρ| ≤ {ROBUST_DELTA_TOL} **and** sign preserved. "
                "⚠︎ marks a feature whose original ρ could not be reproduced from the "
                "stored artifacts — see the baseline reproducibility note above."
            )
            L.append("")

    # ── Check C ───────────────────────────────────────────────────────────────
    L += ["## Check C — alternative confound covariate (robustness)", ""]
    L.append(
        "The **existing, unchanged** per-section pseudotime is used throughout this "
        "check — nothing from Check A feeds into it. Only the partialled-out covariate "
        "changes, from `nuclear_density` (which is also the root-selection criterion) to "
        "`nc_ratio` (which is not). Same algebraic partial-Spearman formula, same "
        "`default_rng(42)` seed, same 1000-shuffle null, same |partial ρ| ≥ 0.1 survival "
        "threshold as the production analysis."
    )
    L.append("")

    for section, res in check_c.items():
        L += [f"### {section}", ""]
        L.append(f"- **ρ(nuclear_density, nc_ratio) = {_fmt(res['rho_nuclear_density_vs_nc_ratio'])}** "
                 f"in this section.")
        L.append(f"- {res['covariate_independence_caveat']}")
        par = res["reproduction_of_saved_confound"]
        L.append(f"- Parity with the saved analysis: **{par['status']}** — {par['note']}")
        L.append(f"- {res['asymmetry_note']}")
        L.append("")
        L.append("| feature | raw ρ | partial ρ \\| nuclear_density | verdict | partial ρ \\| nc_ratio | verdict | changed? |")
        L.append("|---|---|---|---|---|---|---|")
        for feat in res["comparable_features"]:
            c = res["comparison"][feat]
            L.append(
                f"| {feat} | {_fmt(c['raw_rho'])} | {_fmt(c['partial_rho_ctrl_nuclear_density'])} | "
                f"{c['status_ctrl_nuclear_density']} | {_fmt(c['partial_rho_ctrl_nc_ratio'])} | "
                f"{c['status_ctrl_nc_ratio']} | {'**YES**' if c['verdict_changed'] else 'no'} |"
            )
        L.append("")
        nd_only = [f for f in res["control_nuclear_density"]["features"] if f not in res["comparable_features"]]
        nc_only = [f for f in res["control_nc_ratio"]["features"] if f not in res["comparable_features"]]
        for feat in nd_only:
            v = res["control_nuclear_density"]["features"][feat]
            L.append(f"- `{feat}` (tested only under nuclear_density control, no nc_ratio counterpart): "
                     f"raw ρ = {_fmt(v['raw_rho'])}, partial ρ = {_fmt(v['partial_rho'])} → {v['status']}")
        for feat in nc_only:
            v = res["control_nc_ratio"]["features"][feat]
            L.append(f"- `{feat}` (tested only under nc_ratio control, no nuclear_density counterpart): "
                     f"raw ρ = {_fmt(v['raw_rho'])}, partial ρ = {_fmt(v['partial_rho'])} → {v['status']}")
        L.append("")

    # ── Verdicts ──────────────────────────────────────────────────────────────
    L += ["## Verdicts", ""]
    L.append(f"**1. Is the nuclear_density correlation robust to root choice?**  \n{verdicts['1_nuclear_density_robust_to_root_choice']}")
    L.append("")
    L.append(f"**2. Are the other five features robust to root choice?**  \n{verdicts['2_other_five_features_robust_to_root_choice']}")
    L.append("")
    L.append(f"**3. Does the cellularity confound finding hold under an alternative covariate?**  \n{verdicts['3_cellularity_confound_holds_under_alternative_covariate']}")
    L.append("")
    L.append(f"**4. What is the pseudotime axis, geometrically?**  \n{verdicts.get('4_what_the_pseudotime_axis_geometrically_is', 'n/a')}")
    L.append("")

    L += ["## Limitations", ""]
    L.append(
        "- Check A changes the root rule but cannot make pseudotime independent of "
        "cellularity in general: the diffusion geometry itself was built from Phikon "
        "features of the same images, and DC1 may correlate with density for reasons "
        "that have nothing to do with the root rule. Check A bounds the contribution of "
        "**root selection specifically**, not of cellularity as a whole."
    )
    L.append(
        "- `nc_ratio` and `nuclear_density` are correlated (per-section value reported "
        "above), so Check C is a robustness check on covariate choice, not an "
        "independent test of the confound."
    )
    L.append(
        "- Each covariate is excluded from its own test set, so the nuclear_density and "
        "nc_ratio analyses cover overlapping but non-identical feature sets."
    )
    L.append(
        "- The DC1 tail designated direction-matched is labelled using mean "
        "nuclear_density of the root patches. Both tails are reported, so this labelling "
        "affects presentation and the verdict summary, not what was computed."
    )
    L.append(
        "- Permutation p-values are computed per feature with no multiplicity correction, "
        "matching the production validation suite."
    )
    L.append("")

    (output_dir / "root_sensitivity_report.md").write_text("\n".join(L), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Root-selection sensitivity: geometry-seeded DPT (Check A) and "
                    "alternative confound covariate (Check C)."
    )
    parser.add_argument("--sections", nargs="+", required=True,
                        help="Section labels, e.g. 2M-1 2M-2")
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True,
                        help="Existing per-section run dirs, in the same order as --sections. "
                             "READ-ONLY — never written to.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="NEW output directory for the report/JSON/figures.")
    parser.add_argument("--n-roots", type=int, default=20,
                        help="Root count; must match the production run (default: 20)")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Permutation shuffles (default: 1000, matching production)")
    parser.add_argument("--no-fast-null", action="store_true",
                        help="Force the per-root adata.copy() path for null draws. "
                             "By default the null reuses one AnnData, which is "
                             "verified identical on real data before use and is what "
                             "makes a null of useful size affordable. Primary results "
                             "always use the copy path regardless of this flag.")
    parser.add_argument("--n-random-draws", type=int, default=25,
                        help="Random-root null draws per section (default: 5). Each "
                             "draw seeds 20 uniformly-random roots and reports "
                             "rho vs the original pseudotime, calibrating what the "
                             "DC1 numbers mean. 0 disables.")
    parser.add_argument("--dc-index", type=int, default=None,
                        help="Force the X_diffmap column used as DC1, bypassing "
                             "trivial-eigenvector detection. Leave unset (default) "
                             "to auto-detect; the module refuses to seed from a "
                             "column it identifies as the steady-state eigenvector.")
    parser.add_argument("--skip-check-a", action="store_true")
    parser.add_argument("--skip-check-c", action="store_true")
    args = parser.parse_args()

    if len(args.sections) != len(args.run_dirs):
        parser.error(
            f"--sections ({len(args.sections)}) and --run-dirs ({len(args.run_dirs)}) "
            "must have the same length and matching order."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Root-selection sensitivity — Checks A and C")
    print("=" * 64)

    loaded = {}
    for section, run_dir in zip(args.sections, args.run_dirs):
        print(f"\n  Loading {section} from {run_dir} ...")
        adata, validation, confound = load_run(Path(run_dir))
        missing = check_required_columns(adata)
        if missing:
            raise KeyError(
                f"{run_dir}/adata_full.h5ad is missing obs columns {missing}. "
                "This analysis reuses stored morphological features and refuses to "
                "recompute them, which would not be comparable to the saved values."
            )
        print(f"    {adata.n_obs} patches, obs columns present.")
        loaded[section] = (adata, validation, confound)

    check_a, check_c = {}, {}

    if not args.skip_check_a:
        print("\n" + "=" * 64)
        print("  CHECK A — geometry-seeded roots")
        print("=" * 64)
        for section, (adata, validation, _) in loaded.items():
            res = run_check_a_for_section(
                section, adata, validation, args.n_roots, args.n_permutations,
                args.dc_index, args.n_random_draws, not args.no_fast_null,
            )
            res["_original_pt"] = adata.obs["pseudotime"].values.astype(float)
            check_a[section] = res
            write_check_a_figure(section, res, args.output_dir)

    if not args.skip_check_c:
        print("\n" + "=" * 64)
        print("  CHECK C — alternative confound covariate")
        print("=" * 64)
        for section, (adata, _, confound) in loaded.items():
            res = run_check_c_for_section(section, adata, confound, args.n_permutations)
            check_c[section] = res
            write_check_c_figure(section, res, args.output_dir)

    verdicts = build_verdicts(check_a, check_c)

    # Persist the reseeded pseudotime vectors as .npy, then strip the arrays out
    # of the JSON payload.
    for section, res in check_a.items():
        res.pop("_original_pt", None)
        for tail, t in res["tails"].items():
            pt = t.pop("pseudotime", None)
            pt_std = t.pop("pseudotime_std", None)
            if pt is not None:
                np.save(args.output_dir / f"pseudotime_geometry_seeded_{section}_{tail}.npy", pt)
            if pt_std is not None:
                np.save(args.output_dir / f"pseudotime_std_geometry_seeded_{section}_{tail}.npy", pt_std)

    payload = {
        "analysis": "root_sensitivity",
        "why": (
            "compute_dpt_multi_root selects roots as the 20 lowest-nuclear_density "
            "patches; nuclear_density is also a validation feature and the confound "
            "covariate. These checks quantify that circularity."
        ),
        "parameters": {
            "n_roots": args.n_roots,
            "n_permutations": args.n_permutations,
            "dc_index_override": args.dc_index,
            "n_random_draws": args.n_random_draws,
            "fast_null_path": not args.no_fast_null,
            "survive_threshold": SURVIVE_THRESHOLD,
            "robust_delta_tolerance": ROBUST_DELTA_TOL,
            "reproduction_tolerance": REPRO_TOL,
        },
        "check_a_geometry_seeded_roots": check_a,
        "check_c_alternative_covariate": check_c,
        "verdicts": verdicts,
    }
    json_path = args.output_dir / "root_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"\n  JSON: {json_path}")

    write_report(args.output_dir, check_a, check_c, verdicts)
    print(f"  Markdown: {args.output_dir / 'root_sensitivity_report.md'}")

    print("\n" + "=" * 64)
    print("  VERDICTS")
    print("=" * 64)
    for k, v in verdicts.items():
        print(f"\n  [{k}]\n  {v}")


if __name__ == "__main__":
    main()
