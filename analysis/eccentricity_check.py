"""
Is MorphPT's pseudotime a trajectory coordinate, or an eccentricity measure?

WHY THIS EXISTS
---------------
`analysis/root_sensitivity.py` found that the per-section pseudotime correlates
with distance from the diffusion-map centroid at rho = 0.808 (2M-1) / 0.802
(2M-2), far more strongly than with DC1 (0.543 / 0.467), and that 25 uniformly
random root sets reproduce the production pseudotime at |rho| 0.78-0.89. Together
those say the axis is fixed by the manifold rather than by the root rule — and
raise the possibility that what it measures is not "how far along" a patch is but
"how unusual" it is.

That distinction matters more than any root question. An eccentricity measure is
DIRECTIONLESS: 'early' means typical morphology, 'late' means atypical in ANY
direction, so two patches at opposite morphological extremes both score late. A
directed-trajectory reading would not survive it.

THE TAUTOLOGY THIS MODULE IS BUILT AROUND
-----------------------------------------
DPT pseudotime IS a diffusion distance from its roots. With 20 roots
median-aggregated, "distance from the roots" already approximates "distance from
a central location in diffusion space". So rho(PT, diffusion-map centroid
distance) ~= 0.8 is PARTLY TRUE BY CONSTRUCTION and cannot carry the claim.

The informative tests are in spaces DPT is NOT defined in terms of:

  1. PCA space — the representation the diffusion map was built FROM. Eccentricity
     here is not implied by DPT's definition.
  2. Morphological-feature space — the six interpretable descriptors. This is the
     one that matters for the paper: does 'late' mean 'morphologically extreme in
     a consistent direction' (trajectory) or 'extreme in any direction'
     (eccentricity)?

The diffusion-space number is still computed and reported, but explicitly labelled
as partly definitional so it is not read as independent evidence.

WHAT IT DOES
------------
  TASK A — which geometry is the pseudotime?
    Correlates pseudotime against distance-from-centroid in diffusion space
    (definitional), PCA space (informative), and morphological-feature space
    (decisive), plus DC1, a DC1 eccentricity term, and local graph sparsity.
    The key contrast is rho(PT, mean |z|) vs rho(PT, mean signed z) over the six
    features: eccentricity in morphology means the UNSIGNED deviation tracks
    pseudotime while the SIGNED one does not. Repeats the headline within each
    slide so the result cannot be a between-slide batch effect.

  TASK B — is late pseudotime heterogeneous, and in one direction or many?
    Heterogeneity alone does not separate eccentricity from a genuinely BRANCHING
    trajectory — both produce diverse late states. What separates them is
    DIRECTION: a branching trajectory moves away from early consistently within a
    branch, whereas eccentricity puts patches at opposite extremes of the SAME
    feature. So the primary statistic is bidirectional enrichment: among top-decile
    pseudotime patches, are BOTH the top and bottom deciles of a feature enriched?
    Also reports per-decile dispersion, a subclustering of the late patches with
    their signed feature profiles, and the slide composition of each end.

CONSTRAINTS HONOURED
--------------------
NEW module. Imports from `analysis/root_sensitivity.py` and `analysis/clustering.py`
but modifies neither, nor any other existing module. Reads only saved artifacts;
never re-extracts features, re-runs PCA/Leiden/DPT, or writes into an existing
results directory.

CLI
---
  python -m cancer_trajectory_atlas.analysis.eccentricity_check \\
      --sections   2M-1 2M-2 \\
      --run-dirs   $SCRATCH/results/per_section/atlas_2M-1 \\
                   $SCRATCH/results/per_section/atlas_2M-2 \\
      --output-dir $SCRATCH/results/eccentricity
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
from .root_sensitivity import (
    MORPH_FEATURES,
    _fmt,
    _json_default,
    _safe_rho,
    pick_diffusion_component,
    root_provenance,
)

N_BINS_DEFAULT = 10
EXTREME_DECILE = 0.10          # tail fraction used for bidirectional enrichment
ENRICH_THRESHOLD = 1.5         # fold-enrichment counted as "enriched"
ECC_STRONG = 0.50              # |rho| for a convincing eccentricity signal
DIRECTIONAL_GAP = 0.20         # |rho(|z|)| - |rho(signed z)| that implies directionless


# ── Loading ───────────────────────────────────────────────────────────────────

def load_section(run_dir: Path):
    """Load one per-section run's AnnData. Read-only."""
    import anndata as ad

    h5ad = run_dir / "adata_full.h5ad"
    if not h5ad.exists():
        raise FileNotFoundError(
            f"{h5ad} not found — this analysis reuses an existing per-section run "
            "and cannot regenerate it."
        )
    adata = ad.read_h5ad(h5ad)
    missing = [c for c in MORPH_FEATURES + ["pseudotime"] if c not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"{h5ad} is missing obs columns {missing}. This analysis reuses stored "
            "morphological features and refuses to recompute them."
        )
    return adata


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _centroid_distance(X: np.ndarray) -> np.ndarray:
    """Euclidean distance of each row from the column-wise mean."""
    X = np.asarray(X, dtype=float)
    return np.linalg.norm(X - X.mean(axis=0), axis=1)


def _median_centroid_distance(X: np.ndarray) -> np.ndarray:
    """Distance from the column-wise MEDIAN — robust to the outlying lobes that
    root_sensitivity found dominating the DC1 tails, which can drag a mean."""
    X = np.asarray(X, dtype=float)
    return np.linalg.norm(X - np.median(X, axis=0), axis=1)


def _mean_knn_distance(adata) -> np.ndarray:
    """Mean distance to stored k-NN neighbours = a local sparsity measure.

    Eccentric points sit in sparse regions, so this is a second, graph-based
    reading of the same idea. Uses the neighbour graph the pipeline already built;
    nothing is recomputed.
    """
    if "distances" not in adata.obsp:
        return np.full(adata.n_obs, np.nan)
    D = adata.obsp["distances"].tocsr()
    sums = np.asarray(D.sum(axis=1)).ravel()
    counts = np.diff(D.indptr)
    out = np.full(adata.n_obs, np.nan, dtype=float)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    return out


def _feature_z(obs) -> np.ndarray:
    """(N, 6) z-scored morphological features, using median/IQR-free standardisation
    on finite values only. Non-finite entries become 0 (no deviation)."""
    Z = np.zeros((len(obs), len(MORPH_FEATURES)), dtype=float)
    for j, feat in enumerate(MORPH_FEATURES):
        v = obs[feat].values.astype(float)
        finite = np.isfinite(v)
        if finite.sum() < 2:
            continue
        mu, sd = v[finite].mean(), v[finite].std()
        if sd <= 0:
            continue
        z = np.zeros_like(v)
        z[finite] = (v[finite] - mu) / sd
        Z[:, j] = z
    return Z


def check_production_root_provenance(section: str, adata, n_roots: int = 20) -> dict:
    """Which slides do the ACTUAL production roots come from?

    root_sensitivity reported provenance for the DC1 tails and all 50 null draws
    but never for the root set that produced the pseudotime being analysed. That
    matters: tightly-clustered root sets land inside a single slide as a matter of
    course (2M-1 clustered draws 1/18/24 were 100% one slide; both DC1 tails were
    85-95%), and the production rule — the 20 LOWEST-nuclear_density patches — is
    exactly such a clustered set. If those roots sit in one slide's lobe, the
    pseudotime is anchored to a batch direction.

    Reconstructible with no DPT: obs['nuclear_density'] is produced by the same
    code path as compute_nuclear_density_quick (both _deconvolve_hematoxylin ->
    _segment_nuclei_simple -> compute_nuclear_density, since use_stardist=False),
    so argsort(...)[:n] recovers the production root set.

    Self-verifying: DPT pseudotime is zero AT its roots, so if the reconstruction
    is correct the recovered roots must sit at pseudotime ~= 0. That check is
    reported, and a failure means the reconstruction is wrong and the provenance
    below must not be trusted.
    """
    obs = adata.obs
    nd = obs["nuclear_density"].values.astype(float)
    pt = obs["pseudotime"].values.astype(float)

    order = np.argsort(nd)
    roots = [int(i) for i in order[:n_roots]]

    prov = root_provenance(adata, roots)

    root_pt = pt[roots]
    pt_rank = np.array([float((pt < v).mean()) for v in root_pt])
    reconstruction_ok = bool(np.median(root_pt) <= 0.05)

    ties = int((nd <= nd[roots].max()).sum())

    return {
        "section": section,
        "n_roots": n_roots,
        "root_indices": roots,
        "root_nuclear_density_min": float(nd[roots].min()),
        "root_nuclear_density_max": float(nd[roots].max()),
        "n_patches_at_or_below_root_threshold": ties,
        "tie_inflation": ties - n_roots,
        "root_pseudotime_median": float(np.median(root_pt)),
        "root_pseudotime_max": float(root_pt.max()),
        "root_pseudotime_percentile_median": float(np.median(pt_rank)),
        "reconstruction_verified": reconstruction_ok,
        "reconstruction_note": (
            "VERIFIED — the reconstructed roots sit at pseudotime ~0, as DPT "
            "requires of its own roots, so this is the production root set."
            if reconstruction_ok else
            "NOT VERIFIED — the reconstructed roots do NOT sit at pseudotime ~0 "
            f"(median {float(np.median(root_pt)):.4f}). The reconstruction does not "
            "match the roots DPT actually used; treat the provenance below as "
            "unreliable and do not report it."
        ),
        "provenance": prov,
        "verdict": (
            ("SINGLE-SLIDE DOMINATED — "
             f"{prov.get('max_share_from_one_slide', float('nan')):.0%} of the "
             f"{n_roots} production roots come from one slide "
             f"({prov.get('n_distinct_slides')} slides total). The pseudotime "
             "origin is a single slide's local region, so 'early' is defined by "
             "that slide and the axis carries a batch direction. This is a "
             "confound in the production pipeline, not just in the re-analysis."
             if prov.get("single_slide_dominated") else
             f"SPREAD ACROSS SLIDES — {prov.get('n_distinct_slides')} slides among "
             f"the {n_roots} production roots, largest share "
             f"{prov.get('max_share_from_one_slide', float('nan')):.0%}. The origin "
             "is not anchored to one slide, so the axis does not inherit a "
             "single-slide batch direction from its roots.")
            if reconstruction_ok else
            "UNDETERMINED — reconstruction not verified; see reconstruction_note."
        ),
    }


# ── Task A: which geometry is the pseudotime? ─────────────────────────────────

def run_task_a(section: str, adata) -> dict:
    """Correlate pseudotime with eccentricity in three spaces + graph sparsity."""
    obs = adata.obs
    pt = obs["pseudotime"].values.astype(float)

    X_pca = np.asarray(adata.X, dtype=float)
    dc_info = pick_diffusion_component(adata)
    dm = np.asarray(adata.obsm["X_diffmap"], dtype=float)
    trivial = set(dc_info["trivial_columns_detected"])
    nontrivial = [i for i in range(dm.shape[1]) if i not in trivial]
    dm_sub = dm[:, nontrivial]
    dc1 = dm[:, dc_info["dc_index_used"]]

    Z = _feature_z(obs)
    mean_abs_z = np.abs(Z).mean(axis=1)
    mean_signed_z = Z.mean(axis=1)

    n_pcs_match = min(dm_sub.shape[1], X_pca.shape[1])

    measures = {
        "diffmap_centroid_distance": {
            "rho": _safe_rho(pt, _centroid_distance(dm_sub)),
            "space": "diffusion map (non-trivial components)",
            "status": "DEFINITIONAL",
            "note": (
                "DPT pseudotime IS a diffusion distance from its roots, and 20 "
                "median-aggregated roots approximate a central location, so this "
                "correlation is PARTLY TRUE BY CONSTRUCTION. Reported for continuity "
                "with root_sensitivity; it is not evidence."
            ),
        },
        "diffmap_median_centroid_distance": {
            "rho": _safe_rho(pt, _median_centroid_distance(dm_sub)),
            "space": "diffusion map (non-trivial components), median centre",
            "status": "DEFINITIONAL",
            "note": "As above, with a median centre — robust to outlying lobes.",
        },
        "pca_centroid_distance": {
            "rho": _safe_rho(pt, _centroid_distance(X_pca)),
            "space": f"PCA space (all {X_pca.shape[1]} components)",
            "status": "INFORMATIVE",
            "note": (
                "PCA space is what the diffusion map was built FROM. DPT is not "
                "defined in terms of PCA distance, so eccentricity here is not "
                "implied by DPT's construction."
            ),
        },
        "pca_centroid_distance_matched_dims": {
            "rho": _safe_rho(pt, _centroid_distance(X_pca[:, :n_pcs_match])),
            "space": f"PCA space (first {n_pcs_match} components)",
            "status": "INFORMATIVE",
            "note": (
                "Dimension-matched to the diffusion map. Distance-from-centroid "
                "grows with dimensionality, so comparing the all-component PCA "
                "figure against a 9-component diffmap figure would be unfair."
            ),
        },
        "morph_mean_abs_z": {
            "rho": _safe_rho(pt, mean_abs_z),
            "space": "morphological features, mean |z| across the six",
            "status": "DECISIVE",
            "note": (
                "UNSIGNED deviation from typical morphology. If this tracks "
                "pseudotime, 'late' means morphologically extreme."
            ),
        },
        "morph_mean_signed_z": {
            "rho": _safe_rho(pt, mean_signed_z),
            "space": "morphological features, mean signed z across the six",
            "status": "DECISIVE",
            "note": (
                "SIGNED deviation. If the unsigned term tracks pseudotime and this "
                "one does not, late patches are extreme in inconsistent directions "
                "— the eccentricity signature."
            ),
        },
        "dc1": {
            "rho": _safe_rho(pt, dc1),
            "space": "first non-trivial diffusion component",
            "status": "REFERENCE",
            "note": "Position along the leading manifold axis, for contrast.",
        },
        "dc1_eccentricity": {
            "rho": _safe_rho(pt, np.abs(dc1 - np.median(dc1))),
            "space": "|DC1 - median(DC1)|",
            "status": "REFERENCE",
            "note": "Eccentricity along DC1 alone.",
        },
        "mean_knn_distance": {
            "rho": _safe_rho(pt, _mean_knn_distance(adata)),
            "space": "mean distance to stored k-NN neighbours",
            "status": "INFORMATIVE",
            "note": (
                "Local sparsity from the pipeline's own neighbour graph. Eccentric "
                "points sit in sparse regions."
            ),
        },
    }

    # Within-slide: rules out a purely between-slide (batch) explanation.
    within_slide = {}
    if "slide_id" in obs.columns:
        slides = obs["slide_id"].astype(str).values
        d_pca = _centroid_distance(X_pca)
        for key, vec in (("pca_centroid_distance", d_pca), ("morph_mean_abs_z", mean_abs_z)):
            per = {}
            for sl in sorted(set(slides)):
                m = slides == sl
                if m.sum() >= 30:
                    per[sl] = _safe_rho(pt[m], vec[m])
            vals = [v for v in per.values() if np.isfinite(v)]
            within_slide[key] = {
                "per_slide": per,
                "median_within_slide_rho": float(np.median(vals)) if vals else float("nan"),
                "n_slides": len(vals),
                "note": (
                    "Recomputed inside each slide. A within-slide correlation close "
                    "to the cohort-wide one means the relationship is not a "
                    "between-slide batch effect."
                ),
            }

    r_abs = measures["morph_mean_abs_z"]["rho"]
    r_sgn = measures["morph_mean_signed_z"]["rho"]
    r_pca = measures["pca_centroid_distance_matched_dims"]["rho"]
    gap = abs(r_abs) - abs(r_sgn)

    if abs(r_abs) >= ECC_STRONG and gap >= DIRECTIONAL_GAP:
        verdict = (
            f"ECCENTRICITY IN MORPHOLOGY — unsigned deviation tracks pseudotime "
            f"(rho = {r_abs:+.3f}) while signed deviation does not "
            f"(rho = {r_sgn:+.3f}). Late patches are morphologically extreme in "
            "INCONSISTENT directions, which is not a progression."
        )
    elif abs(r_sgn) >= ECC_STRONG and gap <= -DIRECTIONAL_GAP:
        verdict = (
            f"DIRECTIONAL IN MORPHOLOGY — signed deviation tracks pseudotime "
            f"(rho = {r_sgn:+.3f}) more strongly than unsigned "
            f"(rho = {r_abs:+.3f}). Consistent with a directed trajectory."
        )
    elif abs(r_pca) >= ECC_STRONG:
        verdict = (
            f"ECCENTRICITY IN EMBEDDING ONLY — PCA-space eccentricity tracks "
            f"pseudotime (rho = {r_pca:+.3f}) but the morphological signal is "
            f"ambiguous (|z| {r_abs:+.3f} vs signed {r_sgn:+.3f}). The axis is "
            "radial in feature-embedding space without a clean interpretation in "
            "the six descriptors."
        )
    else:
        verdict = (
            f"NOT ECCENTRICITY-DOMINANT outside the diffusion map — PCA-space "
            f"rho = {r_pca:+.3f}, morphology |z| rho = {r_abs:+.3f}. The strong "
            "diffusion-space figure is likely definitional rather than substantive."
        )

    return {
        "section": section,
        "n_patches": int(adata.n_obs),
        "diffusion_component": dc_info,
        "measures": measures,
        "within_slide": within_slide,
        "verdict": verdict,
        "framing": (
            "Read the DECISIVE rows first, then INFORMATIVE. The DEFINITIONAL rows "
            "restate DPT's own construction and must not be counted as evidence."
        ),
    }


# ── Task B: is late pseudotime heterogeneous, and in how many directions? ─────

def run_dispersion_by_bin(adata, n_bins: int) -> dict:
    """Per-pseudotime-bin dispersion in PCA space and in feature space."""
    obs = adata.obs
    pt = obs["pseudotime"].values.astype(float)
    X = np.asarray(adata.X, dtype=float)
    Z = _feature_z(obs)

    edges = np.quantile(pt, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bins = []
    for b in range(n_bins):
        m = (pt >= edges[b]) & (pt < edges[b + 1])
        if m.sum() < 5:
            continue
        Xb = X[m]
        bins.append({
            "bin": b,
            "pt_range": [float(edges[b]), float(edges[b + 1])],
            "n": int(m.sum()),
            "pca_mean_distance_to_bin_centroid": float(
                np.linalg.norm(Xb - Xb.mean(axis=0), axis=1).mean()
            ),
            "pca_total_variance": float(Xb.var(axis=0).sum()),
            "morph_mean_abs_z": float(np.abs(Z[m]).mean()),
            "morph_mean_signed_z": float(Z[m].mean()),
            "morph_feature_sd_mean": float(Z[m].std(axis=0).mean()),
        })

    idx = np.array([b["bin"] for b in bins], dtype=float)
    disp = np.array([b["pca_mean_distance_to_bin_centroid"] for b in bins])
    absz = np.array([b["morph_mean_abs_z"] for b in bins])
    return {
        "n_bins": len(bins),
        "bins": bins,
        "rho_bin_vs_pca_dispersion": _safe_rho(idx, disp),
        "rho_bin_vs_mean_abs_z": _safe_rho(idx, absz),
        "note": (
            "Dispersion rising with pseudotime is consistent with BOTH eccentricity "
            "and a branching trajectory — it does not separate them on its own. The "
            "bidirectional enrichment test below is what separates them."
        ),
    }


def run_bidirectional_enrichment(adata, tail: float) -> dict:
    """THE discriminating test.

    Among top-pseudotime patches, is a feature's HIGH extreme enriched, its LOW
    extreme enriched, or both? A branching trajectory moves away from early
    consistently within a branch, so it enriches one side. Eccentricity puts late
    patches at opposite extremes of the SAME feature, enriching both.
    """
    obs = adata.obs
    pt = obs["pseudotime"].values.astype(float)
    hi_pt = pt >= np.quantile(pt, 1 - tail)
    lo_pt = pt <= np.quantile(pt, tail)

    def _enrich(mask):
        out = {}
        for feat in MORPH_FEATURES:
            v = obs[feat].values.astype(float)
            finite = np.isfinite(v)
            if finite.sum() < 50:
                out[feat] = None
                continue
            hi_cut = np.quantile(v[finite], 1 - tail)
            lo_cut = np.quantile(v[finite], tail)
            sel = mask & finite
            if sel.sum() == 0:
                out[feat] = None
                continue
            p_hi = float((v[sel] >= hi_cut).mean())
            p_lo = float((v[sel] <= lo_cut).mean())
            e_hi, e_lo = p_hi / tail, p_lo / tail
            both = bool(e_hi >= ENRICH_THRESHOLD and e_lo >= ENRICH_THRESHOLD)
            one = bool((e_hi >= ENRICH_THRESHOLD) != (e_lo >= ENRICH_THRESHOLD))
            out[feat] = {
                "frac_in_feature_high_tail": p_hi,
                "frac_in_feature_low_tail": p_lo,
                "enrichment_high": e_hi,
                "enrichment_low": e_lo,
                "bidirectional": both,
                "unidirectional": one,
                "pattern": ("BIDIRECTIONAL — both extremes enriched (eccentricity)"
                            if both else
                            "unidirectional — one extreme enriched (trajectory-like)"
                            if one else
                            "neither extreme enriched"),
            }
        return out

    late, early = _enrich(hi_pt), _enrich(lo_pt)
    n_bi = sum(1 for v in late.values() if v and v["bidirectional"])
    n_uni = sum(1 for v in late.values() if v and v["unidirectional"])
    n_test = sum(1 for v in late.values() if v)

    return {
        "tail_fraction": tail,
        "n_late": int(hi_pt.sum()),
        "n_early": int(lo_pt.sum()),
        "late": late,
        "early": early,
        "n_features_bidirectional": n_bi,
        "n_features_unidirectional": n_uni,
        "n_features_tested": n_test,
        "enrichment_threshold": ENRICH_THRESHOLD,
        "verdict": (
            f"{n_bi} of {n_test} features show BOTH extremes enriched among "
            f"late-pseudotime patches, {n_uni} show one extreme only. "
            + ("Bidirectional enrichment dominates: late patches are extreme in "
               "opposite directions on the same feature, which a directed or "
               "branching trajectory does not produce. This is the eccentricity "
               "signature."
               if n_bi > n_uni else
               "Unidirectional enrichment dominates: late patches move consistently "
               "on each feature, which is what a directed or branching trajectory "
               "produces and eccentricity does not."
               if n_uni > n_bi else
               "Neither pattern dominates; this test is inconclusive here.")
        ),
        "note": (
            "Baseline expectation with no association is 1.0x enrichment on each "
            f"side (both tails are {tail:.0%} of the cohort by construction)."
        ),
    }


def run_late_subclustering(adata, tail: float, k_range: tuple) -> dict:
    """Subcluster the late patches and read their SIGNED feature profiles.

    Heterogeneity is expected either way. What discriminates is whether the
    subclusters sit on OPPOSITE sides of the cohort mean on the same feature.
    """
    from .clustering import cluster_kmeans

    obs = adata.obs
    pt = obs["pseudotime"].values.astype(float)
    mask = pt >= np.quantile(pt, 1 - tail)
    idx = np.where(mask)[0]
    X = np.asarray(adata.X, dtype=float)[idx]
    Z = _feature_z(obs)[idx]

    if len(idx) < max(k_range) * 10:
        return {"attempted": False,
                "reason": f"Only {len(idx)} late patches — too few for a k up to {max(k_range)}."}

    labels, best_k, scores = cluster_kmeans(X, k_range=k_range)

    slides = obs["slide_id"].astype(str).values[idx] if "slide_id" in obs.columns else None
    clusters = {}
    for c in sorted(set(labels.tolist())):
        m = labels == c
        prof = {MORPH_FEATURES[j]: float(Z[m, j].mean()) for j in range(len(MORPH_FEATURES))}
        entry = {"n": int(m.sum()), "mean_z_profile": prof}
        if slides is not None:
            uniq, cnt = np.unique(slides[m], return_counts=True)
            order = np.argsort(-cnt)
            entry["slide_breakdown"] = {str(uniq[i]): int(cnt[i]) for i in order}
            entry["max_share_from_one_slide"] = float(cnt.max() / m.sum())
        clusters[str(c)] = entry

    opposing = {}
    for feat in MORPH_FEATURES:
        vals = [clusters[c]["mean_z_profile"][feat] for c in clusters]
        opposing[feat] = {
            "min_mean_z": float(min(vals)),
            "max_mean_z": float(max(vals)),
            "spans_zero": bool(min(vals) < 0 < max(vals)),
            "opposing_magnitude": float(min(abs(min(vals)), abs(max(vals)))),
        }
    n_opposing = sum(1 for v in opposing.values()
                     if v["spans_zero"] and v["opposing_magnitude"] >= 0.25)

    return {
        "attempted": True,
        "n_late_patches": int(len(idx)),
        "best_k": int(best_k),
        "silhouette_scores": {str(k): float(v) for k, v in scores.items()},
        "clusters": clusters,
        "opposing_directions": opposing,
        "n_features_with_opposing_subclusters": n_opposing,
        "verdict": (
            f"{n_opposing} of {len(MORPH_FEATURES)} features have late subclusters "
            f"on OPPOSITE sides of the cohort mean (|mean z| >= 0.25 both ways). "
            + ("Late pseudotime is not one phenotype but several that differ in "
               "opposing directions — consistent with eccentricity, not with a "
               "single late state."
               if n_opposing >= 2 else
               "Late subclusters mostly deviate in the same direction per feature, "
               "which is compatible with a coherent (possibly branching) late state.")
        ),
    }


def run_task_b(section: str, adata, n_bins: int, tail: float, k_range: tuple) -> dict:
    dispersion = run_dispersion_by_bin(adata, n_bins)
    enrichment = run_bidirectional_enrichment(adata, tail)
    sub = run_late_subclustering(adata, tail, k_range)

    obs = adata.obs
    pt = obs["pseudotime"].values.astype(float)
    slide_composition = {}
    if "slide_id" in obs.columns:
        slides = obs["slide_id"].astype(str).values
        for name, m in (("late", pt >= np.quantile(pt, 1 - tail)),
                        ("early", pt <= np.quantile(pt, tail))):
            uniq, cnt = np.unique(slides[m], return_counts=True)
            order = np.argsort(-cnt)
            slide_composition[name] = {
                "breakdown": {str(uniq[i]): int(cnt[i]) for i in order},
                "max_share_from_one_slide": float(cnt.max() / m.sum()),
                "n_distinct_slides": int(len(uniq)),
            }
        slide_composition["note"] = (
            "root_sensitivity found the DC1 extremes in 2M-1 were 85-95% from a "
            "single slide. If one end of the pseudotime range is likewise "
            "slide-dominated, that end reflects a batch direction and not biology."
        )

    return {
        "section": section,
        "dispersion_by_bin": dispersion,
        "bidirectional_enrichment": enrichment,
        "late_subclustering": sub,
        "slide_composition": slide_composition,
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def write_figures(section: str, task_a: dict, task_b: dict, output_dir: Path) -> None:
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

        ax = axes[0]
        keys = ["diffmap_centroid_distance", "pca_centroid_distance_matched_dims",
                "morph_mean_abs_z", "morph_mean_signed_z", "dc1", "mean_knn_distance"]
        vals = [task_a["measures"][k]["rho"] for k in keys]
        cols = {"DEFINITIONAL": "#BBBBBB", "INFORMATIVE": "#4878CF",
                "DECISIVE": "#D65F5F", "REFERENCE": "#9C9C9C"}
        bar_cols = [cols[task_a["measures"][k]["status"]] for k in keys]
        ax.barh(range(len(keys)), vals, color=bar_cols)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([k.replace("_", "\n") for k in keys], fontsize=7)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Spearman rho with pseudotime")
        ax.set_title(f"{section}: which geometry is the axis?\n"
                     "grey = definitional, blue = informative, red = decisive",
                     fontsize=9)

        ax = axes[1]
        bins = task_b["dispersion_by_bin"]["bins"]
        b = [x["bin"] for x in bins]
        ax.plot(b, [x["pca_mean_distance_to_bin_centroid"] for x in bins],
                "o-", label="PCA dispersion", color="#4878CF")
        ax2 = ax.twinx()
        ax2.plot(b, [x["morph_mean_abs_z"] for x in bins], "s--",
                 label="mean |z| morphology", color="#D65F5F")
        ax.set_xlabel("Pseudotime decile")
        ax.set_ylabel("PCA mean distance to bin centroid", color="#4878CF", fontsize=8)
        ax2.set_ylabel("mean |z| of features", color="#D65F5F", fontsize=8)
        ax.set_title(f"{section}: dispersion across pseudotime\n"
                     "(rising is consistent with BOTH eccentricity and branching)",
                     fontsize=9)

        ax = axes[2]
        enr = task_b["bidirectional_enrichment"]["late"]
        feats = [f for f in MORPH_FEATURES if enr.get(f)]
        x = np.arange(len(feats))
        w = 0.38
        ax.bar(x - w / 2, [enr[f]["enrichment_high"] for f in feats], w,
               label="high tail", color="#6ACC65")
        ax.bar(x + w / 2, [enr[f]["enrichment_low"] for f in feats], w,
               label="low tail", color="#EE854A")
        ax.axhline(1.0, color="k", lw=0.6)
        ax.axhline(ENRICH_THRESHOLD, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_", "\n") for f in feats], fontsize=7)
        ax.set_ylabel("fold enrichment among late patches")
        ax.set_title(f"{section}: both tails enriched = eccentricity\n"
                     f"dashed = {ENRICH_THRESHOLD}x threshold", fontsize=9)
        ax.legend(fontsize=8)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"eccentricity_{section}.{ext}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  WARNING: could not write figures for {section}: {exc}")


# ── Verdicts + report ─────────────────────────────────────────────────────────

def build_verdicts(task_a: dict, task_b: dict) -> dict:
    v1 = " | ".join(
        f"{s}: {r['verdict']}" for s, r in task_a.items()
    )
    v2 = " | ".join(
        f"{s}: {r['bidirectional_enrichment']['verdict']}" for s, r in task_b.items()
    )
    v3 = " | ".join(
        f"{s}: {r['late_subclustering'].get('verdict', r['late_subclustering'].get('reason', 'n/a'))}"
        for s, r in task_b.items()
    )

    # Per-section conclusion first, THEN agreement across sections. Testing only
    # whether the eccentric-section list is non-empty would call a cohort
    # "eccentricity" when one section is eccentric and the other is directional,
    # which is the opposite of what disagreement means.
    per_section = {}
    for s in task_a:
        ecc = task_a[s]["verdict"].startswith("ECCENTRICITY")
        e = task_b[s]["bidirectional_enrichment"]
        bi = e["n_features_bidirectional"] > e["n_features_unidirectional"]
        uni = e["n_features_unidirectional"] > e["n_features_bidirectional"]
        if ecc and bi:
            per_section[s] = "eccentricity"
        elif (not ecc) and uni:
            per_section[s] = "trajectory-compatible"
        else:
            per_section[s] = "ambiguous"

    labels = set(per_section.values())
    detail = "; ".join(f"{s}: {c}" for s, c in per_section.items())

    if labels == {"eccentricity"}:
        overall = (
            f"TRAJECTORY FRAMING NOT SUPPORTED — every section reads as eccentricity "
            f"({detail}). The pseudotime orders patches by how ATYPICAL they are, not "
            "by how far along a progression they are. Report it as a "
            "morphological-atypicality score, not a trajectory, unless a directed "
            "reading can be established some other way."
        )
    elif labels == {"trajectory-compatible"}:
        overall = (
            f"TRAJECTORY FRAMING SURVIVES both tests in every section ({detail}). The "
            "axis is not eccentricity outside the diffusion map, and late patches move "
            "consistently rather than in opposing directions. The strong "
            "diffusion-space correlation found by root_sensitivity was definitional."
        )
    elif "eccentricity" in labels and "trajectory-compatible" in labels:
        overall = (
            f"SECTIONS DISAGREE — {detail}. One section reads as eccentricity and "
            "another as a directed axis, so the two must NOT be pooled and no single "
            "framing covers both. Report them separately and say so."
        )
    else:
        overall = (
            f"MIXED / INCONCLUSIVE — {detail}. At least one section fails to resolve "
            "(the geometry test and the directionality test disagree within it), so "
            "neither framing is established there."
        )

    return {
        "1_which_geometry_is_the_pseudotime": v1,
        "2_is_late_pseudotime_bidirectional": v2,
        "3_are_late_subclusters_opposed": v3,
        "4_overall": overall,
    }


def write_report(output_dir: Path, task_a: dict, task_b: dict, verdicts: dict,
                 root_prov: dict = None) -> None:
    L = ["# Is the pseudotime a trajectory, or an eccentricity measure?", ""]
    L.append(
        "**Why this analysis exists.** `root_sensitivity` found that the per-section "
        "pseudotime correlates with distance from the diffusion-map centroid far more "
        "strongly than with DC1 (0.81/0.80 vs 0.54/0.47), and that 25 random root sets "
        "reproduce it at |rho| 0.78-0.89. That says the axis is fixed by the manifold "
        "rather than by the root rule, and raises the question this module tests: does "
        "it measure how far ALONG a patch is, or how UNUSUAL it is? An eccentricity "
        "measure is directionless — 'late' would mean atypical in any direction, so two "
        "patches at opposite morphological extremes would both score late, and a "
        "directed-trajectory reading would not survive."
    )
    L.append("")
    L.append(
        "**The tautology this is built around.** DPT pseudotime IS a diffusion distance "
        "from its roots, and 20 median-aggregated roots approximate a central location, "
        "so a high correlation with diffusion-space eccentricity is PARTLY TRUE BY "
        "CONSTRUCTION. It is reported below but labelled DEFINITIONAL and must not be "
        "counted as evidence. The informative tests are in PCA space (which the "
        "diffusion map was built from) and in morphological-feature space (which the "
        "paper's claims are actually about) — DPT is not defined in terms of either."
    )
    L.append("")

    L += ["## Task 0 — where do the PRODUCTION roots come from?", ""]
    L.append(
        "root_sensitivity reported root provenance for the DC1 tails and all 50 null "
        "draws, but never for the root set that actually produced this pseudotime. "
        "Tightly-clustered root sets land inside a single slide routinely, and the "
        "production rule (20 lowest-nuclear_density patches) is such a set. Each row "
        "below is self-verified: DPT pseudotime is zero at its own roots, so a correct "
        "reconstruction must sit at pseudotime ~0."
    )
    L.append("")
    for section, rp in (root_prov or {}).items():
        pr = rp["provenance"]
        L.append(
            f"- **{section}** — reconstruction {'VERIFIED' if rp['reconstruction_verified'] else 'NOT VERIFIED'} "
            f"(root pseudotime median {_fmt(rp['root_pseudotime_median'])}); "
            f"{pr.get('n_distinct_slides')} slides among {rp['n_roots']} roots, largest share "
            f"{pr.get('max_share_from_one_slide', float('nan')):.0%}; "
            f"index span {pr.get('index_span')} of {pr.get('n_patches')}. {rp['verdict']}"
        )
    L.append("")

    L += ["## Task A — which geometry is the pseudotime?", ""]
    for section, res in task_a.items():
        L += [f"### {section} (n = {res['n_patches']})", ""]
        L.append("| measure | space | rho | status |")
        L.append("|---|---|---|---|")
        order = ["morph_mean_abs_z", "morph_mean_signed_z",
                 "pca_centroid_distance_matched_dims", "pca_centroid_distance",
                 "mean_knn_distance", "dc1", "dc1_eccentricity",
                 "diffmap_centroid_distance", "diffmap_median_centroid_distance"]
        for key in order:
            m = res["measures"][key]
            L.append(f"| `{key}` | {m['space']} | **{_fmt(m['rho'])}** | {m['status']} |")
        L.append("")
        L.append(f"**Verdict:** {res['verdict']}")
        L.append("")
        for key, ws in res.get("within_slide", {}).items():
            L.append(
                f"- Within-slide `{key}`: median rho across {ws['n_slides']} slides = "
                f"**{_fmt(ws['median_within_slide_rho'])}** (cohort-wide: "
                f"{_fmt(res['measures'][key]['rho'])}). {ws['note']}"
            )
        L.append("")

    L += ["## Task B — is late pseudotime heterogeneous, and in how many directions?", ""]
    L.append(
        "Heterogeneity alone does **not** separate eccentricity from a branching "
        "trajectory — both produce diverse late states. Direction does: a branching "
        "trajectory moves away from early consistently within a branch, whereas "
        "eccentricity puts late patches at opposite extremes of the *same* feature."
    )
    L.append("")
    for section, res in task_b.items():
        L += [f"### {section}", ""]
        d = res["dispersion_by_bin"]
        L.append(
            f"- Dispersion vs pseudotime decile: rho(bin, PCA dispersion) = "
            f"{_fmt(d['rho_bin_vs_pca_dispersion'])}, rho(bin, mean |z|) = "
            f"{_fmt(d['rho_bin_vs_mean_abs_z'])}. {d['note']}"
        )
        L.append("")

        e = res["bidirectional_enrichment"]
        L.append(f"**Bidirectional enrichment among the top {e['tail_fraction']:.0%} of "
                 f"pseudotime (n = {e['n_late']}):**")
        L.append("")
        L.append("| feature | high-tail enrichment | low-tail enrichment | pattern |")
        L.append("|---|---|---|---|")
        for feat in MORPH_FEATURES:
            v = e["late"].get(feat)
            if not v:
                continue
            L.append(f"| {feat} | {v['enrichment_high']:.2f}x | {v['enrichment_low']:.2f}x "
                     f"| {v['pattern']} |")
        L.append("")
        L.append(f"**Verdict:** {e['verdict']} {e['note']}")
        L.append("")

        s = res["late_subclustering"]
        if s.get("attempted"):
            L.append(f"**Late subclustering** (k chosen by silhouette = {s['best_k']}, "
                     f"n = {s['n_late_patches']}):")
            L.append("")
            L.append("| subcluster | n | " + " | ".join(MORPH_FEATURES) + " |")
            L.append("|---|---|" + "---|" * len(MORPH_FEATURES))
            for c, info in s["clusters"].items():
                row = " | ".join(f"{info['mean_z_profile'][f]:+.2f}" for f in MORPH_FEATURES)
                L.append(f"| {c} | {info['n']} | {row} |")
            L.append("")
            L.append(f"**Verdict:** {s['verdict']}")
        else:
            L.append(f"**Late subclustering:** skipped — {s.get('reason')}")
        L.append("")

        sc = res.get("slide_composition") or {}
        if sc:
            for end in ("early", "late"):
                if end in sc:
                    L.append(f"- {end.capitalize()} end: {sc[end]['n_distinct_slides']} "
                             f"slides, largest share "
                             f"{sc[end]['max_share_from_one_slide']:.0%}")
            if "note" in sc:
                L.append(f"- {sc['note']}")
        L.append("")

    L += ["## Verdicts", ""]
    for i, (k, v) in enumerate(verdicts.items(), start=1):
        L.append(f"**{i}. {k.split('_', 1)[1].replace('_', ' ')}**  ")
        L.append(v)
        L.append("")

    L += ["## Limitations", ""]
    L.append(
        "- The diffusion-space eccentricity figures are partly definitional and are "
        "excluded from every verdict here."
    )
    L.append(
        "- Rising dispersion with pseudotime is consistent with both hypotheses and is "
        "reported as context, not as evidence."
    )
    L.append(
        "- Bidirectional enrichment uses fixed decile tails and a 1.5x threshold; both "
        "are conventions, and features with weak marginal signal will read as "
        "'neither extreme enriched' rather than as evidence either way."
    )
    L.append(
        "- The six morphological descriptors are themselves correlated, so the mean-|z| "
        "term is not six independent measurements."
    )
    L.append(
        "- Everything here describes the existing per-section pseudotime. Nothing was "
        "recomputed, and no existing results directory was modified."
    )
    L.append("")

    (output_dir / "eccentricity_report.md").write_text("\n".join(L), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether MorphPT pseudotime is a trajectory coordinate or an "
                    "eccentricity (atypicality) measure."
    )
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True,
                        help="Existing per-section run dirs, same order as --sections. "
                             "READ-ONLY.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="NEW output directory.")
    parser.add_argument("--n-bins", type=int, default=N_BINS_DEFAULT)
    parser.add_argument("--tail-fraction", type=float, default=EXTREME_DECILE,
                        help="Tail fraction defining 'late'/'early' and the feature "
                             "extremes (default: 0.10)")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    args = parser.parse_args()

    if len(args.sections) != len(args.run_dirs):
        parser.error("--sections and --run-dirs must match in length and order")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Eccentricity check — is the pseudotime a trajectory?")
    print("=" * 64)

    task_a, task_b = {}, {}
    for section, run_dir in zip(args.sections, args.run_dirs):
        print(f"\n  Loading {section} from {run_dir} ...")
        adata = load_section(Path(run_dir))
        print(f"    {adata.n_obs} patches.")

        print(f"  [{section}] Task A — which geometry is the pseudotime? ...")
        ta = run_task_a(section, adata)
        task_a[section] = ta
        print(f"    {ta['verdict']}")

        print(f"  [{section}] Task B — late heterogeneity and direction ...")
        tb = run_task_b(section, adata, args.n_bins, args.tail_fraction,
                        (args.k_min, args.k_max))
        task_b[section] = tb
        print(f"    {tb['bidirectional_enrichment']['verdict']}")

        write_figures(section, ta, tb, args.output_dir)

    verdicts = build_verdicts(task_a, task_b)

    payload = {
        "analysis": "eccentricity_check",
        "why": (
            "root_sensitivity found pseudotime tracks diffusion-map centroid distance "
            "(0.81/0.80) far better than DC1 (0.54/0.47), and that random roots "
            "reproduce it at 0.78-0.89. This tests whether the axis measures how far "
            "along a patch is, or how atypical it is — in spaces where DPT's own "
            "construction does not force the answer."
        ),
        "parameters": {
            "n_bins": args.n_bins,
            "tail_fraction": args.tail_fraction,
            "k_range": [args.k_min, args.k_max],
            "enrichment_threshold": ENRICH_THRESHOLD,
            "eccentricity_strong_threshold": ECC_STRONG,
            "directional_gap_threshold": DIRECTIONAL_GAP,
        },
        "task_0_production_root_provenance": root_prov,
        "task_a_which_geometry": task_a,
        "task_b_late_heterogeneity": task_b,
        "verdicts": verdicts,
    }
    with open(args.output_dir / "eccentricity_check.json", "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"\n  JSON: {args.output_dir / 'eccentricity_check.json'}")

    write_report(args.output_dir, task_a, task_b, verdicts, root_prov)
    print(f"  Markdown: {args.output_dir / 'eccentricity_report.md'}")

    print("\n" + "=" * 64)
    print("  VERDICTS")
    print("=" * 64)
    for k, v in verdicts.items():
        print(f"\n  [{k}]\n  {v}")


if __name__ == "__main__":
    main()
