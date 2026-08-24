"""Validation helpers for correlation, permutation, and ordering tests."""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def correlate_features_with_pseudotime(
    pseudotime: np.ndarray,
    morph_features: Dict[str, np.ndarray],
) -> Dict[str, Dict]:
    """
    Spearman correlation between pseudotime and each morphological feature.

    Args:
        pseudotime: (N,) normalized pseudotime values.
        morph_features: {feature_name: (N,) values}.

    Returns:
        {feature_name: {"rho": float, "p_value": float, "interpretation": str}}
    """
    results = {}
    print("\n  Feature–Pseudotime Correlations:")
    print(f"  {'Feature':<25s} {'rho':>8s} {'p-value':>12s}  Interpretation")
    print("  " + "-" * 65)

    for name, values in morph_features.items():
        # Skip invalid values.
        valid = np.isfinite(values) & np.isfinite(pseudotime)
        if valid.sum() < 10:
            results[name] = {"rho": np.nan, "p_value": np.nan, "interpretation": "insufficient data"}
            continue

        rho, p = spearmanr(pseudotime[valid], values[valid])

        if abs(rho) > 0.4:
            interp = "strong — meaningful gradient"
        elif abs(rho) > 0.3:
            interp = "moderate — worth investigating"
        else:
            interp = "weak — no clear gradient"

        results[name] = {"rho": float(rho), "p_value": float(p), "interpretation": interp}
        print(f"  {name:<25s} {rho:>+8.3f} {p:>12.2e}  {interp}")

    return results


def permutation_test(
    pseudotime: np.ndarray,
    morph_features: Dict[str, np.ndarray],
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Permutation test for each feature-versus-pseudotime correlation.

    Shuffles pseudotime labels n_permutations times and computes null
    distribution of |rho|. Reports empirical p-value.

    The empirical p-value is ``mean(null >= |real_rho|)``, the unbiased-downward
    form. It can return exactly 0.0, which means "no permutation reached the
    observed |rho|", NOT "p = 0". With n_permutations=1000 the honest statement
    for a 0.0 result is p < 1/1000. The add-one form, (1 + count) / (1 + n),
    would report 0.001 instead; it is not used here and must not be swapped in
    without re-running every comparison, since it would shift every reported
    p-value.

    Returns:
        {feature_name: {
            "real_rho": float,
            "perm_p_value": float,
            "null_95th": float,
            "significant": bool
        }}
    """
    rng = np.random.RandomState(seed)
    feature_names = list(morph_features.keys())
    null_distributions = {name: [] for name in feature_names}

    print(f"\n  Running {n_permutations} permutations...")
    for _ in tqdm(range(n_permutations), desc="  Permutation test"):
        shuffled_pt = rng.permutation(pseudotime)
        for name in feature_names:
            values = morph_features[name]
            valid = np.isfinite(values) & np.isfinite(shuffled_pt)
            if valid.sum() < 10:
                null_distributions[name].append(0.0)
                continue
            r, _ = spearmanr(shuffled_pt[valid], values[valid])
            null_distributions[name].append(abs(r))

    results = {}
    print(f"\n  {'Feature':<25s} {'real |rho|':>10s} {'null 95th':>10s} {'perm p':>10s}  Sig?")
    print("  " + "-" * 70)

    for name in feature_names:
        values = morph_features[name]
        valid = np.isfinite(values) & np.isfinite(pseudotime)
        if valid.sum() < 10:
            results[name] = {"real_rho": np.nan, "perm_p_value": np.nan,
                             "null_95th": np.nan, "significant": False}
            continue

        real_rho, _ = spearmanr(pseudotime[valid], values[valid])
        null = np.array(null_distributions[name])
        perm_p = float(np.mean(null >= abs(real_rho)))
        null_95 = float(np.percentile(null, 95))
        sig = perm_p < 0.05

        results[name] = {
            "real_rho": float(real_rho),
            "perm_p_value": perm_p,
            "null_95th": null_95,
            "significant": sig,
        }
        sig_str = "YES" if sig else "no"
        print(f"  {name:<25s} {abs(real_rho):>10.3f} {null_95:>10.3f} {perm_p:>10.4f}  {sig_str}")

    return results


def cluster_ordering_analysis(
    pseudotime: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict:
    """
    Check whether clusters occupy distinct pseudotime ranges.

    Returns per-cluster median pseudotime and ordering.
    """
    unique_clusters = sorted(set(cluster_labels))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    cluster_stats = {}
    for c in unique_clusters:
        mask = cluster_labels == c
        pt_cluster = pseudotime[mask]
        cluster_stats[c] = {
            "median_pseudotime": float(np.median(pt_cluster)),
            "mean_pseudotime": float(np.mean(pt_cluster)),
            "std_pseudotime": float(np.std(pt_cluster)),
            "n_patches": int(mask.sum()),
        }

    # Rank clusters by median pseudotime
    ranked = sorted(cluster_stats.keys(), key=lambda c: cluster_stats[c]["median_pseudotime"])

    print("\n  Cluster Ordering (by median pseudotime):")
    print(f"  {'Cluster':>8s} {'Median PT':>10s} {'Std PT':>8s} {'N patches':>10s}")
    print("  " + "-" * 42)
    for c in ranked:
        s = cluster_stats[c]
        print(f"  {c:>8} {s['median_pseudotime']:>10.3f} {s['std_pseudotime']:>8.3f} {s['n_patches']:>10d}")

    return {"cluster_stats": cluster_stats, "ordering": ranked}


def spatial_depth_correlation(
    pseudotime: np.ndarray,
    coords: np.ndarray,
    roi_polygon=None,
) -> Dict:
    """Secondary check: correlation between pseudotime and spatial depth.

    DO NOT CITE THIS NUMBER WITHOUT READING THIS FIRST.

    ``run_all.py`` never passes ``roi_polygon``, so the fallback branch is the
    only one that has ever executed. That fallback computes "depth" as the
    distance from each patch to ``coords.mean(axis=0)``, the mean of ALL patch
    coordinates POOLED ACROSS EVERY SLIDE in the run.

    Patch coordinates are per-slide pixel offsets, each slide having its own
    origin. Averaging them across 8 or 16 slides produces a point that is not
    located in any slide, and distance-to-that-point is not a depth, an
    eccentricity, or any other geometric property of the tissue. It mostly
    reflects how large each slide's cropped PNG is and where its ROIs happen to
    sit in that frame.

    The value is reported as ``spatial_depth_secondary`` in validation.json and
    is printed with a "SECONDARY" label. It does NOT feed the verdict
    (``run_full_validation`` computes the verdict only from
    ``verdict_features``), so no published conclusion rests on it. It is left
    unchanged because fixing it would alter validation.json.

    A meaningful version would compute depth per slide, relative to that slide's
    own ROI boundary. That is what the ``roi_polygon`` branch does, and why it
    exists. That branch also hardcodes ``coords + 56`` as the patch centre, half
    of patch_size=112, so it would be wrong for any other patch size.
    """
    from scipy.spatial.distance import cdist

    if roi_polygon is not None:
        # Compute distance from each patch center to the polygon boundary.
        try:
            from shapely.geometry import Point, Polygon
            poly = Polygon(roi_polygon)
            centers = coords + 56  # Approximate patch center
            depths = np.array([poly.exterior.distance(Point(c)) for c in centers])
        except ImportError:
            print("  WARNING: shapely not installed, skipping spatial depth analysis.")
            return {"rho": np.nan, "p_value": np.nan, "note": "shapely not available"}
    else:
        # Fallback: use distance from the patch centroid.
        centroid = coords.mean(axis=0)
        depths = np.linalg.norm(coords - centroid, axis=1)

    rho, p = spearmanr(pseudotime, depths)

    print(f"\n  Spatial Depth (SECONDARY): rho={rho:+.3f}, p={p:.2e}")
    if abs(rho) > 0.4:
        print("  NOTE: Strong spatial correlation. Only meaningful if morphological "
              "features also correlate. Otherwise → capturing geometry, not biology.")

    return {"rho": float(rho), "p_value": float(p)}


# Full validation suite

def run_full_validation(
    pseudotime: np.ndarray,
    morph_features: Dict[str, np.ndarray],
    cluster_labels: np.ndarray,
    coords: np.ndarray,
    n_permutations: int = 1000,
    roi_polygon=None,
    verdict_features: Optional[List[str]] = None,
) -> Dict:
    """Run the full validation suite and return a results dictionary.

    verdict_features: which features count toward the headline verdict. Defaults
    to all of them. The pipeline passes an explicit list so that alternative
    definitions of the same quantity (h_intensity and h_intensity_wholepatch)
    are both REPORTED while only one is COUNTED. Otherwise a single feature votes
    twice and can push the verdict from CAUTIOUS to POSITIVE on its own.

    THE VERDICT RULE IS NON-MONOTONIC. Know this before quoting it.
    --------------------------------------------------------------
    The three-way branch is::

        n_strong >= 2 and n_sig >= 2   -> POSITIVE
        n_strong == 1 or  n_sig == 1   -> CAUTIOUS
        otherwise                      -> NULL RESULT

    The middle test matches EXACTLY one, not "at least one", so adding evidence
    can move the verdict backwards::

        n_strong=0, n_sig=1  -> CAUTIOUS
        n_strong=0, n_sig=2  -> NULL RESULT     (more significance, weaker verdict)
        n_strong=1, n_sig=0  -> CAUTIOUS
        n_strong=2, n_sig=0  -> NULL RESULT     (more strong rhos, weaker verdict)
        n_strong=3, n_sig=0  -> NULL RESULT

    Note also that with several thousand patches, permutation significance is
    easy to reach at trivial effect sizes, so n_sig is expected to be large and
    n_strong is the discriminating term in practice.

    This is left exactly as written: every published verdict for this project was
    produced by this rule, and changing it would silently reinterpret past runs.
    Treat the verdict string as a coarse label and read
    ``summary.n_strong_correlations`` / ``summary.n_significant_permutations``
    for anything that matters. Phase 8's regression compares the verdict string,
    so it must not change before that run.
    """
    print("\n" + "=" * 60)
    print("VALIDATION SUITE")
    print("=" * 60)

    correlations = correlate_features_with_pseudotime(pseudotime, morph_features)
    perm_results = permutation_test(pseudotime, morph_features, n_permutations)
    cluster_order = cluster_ordering_analysis(pseudotime, cluster_labels)
    spatial = spatial_depth_correlation(pseudotime, coords, roi_polygon)

    counted = list(morph_features.keys()) if verdict_features is None else [
        f for f in verdict_features if f in correlations
    ]

    # Overall interpretation. np.isfinite guards are explicit: abs(nan) > 0.4 is
    # False, so without them a feature that could not be computed would be
    # silently counted as "measured, weak" rather than excluded.
    n_strong = sum(1 for f in counted
                   if np.isfinite(correlations[f].get("rho", np.nan))
                   and abs(correlations[f]["rho"]) > 0.4)
    n_sig = sum(1 for f in counted if perm_results.get(f, {}).get("significant", False))
    n_uncomputable = sum(1 for f in counted
                         if not np.isfinite(correlations[f].get("rho", np.nan)))
    if n_uncomputable:
        print(f"\n  NOTE: {n_uncomputable} of {len(counted)} counted feature(s) had "
              "a non-finite correlation and were EXCLUDED from the verdict, not "
              "counted as weak.")

    if n_strong >= 2 and n_sig >= 2:
        verdict = ("POSITIVE: Multiple features show strong, significant correlation "
                   "with pseudotime. The trajectory captures a coherent morphological gradient.")
    elif n_strong == 1 or n_sig == 1:
        verdict = ("CAUTIOUS: Only one feature correlates. May be tracking a single confound "
                   "(e.g., staining intensity) rather than a true morphological transition.")
    else:
        verdict = ("NULL RESULT: No features correlate with pseudotime. The data does not "
                   "contain a progression signal detectable by this method. "
                   "This is still a valid and reportable finding.")

    print(f"\n  VERDICT: {verdict}")

    return {
        "feature_correlations": correlations,
        "permutation_tests": perm_results,
        "cluster_ordering": cluster_order,
        "spatial_depth_secondary": spatial,
        "summary": {
            "n_strong_correlations": n_strong,
            "n_significant_permutations": n_sig,
            "n_uncomputable_correlations": n_uncomputable,
            "features_counted_toward_verdict": counted,
            "verdict": verdict,
        },
    }
