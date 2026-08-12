"""
Phase 5 — Morphological Feature Extraction

Compute interpretable, low-level morphological descriptors from raw
(stain-normalized) patch images. These are used to validate whether
pseudotime captures a coherent morphological gradient.

Features:
  - Nuclear density          (cell crowding / proliferation)
  - Mean nuclear area        (atypia indicator)
  - Nuclear-to-cytoplasm ratio (relative nuclear size)
  - Texture entropy          (tissue disorganization via GLCM, multi-angle)
  - Hematoxylin intensity    (chromatin density, masked to segmented nuclei)
  - Cell packing irregularity (spatial disorder of nuclear centroids)

MISSING-VALUE CONVENTION
    Every function here returns np.nan for "could not be measured", never 0.0.
    That distinction is load-bearing: DPT roots are selected as
    argsort(nuclear_density)[:20] (analysis/diffusion.py), so a failure encoded
    as 0.0 is not merely lost — it is PREFERENTIALLY PROMOTED TO A ROOT, and the
    pseudotime origin ends up anchored on whichever patches crashed the
    segmenter. NaN is excluded by the np.isfinite() filters in
    validation/correlations.py and by explicit masking in the root selection.

    Two functions deliberately keep 0.0: compute_mean_nuclear_area and
    compute_nc_ratio return 0.0 for an empty mask, because "no nuclei" really
    does mean zero nuclear area and zero nuclear fraction. compute_hematoxylin_
    intensity does NOT, because the mean of an empty selection is undefined
    rather than zero.

    Two further non-nan returns exist and are NOT exceptions to the convention,
    because both are unreachable or self-excluding in practice:
      * compute_nuclear_density returns 0.0 when patch_area <= 0. patch_area is
        patch_size**2 (12544), so this cannot fire from the pipeline.
      * compute_nc_ratio returns +inf when a mask covers 100% of the patch
        (zero cytoplasm pixels). np.isfinite(inf) is False, so such a patch is
        excluded by the same filters that exclude nan — the convention holds
        even though the sentinel differs.

DIAGNOSTIC ARITHMETIC HAS A KNOWN EDGE CASE
    compute_morphological_features derives two reported counts by subtraction:
        n_empty_mask = nan_counts["h_intensity"]          - n_failed
        n_lt3_nuclei = nan_counts["packing_irregularity"] - n_failed
    This assumes a failed patch contributes a nan to EVERY feature, which holds
    only when the exception is raised at or before nuclear segmentation. Features
    are assigned incrementally inside one try block and texture_entropy is
    computed LAST, so a patch that succeeds through packing_irregularity and then
    dies in rgb2gray/compute_texture_entropy keeps six real values while still
    incrementing n_failed. Both subtractions are then understated and can go
    NEGATIVE.

    The FEATURE VALUES are correct in that case — partial results are genuinely
    valid and are deliberately kept. Only the two derived counts in the printout
    and in feature_failures.json are affected. With n_failed == 0 (every run to
    date, including per_section_v2) the arithmetic is exact. Left as-is because
    changing it would change feature_failures.json's contents; see reports/
    codebase_inventory.md.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Fix 1b: GLCM angles. Tissue disorganisation has no preferred axis, and a
# single-angle GLCM measures slide mounting orientation as much as morphology.
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_DISTANCES = (1, 3, 5)


def _deconvolve_hematoxylin(patch_rgb: np.ndarray) -> np.ndarray:
    """
    Extract hematoxylin channel via color deconvolution (HED space).

    Returns a 2D float array where higher values = more hematoxylin staining.
    """
    from skimage.color import rgb2hed
    hed = rgb2hed(patch_rgb)
    return hed[:, :, 0]  # H channel


def _segment_nuclei_simple(h_channel: np.ndarray, threshold: str = "otsu") -> np.ndarray:
    """
    Simple nuclear segmentation: Otsu on hematoxylin channel + connected components.

    For higher quality, use StarDist (see _segment_nuclei_stardist).
    Returns a labeled mask (0 = background, 1..N = nuclei).
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, binary_opening, disk
    from skimage.measure import label

    # Otsu on hematoxylin channel
    thresh = threshold_otsu(h_channel)
    binary = h_channel > thresh

    # Clean up
    binary = binary_opening(binary, disk(1))
    binary = remove_small_objects(binary, min_size=20)

    return label(binary)


def _segment_nuclei_stardist(patch_rgb: np.ndarray) -> np.ndarray:
    """
    Nuclear segmentation using StarDist (higher quality, slower).

    Requires: pip install stardist tensorflow
    Returns a labeled mask.
    """
    try:
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize
    except ImportError:
        print("  StarDist not available, falling back to Otsu segmentation.")
        h_channel = _deconvolve_hematoxylin(patch_rgb)
        return _segment_nuclei_simple(h_channel)

    model = StarDist2D.from_pretrained("2D_versatile_he")
    labels, _ = model.predict_instances(normalize(patch_rgb))
    return labels


def compute_nuclear_density(labeled_mask: np.ndarray, patch_area: float) -> float:
    """Count of nuclei per unit area."""
    n_nuclei = labeled_mask.max()  # labels are 1..N
    return n_nuclei / patch_area if patch_area > 0 else 0.0


def compute_mean_nuclear_area(labeled_mask: np.ndarray) -> float:
    """Mean pixel area of segmented nuclei."""
    from skimage.measure import regionprops
    props = regionprops(labeled_mask)
    if len(props) == 0:
        return 0.0
    areas = [p.area for p in props]
    return float(np.mean(areas))


def compute_nc_ratio(labeled_mask: np.ndarray) -> float:
    """Nuclear-to-cytoplasm ratio (total nuclear pixels / total non-nuclear pixels)."""
    nuclear_pixels = (labeled_mask > 0).sum()
    total_pixels = labeled_mask.size
    cytoplasm_pixels = total_pixels - nuclear_pixels
    if cytoplasm_pixels == 0:
        return float("inf")
    return nuclear_pixels / cytoplasm_pixels


def compute_texture_entropy(
    patch_gray: np.ndarray,
    distances: List[int] = GLCM_DISTANCES,
    angles: List[float] = GLCM_ANGLES,
) -> float:
    """Shannon entropy of the GLCM, averaged over distances AND angles.

    FIX 1b — was angles=[0] only, i.e. horizontal pixel pairs. Tissue
    disorganisation has no preferred axis, so a single-angle GLCM partly measures
    how the section happened to be mounted.

    AVERAGING CHOICE (explicit): entropy is computed per (distance, angle) pair
    and the 12 scalars are then averaged. It is NOT computed once on a pooled
    GLCM. Shannon entropy is concave, so averaging the matrices first yields a
    systematically HIGHER value than the mean of the parts (Jensen), which would
    add an offset unrelated to this fix and break comparability with the baseline
    run. Because the 3x4 design is balanced, averaging over angles then distances
    is numerically identical to this flat mean — that ordering is not a real
    choice; entropy-then-average versus average-then-entropy is.
    """
    from skimage.feature import graycomatrix

    # Quantize to 64 levels for GLCM
    if patch_gray.dtype != np.uint8:
        patch_gray = (patch_gray * 255).astype(np.uint8)
    patch_q = (patch_gray // 4).astype(np.uint8)

    # One call covers the full grid; graycomatrix returns (levels, levels, nd, na).
    glcm = graycomatrix(patch_q, distances=list(distances), angles=list(angles),
                        levels=64, symmetric=True, normed=True)

    entropies = []
    for di in range(glcm.shape[2]):
        for ai in range(glcm.shape[3]):
            p = glcm[:, :, di, ai]
            p_nonzero = p[p > 0]
            entropies.append(-np.sum(p_nonzero * np.log2(p_nonzero)))

    if not entropies:
        return float("nan")
    return float(np.mean(entropies))


def compute_hematoxylin_intensity(h_channel: np.ndarray,
                                  labeled_mask: Optional[np.ndarray] = None) -> float:
    """Mean hematoxylin optical density WITHIN segmented nuclei — chromatin density.

    FIX 1c — previously averaged over every pixel in the patch, background
    included. That makes the value rise mechanically with the fraction of the
    patch covered by nuclei, i.e. with nuclear density, rather than measuring how
    darkly the chromatin itself stains. The module docstring has always called
    this "chromatin density"; only now does the implementation deliver it.

    Returns nan when the mask is empty: the mean of an empty selection is
    undefined, not zero (see the missing-value convention in the module
    docstring). Pass labeled_mask=None to get the legacy whole-patch value, which
    the pipeline retains alongside as `h_intensity_wholepatch` so the two
    definitions can be compared within a single run.
    """
    if labeled_mask is None:
        return float(np.mean(h_channel))
    sel = h_channel[labeled_mask > 0]
    if sel.size == 0:
        return float("nan")
    return float(np.mean(sel))


def compute_packing_irregularity(labeled_mask: np.ndarray) -> float:
    """
    Coefficient of variation of nearest-neighbor distances between nuclear centroids.
    Higher = more spatially disordered.

    FIX 1d — sentinel only. The <3-nuclei limitation is intrinsic to a
    coefficient of variation over nearest-neighbour distances and is NOT changed.
    What changed is that sparse patches now return nan instead of 0.0, so they are
    excluded from correlations rather than counted as "perfectly regular packing"
    — an artificial value that tied this feature structurally to nuclear density.
    """
    from skimage.measure import regionprops
    from scipy.spatial import KDTree

    props = regionprops(labeled_mask)
    if len(props) < 3:
        return float("nan")

    centroids = np.array([p.centroid for p in props])
    tree = KDTree(centroids)
    # Nearest neighbor distance for each nucleus (k=2 because first is self)
    distances, _ = tree.query(centroids, k=2)
    nn_dists = distances[:, 1]

    mean_dist = nn_dists.mean()
    if mean_dist < 1e-10:
        return float("nan")
    return float(nn_dists.std() / mean_dist)


# ── Lightweight nuclear density (for multi-root DPT root selection) ──

def compute_nuclear_density_quick(patches: np.ndarray,
                                  return_diagnostics: bool = False):
    """Compute nuclear density for each patch using only hematoxylin + Otsu.

    Faster than the full morphological feature suite; used to select the
    n lowest-cellularity root candidates before running multi-root DPT.

    FIX 1a — failures now yield np.nan, not 0.0. This function feeds DPT root
    selection directly, so the old behaviour meant every crashed patch became a
    root candidate ahead of every real patch: the pseudotime origin was biased
    toward segmentation failures by construction.

    Args:
        return_diagnostics: if True, return (densities, diagnostics) instead of
            just densities. Default False keeps the existing call signature.

    Returns:
        (N,) float64 array, nan where extraction failed.
    """
    n = len(patches)
    patch_area = patches.shape[1] * patches.shape[2]
    densities = np.full(n, np.nan, dtype=np.float64)

    failed_indices: List[int] = []
    exc_types: Dict[str, int] = {}

    for i in range(n):
        try:
            h_channel = _deconvolve_hematoxylin(patches[i])
            labeled = _segment_nuclei_simple(h_channel)
            densities[i] = compute_nuclear_density(labeled, patch_area)
        except Exception as exc:
            failed_indices.append(i)
            key = type(exc).__name__
            exc_types[key] = exc_types.get(key, 0) + 1
            if len(failed_indices) <= 5:
                print(f"  WARNING: quick nuclear density failed for patch {i}: {exc}")

    n_failed = len(failed_indices)
    rate = n_failed / n if n else 0.0
    if n_failed:
        print(f"  compute_nuclear_density_quick: {n_failed}/{n} patches failed "
              f"({rate:.3%}); recorded as nan and EXCLUDED from root selection.")
    else:
        print(f"  compute_nuclear_density_quick: 0/{n} patches failed.")

    if not return_diagnostics:
        return densities
    return densities, {
        "function": "compute_nuclear_density_quick",
        "n_patches": int(n),
        "n_failed": int(n_failed),
        "failure_rate": float(rate),
        "failed_indices": failed_indices,
        "exception_types": exc_types,
        "n_nan_out": int(np.isnan(densities).sum()),
    }


# ── Main feature extraction loop ────────────────────────────────────

FEATURE_NAMES = (
    "nuclear_density",
    "mean_nuclear_area",
    "nc_ratio",
    "texture_entropy",
    "h_intensity",
    "h_intensity_wholepatch",
    "packing_irregularity",
)


def compute_morphological_features(
    patches: np.ndarray,
    use_stardist: bool = False,
    return_diagnostics: bool = False,
):
    """
    Compute all morphological features for a batch of patches.

    FIX 1a — features initialise to np.nan and a failed patch STAYS nan, instead
    of silently reading as 0.0. Failures are counted, indexed and reported rather
    than warned about five times and then forgotten.

    FIX 1c — `h_intensity` is now masked to segmented nuclei. The legacy
    whole-patch value is retained as `h_intensity_wholepatch` so the two
    definitions are directly comparable inside a single run.

    Args:
        patches: (N, H, W, 3) uint8 RGB arrays (should be stain-normalized).
        use_stardist: If True, use StarDist for nuclear segmentation (slower but better).
        return_diagnostics: if True, return (features, diagnostics). Default False
            preserves the existing call signature.

    Returns:
        Dict mapping feature name -> (N,) float array, nan where unmeasurable.
    """
    from skimage.color import rgb2gray

    n_patches = len(patches)
    patch_h, patch_w = patches.shape[1], patches.shape[2]
    patch_area = patch_h * patch_w

    features = {name: np.full(n_patches, np.nan, dtype=np.float64)
                for name in FEATURE_NAMES}

    failed_indices: List[int] = []
    exc_types: Dict[str, int] = {}

    for i in tqdm(range(n_patches), desc="Computing morphological features"):
        patch = patches[i]

        try:
            # Hematoxylin deconvolution
            h_channel = _deconvolve_hematoxylin(patch)

            # Nuclear segmentation
            if use_stardist:
                labeled = _segment_nuclei_stardist(patch)
            else:
                labeled = _segment_nuclei_simple(h_channel)

            # Compute features
            features["nuclear_density"][i] = compute_nuclear_density(labeled, patch_area)
            features["mean_nuclear_area"][i] = compute_mean_nuclear_area(labeled)
            features["nc_ratio"][i] = compute_nc_ratio(labeled)
            features["h_intensity"][i] = compute_hematoxylin_intensity(h_channel, labeled)
            features["h_intensity_wholepatch"][i] = compute_hematoxylin_intensity(h_channel)
            features["packing_irregularity"][i] = compute_packing_irregularity(labeled)

            # Texture entropy (on grayscale)
            gray = rgb2gray(patch)
            features["texture_entropy"][i] = compute_texture_entropy(
                (gray * 255).astype(np.uint8)
            )

        except Exception as exc:
            # Features for this patch stay nan — NOT 0.0. See the missing-value
            # convention in the module docstring for why that distinction matters.
            failed_indices.append(i)
            key = type(exc).__name__
            exc_types[key] = exc_types.get(key, 0) + 1
            if len(failed_indices) <= 5:  # Only warn in detail for the first few
                print(f"  WARNING: Feature extraction failed for patch {i}: {exc}")

    n_failed = len(failed_indices)
    rate = n_failed / n_patches if n_patches else 0.0

    # Per-feature nan counts separate the three distinct causes of a missing
    # value: a crashed patch (all features nan), an empty nuclear mask
    # (h_intensity only), and <3 nuclei (packing_irregularity only).
    nan_counts = {name: int(np.isnan(v).sum()) for name, v in features.items()}
    n_empty_mask = nan_counts["h_intensity"] - n_failed
    n_lt3_nuclei = nan_counts["packing_irregularity"] - n_failed

    print(f"\n  Feature extraction: {n_failed}/{n_patches} patches failed "
          f"({rate:.3%}).")
    print(f"    h_intensity nan from empty nuclear mask : {n_empty_mask}")
    print(f"    packing_irregularity nan from <3 nuclei : {n_lt3_nuclei}")
    if exc_types:
        print(f"    exception types: {exc_types}")

    if not return_diagnostics:
        return features
    return features, {
        "function": "compute_morphological_features",
        "n_patches": int(n_patches),
        "n_failed": int(n_failed),
        "failure_rate": float(rate),
        "failed_indices": failed_indices,
        "exception_types": exc_types,
        "nan_counts_per_feature": nan_counts,
        "n_nan_h_intensity_empty_mask": int(n_empty_mask),
        "n_nan_packing_irregularity_lt3_nuclei": int(n_lt3_nuclei),
        "glcm_angles": [float(a) for a in GLCM_ANGLES],
        "glcm_distances": [int(d) for d in GLCM_DISTANCES],
        "h_intensity_definition": "mean(h_channel[labeled_mask > 0]) — masked to nuclei",
        "h_intensity_wholepatch_definition": "mean(h_channel) — legacy, all pixels",
    }
