"""
Phase 5 — Morphological Feature Extraction

Compute interpretable, low-level morphological descriptors from raw
(stain-normalized) patch images. These are used to validate whether
pseudotime captures a coherent morphological gradient.

Features:
  - Nuclear density          (cell crowding / proliferation)
  - Mean nuclear area        (atypia indicator)
  - Nuclear-to-cytoplasm ratio (relative nuclear size)
  - Texture entropy          (tissue disorganization via GLCM)
  - Hematoxylin intensity    (chromatin density)
  - Cell packing irregularity (spatial disorder of nuclear centroids)
"""

import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm


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
    distances: List[int] = (1, 3, 5),
) -> float:
    """
    Shannon entropy of the GLCM (gray-level co-occurrence matrix).
    Averaged over multiple distances.
    """
    from skimage.feature import graycomatrix, graycoprops

    # Quantize to 64 levels for GLCM
    if patch_gray.dtype != np.uint8:
        patch_gray = (patch_gray * 255).astype(np.uint8)
    patch_q = (patch_gray // 4).astype(np.uint8)

    entropies = []
    for d in distances:
        glcm = graycomatrix(patch_q, distances=[d], angles=[0], levels=64, symmetric=True, normed=True)
        # Shannon entropy of the normalized GLCM
        p = glcm[:, :, 0, 0]
        p_nonzero = p[p > 0]
        entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
        entropies.append(entropy)

    return float(np.mean(entropies))


def compute_hematoxylin_intensity(h_channel: np.ndarray) -> float:
    """Mean optical density in the hematoxylin channel."""
    return float(np.mean(h_channel))


def compute_packing_irregularity(labeled_mask: np.ndarray) -> float:
    """
    Coefficient of variation of nearest-neighbor distances between nuclear centroids.
    Higher = more spatially disordered.
    """
    from skimage.measure import regionprops
    from scipy.spatial import KDTree

    props = regionprops(labeled_mask)
    if len(props) < 3:
        return 0.0

    centroids = np.array([p.centroid for p in props])
    tree = KDTree(centroids)
    # Nearest neighbor distance for each nucleus (k=2 because first is self)
    distances, _ = tree.query(centroids, k=2)
    nn_dists = distances[:, 1]

    mean_dist = nn_dists.mean()
    if mean_dist < 1e-10:
        return 0.0
    return float(nn_dists.std() / mean_dist)


# ── Lightweight nuclear density (for multi-root DPT root selection) ──

def compute_nuclear_density_quick(patches: np.ndarray) -> np.ndarray:
    """Compute nuclear density for each patch using only hematoxylin + Otsu.

    Faster than the full morphological feature suite; used to select the
    n lowest-cellularity root candidates before running multi-root DPT.

    Returns:
        (N,) float array of nuclei-per-pixel-area values.
    """
    n = len(patches)
    patch_area = patches.shape[1] * patches.shape[2]
    densities = np.zeros(n, dtype=np.float32)

    for i in range(n):
        try:
            h_channel = _deconvolve_hematoxylin(patches[i])
            labeled = _segment_nuclei_simple(h_channel)
            densities[i] = compute_nuclear_density(labeled, patch_area)
        except Exception:
            pass  # leave as 0.0 on failure

    return densities


# ── Main feature extraction loop ────────────────────────────────────

def compute_morphological_features(
    patches: np.ndarray,
    use_stardist: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute all morphological features for a batch of patches.

    Args:
        patches: (N, H, W, 3) uint8 RGB arrays (should be stain-normalized).
        use_stardist: If True, use StarDist for nuclear segmentation (slower but better).

    Returns:
        Dict mapping feature name → (N,) float array.
    """
    from skimage.color import rgb2gray

    n_patches = len(patches)
    patch_h, patch_w = patches.shape[1], patches.shape[2]
    patch_area = patch_h * patch_w

    features = {
        "nuclear_density": np.zeros(n_patches),
        "mean_nuclear_area": np.zeros(n_patches),
        "nc_ratio": np.zeros(n_patches),
        "texture_entropy": np.zeros(n_patches),
        "h_intensity": np.zeros(n_patches),
        "packing_irregularity": np.zeros(n_patches),
    }

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
            features["h_intensity"][i] = compute_hematoxylin_intensity(h_channel)
            features["packing_irregularity"][i] = compute_packing_irregularity(labeled)

            # Texture entropy (on grayscale)
            gray = rgb2gray(patch)
            features["texture_entropy"][i] = compute_texture_entropy(
                (gray * 255).astype(np.uint8)
            )

        except Exception as exc:
            # Silently skip failed patches (features stay at 0.0)
            if i < 5:  # Only warn for first few
                print(f"  WARNING: Feature extraction failed for patch {i}: {exc}")

    return features
