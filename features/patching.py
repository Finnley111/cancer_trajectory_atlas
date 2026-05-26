"""Patch extraction and basic tissue filtering for whole-slide images."""

import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Tuple, Optional, List

Image.MAX_IMAGE_PIXELS = None


# Background and tissue filters

def _is_mostly_white(patch_arr: np.ndarray,
                     white_thresh: int = 220,
                     white_frac: float = 0.70) -> bool:
    """Reject patches where most pixels are above the white threshold."""
    white_mask = np.all(patch_arr > white_thresh, axis=-1)
    return white_mask.mean() > white_frac


def _has_tissue_hsv(patch_pil: Image.Image,
                    sat_thresh: int = 15,
                    val_thresh: int = 230,
                    tissue_threshold: float = 0.5) -> bool:
    """Return True if the patch has enough saturated, non-bright pixels."""
    hsv = np.array(patch_pil.convert("HSV"))
    has_color = hsv[:, :, 1] > sat_thresh
    is_dense = hsv[:, :, 2] < val_thresh
    tissue_frac = (has_color & is_dense).mean()
    return tissue_frac >= tissue_threshold


# ROI helpers

def load_roi_polygons(
    annotation_path: str,
    coordinate_space: str = "ratio",
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
    original_full_width: Optional[int] = None,
    original_full_height: Optional[int] = None,
    cropped_w: Optional[int] = None,
    cropped_h: Optional[int] = None,
) -> List:
    """Load ROI polygons from JSON and convert them to pixel coordinates."""
    import json
    from matplotlib.path import Path as MplPath

    # Decide which dimensions to use for ratio scaling.
    if coordinate_space == "ratio":
        if original_full_width is not None:
            scale_w = original_full_width
            scale_h = original_full_height if original_full_height is not None else img_h
        elif img_w is not None:
            scale_w, scale_h = img_w, img_h
        else:
            raise ValueError(
                "ratio coordinate_space requires img_w/img_h or original_full_width/height"
            )
    else:
        scale_w = scale_h = None  # pixel coords — no scaling needed

    with open(annotation_path) as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        features_list = data["features"]
    elif isinstance(data, list):
        features_list = data
    else:
        features_list = [data]

    def scale_ring(ring):
        arr = np.asarray(ring, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        if coordinate_space == "ratio":
            arr[:, 0] *= scale_w
            arr[:, 1] *= scale_h
        return arr[:, :2]

    polygons = []
    for feat in features_list:
        geom = feat.get("geometry", {})
        geom_type = geom.get("type", "")

        if geom_type == "Polygon":
            ring = scale_ring(geom["coordinates"][0])
            if ring is not None and len(ring) >= 3:
                polygons.append(MplPath(ring))
        elif geom_type == "MultiPolygon":
            for poly_coords in geom["coordinates"]:
                ring = scale_ring(poly_coords[0])
                if ring is not None and len(ring) >= 3:
                    polygons.append(MplPath(ring))

    # Discard polygons whose centroid falls outside the cropped region.
    if cropped_w is not None and original_full_width is not None:
        filtered = []
        n_discarded = 0
        for poly in polygons:
            cx, cy = poly.vertices[:, 0].mean(), poly.vertices[:, 1].mean()
            if cx <= cropped_w and cy <= (cropped_h or scale_h):
                filtered.append(poly)
            else:
                n_discarded += 1
        if n_discarded > 0:
            print(f"    Discarded {n_discarded} ROI polygons outside cropped region")
        polygons = filtered

    return polygons


def _point_in_any_roi(x: float, y: float, roi_polygons: List) -> bool:
    """Check whether a point falls inside any ROI polygon."""
    for poly in roi_polygons:
        if poly.contains_point((x, y)):
            return True
    return False


# Main extraction

def get_patches_from_array(
    img_arr: np.ndarray,
    patch_size: int = 112,
    stride: int = 96,
    sat_thresh: int = 15,
    val_thresh: int = 230,
    tissue_threshold: float = 0.5,
    white_thresh: int = 220,
    white_frac: float = 0.70,
    image_name: str = "<array>",
    roi_polygons: Optional[List] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract filtered tissue patches from an in-memory RGB image."""
    h, w = img_arr.shape[:2]
    patches, coords = [], []

    y_steps = range(0, h - patch_size + 1, stride)
    x_steps = range(0, w - patch_size + 1, stride)
    total = len(y_steps) * len(x_steps)

    half = patch_size / 2.0
    n_roi_rejected = 0

    with tqdm(total=total, desc=f"Patching {image_name}") as pbar:
        for y in y_steps:
            for x in x_steps:
                pbar.update(1)

                # ROI containment check on the patch center.
                if roi_polygons is not None:
                    cx, cy = x + half, y + half
                    if not _point_in_any_roi(cx, cy, roi_polygons):
                        n_roi_rejected += 1
                        continue

                patch_arr = img_arr[y : y + patch_size, x : x + patch_size]

                # White-pixel rejection.
                if _is_mostly_white(patch_arr, white_thresh, white_frac):
                    continue

                # HSV tissue check.
                patch_pil = Image.fromarray(patch_arr)
                if not _has_tissue_hsv(patch_pil, sat_thresh, val_thresh, tissue_threshold):
                    continue

                patches.append(patch_arr)
                coords.append((x, y))

    if roi_polygons is not None:
        print(f"  ROI filter: {n_roi_rejected} patches outside hotspots")

    if total > 0:
        print(f"  Kept {len(patches)} / {total} patches ({len(patches)/total:.1%})")

    if len(patches) == 0:
        return np.array([]), np.array([])

    return np.array(patches), np.array(coords)


def sample_patches(patches, coords, max_n, base_seed, slide_name):
    """Subsample patches to at most max_n, reproducibly by slide name."""
    import hashlib
    if max_n is None or max_n <= 0 or len(patches) <= max_n:
        return patches, coords, None
    name_seed = int(hashlib.md5(slide_name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(int(base_seed) ^ name_seed)
    idx = np.sort(rng.choice(len(patches), max_n, replace=False))
    return patches[idx], coords[idx], idx


def get_patches(
    image_path: str,
    patch_size: int = 112,
    stride: int = 96,
    sat_thresh: int = 15,
    val_thresh: int = 230,
    white_thresh: int = 220,
    white_frac: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image from disk and extract patches."""
    print(f"Scanning image: {image_path}")
    try:
        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return np.array([]), np.array([])

    return get_patches_from_array(
        img_arr,
        patch_size=patch_size,
        stride=stride,
        sat_thresh=sat_thresh,
        val_thresh=val_thresh,
        white_thresh=white_thresh,
        white_frac=white_frac,
        image_name=image_path,
    )