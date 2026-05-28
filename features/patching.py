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

def _make_path(outer: np.ndarray, inner_rings: list):
    """Return an MplPath that correctly handles inner rings (holes) as a compound path."""
    from matplotlib.path import Path as MplPath
    if not inner_rings:
        return MplPath(outer)
    all_rings = [outer] + inner_rings
    verts, codes = [], []
    for ring in all_rings:
        n = len(ring)
        verts.append(ring)
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (n - 2) + [MplPath.CLOSEPOLY]
    return MplPath(np.concatenate(verts, axis=0), np.array(codes, dtype=np.uint8))


def load_roi_polygons(
    annotation_path: str,
    coordinate_space: str = "ratio",
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
    original_full_width: Optional[int] = None,
    original_full_height: Optional[int] = None,
    cropped_w: Optional[int] = None,
    cropped_h: Optional[int] = None,
) -> Tuple[List, List]:
    """Load ROI polygons from JSON, split into (include, exclude) lists.

    Polygons with classification "Tumor" or no classification are treated as
    inclusion zones (ROI hotspots).  All other named classifications
    (e.g. "Ignore*", "Necrosis", "Region*") are treated as exclusion zones —
    patches whose centre falls inside any exclusion polygon are dropped even
    if they are also inside an inclusion polygon.

    Returns
    -------
    include_polys : list of MplPath
        Patches must be inside at least one of these to be kept.
    exclude_polys : list of MplPath
        Patches inside any of these are always dropped.
    """
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

    def _class_name(feat):
        cls = feat.get("properties", {}).get("classification")
        if cls is None:
            return None
        return cls.get("name") if isinstance(cls, dict) else cls

    include_polys: List = []
    exclude_polys: List = []

    for feat in features_list:
        geom = feat.get("geometry", {})
        geom_type = geom.get("type", "")
        name = _class_name(feat)
        # None (unclassified) or "Tumor" → inclusion; anything else → exclusion.
        is_include = name is None or name == "Tumor"

        def _add(all_rings_list):
            for all_rings in all_rings_list:
                outer = scale_ring(all_rings[0])
                if outer is None or len(outer) < 3:
                    continue
                inner = [r for rc in all_rings[1:]
                         if (r := scale_ring(rc)) is not None and len(r) >= 3]
                path = _make_path(outer, inner)
                if is_include:
                    include_polys.append(path)
                else:
                    exclude_polys.append(path)

        if geom_type == "Polygon":
            _add([geom["coordinates"]])
        elif geom_type == "MultiPolygon":
            _add(geom["coordinates"])

    # Discard polygons whose centroid falls outside the cropped region.
    if cropped_w is not None and original_full_width is not None:
        def _in_crop(poly):
            cx, cy = poly.vertices[:, 0].mean(), poly.vertices[:, 1].mean()
            return cx <= cropped_w and cy <= (cropped_h or scale_h)

        n_before = len(include_polys) + len(exclude_polys)
        include_polys = [p for p in include_polys if _in_crop(p)]
        exclude_polys = [p for p in exclude_polys if _in_crop(p)]
        n_discarded = n_before - len(include_polys) - len(exclude_polys)
        if n_discarded > 0:
            print(f"    Discarded {n_discarded} ROI polygons outside cropped region")

    return include_polys, exclude_polys


def _find_containing_roi(x: float, y: float, roi_polygons: List):
    """Return the first polygon in roi_polygons that contains (x, y), or None."""
    for poly in roi_polygons:
        if poly.contains_point((x, y)):
            return poly
    return None


def _coverage_in_rois(x: float, y: float, patch_size: int, roi_polygons: List) -> float:
    """Fraction of a 3x3 sample grid (at 1/4, 1/2, 3/4 offsets) inside ANY ROI polygon.

    Checks all polygons rather than just the containing one so that grid points
    landing in completely unannotated territory are correctly counted as outside.
    """
    step = patch_size / 4.0
    inside = sum(
        1 for gi in range(1, 4) for gj in range(1, 4)
        if _find_containing_roi(x + gi * step, y + gj * step, roi_polygons) is not None
    )
    return inside / 9


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
    exclude_polygons: Optional[List] = None,
    min_roi_coverage: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract filtered tissue patches from an in-memory RGB image.

    roi_polygons     — inclusion zones: patch centre must be inside at least one.
    exclude_polygons — exclusion zones (Ignore*, Necrosis, etc.): patches whose
                       centre falls inside any of these are always dropped.
    min_roi_coverage — if set, patches where less than this fraction of a 3x3
                       sample grid lies inside the containing ROI polygon are
                       dropped (catches boundary patches that are mostly outside
                       the annotation). None = centre-point check only.
    """
    h, w = img_arr.shape[:2]
    patches, coords = [], []

    y_steps = range(0, h - patch_size + 1, stride)
    x_steps = range(0, w - patch_size + 1, stride)
    total = len(y_steps) * len(x_steps)

    half = patch_size / 2.0
    n_roi_rejected = 0
    n_exclude_rejected = 0
    n_coverage_rejected = 0

    with tqdm(total=total, desc=f"Patching {image_name}") as pbar:
        for y in y_steps:
            for x in x_steps:
                pbar.update(1)
                cx, cy = x + half, y + half

                # ROI inclusion check + optional coverage filter.
                if roi_polygons is not None:
                    containing = _find_containing_roi(cx, cy, roi_polygons)
                    if containing is None:
                        n_roi_rejected += 1
                        continue
                    if min_roi_coverage is not None:
                        if _coverage_in_rois(x, y, patch_size, roi_polygons) < min_roi_coverage:
                            n_coverage_rejected += 1
                            continue

                # Exclusion check (Ignore*, Necrosis, etc.).
                if exclude_polygons:
                    if _find_containing_roi(cx, cy, exclude_polygons) is not None:
                        n_exclude_rejected += 1
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
    if n_coverage_rejected:
        print(f"  Coverage filter: {n_coverage_rejected} patches below {min_roi_coverage:.0%} ROI coverage")
    if n_exclude_rejected:
        print(f"  Exclude filter: {n_exclude_rejected} patches inside Ignore/Necrosis regions")

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