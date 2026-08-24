"""Patch extraction and basic tissue filtering for whole-slide images.

The five filters in ``get_patches_from_array`` run in a fixed order and every
published run used all five. ``apply_white_filter`` / ``apply_tissue_filter``
were added for the v3 relaxed-filter experiment and default to True, so with no
argument passed this module behaves exactly as it did for the v2 per-section
runs. Nothing on the production path sets them.
"""

import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Tuple, Optional, List

Image.MAX_IMAGE_PIXELS = None


def _is_mostly_white(patch_arr: np.ndarray,
                     white_thresh: int = 220,
                     white_frac: float = 0.70) -> bool:
    """True when the patch is mostly background glass.

    A pixel counts as white only if ALL THREE channels exceed ``white_thresh``,
    so a strongly tinted bright pixel is not white.

    Both thresholds are function defaults that no caller overrides and that are
    not exposed on the CLI, so 220 and 0.70 are what every recorded run used.
    Neither value was tuned; they are the originals.
    """
    white_mask = np.all(patch_arr > white_thresh, axis=-1)
    return white_mask.mean() > white_frac


def _has_tissue_hsv(patch_pil: Image.Image,
                    sat_thresh: int = 15,
                    val_thresh: int = 230,
                    tissue_threshold: float = 0.5) -> bool:
    """True when enough of the patch is saturated and not too bright.

    Catches what ``_is_mostly_white`` misses: pale or out-of-focus tissue that is
    not white enough to reject on brightness alone but carries no usable
    morphology. A pixel counts as tissue when its saturation exceeds
    ``sat_thresh`` AND its value is below ``val_thresh``, and the patch passes
    when at least ``tissue_threshold`` of its pixels qualify.

    Note the comparison is >=, so a patch exactly at the threshold is kept.

    All three thresholds (15, 230, 0.5) are function defaults, not in the config
    and not on the CLI. They were never tuned.
    """
    hsv = np.array(patch_pil.convert("HSV"))
    has_color = hsv[:, :, 1] > sat_thresh
    is_dense = hsv[:, :, 2] < val_thresh
    tissue_frac = (has_color & is_dense).mean()
    return tissue_frac >= tissue_threshold


def _make_path(outer: np.ndarray, inner_rings: list):
    """Build an MplPath from an outer ring plus any inner rings.

    Inner rings become genuine holes rather than separate filled shapes, which
    matters because a duct annotated with a lumen would otherwise test as
    containing points that lie in the lumen.

    Returns a plain path when there are no inner rings, and a compound path with
    explicit MOVETO/LINETO/CLOSEPOLY codes when there are. Matplotlib applies the
    even-odd rule to compound paths, which is what makes the holes holes.

    Assumes every ring is already closed (last vertex repeating the first); the
    code array allots exactly one CLOSEPOLY per ring on that basis.
    """
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

    Coordinate system
    -----------------
    Three spaces exist in this pipeline:

    1. Full-NDPI pixel space. Width is the NDPI level-0 dimension, which includes
       BOTH slide copies side by side. ``original_full_width`` stores this value.

    2. Cropped-PNG pixel space. Width is ``original_full_width // 2``, the left
       half only. Patch (x, y) coords from get_patches_from_array live here.

    3. Ratio space, coordinates in [0, 1] relative to full-NDPI dimensions.
       QuPath annotates the left half of the NDPI, so left-half annotation
       x-values fall in [0, 0.5].

    When coordinate_space="ratio":
        polygon x *= original_full_width  →  full-NDPI pixel space
        Left-half annotations therefore land in [0, original_full_width/2]
        which equals [0, cropped_width].  This is the SAME as patch x-space,
        so no further offset is needed.

    Right-half polygons are discarded; they would correspond to the duplicate
    slide copy that is cropped out. Precisely: the discard runs only when BOTH
    ``cropped_w`` and ``original_full_width`` are supplied, and it drops any
    polygon whose centroid has ``cx > cropped_w`` OR ``cy > cropped_h``. The
    y-condition never fires in practice because the crop is horizontal only
    (``cropped_h == original_full_height``), but it is there.

    "Centroid" here is the mean of the path's vertices, not the area centroid,
    and for a polygon with holes the inner-ring vertices are included in that
    mean. This is accurate enough to separate left-half from right-half copies,
    which is all it is used for.

    Classification rules
    --------------------
    "Tumor" or unclassified (no ``properties.classification``) → inclusion zone.
    Any other name (Ignore*, Necrosis, Region*, …) → exclusion zone.
    The match on "Tumor" is exact and case-sensitive.

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
        # Already in pixels, so scale_ring leaves the coordinates alone.
        scale_w = scale_h = None

    with open(annotation_path) as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        features_list = data["features"]
    elif isinstance(data, list):
        features_list = data
    else:
        features_list = [data]

    def scale_ring(ring):
        """Convert one polygon ring to full-NDPI pixel coordinates.

        Returns an (n, 2) float array, or None when the ring is malformed
        (wrong dimensionality, or fewer than two coordinates per vertex).
        Callers must check for None; a malformed ring is skipped, not fatal.

        Trailing coordinates beyond x and y are dropped, so GeoJSON rings
        carrying a z value are accepted.
        """
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
    apply_white_filter: bool = True,
    apply_tissue_filter: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract filtered tissue patches from an in-memory RGB image.

    Returns ``(patches, coords)``: an (N, patch_size, patch_size, 3) uint8 array
    and an (N, 2) array of top-left (x, y) positions in CROPPED-PNG pixel space.
    Row i of one corresponds to row i of the other, and that alignment is what
    every downstream join relies on.

    Patches tile the image left to right, top to bottom. With ``stride`` below
    ``patch_size`` they overlap, so patches are not independent samples. The
    trailing partial row and column are dropped: the ranges stop at
    ``dimension - patch_size``, so up to ``patch_size - 1`` pixels along the
    right and bottom edges are never covered by any patch.

    Args:
        roi_polygons: inclusion zones. A patch is kept only if its centre lies
            inside at least one. None disables the check and keeps everything.
        exclude_polygons: exclusion zones (Ignore*, Necrosis, and so on). A patch
            whose centre falls inside any of these is dropped regardless of
            inclusion.
        min_roi_coverage: when set, drops patches where less than this fraction
            of a 3x3 sample grid lies inside ANY inclusion polygon, which catches
            boundary patches that are mostly outside the annotation. None applies
            the centre-point check only, and None is what every recorded run
            used.

            The grid is tested against the whole ``roi_polygons`` list rather
            than only the polygon containing the centre (see
            ``_coverage_in_rois``). A patch straddling two adjacent ROIs
            therefore counts as covered, which is intended.
        apply_white_filter, apply_tissue_filter: both default True, the only
            behaviour any published run has used. They exist for the v3
            relaxed-filter experiment (Configs B and C of jobs/run_v3b_relaxed.sh
            and run_v3c_both.sh), which asks what the manifold looks like when
            background is not removed.

            WARNING: with either set False this emits patches the production
            pipeline would never have seen, meaning background, slide edge and
            out-of-focus glass. That changes the patch count, hence the PCA
            basis, hence every downstream number, so such a run is NOT comparable
            to a production run on absolute values. Leave both True unless you
            are running that experiment.

    The five filters run in a fixed order, and the order matters for the printed
    counts: ROI inclusion, then coverage, then exclusion, then white-pixel, then
    HSV tissue. A patch rejected by an earlier filter is never seen by a later
    one, so the tallies are disjoint rather than overlapping, and no single count
    tells you how many patches a given filter would reject on its own.

    Assumes ``img_arr`` is at least ``patch_size`` in both dimensions. A smaller
    image yields empty ranges and returns without raising.

    TRAP: when nothing survives the filters, the returned arrays are
    ``np.array([])``, shape (0,), NOT the (0, size, size, 3) and (0, 2) the
    non-empty case gives. Callers that index columns, such as ``coords[:, 0]``,
    fail on the empty case, so check ``len()`` before indexing.
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
    n_white_rejected = 0
    n_tissue_rejected = 0

    if not (apply_white_filter and apply_tissue_filter):
        print(f"    RELAXED FILTERS: white={apply_white_filter}, "
              f"tissue_hsv={apply_tissue_filter} — background patches will be kept. "
              "Output is NOT comparable to a production run.")

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
                if apply_white_filter and _is_mostly_white(patch_arr, white_thresh, white_frac):
                    n_white_rejected += 1
                    continue

                # HSV tissue check.
                if apply_tissue_filter:
                    patch_pil = Image.fromarray(patch_arr)
                    if not _has_tissue_hsv(patch_pil, sat_thresh, val_thresh, tissue_threshold):
                        n_tissue_rejected += 1
                        continue

                patches.append(patch_arr)
                coords.append((x, y))

    if roi_polygons is not None:
        print(f"  ROI filter: {n_roi_rejected} patches outside hotspots")
    if n_coverage_rejected:
        print(f"  Coverage filter: {n_coverage_rejected} patches below {min_roi_coverage:.0%} ROI coverage")
    if n_exclude_rejected:
        print(f"  Exclude filter: {n_exclude_rejected} patches inside Ignore/Necrosis regions")

    # Printed only under relaxed filters, so production logs stay byte-identical.
    # With both filters on these two counts are recoverable anyway, as
    # total - roi - coverage - exclude - kept.
    if not (apply_white_filter and apply_tissue_filter):
        print(f"  White filter:  {n_white_rejected} rejected "
              f"({'ON' if apply_white_filter else 'DISABLED'})")
        print(f"  Tissue filter: {n_tissue_rejected} rejected "
              f"({'ON' if apply_tissue_filter else 'DISABLED'})")

    if total > 0:
        print(f"  Kept {len(patches)} / {total} patches ({len(patches)/total:.1%})")

    if len(patches) == 0:
        return np.array([]), np.array([])

    return np.array(patches), np.array(coords)


def sample_patches(patches, coords, max_n, base_seed, slide_name):
    """Subsample patches to at most max_n, reproducibly per slide.

    Returns ``(patches, coords, idx)``. ``idx`` is the array of selected
    positions into the ORIGINAL arrays, or None when no cap was applied. Callers
    need it to subset anything else aligned with the patches; without it the
    correspondence to the original rows is unrecoverable.

    The seed mixes ``base_seed`` with an MD5 hash of the slide name, so each
    slide draws a different subset while the whole thing stays reproducible from
    the seed alone. Seeding on ``base_seed`` alone would make every slide select
    the same positions, which is a real risk here since patches are emitted in
    raster order.

    Shuffling before truncating is what keeps the kept patches spatially spread.
    Taking the first max_n in raster order would return the top strip of the
    slide.

    ``max_n`` of 0 or None disables the cap, as does a patch count already at or
    below it; all three return the inputs unchanged with idx None.
    """
    import hashlib
    if max_n is None or max_n == 0 or len(patches) <= max_n:
        return patches, coords, None
    name_seed = int(hashlib.md5(slide_name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(int(base_seed) ^ name_seed)
    idx = np.arange(len(patches))
    rng.shuffle(idx)
    idx = idx[:max_n]
    return patches[idx], coords[idx], idx


def get_patches(
    image_path: str,
    patch_size: int = 112,
    stride: int = 96,
    sat_thresh: int = 15,
    val_thresh: int = 230,
    white_thresh: int = 220,
    white_frac: float = 0.70,
    apply_white_filter: bool = True,
    apply_tissue_filter: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image from disk and extract patches from it.

    Thin wrapper over :func:`get_patches_from_array`. Takes no ROI arguments, so
    it cannot restrict extraction to an annotation; it patches the whole image.
    The annotated path in ``run_all.py`` loads the image itself and calls
    ``get_patches_from_array`` directly.

    TRAP: this SWALLOWS every exception from loading. A missing file, a truncated
    PNG, or a decode error all print a line and return two empty arrays, which is
    the same value a real image containing no tissue returns. The caller cannot
    tell "failed to load" from "loaded fine, nothing survived the filters", and
    neither case raises. A slide silently contributing zero patches to a cohort
    is the failure mode to watch for; check the log for "Error loading image".

    ``tissue_threshold`` is not exposed here, so the HSV filter always uses the
    0.5 default even when a caller has tuned it elsewhere.
    """
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
        apply_white_filter=apply_white_filter,
        apply_tissue_filter=apply_tissue_filter,
    )