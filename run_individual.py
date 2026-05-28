#!/usr/bin/env python3
"""Run pseudotime per-slide and save visuals to individual_pseudotime_runs/.

For each slide in PNG_DIR, fits its own PCA + Leiden + diffusion map and
computes a pseudotime trajectory in isolation. Useful for spotting per-slide
trajectory structure before pooling, and for comparing how pseudotime
distributes across patients.

Usage:
    python -m cancer_trajectory_atlas.run_individual              # all slides
    python -m cancer_trajectory_atlas.run_individual --slide 6027-4L-2M-1
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter

Image.MAX_IMAGE_PIXELS = None


# Default configuration loader

_SCRIPT_DIR = Path(__file__).parent
_PATHS_FILE = _SCRIPT_DIR / "paths.json"

def _load_default_paths():
    """Load paths from paths.json or use local development paths."""
    if _PATHS_FILE.exists():
        with open(_PATHS_FILE) as f:
            config = json.load(f)
        return {
            "png_dir": Path(config["cropped_png"]).expanduser(),
            "annotation_dir": Path(config["annotations"]).expanduser(),
            "output_dir": Path(config["results"]).expanduser() / "individual_pseudotime_runs",
        }
    else:
        return {
            "png_dir": _SCRIPT_DIR / "data" / "MCF7_x5_cropped",
            "annotation_dir": _SCRIPT_DIR / "data" / "annotations",
            "output_dir": _SCRIPT_DIR.parent / "individual_pseudotime_runs",
        }

from .data.slide_registry import KNOWN_NDPI_DIMENSIONS

# Global variables (will be set by CLI arguments)
PNG_DIR = None
ANNOTATION_DIR = None
OUTPUT_DIR = None
MODEL = None
PATCH_SIZE = None
STRIDE = None
LEIDEN_RESOLUTION = None
STAIN_NORMALIZATION = None
MIN_ROI_COVERAGE = None


# Slide discovery

def _get_known_dimensions(png_name, ndpi_scale):
    stem = Path(png_name).stem
    base_stem = stem.replace("_x5", "")
    dims = KNOWN_NDPI_DIMENSIONS.get(base_stem)
    if dims is None:
        return None, None
    full_w, full_h = dims
    if ndpi_scale != 1.0:
        full_w = int(full_w * ndpi_scale)
        full_h = int(full_h * ndpi_scale)
    return full_w, full_h


def discover_slides(ndpi_scale=1.0, filter_name=None):
    png_dir = Path(PNG_DIR)
    ann_dir = Path(ANNOTATION_DIR)

    if not png_dir.exists():
        print(f"ERROR: PNG directory not found: {png_dir}")
        sys.exit(1)

    png_files = sorted(png_dir.glob("*.png"))
    if not png_files:
        print(f"ERROR: No PNG files found in {png_dir}")
        sys.exit(1)

    # Sidecar dimensions if present
    dims_path = png_dir / "slide_dimensions.json"
    dims_log = {}
    if dims_path.exists():
        with open(dims_path) as f:
            dims_log = json.load(f)

    slides = []
    for png_path in png_files:
        stem = png_path.stem
        base_stem = stem.replace("_x5", "")

        if filter_name and filter_name not in stem:
            continue

        ann_path = None
        for cand in [
            ann_dir / f"{stem}.json",
            ann_dir / f"{base_stem}.json",
            ann_dir / f"{stem}.geojson",
            ann_dir / f"{base_stem}.geojson",
        ]:
            if cand.exists():
                ann_path = str(cand)
                break

        sidecar = dims_log.get(png_path.name, {})
        if sidecar:
            fw = sidecar["original_full_width"]
            fh = sidecar["original_full_height"]
        else:
            fw, fh = _get_known_dimensions(png_path.name, ndpi_scale)

        slides.append({
            "image": str(png_path),
            "name": stem,
            "annotation": ann_path,
            "original_full_width": fw,
            "original_full_height": fh,
        })

    if filter_name and not slides:
        print(f"ERROR: No slides matched filter '{filter_name}'")
        sys.exit(1)

    return slides


# Pseudotime heatmap overlay

def plot_pseudotime_heatmap_overlay(
    image_path, coords, pseudotime, patch_size, stride, save_path,
    alpha=0.55, smooth_sigma=1.0,
):
    """Overlay per-patch pseudotime on the cropped slide image.

    Builds a stride-resolution grid of pseudotime values, smooths it with
    a Gaussian, and bilinear-upsamples to the original image dimensions.
    Non-patched regions are masked transparent.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]

    # Patch centers (coords are top-left corners)
    half = patch_size // 2
    cx = coords[:, 0] + half
    cy = coords[:, 1] + half

    # Grid at stride resolution — use ceil so we don't clip the right/bottom edge
    gw = int(np.ceil(W / stride))
    gh = int(np.ceil(H / stride))
    pt_sum = np.zeros((gh, gw), dtype=np.float64)
    pt_count = np.zeros((gh, gw), dtype=np.float64)

    for x, y, pt in zip(cx, cy, pseudotime):
        gx = int(x / stride)
        gy = int(y / stride)
        if 0 <= gx < gw and 0 <= gy < gh:
            pt_sum[gy, gx] += pt
            pt_count[gy, gx] += 1

    valid = pt_count > 0
    grid = np.where(valid, pt_sum / np.maximum(pt_count, 1), 0.0)

    # Slight smoothing on the valid region only (avoids bleeding into NaN)
    if smooth_sigma > 0:
        weighted = gaussian_filter(grid * valid.astype(float), sigma=smooth_sigma)
        weights = gaussian_filter(valid.astype(float), sigma=smooth_sigma)
        grid = np.where(weights > 1e-6, weighted / np.maximum(weights, 1e-6), 0.0)

    # Upsample grid + mask to image resolution
    zh, zw = H / gh, W / gw
    pt_full = zoom(grid, (zh, zw), order=1)
    # Upsample the boolean validity mask with nearest-neighbor to avoid
    # linear interpolation producing partial values outside annotated areas.
    mask_full = zoom(valid.astype(float), (zh, zw), order=0) > 0.5

    # Trim to image dims (zoom can produce ±1 pixel rounding)
    pt_full = pt_full[:H, :W]
    mask_full = mask_full[:H, :W]
    pt_full[~mask_full] = np.nan

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    masked = np.ma.masked_invalid(pt_full)
    sc = ax.imshow(masked, cmap="plasma", alpha=alpha, vmin=0, vmax=1)
    ax.set_title(f"Pseudotime Heatmap — {Path(image_path).stem}")
    ax.axis("off")
    plt.colorbar(sc, ax=ax, label="Pseudotime", fraction=0.04, pad=0.02)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


# Pseudotime distribution histogram

def plot_pseudotime_histogram(pseudotime, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pseudotime, bins=40, color="#7e3a92", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Patch count")
    ax.set_title("Pseudotime Distribution")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# Per-slide run

def run_one_slide(slide_cfg, stain_normalizer, out_root, leiden_resolution):
    """Compute pseudotime for a single slide and save all visuals."""

    from .data.stain_normalization import normalize_slide
    from .features.patching import get_patches_from_array, load_roi_polygons
    from .features.extractors import extract_features
    from .analysis.clustering import (
        fit_pca, run_umap, cluster, get_cluster_centroids,
    )
    from .analysis.diffusion import (
        build_adata, compute_diffusion_map, compute_dpt,
    )
    from .utils import viz

    name = slide_cfg["name"]
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Load and stain-normalize
    img = Image.open(slide_cfg["image"]).convert("RGB")
    img_arr = np.array(img)
    print(f"  Image: {img_arr.shape[1]} x {img_arr.shape[0]}")
    img_arr = normalize_slide(img_arr, stain_normalizer, name)

    # ROI polygons (if annotation exists)
    roi_polys = None
    exclude_polys = None
    ann_path = slide_cfg.get("annotation")
    if ann_path is not None:
        roi_polys, exclude_polys = load_roi_polygons(
            ann_path,
            coordinate_space="ratio",
            original_full_width=slide_cfg.get("original_full_width"),
            original_full_height=slide_cfg.get("original_full_height"),
            cropped_w=img_arr.shape[1],
            cropped_h=img_arr.shape[0],
        )
        print(f"  ROIs: {len(roi_polys)} include, {len(exclude_polys)} exclude polygons")
    else:
        print(f"  No annotation — using full slide")

    # Patches
    patches, coords = get_patches_from_array(
        img_arr,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        image_name=name,
        roi_polygons=roi_polys,
        exclude_polygons=exclude_polys,
        min_roi_coverage=MIN_ROI_COVERAGE,
    )
    if len(patches) < 50:
        print(f"  SKIP: only {len(patches)} patches — too few for trajectory")
        return None

    # Features
    print(f"  Extracting {MODEL} features ({len(patches)} patches)...")
    features = extract_features(patches, model_name=MODEL)

    # PCA + UMAP + cluster
    scaler, pca, X_pca = fit_pca(features, variance_target=0.95)
    _, X_umap = run_umap(X_pca)
    labels = cluster(X_pca, method="leiden", resolution=leiden_resolution)

    n_clusters = len(set(labels) - {-1})
    print(f"  Clusters: {n_clusters}")
    if n_clusters < 2:
        print(f"  SKIP: need at least 2 clusters for diffusion pseudotime")
        return None

    centroids = get_cluster_centroids(X_pca, labels)

    # Diffusion + DPT.
    # Per-slide we don't get to inspect cluster patches before picking a root,
    # so use cluster 0. Re-run with --leiden_resolution if it looks weird.
    slide_ids = np.zeros(len(features), dtype=int)
    adata = build_adata(X_pca, labels, slide_ids, X_umap)
    compute_diffusion_map(adata, n_neighbors=min(30, len(features) - 1), n_comps=10)

    valid_clusters = sorted([c for c in set(labels) if c != -1])
    root_cluster = str(valid_clusters[0])
    print(f"  Root cluster: {root_cluster} (auto)")

    compute_dpt(adata, root_cluster=root_cluster)
    pseudotime = adata.obs["pseudotime"].values

    # Visuals

    # 1) Pseudotime heatmap overlay on cropped image
    plot_pseudotime_heatmap_overlay(
        slide_cfg["image"], coords, pseudotime,
        patch_size=PATCH_SIZE, stride=STRIDE,
        save_path=out_dir / "pseudotime_heatmap.png",
    )

    # 2) Top 3 diffusion components, colored by pseudotime
    if "X_diffmap" in adata.obsm and adata.obsm["X_diffmap"].shape[1] >= 3:
        viz.plot_3d_manifold(
            adata.obsm["X_diffmap"], pseudotime,
            out_dir / "diffusion_3d.png",
            title=f"3D Diffusion Manifold — {name}",
        )

    # 3) UMAP — clusters and pseudotime
    if X_umap is not None:
        viz.plot_umap_clusters(
            X_umap, labels, out_dir / "umap_clusters.png",
            title=f"UMAP Clusters — {name}",
        )
        viz.plot_umap_pseudotime(
            X_umap, pseudotime, out_dir / "umap_pseudotime.png",
            title=f"UMAP Pseudotime — {name}",
        )

    # 4) Cluster patch grid (essential for sanity-checking root choice)
    if len(centroids) > 0:
        viz.plot_cluster_patch_grid(
            patches, labels, centroids,
            out_dir / "cluster_patches.png",
        )

    # 5) Pseudotime violins by cluster
    viz.plot_pseudotime_violins(
        pseudotime, labels, out_dir / "pseudotime_violins.png",
    )

    # 6) Pseudotime histogram
    plot_pseudotime_histogram(pseudotime, out_dir / "pseudotime_histogram.png")

    # 7) Spatial cluster scatter (alongside the heatmap, easy to compare)
    viz.plot_spatial_clusters(coords, labels, slide_ids, out_dir, prefix="spatial_clusters")

    # CSV results
    import pandas as pd
    df = pd.DataFrame({
        "slide_name": name,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": labels,
        "pseudotime": pseudotime,
    })
    df.to_csv(out_dir / "results.csv", index=False)

    print(f"  Saved: {out_dir}")

    return {
        "name": name,
        "n_patches": int(len(patches)),
        "n_clusters": int(n_clusters),
        "pt_mean": float(pseudotime.mean()),
        "pt_std": float(pseudotime.std()),
        "root_cluster": root_cluster,
    }


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Per-slide pseudotime runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cancer_trajectory_atlas.run_individual                    # all slides
  python -m cancer_trajectory_atlas.run_individual --slide 6027       # single slide
  python -m cancer_trajectory_atlas.run_individual --png-dir ~/scratch/data/png
  python -m cancer_trajectory_atlas.run_individual --leiden-resolution 1.0
  python -m cancer_trajectory_atlas.run_individual --stain-method macenko
        """,
    )
    
    # Filter and output
    parser.add_argument("--slide", default=None,
                        help="Substring filter — only run slides whose name contains this.")
    
    # Paths
    default_paths = _load_default_paths()
    parser.add_argument("--png-dir", type=Path, default=default_paths["png_dir"],
                        help=f"Directory with cropped PNG slides (default: {default_paths['png_dir']})")
    parser.add_argument("--annotation-dir", type=Path, default=default_paths["annotation_dir"],
                        help=f"Directory with annotation JSON files (default: {default_paths['annotation_dir']})")
    parser.add_argument("--output-dir", type=Path, default=default_paths["output_dir"],
                        help=f"Output directory for results (default: {default_paths['output_dir']})")
    
    # Pipeline settings
    parser.add_argument("--model", type=str, default="phikon", choices=["phikon", "resnet50"],
                        help="Feature extraction model (default: phikon)")
    parser.add_argument("--patch-size", type=int, default=112,
                        help="Patch size in pixels (default: 112)")
    parser.add_argument("--stride", type=int, default=96,
                        help="Stride between patches (default: 96)")
    parser.add_argument("--leiden-resolution", type=float, default=0.5,
                        help="Leiden resolution (higher=more clusters; default: 0.5)")
    parser.add_argument("--stain-method", type=str, default="reinhard",
                        choices=["reinhard", "macenko", "none"],
                        help="Stain normalization method (default: reinhard)")
    parser.add_argument("--ndpi-scale", type=float, default=1.0,
                        help="Downscale factor used when PNGs were created (default: 1.0)")
    parser.add_argument("--min-roi-coverage", type=float, default=None,
                        help="Minimum fraction of a patch inside its ROI polygon (3x3 grid). "
                             "Default: None. Use 0.5 to drop boundary patches.")

    args = parser.parse_args()

    # Set global variables from CLI arguments
    global PNG_DIR, ANNOTATION_DIR, OUTPUT_DIR, MODEL, PATCH_SIZE, STRIDE, LEIDEN_RESOLUTION, STAIN_NORMALIZATION, MIN_ROI_COVERAGE
    PNG_DIR = args.png_dir
    ANNOTATION_DIR = args.annotation_dir
    OUTPUT_DIR = args.output_dir
    MODEL = args.model
    PATCH_SIZE = args.patch_size
    STRIDE = args.stride
    LEIDEN_RESOLUTION = args.leiden_resolution
    STAIN_NORMALIZATION = args.stain_method
    MIN_ROI_COVERAGE = args.min_roi_coverage

    slides = discover_slides(ndpi_scale=args.ndpi_scale, filter_name=args.slide)
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning per-slide pseudotime on {len(slides)} slides")
    print(f"Output: {out_root}\n")

    # Build the stain normalizer once, using the first slide as reference.
    # Same convention as run_all.py — keeps results comparable to the pooled atlas.
    from .data.stain_normalization import build_normalizer
    print(f"Building {STAIN_NORMALIZATION} normalizer (ref: {slides[0]['name']})")
    stain_normalizer = build_normalizer(STAIN_NORMALIZATION, slides[0]["image"])

    t_start = time.time()
    summaries = []

    for slide_cfg in slides:
        try:
            summary = run_one_slide(
                slide_cfg, stain_normalizer, out_root,
                leiden_resolution=LEIDEN_RESOLUTION,
            )
            if summary is not None:
                summaries.append(summary)
        except Exception as e:
            print(f"  ERROR on {slide_cfg['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary CSV — one row per slide
    if summaries:
        import pandas as pd
        pd.DataFrame(summaries).to_csv(out_root / "summary.csv", index=False)
        print(f"\nSummary: {out_root / 'summary.csv'}")

    elapsed = time.time() - t_start
    print(f"\nDone! ({elapsed / 60:.1f} min, {len(summaries)}/{len(slides)} slides)")


if __name__ == "__main__":
    main()
