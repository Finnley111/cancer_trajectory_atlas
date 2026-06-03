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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter

Image.MAX_IMAGE_PIXELS = None

from .data.slide_registry import KNOWN_NDPI_DIMENSIONS


# ── Configuration ──────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent
_PATHS_FILE = _SCRIPT_DIR / "paths.json"


@dataclass
class IndividualConfig:
    png_dir: Path
    annotation_dir: Path
    output_dir: Path
    model: str = "phikon"
    patch_size: int = 112
    stride: int = 96
    leiden_resolution: float = 0.5
    stain_method: str = "reinhard"
    ndpi_scale: float = 1.0
    min_roi_coverage: Optional[float] = None
    root_cluster: Optional[str] = None


def _load_default_paths() -> dict:
    if _PATHS_FILE.exists():
        with open(_PATHS_FILE) as f:
            config = json.load(f)
        return {
            "png_dir": Path(config["cropped_png"]).expanduser(),
            "annotation_dir": Path(config["annotations"]).expanduser(),
            "output_dir": Path(config["results"]).expanduser() / "individual_pseudotime_runs",
        }
    return {
        "png_dir": _SCRIPT_DIR / "data" / "MCF7_x5_cropped",
        "annotation_dir": _SCRIPT_DIR / "data" / "annotations_ratio",
        "output_dir": _SCRIPT_DIR.parent / "individual_pseudotime_runs",
    }


# ── Slide discovery ─────────────────────────────────────────────────────────

def _get_known_dimensions(png_name: str, ndpi_scale: float):
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


def discover_slides(cfg: IndividualConfig, filter_name: Optional[str] = None) -> list:
    png_dir = Path(cfg.png_dir)
    ann_dir = Path(cfg.annotation_dir)

    if not png_dir.exists():
        print(f"ERROR: PNG directory not found: {png_dir}")
        sys.exit(1)

    png_files = sorted(png_dir.glob("*.png"))
    if not png_files:
        print(f"ERROR: No PNG files found in {png_dir}")
        sys.exit(1)

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
            fw, fh = _get_known_dimensions(png_path.name, cfg.ndpi_scale)

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


# ── Visualization helpers ───────────────────────────────────────────────────

def plot_pseudotime_heatmap_overlay(
    image_path, coords, pseudotime, patch_size, stride, save_path,
    alpha=0.55, smooth_sigma=1.0,
):
    """Overlay per-patch pseudotime on the cropped slide image."""
    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]

    half = patch_size // 2
    cx = coords[:, 0] + half
    cy = coords[:, 1] + half

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

    if smooth_sigma > 0:
        weighted = gaussian_filter(grid * valid.astype(float), sigma=smooth_sigma)
        weights = gaussian_filter(valid.astype(float), sigma=smooth_sigma)
        grid = np.where(weights > 1e-6, weighted / np.maximum(weights, 1e-6), 0.0)

    zh, zw = H / gh, W / gw
    pt_full = zoom(grid, (zh, zw), order=1)
    mask_full = zoom(valid.astype(float), (zh, zw), order=0) > 0.5
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


# ── Per-slide runner ────────────────────────────────────────────────────────

def run_one_slide(slide_cfg: dict, stain_normalizer, out_root: Path,
                  cfg: IndividualConfig) -> Optional[dict]:
    """Compute pseudotime for a single slide and save all visuals."""

    from .data.stain_normalization import normalize_slide
    from .features.patching import get_patches_from_array, load_roi_polygons
    from .features.extractors import extract_features
    from .analysis.clustering import fit_pca, run_umap, cluster, get_cluster_centroids
    from .analysis.diffusion import build_adata, compute_diffusion_map, compute_dpt
    from .utils import viz

    name = slide_cfg["name"]
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    img = Image.open(slide_cfg["image"]).convert("RGB")
    img_arr = np.array(img)
    print(f"  Image: {img_arr.shape[1]} x {img_arr.shape[0]}")
    img_arr = normalize_slide(img_arr, stain_normalizer, name)

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

    patches, coords = get_patches_from_array(
        img_arr,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        image_name=name,
        roi_polygons=roi_polys,
        exclude_polygons=exclude_polys,
        min_roi_coverage=cfg.min_roi_coverage,
    )
    if len(patches) < 50:
        print(f"  SKIP: only {len(patches)} patches — too few for trajectory")
        return None

    print(f"  Extracting {cfg.model} features ({len(patches)} patches)...")
    features = extract_features(patches, model_name=cfg.model)

    scaler, pca, X_pca = fit_pca(features, variance_target=0.95)
    _, X_umap = run_umap(X_pca)
    labels = cluster(X_pca, method="leiden", resolution=cfg.leiden_resolution)

    n_clusters = len(set(labels) - {-1})
    print(f"  Clusters: {n_clusters}")
    if n_clusters < 2:
        print(f"  SKIP: need at least 2 clusters for diffusion pseudotime")
        return None

    centroids = get_cluster_centroids(X_pca, labels)

    slide_ids = np.zeros(len(features), dtype=int)
    adata = build_adata(X_pca, labels, slide_ids, X_umap)
    compute_diffusion_map(adata, n_neighbors=min(30, len(features) - 1), n_comps=10)

    valid_clusters = sorted([c for c in set(labels) if c != -1])
    if cfg.root_cluster is not None:
        root_cluster = str(cfg.root_cluster)
        if int(root_cluster) not in valid_clusters:
            print(f"  WARNING: --root-cluster {root_cluster} not in {valid_clusters}; using cluster 0.")
            root_cluster = str(valid_clusters[0])
        else:
            print(f"  Root cluster: {root_cluster} (from --root-cluster)")
    else:
        root_cluster = str(valid_clusters[0])
        print(f"  Root cluster: {root_cluster} (auto — re-run with --root-cluster N)")

    compute_dpt(adata, root_cluster=root_cluster)
    pseudotime = adata.obs["pseudotime"].values

    plot_pseudotime_heatmap_overlay(
        slide_cfg["image"], coords, pseudotime,
        patch_size=cfg.patch_size, stride=cfg.stride,
        save_path=out_dir / "pseudotime_heatmap.png",
    )

    if "X_diffmap" in adata.obsm and adata.obsm["X_diffmap"].shape[1] >= 3:
        viz.plot_3d_manifold(
            adata.obsm["X_diffmap"], pseudotime,
            out_dir / "diffusion_3d.png",
            title=f"3D Diffusion Manifold — {name}",
        )

    if X_umap is not None:
        viz.plot_umap_clusters(X_umap, labels, out_dir / "umap_clusters.png",
                               title=f"UMAP Clusters — {name}")
        viz.plot_umap_pseudotime(X_umap, pseudotime, out_dir / "umap_pseudotime.png",
                                 title=f"UMAP Pseudotime — {name}")

    if len(centroids) > 0:
        viz.plot_cluster_patch_grid(patches, labels, centroids,
                                    out_dir / "cluster_patches.png")

    viz.plot_pseudotime_violins(pseudotime, labels, out_dir / "pseudotime_violins.png")
    plot_pseudotime_histogram(pseudotime, out_dir / "pseudotime_histogram.png")
    viz.plot_spatial_clusters(coords, labels, slide_ids, out_dir, prefix="spatial_clusters",
                              slide_name_map={0: name}, pseudotime=pseudotime)

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


# ── Entry point ─────────────────────────────────────────────────────────────

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

    parser.add_argument("--slide", default=None,
                        help="Substring filter — only run slides whose name contains this.")

    default_paths = _load_default_paths()
    parser.add_argument("--png-dir", type=Path, default=default_paths["png_dir"])
    parser.add_argument("--annotation-dir", type=Path, default=default_paths["annotation_dir"])
    parser.add_argument("--output-dir", type=Path, default=default_paths["output_dir"])
    parser.add_argument("--model", type=str, default="phikon", choices=["phikon", "resnet50"])
    parser.add_argument("--patch-size", type=int, default=112)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--leiden-resolution", type=float, default=0.5)
    parser.add_argument("--stain-method", type=str, default="reinhard",
                        choices=["reinhard", "macenko", "none"])
    parser.add_argument("--ndpi-scale", type=float, default=1.0)
    parser.add_argument("--min-roi-coverage", type=float, default=None,
                        help="Minimum fraction of a patch inside its ROI polygon (3x3 grid). "
                             "Default: None. Use 0.5 to drop boundary patches.")
    parser.add_argument("--root-cluster", type=int, default=None,
                        help="Cluster to use as pseudotime root. "
                             "Default: None (auto cluster 0). Re-run after inspecting cluster_patches.png.")

    args = parser.parse_args()

    cfg = IndividualConfig(
        png_dir=args.png_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        model=args.model,
        patch_size=args.patch_size,
        stride=args.stride,
        leiden_resolution=args.leiden_resolution,
        stain_method=args.stain_method,
        ndpi_scale=args.ndpi_scale,
        min_roi_coverage=args.min_roi_coverage,
        root_cluster=str(args.root_cluster) if args.root_cluster is not None else None,
    )

    slides = discover_slides(cfg, filter_name=args.slide)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning per-slide pseudotime on {len(slides)} slides")
    print(f"Output: {out_root}\n")

    from .data.stain_normalization import build_normalizer
    print(f"Building {cfg.stain_method} normalizer (ref: {slides[0]['name']})")
    stain_normalizer = build_normalizer(cfg.stain_method, slides[0]["image"])

    t_start = time.time()
    summaries = []

    for slide_cfg in slides:
        try:
            summary = run_one_slide(slide_cfg, stain_normalizer, out_root, cfg)
            if summary is not None:
                summaries.append(summary)
        except Exception as e:
            import traceback
            print(f"  ERROR on {slide_cfg['name']}: {e}")
            traceback.print_exc()
            continue

    if summaries:
        import pandas as pd
        pd.DataFrame(summaries).to_csv(out_root / "summary.csv", index=False)
        print(f"\nSummary: {out_root / 'summary.csv'}")

    elapsed = time.time() - t_start
    print(f"\nDone! ({elapsed / 60:.1f} min, {len(summaries)}/{len(slides)} slides)")


if __name__ == "__main__":
    main()
