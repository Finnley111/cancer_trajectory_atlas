#!/usr/bin/env python3
"""Convert NDPI slides and run the full atlas pipeline on all slides.

Usage:
    python -m cancer_trajectory_atlas.run_all --convert
    python -m cancer_trajectory_atlas.run_all --run
    python -m cancer_trajectory_atlas.run_all --convert --run
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from .pipeline_config import PipelineConfig
from .data.slide_registry import KNOWN_NDPI_DIMENSIONS


# Default configuration loader

_SCRIPT_DIR = Path(__file__).parent
_PATHS_FILE = _SCRIPT_DIR / "paths.json"

def _load_default_paths():
    """Load paths from paths.json or use local development paths."""
    if _PATHS_FILE.exists():
        with open(_PATHS_FILE) as f:
            config = json.load(f)
        return {
            "ndpi_dir": Path(config["raw_ndpi"]).expanduser(),
            "png_dir": Path(config["cropped_png"]).expanduser(),
            "annotation_dir": Path(config["annotations"]).expanduser(),
            "output_dir": Path(config["results"]).expanduser() / "atlas_full",
        }
    else:
        return {
            "ndpi_dir": _SCRIPT_DIR / "data" / "MCF7_x5",
            "png_dir": _SCRIPT_DIR / "data" / "MCF7_x5_cropped",
            "annotation_dir": _SCRIPT_DIR / "data" / "annotations",
            "output_dir": _SCRIPT_DIR.parent / "results" / "atlas_full",
        }


# NDPI to PNG conversion

def convert_ndpi_to_left_half_png(cfg: PipelineConfig):
    """
    Convert all NDPI files to PNG, keeping only the left half.
    Your NDPIs contain two copies of the same slide side by side —
    annotations were done on the left, so we discard the right.
    """
    ndpi_dir = Path(cfg.ndpi_dir)
    out_dir = Path(cfg.png_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import openslide
    except ImportError:
        print("ERROR: openslide-python is required for NDPI conversion.")
        print("  pip install openslide-python")
        print("  (Also need system lib: apt install libopenslide0 on Ubuntu)")
        sys.exit(1)

    ndpi_files = sorted(ndpi_dir.glob("*.ndpi"))
    if not ndpi_files:
        print(f"No .ndpi files found in {ndpi_dir}")
        sys.exit(1)

    print(f"\nFound {len(ndpi_files)} NDPI files in {ndpi_dir}")
    print(f"Output: {out_dir}")
    print(f"Level: {cfg.ndpi_level}, Scale: {cfg.ndpi_scale}")
    print("=" * 60)

    dimensions_log = {}

    for ndpi_path in ndpi_files:
        out_name = f"{ndpi_path.stem}_x5.png"
        out_path = out_dir / out_name

        # Read dimensions even if the PNG already exists.
        slide = openslide.OpenSlide(str(ndpi_path))
        dims = slide.level_dimensions[cfg.ndpi_level]

        if out_path.exists():
            print(f"\n  SKIP (exists): {out_name}")
            slide.close()
        else:
            print(f"\n  Converting: {ndpi_path.name}")
            print(f"    Full size: {dims[0]} x {dims[1]}")

            img = slide.read_region((0, 0), cfg.ndpi_level, dims).convert("RGB")
            slide.close()

            if cfg.ndpi_scale != 1.0:
                new_size = (int(dims[0] * cfg.ndpi_scale), int(dims[1] * cfg.ndpi_scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"    Scaled to: {new_size[0]} x {new_size[1]}")

            full_w, full_h = img.size
            img_left = img.crop((0, 0, full_w // 2, full_h))
            print(f"    Left half: {img_left.size[0]} x {img_left.size[1]}")

            img_left.save(str(out_path), "PNG", compress_level=6)
            print(f"    Saved: {out_name}")

        # Log dimensions for later ratio-to-pixel conversion.
        full_w = int(dims[0] * cfg.ndpi_scale) if cfg.ndpi_scale != 1.0 else dims[0]
        full_h = int(dims[1] * cfg.ndpi_scale) if cfg.ndpi_scale != 1.0 else dims[1]
        dimensions_log[out_name] = {
            "original_full_width": full_w,
            "original_full_height": full_h,
            "cropped_width": full_w // 2,
            "cropped_height": full_h,
        }

    # Save dimensions sidecar
    dims_path = out_dir / "slide_dimensions.json"
    with open(dims_path, "w") as f:
        json.dump(dimensions_log, f, indent=2)
    print(f"\n  Dimensions log: {dims_path}")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"PNGs in: {out_dir}")


# Pipeline setup

def _get_known_dimensions(png_name, ndpi_scale):
    """Look up original NDPI dimensions from the fallback table."""
    stem = Path(png_name).stem              # "6027-4L-2M-1_x5"
    base_stem = stem.replace("_x5", "")     # "6027-4L-2M-1"

    dims = KNOWN_NDPI_DIMENSIONS.get(base_stem)
    if dims is not None:
        full_w, full_h = dims
        if ndpi_scale != 1.0:
            full_w = int(full_w * ndpi_scale)
            full_h = int(full_h * ndpi_scale)
        return full_w, full_h
    return None, None


def discover_slides(cfg: PipelineConfig):
    """Discover slides, match annotations, and load original dimensions."""
    png_dir = Path(cfg.png_dir)
    ann_dir = Path(cfg.annotation_dir)

    if not png_dir.exists():
        print(f"ERROR: PNG directory not found: {png_dir}")
        print("  Run with --convert first, or create PNGs manually.")
        sys.exit(1)

    png_files = sorted(png_dir.glob("*.png"))
    if not png_files:
        print(f"ERROR: No PNG files found in {png_dir}")
        sys.exit(1)

    # Try loading the sidecar first.
    dims_path = png_dir / "slide_dimensions.json"
    dims_log = {}
    if dims_path.exists():
        with open(dims_path) as f:
            dims_log = json.load(f)
        print(f"  Loaded slide_dimensions.json ({len(dims_log)} entries)")
    else:
        print(f"  No slide_dimensions.json found — using hardcoded NDPI dimensions")

    slides = []
    for png_path in png_files:
        slide_entry = {"image": str(png_path)}

        # Match annotation file.
        stem = png_path.stem                    # e.g. "6027-4L-2M-1_x5"
        base_stem = stem.replace("_x5", "")     # e.g. "6027-4L-2M-1"

        ann_path = None
        for candidate in [
            ann_dir / f"{stem}.json",
            ann_dir / f"{base_stem}.json",
            ann_dir / f"{stem}.geojson",
            ann_dir / f"{base_stem}.geojson",
        ]:
            if candidate.exists():
                ann_path = str(candidate)
                break

        slide_entry["annotation"] = ann_path

        # Look up original dimensions: sidecar first, then the fallback table.
        sidecar_dims = dims_log.get(png_path.name, {})
        if sidecar_dims:
            slide_entry["original_full_width"] = sidecar_dims["original_full_width"]
            slide_entry["original_full_height"] = sidecar_dims["original_full_height"]
        else:
            fw, fh = _get_known_dimensions(png_path.name, cfg.ndpi_scale)
            slide_entry["original_full_width"] = fw
            slide_entry["original_full_height"] = fh

        slides.append(slide_entry)

    # Report discovered slides.
    n_ann = sum(1 for s in slides if s["annotation"] is not None)
    n_dims = sum(1 for s in slides if s["original_full_width"] is not None)
    print(f"\nDiscovered {len(slides)} slides "
          f"({n_ann} with annotations, {n_dims} with original dims)")
    for s in slides:
        ann_str = "✓ ann" if s["annotation"] else "✗ ann"
        dim_str = (f"full={s['original_full_width']}x{s['original_full_height']}"
                   if s["original_full_width"] else "NO DIMS")
        print(f"  {ann_str}  {dim_str}  {Path(s['image']).name}")

    # Warn if annotated slides are missing dimensions.
    missing = [s for s in slides
               if s["annotation"] is not None and s["original_full_width"] is None]
    if missing:
        print(f"\n  WARNING: {len(missing)} annotated slides have no original dimensions!")
        print(f"  Ratio annotation coordinates will be INCORRECT for these slides.")
        print(f"  Fix: run --convert, or add entries to KNOWN_NDPI_DIMENSIONS.")

    # Apply subset filter if provided.
    if cfg.slide_filter:
        total = len(slides)
        requested = set(cfg.slide_filter)
        found = {Path(s["image"]).stem for s in slides}
        not_found = requested - found
        if not_found:
            print(f"\nERROR: {len(not_found)} requested slide(s) not found in {png_dir}:")
            for name in sorted(not_found):
                print(f"  {name}")
            sys.exit(1)
        slides = [s for s in slides if Path(s["image"]).stem in requested]
        print(f"\n  Subset mode: running on {len(slides)} of {total} slides")

    return slides


def run_pipeline(cfg: PipelineConfig):
    """Run the full pipeline on all discovered slides."""
    slides = discover_slides(cfg)

    if len(slides) == 0:
        return

    output_dir = Path(cfg.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if cfg.use_harmony and "harmony" not in str(output_dir).lower():
        print(f"  NOTE: --harmony is active but 'harmony' is not in the output dir name.")
        print(f"  Output: {output_dir}")
        print(f"  Consider passing --output-dir with a distinct name to avoid clobbering.\n")

    t_start = time.time()

    # Imports (relative to this module)
    from .data.stain_normalization import build_normalizer, normalize_slide
    from .features.patching import get_patches_from_array, load_roi_polygons, sample_patches
    from .features.extractors import extract_features
    from .analysis.clustering import (
        fit_pca, run_umap, cluster, check_slide_independence, get_cluster_centroids,
    )
    from .analysis.diffusion import (
        build_adata, compute_diffusion_map, compute_dpt_multi_root,
    )
    from .validation.morphological_features import compute_morphological_features
    from .validation.correlations import run_full_validation
    from .utils import viz, io

    # Stain normalization
    print(f"\n{'='*60}")
    print("PHASE 1: Stain Normalization")
    print(f"{'='*60}")

    reference_image = slides[0]["image"]
    stain_normalizer = build_normalizer(cfg.stain_method, reference_image)

    if stain_normalizer is not None:
        import shutil
        shutil.copy2(reference_image, output_dir / "stain_reference.png")

    # Patch extraction and features
    print(f"\n{'='*60}")
    print("PHASE 2: Patch Extraction & Feature Embedding")
    print(f"{'='*60}")

    all_patches_list = []
    all_coords_list = []
    all_slide_ids = []
    slide_names = []
    sampling_log_rows = []
    sampling_indices = {}

    # Per-slide feature cache support: model loaded lazily on first cache miss.
    all_features_list = []
    _cache_model = _cache_processor = _cache_device = None
    _cache_model_loaded = False

    if cfg.features_cache_dir:
        Path(cfg.features_cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"  Feature cache enabled: {cfg.features_cache_dir}")

    # ── Pass 1: Extract & Cache (full uncapped features per slide) ────────────
    # Sampling is deferred to Pass 2 so the cohort median can be computed first.
    slide_data = []

    for i, slide_entry in enumerate(slides):
        image_path = slide_entry["image"]
        slide_name = Path(image_path).stem
        slide_names.append(slide_name)

        print(f"\n  [{i+1}/{len(slides)}] {slide_name}")

        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img)
        print(f"    Image size: {img_arr.shape[1]} x {img_arr.shape[0]}")

        img_arr = normalize_slide(img_arr, stain_normalizer, slide_name)

        # Load ROI polygons if annotations exist.
        roi_polys = None
        exclude_polys = None
        ann_path = slide_entry.get("annotation")
        if ann_path is not None:
            cropped_w, cropped_h = img_arr.shape[1], img_arr.shape[0]
            roi_polys, exclude_polys = load_roi_polygons(
                ann_path,
                coordinate_space="ratio",
                original_full_width=slide_entry.get("original_full_width"),
                original_full_height=slide_entry.get("original_full_height"),
                cropped_w=cropped_w,
                cropped_h=cropped_h,
            )
            print(f"    Loaded {len(roi_polys)} ROI hotspot polygons, "
                  f"{len(exclude_polys)} exclusion polygons")
        else:
            print(f"    No annotation — using full slide")

        patches, coords = get_patches_from_array(
            img_arr,
            patch_size=cfg.patch_size,
            stride=cfg.stride,
            image_name=slide_name,
            roi_polygons=roi_polys,
            exclude_polygons=exclude_polys,
            min_roi_coverage=cfg.min_roi_coverage,
        )

        if len(patches) == 0:
            print(f"    WARNING: No patches found — skipping")
            continue

        orig_count = len(patches)

        # Cache always stores full (uncapped) features — cap is applied in Pass 2
        # after the cohort median is known. Do not sample here.
        slide_feats = None
        if cfg.features_cache_dir:
            cache_file = Path(cfg.features_cache_dir) / f"{slide_name}_features.npy"
            if cache_file.exists():
                print(f"    Cache hit: {cache_file.name}")
                slide_feats = np.load(cache_file)
                if len(slide_feats) != orig_count:
                    raise RuntimeError(
                        f"Feature cache size mismatch for '{slide_name}': "
                        f"cache has {len(slide_feats)} features but {orig_count} patches were extracted.\n"
                        "The cache is stale (patch extraction settings changed). "
                        "Delete the stale .npy file and re-run with GPU to rebuild it, "
                        "or re-run run_cache_population.sh with the current extraction settings."
                    )
            else:
                from .features.extractors import load_model_components, extract_features_from_model
                if not _cache_model_loaded:
                    print(f"    Pre-loading {cfg.model} model...")
                    _cache_model, _cache_processor, _cache_device = load_model_components(cfg.model)
                    _cache_model_loaded = True
                slide_feats = extract_features_from_model(
                    patches, _cache_model, _cache_processor, _cache_device
                )
                np.save(cache_file, slide_feats)
                print(f"    Saved to cache: {cache_file.name}")

        slide_data.append({
            "slide_idx": i,
            "slide_name": slide_name,
            "patches": patches,
            "coords": coords,
            "slide_feats": slide_feats,
            "orig_count": orig_count,
        })

    if not slide_data:
        print("ERROR: No patches extracted from any slide!")
        return

    # ── Pass 2: Calculate active cap & sample ─────────────────────────────────
    # Shuffle slide order before sampling to prevent systematic ordering bias.
    np.random.default_rng(cfg.patch_sample_seed).shuffle(slide_data)

    if cfg.cap_strategy == "median":
        active_cap = int(np.median([d["orig_count"] for d in slide_data]))
        print(f"\n  Cap strategy: median → {active_cap} patches/slide "
              f"(computed across {len(slide_data)} slides)")
    elif cfg.cap_strategy == "fixed":
        active_cap = cfg.max_patches_per_slide
        print(f"\n  Cap strategy: fixed → {active_cap} patches/slide")
    else:  # 'none' — fall back to max_patches_per_slide for backward compat
        active_cap = cfg.max_patches_per_slide

    for d in slide_data:
        slide_name = d["slide_name"]
        slide_idx = d["slide_idx"]
        patches = d["patches"]
        coords = d["coords"]
        slide_feats = d["slide_feats"]
        orig_count = d["orig_count"]

        patches, coords, selected_idx = sample_patches(
            patches, coords, active_cap, cfg.patch_sample_seed, slide_name
        )
        was_capped = selected_idx is not None
        n_sampled = len(patches)
        if was_capped:
            print(f"    [{slide_name}] Sampled {n_sampled} / {orig_count} patches "
                  f"(cap={active_cap})")
        sampling_log_rows.append({
            "slide_id": slide_name,
            "patches_before_cap": orig_count,
            "patches_after_cap": n_sampled,
            "was_capped": was_capped,
            "seed": cfg.patch_sample_seed,
            "pct_retained": round(n_sampled / orig_count * 100, 1) if orig_count > 0 else 0.0,
        })
        if was_capped:
            sampling_indices[slide_name] = selected_idx

        all_patches_list.append(patches)
        all_coords_list.append(coords)
        all_slide_ids.extend([slide_idx] * len(patches))

        if slide_feats is not None:
            if selected_idx is not None:
                slide_feats = slide_feats[selected_idx]
            all_features_list.append(slide_feats)

    all_patches = np.concatenate(all_patches_list)
    all_coords = np.concatenate(all_coords_list)
    slide_ids = np.array(all_slide_ids)

    n_total = len(all_patches)
    pct_of_target = n_total / cfg.target_total * 100 if cfg.target_total else 0
    print(f"\n  Total: {n_total} patches from {len(slide_names)} slides "
          f"(target: {cfg.target_total}, {pct_of_target:.0f}%)")

    # Feature extraction (from cache or bulk inference)
    print(f"\n  Extracting {cfg.model} features...")
    if cfg.features_cache_dir and all_features_list:
        features = np.concatenate(all_features_list)
        print(f"  Loaded from cache ({len(all_features_list)} slides).")
    else:
        features = extract_features(all_patches, model_name=cfg.model)
    print(f"  Feature shape: {features.shape}")

    # Clustering
    print(f"\n{'='*60}")
    print("PHASE 3: Morphological Clustering")
    print(f"{'='*60}")

    scaler, pca, X_pca = fit_pca(features, variance_target=0.95)

    # Optional Harmony batch correction — applied in PCA space before clustering
    # and DPT so both use the same corrected representation.
    X_embed = X_pca
    if cfg.use_harmony:
        from .analysis.harmony import apply_harmony
        X_embed = apply_harmony(X_pca, slide_names, slide_ids, key=cfg.harmony_key)

    umap_reducer, X_umap = run_umap(X_embed)

    cluster_labels = cluster(
        X_embed,
        method=cfg.clustering_method,
        resolution=cfg.leiden_resolution,
    )

    slide_check = check_slide_independence(cluster_labels, slide_ids)
    centroids = get_cluster_centroids(X_embed, cluster_labels)

    # Clustering figures
    if X_umap is not None:
        viz.plot_umap_clusters(X_umap, cluster_labels, fig_dir / "fig1_umap_clusters.png")
        viz.plot_umap_by_slide(X_umap, slide_ids, fig_dir / "qc_umap_by_slide.png")

    if len(centroids) > 0:
        viz.plot_cluster_patch_grid(
            all_patches, cluster_labels, centroids,
            fig_dir / "fig2_cluster_patches.png",
        )

    # Diffusion pseudotime
    print(f"\n{'='*60}")
    print("PHASE 4: Diffusion Pseudotime")
    print(f"{'='*60}")

    adata = build_adata(X_embed, cluster_labels, slide_ids, X_umap)

    # Add slide-level metadata (useful for QC batch plots and harmony ablations)
    def _parse_section(name):
        parts = name.replace("_x5", "").split("-")   # ["6027", "4L", "2M", "1"]
        return f"{parts[-2]}-{parts[-1]}"             # "2M-1"

    adata.obs["mouse_id"]       = [slide_names[sid].split("-")[0] for sid in slide_ids]
    adata.obs["section_number"] = [_parse_section(slide_names[sid]) for sid in slide_ids]

    if cfg.use_harmony:
        adata.obsm["X_pca_original"] = X_pca.astype(np.float32)
        adata.obsm["X_pca_harmony"]  = X_embed.astype(np.float32)

    compute_diffusion_map(adata, n_neighbors=cfg.diffmap_neighbors, n_comps=cfg.diffmap_comps)

    # Pre-compute nuclear density for multi-root DPT root candidate selection.
    from .validation.morphological_features import compute_nuclear_density_quick
    print(f"  Computing nuclear density for {cfg.n_roots}-root DPT candidate selection...")
    t_nd = time.time()
    nuclear_density_quick = compute_nuclear_density_quick(all_patches)
    print(f"  Nuclear density done: {time.time() - t_nd:.1f}s")
    compute_dpt_multi_root(adata, nuclear_density_quick, n_roots=cfg.n_roots)

    pseudotime = adata.obs["pseudotime"].values
    pseudotime_std = adata.obs["pseudotime_std"].values

    # Diffusion figures
    if X_umap is not None:
        viz.plot_umap_pseudotime(X_umap, pseudotime, fig_dir / "fig4_umap_pseudotime.png")
        viz.plot_umap_pseudotime_std(X_umap, pseudotime_std,
                                     fig_dir / "fig4b_umap_pseudotime_std.png")
    viz.plot_pseudotime_violins(pseudotime, cluster_labels, fig_dir / "fig5_pt_violins.png")
    viz.plot_spatial_clusters(all_coords, cluster_labels, slide_ids, fig_dir,
                              slide_name_map=dict(enumerate(slide_names)),
                              pseudotime=pseudotime)

    if "X_diffmap" in adata.obsm and adata.obsm["X_diffmap"].shape[1] >= 3:
        viz.plot_3d_manifold(
            adata.obsm["X_diffmap"], pseudotime,
            fig_dir / "diffusion_3d.png",
        )

    # Validation
    print(f"\n{'='*60}")
    print("PHASE 5: Morphological Feature Validation")
    print(f"{'='*60}")

    morph_features = compute_morphological_features(
        all_patches, use_stardist=cfg.use_stardist,
    )

    for name, values in morph_features.items():
        adata.obs[name] = values

    validation = run_full_validation(
        pseudotime, morph_features, cluster_labels, all_coords,
        n_permutations=cfg.n_permutations,
    )

    viz.plot_feature_vs_pseudotime(
        pseudotime, morph_features,
        validation["feature_correlations"],
        fig_dir / "fig6_features_vs_pt.png",
    )
    viz.plot_permutation_nulls(
        validation["permutation_tests"],
        fig_dir / "fig7_permutation_nulls.png",
    )

    # Save everything
    print(f"\n{'='*60}")
    print("Saving Results")
    print(f"{'='*60}")

    import pandas as pd

    adata.write(output_dir / "adata_full.h5ad")
    print(f"  AnnData: {output_dir / 'adata_full.h5ad'}")

    df = pd.DataFrame({
        "x": all_coords[:, 0],
        "y": all_coords[:, 1],
        "slide_id": slide_ids,
        "slide_name": [slide_names[sid] for sid in slide_ids],
        "cluster": cluster_labels,
        "pseudotime": pseudotime,
        "pseudotime_std": pseudotime_std,
    })
    for name, values in morph_features.items():
        df[name] = values
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"  CSV: {output_dir / 'results.csv'}")

    if sampling_log_rows:
        sampling_df = pd.DataFrame(sampling_log_rows)
        sampling_df.to_csv(output_dir / "sampling_manifest.csv", index=False)
        print(f"  Sampling manifest: {output_dir / 'sampling_manifest.csv'}")
        if sampling_indices:
            sampling_dir = output_dir / "sampling"
            sampling_dir.mkdir(exist_ok=True)
            for sname, idx in sampling_indices.items():
                np.save(sampling_dir / f"{sname}_sample_idx.npy", idx)
            print(f"  Sampling indices: {sampling_dir}/")

    io.save_json(validation, output_dir / "validation.json")
    print(f"  Validation: {output_dir / 'validation.json'}")

    io.save_pickle(scaler, output_dir / "scaler.pkl")
    io.save_pickle(pca, output_dir / "pca.pkl")
    io.save_pickle(umap_reducer, output_dir / "umap_reducer.pkl")

    io.save_json(slide_check, output_dir / "slide_independence.json")

    # Save AtlasProjector for LOO projection analysis (Experiment 2)
    print("  Training and saving AtlasProjector...")
    from .analysis.projector import AtlasProjector
    projector = AtlasProjector.from_training(scaler, pca, umap_reducer, adata, centroids)
    projector.save(output_dir / "projector")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE! ({elapsed / 60:.1f} minutes)")
    print(f"{'='*60}")
    print(f"  Results:    {output_dir}")
    print(f"  Figures:    {fig_dir}")
    print(f"  Patches:    {len(all_patches)}")
    print(f"  Clusters:   {len(set(cluster_labels) - {-1})}")
    print(f"")
    print(f"  VERDICT: {validation['summary']['verdict']}")
    print(f"")
    print(f"  NEXT STEPS:")
    print(f"  1. Open {fig_dir / 'fig2_cluster_patches.png'}")
    print(f"     Identify the most organized/regular morphology cluster.")
    print(f"  2. Re-run with a different --leiden-resolution if clusters look off.")
    print(f"  3. Re-run to get biologically anchored pseudotime.")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cancer Trajectory Atlas — Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cancer_trajectory_atlas.run_all --convert
  python -m cancer_trajectory_atlas.run_all --run
  python -m cancer_trajectory_atlas.run_all --convert --run
  python -m cancer_trajectory_atlas.run_all --run --png-dir ~/scratch/data/png --output-dir ~/scratch/results
  python -m cancer_trajectory_atlas.run_all --run --model phikon --leiden-resolution 1.0 --stain-method macenko
        """,
    )

    # Pipeline steps
    parser.add_argument("--convert", action="store_true",
                        help="Convert NDPI files to left-half PNGs")
    parser.add_argument("--run", action="store_true",
                        help="Run the analysis pipeline")

    # Slide subset
    parser.add_argument("--slides", type=str, default=None,
                        help="Comma-separated slide stems to process, e.g. "
                             "6027-4L-2M-1_x5,6028-4L-2M-1_x5. "
                             "Default: all slides in --png-dir.")
    parser.add_argument("--slides-from-file", type=Path, default=None,
                        help="Text file with one slide stem per line. "
                             "Mutually exclusive with --slides.")

    # Path arguments (with intelligent defaults)
    default_paths = _load_default_paths()
    parser.add_argument("--ndpi-dir", type=Path, default=default_paths["ndpi_dir"],
                        help=f"Directory with NDPI files (default: {default_paths['ndpi_dir']})")
    parser.add_argument("--png-dir", type=Path, default=default_paths["png_dir"],
                        help=f"Directory for cropped PNGs (default: {default_paths['png_dir']})")
    parser.add_argument("--annotation-dir", type=Path, default=default_paths["annotation_dir"],
                        help=f"Directory with annotation JSON files (default: {default_paths['annotation_dir']})")
    parser.add_argument("--output-dir", type=Path, default=default_paths["output_dir"],
                        help=f"Output directory for results (default: {default_paths['output_dir']})")

    # NDPI conversion settings
    parser.add_argument("--ndpi-level", type=int, default=0,
                        help="NDPI pyramid level (0=full res; default: 0)")
    parser.add_argument("--ndpi-scale", type=float, default=1.0,
                        help="Additional downscale factor applied after level selection (default: 1.0)")

    # Pipeline settings
    parser.add_argument("--model", type=str, default="phikon", choices=["phikon", "resnet50"],
                        help="Feature extraction model (default: phikon)")
    parser.add_argument("--patch-size", type=int, default=112,
                        help="Patch size in pixels (default: 112)")
    parser.add_argument("--stride", type=int, default=96,
                        help="Stride between patches (default: 96)")
    parser.add_argument("--clustering-method", type=str, default="leiden",
                        choices=["leiden", "hdbscan", "kmeans"],
                        help="Clustering algorithm (default: leiden)")
    parser.add_argument("--leiden-resolution", type=float, default=0.5,
                        help="Leiden resolution (higher=more clusters; default: 0.5)")
    parser.add_argument("--stain-method", type=str, default="reinhard",
                        choices=["reinhard", "macenko", "none"],
                        help="Stain normalization method (default: reinhard)")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Number of permutations for validation (default: 1000)")
    parser.add_argument("--use-stardist", action="store_true",
                        help="Use StarDist for nuclear segmentation (slower, more accurate)")
    parser.add_argument("--harmony", action="store_true",
                        help="Apply Harmony batch correction after PCA (default: off)")
    parser.add_argument("--harmony-key", type=str, default="section_number",
                        choices=["slide_id", "section_number", "mouse_id"],
                        help="Batch grouping variable for Harmony (default: section_number)")
    parser.add_argument("--diffmap-neighbors", type=int, default=30,
                        help="k-NN neighbors for diffusion map (default: 30)")
    parser.add_argument("--diffmap-comps", type=int, default=10,
                        help="Number of diffusion map components (default: 10)")
    parser.add_argument("--features-cache-dir", type=Path, default=None,
                        help="Directory for per-slide Phikon feature cache (.npy). "
                             "Features are saved on first run and loaded on subsequent runs, "
                             "avoiding redundant GPU inference across LOO runs.")
    parser.add_argument("--cap-strategy", type=str, default="fixed",
                        choices=["none", "fixed", "median"],
                        help="Patch count cap strategy: 'fixed' = cap at "
                             "--max-patches-per-slide (default); 'median' = cap at cohort "
                             "median computed after full extraction; 'none' = use "
                             "--max-patches-per-slide if set (backward compat). "
                             "(default: fixed)")
    parser.add_argument("--max-patches-per-slide", type=int, default=200,
                        help="Per-slide patch cap (Vig et al. slide-aware sampling). "
                             "0 = no cap. Used when --cap-strategy is 'fixed' or 'none'. "
                             "(default: 200)")
    parser.add_argument("--target-total", type=int, default=3200,
                        help="Informational target for total patches across all slides. "
                             "Logged only — never used in sampling logic. (default: 3200)")
    parser.add_argument("--patch-sample-seed", type=int, default=42,
                        help="Base random seed for per-slide patch subsampling (default: 42). "
                             "Combined with a slide-name-derived hash for order independence.")
    parser.add_argument("--min-roi-coverage", type=float, default=None,
                        help="Minimum fraction of a patch that must lie inside an ROI polygon "
                             "(3x3 grid check). Default: None (centre-point only). "
                             "Use 0.5 to drop boundary patches that are mostly outside the annotation.")
    parser.add_argument("--root-cluster", type=int, default=None,
                        help="Legacy arg; unused with multi-root DPT.")
    parser.add_argument("--n-roots", type=int, default=20,
                        help="Number of root candidates for multi-root DPT averaging. "
                             "Candidates are the n patches with lowest nuclear density. "
                             "(default: 20)")
    parser.add_argument("--root-metric", type=str, default="cellularity",
                        choices=["cellularity"],
                        help="Metric used to rank root candidates. (default: cellularity)")

    args = parser.parse_args()

    # Resolve slide filter.
    if args.slides and args.slides_from_file:
        parser.error("--slides and --slides-from-file are mutually exclusive")

    slide_filter = None
    if args.slides:
        slide_filter = [s.strip() for s in args.slides.split(",") if s.strip()]
        if not slide_filter:
            parser.error("--slides was provided but contained no slide names")
    elif args.slides_from_file:
        if not args.slides_from_file.exists():
            parser.error(f"--slides-from-file path not found: {args.slides_from_file}")
        with open(args.slides_from_file) as f:
            slide_filter = [line.strip() for line in f if line.strip()]
        if not slide_filter:
            parser.error(f"--slides-from-file is empty: {args.slides_from_file}")

    if not args.convert and not args.run:
        parser.print_help()
        print("\n  Specify --convert, --run, or both.")
        sys.exit(0)

    cfg = PipelineConfig(
        ndpi_dir=args.ndpi_dir,
        png_dir=args.png_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        ndpi_level=args.ndpi_level,
        ndpi_scale=args.ndpi_scale,
        model=args.model,
        patch_size=args.patch_size,
        stride=args.stride,
        clustering_method=args.clustering_method,
        leiden_resolution=args.leiden_resolution,
        stain_method=args.stain_method,
        n_permutations=args.n_permutations,
        use_stardist=args.use_stardist,
        use_harmony=args.harmony,
        harmony_key=args.harmony_key,
        diffmap_neighbors=args.diffmap_neighbors,
        diffmap_comps=args.diffmap_comps,
        slide_filter=slide_filter,
        features_cache_dir=args.features_cache_dir,
        max_patches_per_slide=args.max_patches_per_slide,
        patch_sample_seed=args.patch_sample_seed,
        cap_strategy=args.cap_strategy,
        target_total=args.target_total,
        min_roi_coverage=args.min_roi_coverage,
        root_cluster=str(args.root_cluster) if args.root_cluster is not None else None,
        n_roots=args.n_roots,
        root_metric=args.root_metric,
    )

    if args.convert:
        convert_ndpi_to_left_half_png(cfg)

    if args.run:
        run_pipeline(cfg)
