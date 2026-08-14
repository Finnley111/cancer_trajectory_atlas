"""Typed configuration object for the full atlas pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    # Required paths — no defaults, must be supplied via CLI
    png_dir: Path
    annotation_dir: Path
    output_dir: Path

    # NDPI conversion (only used with --convert)
    ndpi_dir: Path = None
    ndpi_level: int = 0
    ndpi_scale: float = 1.0

    # Feature embedding
    model: str = "phikon"
    patch_size: int = 112
    stride: int = 96

    # Clustering
    clustering_method: str = "leiden"
    leiden_resolution: float = 0.5

    # Diffusion pseudotime
    diffmap_neighbors: int = 30
    diffmap_comps: int = 10

    # Stain normalization
    stain_method: str = "reinhard"

    # Morphological validation
    n_permutations: int = 1000
    use_stardist: bool = False

    # Harmony batch correction
    use_harmony: bool = False
    harmony_key: str = "section_number"

    # Batch correction backend selector. None = fall back to use_harmony
    # (legacy behavior, keeps existing job scripts unchanged). "none",
    # "harmony", or "scvi" override use_harmony when explicitly set.
    batch_method: Optional[str] = None

    # scVI batch correction (alternative to Harmony). Operates on the same
    # post-PCA matrix Harmony receives, batch key is always section_number.
    scvi_n_latent: int = 30
    scvi_n_layers: int = 2
    scvi_n_hidden: int = 128
    scvi_max_epochs: int = 400

    # Slide subset filter (None = all slides)
    slide_filter: list = None

    # Per-slide Phikon feature cache (None = disabled; set to a Path to enable)
    features_cache_dir: Path = None

    # Per-slide patch count cap (Vig et al. slide-aware sampling).
    # cap_strategy controls how the cap is computed:
    #   'fixed'  — cap each slide at max_patches_per_slide (default)
    #   'median' — cap at cohort median patch count, computed after full extraction
    #   'none'   — use max_patches_per_slide if set (backward compat), else no cap
    # max_patches_per_slide=0 means no cap regardless of strategy.
    # target_total is informational only (logged; never used in sampling logic).
    max_patches_per_slide: Optional[int] = 200
    patch_sample_seed: int = 42
    cap_strategy: str = "median"
    target_total: int = 3200

    # ── v3 root/filter experiment (2M-1). All default to production behaviour ──
    # relaxed_tissue_filters=False and root_source="cellularity" reproduce every
    # run made before 2026-08-13; nothing on the production path sets either.
    #   relaxed_tissue_filters — disable BOTH the white and HSV tissue filters in
    #     features/patching.py. Changes the patch count, hence the PCA basis and
    #     every downstream number, so such a run is not comparable to a
    #     production run on absolute values.
    #   root_source="holeyness" — select DPT roots from expert-annotated per-duct
    #     hole %, via analysis/holeyness_roots.py, instead of nuclear density.
    #     Requires holeyness_export and holeyness_slide_dims.
    relaxed_tissue_filters: bool = False
    root_source: str = "cellularity"
    holeyness_export: Path = None
    holeyness_slide_dims: Path = None
    holeyness_percentile: float = 10.0
    holeyness_min_patches: int = 1
    #   holeyness_assignment="overlap" — assign patches to ducts by AREA OVERLAP
    #     instead of centre-in-polygon, which recovers the small/least-holey ducts
    #     the centre rule structurally drops (571/2173 = 26% previously).
    #   holeyness_max_roots_per_duct — >1 allows several roots per duct, filled
    #     round-robin so duct diversity is maximised first.
    #   holeyness_allow_degenerate_pool — by default a pool whose ducts all share
    #     the threshold hole % is a hard error, because then the arbitrary UUID
    #     tie-break, not holeyness, is picking the roots.
    holeyness_assignment: str = "centre"
    holeyness_overlap_min_fraction: float = 0.25
    holeyness_max_roots_per_duct: int = 1
    holeyness_allow_degenerate_pool: bool = False

    # Minimum fraction of the patch area that must lie inside an ROI polygon.
    # None = centre-point check only (original behaviour).
    # 0.5 = drop patches where more than half the area is outside the annotation.
    min_roi_coverage: Optional[float] = None

    # ── VESTIGIAL: root_cluster and root_metric are NO-OPS ────────────────────
    # Both are populated by run_all.py from --root-cluster / --root-metric and
    # then never read. Nothing on the --run path consumes either field; verified
    # by grep for `cfg.root_cluster` / `cfg.root_metric` (Phase 5, 2026-08-12).
    #
    # They predate multi-root DPT. Single-root DPT chose its origin as the patch
    # nearest a named cluster's centroid; compute_dpt_multi_root replaced that
    # with a fixed rule (lowest measured nuclear density), so neither field has
    # anything left to control. Setting them changes nothing about any output.
    #
    # Kept rather than removed so existing command lines and notes keep working.
    # Safe to delete once nothing references the flags — but note that
    # run_individual.py has its OWN, LIVE --root-cluster backed by
    # IndividualConfig.root_cluster. That one is real and must not be touched.
    # NOTE: field order below is deliberate and must not be rearranged — these
    # are dataclass fields, so their order defines PipelineConfig's positional
    # __init__ signature.
    root_cluster: Optional[str] = None

    # Multi-root DPT. n_roots candidates are the n patches with the lowest
    # MEASURED nuclear density (non-finite densities are excluded first); DPT is
    # run once per candidate and the per-patch results are median-aggregated.
    # This is the only root parameter that does anything.
    n_roots: int = 20

    # (vestigial — see the block above)
    root_metric: str = "cellularity"
