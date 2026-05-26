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
    ndpi_scale: float = 0.5

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

    # Slide subset filter (None = all slides)
    slide_filter: list = None

    # Per-slide Phikon feature cache (None = disabled; set to a Path to enable)
    features_cache_dir: Path = None

    # Per-slide patch count cap (None or 0 = no cap; applied after ROI filter, before Phikon)
    max_patches_per_slide: Optional[int] = None
    patch_sample_seed: int = 42
