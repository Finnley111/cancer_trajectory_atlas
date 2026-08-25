"""Typed configuration object for the full atlas pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Every knob the ``--convert`` and ``--run`` paths read, in one object.

    Constructed in ``run_all.py`` from parsed CLI arguments, by keyword, and passed
    down unchanged. Nothing mutates it after construction.

    It is never serialised as a whole. A few individual settings do reach the
    output: ``active_cap.txt`` holds the effective cap, ``sampling_manifest.csv``
    holds ``patch_sample_seed``, and ``adata.uns`` holds ``dpt_root_source`` plus
    the root set (so ``n_roots`` is recoverable from its length).

    Everything else is unrecorded, including the group that decides what the
    feature cache contains: ``model``, ``patch_size``, ``stride`` and
    ``stain_method``. Those are exactly the parameters a cache directory must be
    consistent in, and nothing in a run directory states which ones it used. The
    job script and the SLURM log are the only provenance for them, so keep the
    scripts in ``jobs/`` alongside their results.

    Two things to know before editing this class.

    Field order is part of the public signature. These are dataclass fields, so
    their order defines the positional ``__init__``. Reordering them silently
    changes what a positional call means. Add new fields at the end of their
    section, and give anything after the first default a default of its own.

    Most defaults here are not the values production uses. ``run_all.py``'s argparse
    layer carries its own defaults and passes them explicitly, so a default here is
    the fallback for a direct constructor call, not a description of the reference
    run. Where the two disagree the argparse value is what shipped. ``stain_method``
    is the clearest case: it defaults to ``"reinhard"`` below, but every run since
    2026-05 passes ``--stain-method none``.
    """

    # Required. No defaults, so a caller cannot forget them.
    png_dir: Path
    annotation_dir: Path
    output_dir: Path

    # NDPI conversion (only used with --convert)
    ndpi_dir: Path = None
    ndpi_level: int = 0
    ndpi_scale: float = 1.0

    # Feature embedding. All three are passed explicitly by every job script, so
    # they are choices rather than defaults, but no note in the repo records WHY
    # 112 and 96. What can be said is the consequence: a stride below the patch
    # size makes adjacent patches overlap by 16 px, so they are not independent
    # samples, which matters for any test that assumes they are.
    #
    # TRAP: model, patch_size, stride, min_roi_coverage and stain_method together
    # determine what is in the feature cache, but the cache is keyed on slide name
    # alone. Changing any of them while pointing at an existing cache silently
    # reuses features computed under the old settings. See features/extractors.py.
    model: str = "phikon"
    patch_size: int = 112
    stride: int = 96

    # Clustering. leiden_resolution is genuinely chosen: the function default is
    # 1.0 and every job passes 0.5 explicitly.
    #
    # The Leiden graph's k (15) and metric (cosine) live in analysis/clustering.py
    # as function defaults. They are not fields here and not on the CLI, so they
    # cannot be varied without editing that module.
    clustering_method: str = "leiden"
    leiden_resolution: float = 0.5

    # Diffusion pseudotime. Both values are set here and neither is passed by the
    # reference job, so these defaults are what every recorded run used.
    #
    # The diffusion graph's METRIC is not represented here at all. There is no
    # field and no CLI flag for it, so it stays at scanpy's default of euclidean.
    # That was never a decision. It also makes the diffusion graph a different
    # graph from the Leiden one, which is cosine. See analysis/clustering.py.
    diffmap_neighbors: int = 30
    diffmap_comps: int = 10

    # Stain normalization. This default is NOT what production uses: every run
    # since 2026-05 passes --stain-method none. Reinhard survives here as the
    # original default and is only reachable by asking for it.
    stain_method: str = "reinhard"

    # Morphological validation. use_stardist=False segments nuclei by Otsu
    # thresholding, which is what every recorded run used.
    #
    # WARNING: StarDist is not in requirements.txt. If it is unavailable,
    # _segment_nuclei_stardist prints one line and returns Otsu output instead
    # (validation/morphological_features.py:114). A run launched with
    # --use-stardist on a machine without it therefore SUCCEEDS while producing
    # Otsu features, and nothing in the output directory records which segmenter
    # actually ran. Check the SLURM log before trusting a StarDist run.
    n_permutations: int = 1000
    use_stardist: bool = False

    # Harmony batch correction. Superseded by batch_method below, kept because
    # job scripts written before batch_method existed still set it.
    use_harmony: bool = False
    harmony_key: str = "section_number"

    # Batch correction backend selector, and the reason use_harmony still exists.
    # None means defer to use_harmony, which is what the older scripts rely on.
    # Setting this to "none", "harmony" or "scvi" overrides use_harmony outright.
    # Two switches for one decision is a migration that was left half-finished
    # rather than a design. Prefer batch_method in anything new.
    batch_method: Optional[str] = None

    # scVI batch correction, an alternative to Harmony. It operates on the same
    # post-PCA matrix Harmony receives, and its batch key is always
    # section_number regardless of harmony_key.
    #
    # WARNING: scVI expects a counts matrix and this feeds it standardized,
    # PCA'd Phikon activations. See analysis/scvi_integration.py for what that
    # costs and why the likelihood had to be changed.
    scvi_n_latent: int = 30
    scvi_n_layers: int = 2
    scvi_n_hidden: int = 128
    scvi_max_epochs: int = 400

    # Restrict the run to a subset of slides. None means every slide found in
    # png_dir. Populated from --slides, which takes a CSV of slide names.
    slide_filter: list = None

    # Per-slide Phikon feature cache. None disables caching; a Path enables it.
    # Extraction is the expensive stage, so the per_section runs share one cache
    # across both sections and every re-run.
    features_cache_dir: Path = None

    # Per-slide patch count cap (Vig et al. slide-aware sampling). Without a cap a
    # large slide contributes proportionally more patches, so it dominates the PCA
    # basis and the Leiden partition built on top of it.
    #
    # cap_strategy controls how the cap is computed:
    #   'fixed'   cap each slide at max_patches_per_slide
    #   'median'  cap at the cohort median patch count, computed only after every
    #             slide has been extracted, so it cannot be known in advance
    #   'none'    use max_patches_per_slide if set (backward compat), else no cap
    #
    # max_patches_per_slide=0 means no cap regardless of strategy.
    # target_total is informational only. It is logged and never read by the
    # sampling logic, so changing it cannot change a result.
    #
    # TRAP: passing --max-patches-per-slide without also passing
    # --cap-strategy fixed leaves the strategy at 'median' and the cap value inert.
    # jobs/run_all_capped.sh did exactly that and produced median-capped output
    # while claiming a 1900 cap. See archive/README.md.
    max_patches_per_slide: Optional[int] = 200
    patch_sample_seed: int = 42
    cap_strategy: str = "median"
    target_total: int = 3200

    # ── v3 root/filter experiment (2M-1) ─────────────────────────────────────
    # Both fields default to production behaviour. relaxed_tissue_filters=False
    # with root_source="cellularity" reproduces every run made before 2026-08-13,
    # and nothing on the production path sets either.
    #
    #   relaxed_tissue_filters
    #     Disables BOTH the white and HSV tissue filters in features/patching.py.
    #     Doing so changes the patch count, which changes the PCA basis, which
    #     changes every number downstream of it. Absolute values from such a run
    #     cannot be compared against a production run. Rank-based comparisons
    #     within the run are still meaningful.
    #
    #   root_source="holeyness"
    #     Selects DPT roots from expert-annotated per-duct hole %, via
    #     analysis/holeyness_roots.py, rather than from nuclear density. The point
    #     is to break the circularity of rooting the trajectory on a quantity the
    #     pipeline measured itself. Requires holeyness_export and
    #     holeyness_slide_dims; the run fails loudly if either is missing.
    relaxed_tissue_filters: bool = False
    root_source: str = "cellularity"
    holeyness_export: Path = None
    holeyness_slide_dims: Path = None
    holeyness_percentile: float = 10.0
    holeyness_min_patches: int = 1
    #   holeyness_assignment="overlap"
    #     Assigns patches to ducts by AREA OVERLAP rather than centre-in-polygon.
    #     The centre rule structurally drops any duct smaller than a patch, which
    #     is not a random subset: it removes the small ducts, and small ducts are
    #     the least holey ones. That is 571 of 2173 ducts, 26%, and it biases the
    #     root pool in exactly the direction the root rule selects for.
    #
    #   holeyness_max_roots_per_duct
    #     Values above 1 allow several roots per duct, filled round-robin so every
    #     duct gets a first root before any duct gets a second. Duct diversity is
    #     maximised ahead of depth.
    #
    #   holeyness_allow_degenerate_pool
    #     A pool whose ducts all share the threshold hole % is a hard error by
    #     default. In that situation the UUID tie-break is choosing the roots and
    #     holeyness is choosing nothing, which would make the anchor arbitrary
    #     while still appearing annotation-driven.
    holeyness_assignment: str = "centre"
    holeyness_overlap_min_fraction: float = 0.25
    holeyness_max_roots_per_duct: int = 1
    holeyness_allow_degenerate_pool: bool = False

    # Minimum fraction of the patch area that must lie inside an ROI polygon.
    # None applies the centre-point check only, which is what every recorded run
    # used. A value of 0.5 drops patches with more than half their area outside
    # the annotation.
    #
    # Leaving this at None means a patch straddling an ROI boundary is kept whole
    # as long as its centre is inside, so its embedding mixes annotated tissue
    # with whatever lies beyond the boundary.
    min_roi_coverage: Optional[float] = None

    # ── VESTIGIAL: root_cluster and root_metric are NO-OPS ────────────────────
    # Both are populated by run_all.py from --root-cluster / --root-metric and
    # then never read. Nothing on the --run path consumes either field, verified
    # by grep for `cfg.root_cluster` and `cfg.root_metric` (Phase 5, 2026-08-12,
    # re-confirmed 2026-08-24).
    #
    # They predate multi-root DPT. Single-root DPT chose its origin as the patch
    # nearest a named cluster's centroid, and compute_dpt_multi_root replaced
    # that with a fixed rule: the n patches of lowest measured nuclear density.
    # Neither field has anything left to control, so setting either changes
    # nothing about any output. A run that passes them is not misconfigured, it
    # is merely carrying a flag that stopped meaning anything.
    #
    # Kept rather than removed so existing command lines and notes keep working.
    # Safe to delete once nothing references the flags.
    #
    # WARNING: run_individual.py has its OWN, LIVE --root-cluster, backed by
    # IndividualConfig.root_cluster. That one is real and must not be touched.
    # The two share a name and nothing else.
    #
    # Field order below is deliberate. See the class docstring.
    root_cluster: Optional[str] = None

    # Multi-root DPT. n_roots candidates are the n patches with the lowest
    # MEASURED nuclear density (non-finite densities are excluded first); DPT is
    # run once per candidate and the per-patch results are median-aggregated.
    # This is the only root parameter that does anything.
    n_roots: int = 20

    # Vestigial. See the block above root_cluster.
    root_metric: str = "cellularity"
