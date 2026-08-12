# Cancer Trajectory Atlas

Morphology analysis and diffusion pseudotime for whole-slide histopathology images.

This pipeline constructs a diffusion-based pseudotime from H&E morphology features across a cohort of whole-slide images. It identifies malignancy-associated morphological trajectories—progression patterns captured by nuclear density, size ratios, and texture—and projects new slides into the learned trajectory space for diagnosis support and biomarker discovery.

## Quick Start

> **How to invoke this project.** Every entry point uses relative imports, so it must be
> run as a module **from the directory that *contains* the repo**, not from inside it:
>
> ```bash
> cd ~                     # parent of cancer_trajectory_atlas/
> python -m cancer_trajectory_atlas.run_all --run
> ```
>
> `python run_all.py` fails with `ImportError: attempted relative import with no known
> parent package`. There is no form of this project that runs a `.py` file directly.
> The one exception is `converters/batch_convert.py` (Step 2), which is a standalone
> script and must be run *from* the repo root.

**Step 1: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 2: Convert NDPIs to cropped PNGs**

Place NDPI files in `data/MCF7_x5/` (or point `--ndpi-dir` elsewhere), then:

```bash
cd ~ && python -m cancer_trajectory_atlas.run_all --convert \
  --ndpi-dir ~/cancer_trajectory_atlas/data/MCF7_x5 \
  --png-dir  ~/cancer_trajectory_atlas/data/MCF7_x5_cropped
```

Each NDPI holds two side-by-side copies of the same slide; only the **left half** is
kept, because that is the half that was annotated. This also writes
`slide_dimensions.json` into the PNG directory, recording each slide's original
full-NDPI dimensions — required for ratio-coordinate conversion.

**Step 3: Prepare annotations**

Export ROI annotations from QuPath as GeoJSON into `data/annotations/`, then convert
them to the ratio-coordinate JSON the pipeline consumes:

```bash
cd ~/cancer_trajectory_atlas && python converters/batch_convert.py
```

This reads `data/annotations/*.geojson` plus `converters/img_dims.txt` and writes
`data/annotations_ratio/*.json`. **Run it from the repo root** — its paths are hardcoded
and relative. New slides must be added to both `converters/img_dims.txt` and
`data/slide_registry.py:KNOWN_NDPI_DIMENSIONS`; see `NOTES.md` → *Annotation directory*.

**Step 4: Run the pipeline**

```bash
cd ~ && python -m cancer_trajectory_atlas.run_all --run \
  --png-dir        ~/cancer_trajectory_atlas/data/MCF7_x5_cropped \
  --annotation-dir ~/cancer_trajectory_atlas/data/annotations_ratio \
  --output-dir     ~/results/atlas_full
```

The pipeline will:
1. Stain-normalize all slides (or skip it — current runs use `--stain-method none`)
2. Extract patches inside the ROI polygons and embed them (Phikon, 768-dim)
3. Cap each slide at the cohort median patch count, then PCA to 95% variance
4. Optionally batch-correct (Harmony or scVI), then cluster with Leiden
5. Build a diffusion map and compute multi-root DPT pseudotime (20 roots, median-aggregated)
6. Validate by correlating morphological features against pseudotime, with permutation tests

Outputs (`adata_full.h5ad`, `results.csv`, `validation.json`, figures) go to `--output-dir`.

On the cluster, use the job scripts in `jobs/` rather than invoking `run_all` by hand —
they carry the exact flag sets used for published runs. `jobs/run_per_section_v2.sh` is
the current reference configuration.

## Command-Line Options

### Full Pipeline (`run_all.py`)

All paths and parameters are configurable via CLI arguments. If `paths.json` exists, its defaults will be used as fallbacks; otherwise, local relative paths are used.

All examples assume `cd ~` first (see the invocation note under *Quick Start*).

```bash
# Convert NDPI → PNG only
python -m cancer_trajectory_atlas.run_all --convert

# Run pipeline on existing PNGs
python -m cancer_trajectory_atlas.run_all --run

# Convert then run
python -m cancer_trajectory_atlas.run_all --convert --run

# Specify custom paths
python -m cancer_trajectory_atlas.run_all --run \
  --ndpi-dir ~/data/ndpi \
  --png-dir ~/data/png \
  --annotation-dir ~/data/annotations_ratio \
  --output-dir ~/results

# Run on a subset of slides
python -m cancer_trajectory_atlas.run_all --run \
  --slides 6027-4L-2M-1_x5,6028-4L-2M-1_x5

# Reuse cached Phikon features (skips GPU inference on a cache hit)
python -m cancer_trajectory_atlas.run_all --run \
  --features-cache-dir $SCRATCH/data/features_cache
```

**Paths and steps**
- `--convert` — Convert NDPI files to cropped left-half PNGs
- `--run` — Run the full analysis pipeline
- `--ndpi-dir PATH` — Input NDPI directory (default: from `paths.json`)
- `--png-dir PATH` — Cropped PNG directory (default: from `paths.json`)
- `--annotation-dir PATH` — Annotation directory (default: from `paths.json`, i.e. `data/annotations_ratio`)
- `--output-dir PATH` — Output directory (default: from `paths.json`)
- `--slides STR` — Comma-separated slide stems to process (default: all)
- `--slides-from-file PATH` — Same, one stem per line. Mutually exclusive with `--slides`

**Conversion**
- `--ndpi-level INT` — NDPI pyramid level (0 = full res; default: `0`)
- `--ndpi-scale FLOAT` — Additional downscale applied after level selection (default: `1.0`)

**Patching and features**
- `--model STR` — Feature model (`phikon`, `resnet50`; default: `phikon`)
- `--patch-size INT` — Patch size in pixels (default: `112`)
- `--stride INT` — Stride between patches (default: `96`)
- `--min-roi-coverage FLOAT` — Minimum fraction of a patch inside an ROI polygon, via a
  3×3 grid check (default: `None` = centre-point test only)
- `--features-cache-dir PATH` — Per-slide Phikon feature cache (default: `None` = disabled)

**Patch-count cap** (Vig et al. slide-aware sampling)
- `--cap-strategy STR` — `median` (cap at cohort median, computed after full extraction),
  `fixed` (cap at `--fixed-cap`), or `none` (no cap). **Default: `median`.**
- `--fixed-cap INT` — Per-slide cap, **only read when `--cap-strategy fixed`** (default: `200`).
  Also accepted as `--max-patches-per-slide`. Passing it without `--cap-strategy fixed`
  has no effect.
- `--patch-sample-seed INT` — Base seed, combined with a slide-name hash (default: `42`)
- `--target-total INT` — Informational only; logged, never used in sampling (default: `3200`)

**Clustering and batch correction**
- `--clustering-method STR` — `leiden`, `hdbscan`, `kmeans` (default: `leiden`)
- `--leiden-resolution FLOAT` — Higher = more clusters (default: `0.5`)
- `--batch-method STR` — `none`, `harmony`, `scvi`. Overrides `--harmony` when set.
  Default `None`, which falls back to `--harmony`.
- `--harmony` / `--harmony-key STR` — Legacy Harmony toggle and batch key
  (default: off / `section_number`)
- `--scvi-n-latent`, `--scvi-n-layers`, `--scvi-n-hidden`, `--scvi-max-epochs` —
  scVI hyperparameters (defaults: `30`, `2`, `128`, `400`)

**Diffusion pseudotime**
- `--diffmap-neighbors INT` — k for the diffusion-map kNN graph (default: `30`)
- `--diffmap-comps INT` — Diffusion map components (default: `10`)
- `--n-roots INT` — Root candidates for multi-root DPT; the n lowest-nuclear-density
  patches, results median-aggregated (default: `20`)

**Validation**
- `--stain-method STR` — `reinhard`, `macenko`, `none` (default: `reinhard`; current
  runs pass `none`)
- `--n-permutations INT` — Permutations for validation (default: `1000`)
- `--use-stardist` — Enable StarDist nuclear segmentation (slower; never used in any run to date)

**Vestigial**
- `--root-cluster`, `--root-metric` — parsed and stored but never read on the `--run`
  path; the multi-root DPT root rule is fixed. Do not rely on them.

### Per-Slide Analysis (`run_individual.py`)

Run pseudotime independently on each slide:

```bash
# All slides
python -m cancer_trajectory_atlas.run_individual

# Specific slide (substring match)
python -m cancer_trajectory_atlas.run_individual --slide 6027-4L-2M-1

# Custom paths
python -m cancer_trajectory_atlas.run_individual \
  --png-dir ~/scratch/data/png \
  --annotation-dir ~/scratch/data/annotations \
  --output-dir ~/results/per_slide

# Adjust clustering resolution
python -m cancer_trajectory_atlas.run_individual --leiden-resolution 1.0

# Change stain normalization
python -m cancer_trajectory_atlas.run_individual --stain-method macenko
```

**Available flags:**
- `--slide STR` — Substring filter (only run slides matching this name)
- `--png-dir PATH` — PNG directory (default: from `paths.json` or local `data/MCF7_x5_cropped`)
- `--annotation-dir PATH` — Annotation directory (default: from `paths.json`, i.e. `data/annotations_ratio`)
- `--output-dir PATH` — Output directory (default: from `paths.json` or local `individual_pseudotime_runs`)
- `--model STR` — Feature model (`phikon`, `resnet50`; default: `phikon`)
- `--patch-size INT` — Patch size (default: 112)
- `--stride INT` — Stride (default: 96)
- `--leiden-resolution FLOAT` — Leiden resolution (default: 0.5)
- `--stain-method STR` — Stain normalization (`reinhard`, `macenko`, `none`; default: `reinhard`)

To analyse a single slide, use `run_all.py --run --slides <stem>` or
`run_individual.py --slide <substring>`. There is no separate single-slide entry point.

## Configuration File (Optional)

`paths.json` in the repository root supplies defaults so paths need not be passed every
run. It takes **exactly these four keys** — any others are ignored:

```json
{
    "raw_ndpi": "~/scratch/data/ndpi",
    "cropped_png": "~/scratch/data/MCF7_x5_cropped",
    "annotations": "~/cancer_trajectory_atlas/data/annotations_ratio",
    "results": "~/scratch/results/cancer_trajectory_atlas"
}
```

`annotations` must point at the **ratio-coordinate** JSON directory
(`data/annotations_ratio`), not the QuPath GeoJSON source (`data/annotations`) —
see `NOTES.md` → *Annotation directory*.

`results` is used as a base; `run_all.py` appends `/atlas_full` to it for the default
`--output-dir`. The `~` is expanded to your home directory. If `paths.json` is absent,
the pipeline falls back to local relative paths suitable for development.

## Project Structure

Only the modules on the main pipeline path are listed. `analysis/` also holds ~25
single-purpose analysis and diagnostic modules (holeyness, timepoint, root sensitivity,
LOO, …), each driven by its own script in `jobs/`.

```
cancer_trajectory_atlas/
├── run_all.py                # Main entry point (--convert and --run)
├── run_individual.py         # Per-slide pseudotime, independent of the cohort
├── pipeline_config.py        # PipelineConfig dataclass
├── paths.json                # Default path configuration
│
├── data/
│   ├── slide_registry.py        # KNOWN_NDPI_DIMENSIONS fallback table
│   ├── stain_normalization.py   # Reinhard / Macenko normalization
│   ├── annotations/             # QuPath GeoJSON exports (source)
│   └── annotations_ratio/       # Ratio-coordinate JSON (what the pipeline reads)
│
├── converters/
│   ├── batch_convert.py         # GeoJSON → ratio JSON (run from repo root)
│   ├── img_dims.txt             # Per-slide NDPI dimensions used by batch_convert
│   ├── ndpi_to_img.py           # Standalone NDPI converter (not used by run_all)
│   └── tiff_to_img.py           # Standalone TIFF converter (not used by run_all)
│
├── features/
│   ├── patching.py              # Patch extraction, ROI polygons, sampling
│   └── extractors.py            # Phikon / ResNet50 feature extraction
│
├── analysis/
│   ├── clustering.py            # PCA, UMAP, Leiden
│   ├── harmony.py               # Harmony batch correction
│   ├── scvi_integration.py      # scVI batch correction
│   ├── diffusion.py             # Diffusion map, multi-root DPT, PAGA
│   └── projector.py             # AtlasProjector for LOO projection
│
├── validation/
│   ├── morphological_features.py  # Nuclear density, NC ratio, texture, …
│   └── correlations.py            # Correlation and permutation tests
│
├── utils/
│   ├── viz.py                    # Plotting helpers
│   └── io.py                     # Save/load helpers
│
├── qc/                       # Post-run QC (submit_qc.sh)
├── diagnostics/              # One-off diagnostic scripts
├── figures/                  # Paper figure generation
├── visualize/                # Interactive overlays and patch export
├── jobs/                     # SLURM submission scripts — the canonical run recipes
└── reports/                  # Written analysis reports
```