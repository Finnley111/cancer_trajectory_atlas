# Design Notes & Decisions

Running record of non-obvious decisions, known constraints, and experiment
rationale. Things that should have been written down but only lived in chat.

---

## Current scientific status (read this first)

The pipeline produces a morphological pseudotime that orders patches from
densely cellular tumor regions (low PT) to stromal/acellular-looking regions
(high PT). This ordering is consistent across stain normalization choices
and batch correction approaches. Feature correlations against pseudotime
range rho 0.2-0.5 depending on analysis, all significant against permutation
null.

**Critical unresolved question:** it has not been demonstrated that pseudotime
captures anything beyond gross tumor cellularity per patch. A patch with more
tumor cells looks different from a patch with more stroma, even if the tumor
itself is biologically identical. Until the cellularity confound is tested,
"morphological trajectory" claims are not defensible. This is the next
experiment to run.

---

## The section split — what 2M-1 and 2M-2 actually mean

Originally suspected to be (a) serial sections of the same tissue stained with
different reagents, or (b) different anatomical regions of each tumor.
Confirmed with Miranda: scenario (b). 2M-1 and 2M-2 sections were sampled
from different anatomical regions, not serial sections.

Implication: the two sections are not clean cross-validation replicates.
Differences between section 1 and section 2 analyses can reflect both
different staining and different sampled biology, confounded together.
This affects how we can frame any cross-section comparison.

There was also confusion about whether the two sections used different stain
reagents. Miranda confirmed they did. So both factors (anatomy and stain)
differ between sections — neither can be isolated with the current data.

---

## Two-island UMAP problem

When all 16 slides are pooled without batch correction, the UMAP produces
two distinct islands. QC pseudotime violins showed PT ≈ 0.8 for all 2M-1
slides and PT ≈ 0.05 for all 2M-2 slides, confirming the section divide
dominates the embedding.

Classical stain normalization (Macenko, Reinhard) did not resolve this and
in some configurations made it worse — Macenko amplified the section split
into more aggressively bimodal pseudotime. This is because the two sections
used different stain reagents, which is outside what Macenko/Reinhard can
correct for (they assume a single underlying dye chemistry).

Resolution: Harmony at the feature level (with `section_number` as batch
key) successfully merges the two islands into a connected manifold.
No-normalization + Harmony works as well as Macenko + Harmony — stain
normalization adds little once feature-space correction is applied.

---

## Experiment runs (completed)

| Run | Output dir | Result |
|---|---|---|
| Section 1 only, no norm | `atlas_none_section1` | Connected manifold. Leading features: h_intensity (+0.50), nuclear_density (+0.44), nc_ratio (+0.38) |
| Section 2 only, no norm | `atlas_none_section2` | Connected manifold. Leading features: texture_entropy (+0.43), mean_nuclear_area (−0.38) |
| All slides + Harmony, no norm | `atlas_none_harmony` | Connected manifold, islands merged. Confirms feature-space correction is the right intervention |
| All slides + Harmony + Macenko | `atlas_macenko_harmony` | Similar to above. Features: h_intensity (+0.35), texture_entropy (+0.28), nc_ratio (+0.21), nuclear_density (+0.20) |

Different feature loadings between section 1 and section 2 reflect either
the different stain reagents (each reagent making different features more
visible) or the different anatomical sampling, or both. Cannot be
disentangled with current data.

---

## Pending experiments

In priority order. Do these in sequence, not parallel — each one's result
affects whether the next is worth running.

1. **Cellularity confound test.** Compute per-patch nuclear coverage from
   existing nuclear segmentation outputs, correlate with pseudotime across
   all completed runs. If rho > 0.7, pseudotime is mostly cellularity and
   the "trajectory" framing is not defensible. If rho < 0.3, there's real
   signal beyond cellularity. If 0.3-0.7, run within-cellularity-stratum
   analysis to find what the residual is. This is a post-hoc analysis on
   existing results, does not require pipeline reruns.

2. **Leave-one-out projection stability.** For each of 16 slides, rebuild
   the manifold on the other 15 and project the held-out slide. Compare
   projected vs in-manifold pseudotime distributions per slide. Foundational
   experiment for any atlas-style claims. Requires 16 pipeline runs — needs
   Phikon feature caching to be reasonable on Narval.

3. **Within-cellularity-stratum trajectory.** Conditional on cellularity
   quartile, does a sub-trajectory exist? Only worth running if experiment 1
   returns rho between 0.3 and 0.7.

### Per-reagent runs — superseded

Earlier plans called for running the pipeline on each reagent group
separately. This is now equivalent to the per-section runs (since reagent
maps onto section), and those are already done. No separate per-reagent
runs needed.

---

## harmonypy version — must stay on 0.0.9

**Do not upgrade harmonypy without re-running the synthetic shape test.**

### Bug history

| Attempt | What was tried | Why it failed |
|---|---|---|
| 1 | `backend='numpy'` kwarg | Parameter does not exist in any harmonypy version. TypeError before harmony ran. |
| 2 | Removed the kwarg entirely | No mechanism to suppress CUDA. harmonypy 0.2.0 auto-selected CUDA and hit the shape bug again. |
| 3 | `use_gpu=False` kwarg | Correct for 0.2.0 API, forced CPU. But 0.2.0 has the same shape-squeezing bug on CPU too when convergence happens in ≤2 iterations. |
| 4 ✓ | Downgraded to 0.0.9 | Pure numpy, no GPU path exists. Shape bug does not exist in 0.0.9. |

### Root cause

harmonypy 0.2.0 is a PyTorch rewrite. When Harmony converges in ≤2 iterations
(which happens with only 2 batches and small `nclust`), the PyTorch tensor
operations return `Z_corr` as a 1D array of shape `(k,)` instead of `(k, N)`.
Scanpy's `harmony_integrate` then does `.T` and tries to write a 1D array to
`obsm`, which anndata rejects.

### 0.0.9 orientation

harmonypy 0.0.9 returns `Z_corr` with shape `(k, N)` (PCs × cells). Scanpy's
`harmony_integrate` does `.T` → `(N, k)` before writing to `obsm`. This is
correct and expected.

### Synthetic test to re-run before any harmonypy upgrade

```bash
source ~/envs/atlas/bin/activate
python -c "
import numpy as np, pandas as pd, harmonypy
np.random.seed(0)
X = np.random.randn(100, 20).astype(np.float32)
meta = pd.DataFrame({'batch': ['A']*50 + ['B']*50})
ho = harmonypy.run_harmony(X, meta, 'batch', nclust=10)
print('version:', harmonypy.__version__)
print('Z_corr shape:', ho.Z_corr.shape)   # expect (20, 100) for 0.0.9
print('2D?', ho.Z_corr.ndim == 2)         # must be True
"
```

---

## Harmony nclust=10 rationale

harmonypy's default `nclust = min(N/30, 100)` gives 100 for 65k+ cells.
With only 2 batches (`section_number`), 100 internal K-means clusters is
massively over-parameterized — the batch model over-fits trivially and
converges in 2 iterations. `nclust=10` is set in `apply_harmony()` as the
default, exposed as a parameter so it can be increased if switching to
`slide_id` key (16 batches) or `mouse_id` (4 batches).

---

## Stain normalizers tried and conclusions

Macenko, Reinhard, and no-normalization all evaluated. Conclusions:

- Macenko and Reinhard cannot correct for the section batch effect because
  the two sections used different stain reagents (different dye chemistries),
  which violates the single-stain assumption these methods rely on.
- No-normalization + Harmony works as well as any normalized + Harmony
  combination. Stain normalization adds little when feature-space batch
  correction is in place.
- Deep-learning normalizers (CycleGAN-based, etc.) are documented to
  hallucinate biologically inaccurate features when source and target domains
  differ substantially — Cohen et al. (MICCAI 2018), Rivenson et al. (Nature
  Communications 2021), Salahuddin et al. (Scientific Reports 2026). Would
  require pathologist-level per-image verification to defend in a paper. Not
  pursued for now.

For new analyses: skip stain normalization unless there's a specific reason
to add it. Use no-norm + Harmony as the default pipeline configuration.

---

## Annotation directory

Four annotation directories exist under `~/cancer_trajectory_atlas/data/`:

- `annotations/` — raw QuPath GeoJSON exports, **absolute full-NDPI pixel coordinates**.
  This is the *source* the converter reads, not an archive.
- `annotations_ratio/` — ratio-coordinate JSON, **generated by
  `converters/batch_convert.py`**. This is what the pipeline consumes.
- `old_annotations/` — 16 ratio JSONs from an older, un-updated annotation round.
  Superseded but genuinely different content. Keep as historical record; do not use.
- `annot_check_test/` — PNG overlays written by `jobs/check_annotations.py` for visual QC.

**Always use `annotations_ratio/` for pipeline runs.** `paths.json` points here. The
ratio format is what `load_roi_polygons()` has been tested with.

### Regenerating the ratio annotations

```bash
cd ~/cancer_trajectory_atlas && python converters/batch_convert.py
```

Takes no arguments — paths are hardcoded relative to the repo root, so the `cd` matters.

### Round-trip invariant (important)

`batch_convert.py` **divides** polygon coordinates by `converters/img_dims.txt`.
`load_roi_polygons()` **multiplies** them back by `original_full_width`, sourced from
`slide_dimensions.json` (written by `run_all.py --convert`) or, failing that, from
`data/slide_registry.py:KNOWN_NDPI_DIMENSIONS`.

Verified 2026-08-12: `img_dims.txt` and `KNOWN_NDPI_DIMENSIONS` contain the same 16
keys with identical values — zero mismatches — so the round trip is exact and the
sidecar and fallback paths are equivalent. **If a slide is ever added, add it to both.**

### History: `annotations/` used to hold the ratio files

Until commit `f050e4a`, `data/annotations/` contained the ratio `.json` files; that
commit replaced them with `.geojson` and the ratio files moved to
`annotations_ratio/`. Several older job scripts still pass
`--annotation-dir .../data/annotations` — `run_all_none.sh`, `run_all_macenko.sh`,
`run_all_reinhard.sh`, `run_all_none_section.sh`, `submit_harmony*.sh`,
`run_individual_pseudotime.sh`, `run_timepoint_stage1_convert.sh`.

**Those scripts were correct when they ran; results produced by them are not suspect.**
But re-running one today would feed absolute-pixel GeoJSON to a loader that always
assumes ratio coordinates, multiplying every coordinate by `original_full_width` a
second time and putting every ROI off-canvas — yielding few or no in-ROI patches.
Treat them as historical, not as runnable.

If annotations are missing from a run, the pipeline reports "0 with annotations" — this
is the first thing to check if ROI filtering silently stops working.

---

## SLURM slides file path

`${BASH_SOURCE[0]}` resolves to SLURM's temporary job directory
(`/localscratch/spool/slurmd/jobXXX/`), not the repo. Any job script that
references sibling files must use hardcoded absolute paths
(`~/cancer_trajectory_atlas/jobs/...`), not paths derived from BASH_SOURCE.

This bit `run_all_none_section.sh` on first submission and has been fixed.

---

## Results directory naming convention

Under `$SCRATCH/results/` on Narval, use these prefixes to distinguish run types:

| Prefix | Purpose | Examples |
|---|---|---|
| `atlas_*` | Pooled baseline and ablation runs (all 16 slides) | `atlas_none_harmony`, `atlas_macenko_harmony`, `atlas_none_harmony_cap1900` |
| `section_*` | Per-section subset runs (2M-1 or 2M-2 only) | `section_none_2m1`, `section_none_2m2` |
| `loo_*` | LOO training + projection per held-out slide | `loo_6027-4L-2M-1_x5` |
| `loo_summary` | Aggregated LOO results (CSV + stability figure) | `loo_summary` |
| `individual_*` | Per-slide standalone pseudotime runs | `individual_pseudotime_runs` |
| `diagnostic_*` | QC and post-hoc analyses | `diagnostic_6028-4L-2M-2_x5`, `diagnostic_confound` |

**Rules:**
- Never reuse an existing directory name for a new experiment. Pass a distinct `--output-dir`.
- Canonical reference run for LOO: `atlas_none_harmony` (no stain, Harmony, uncapped).
  The `FULL_RUN` env var in all LOO job scripts defaults to this directory.
- The `qc/` subdirectory inside any `atlas_*` run contains QC outputs from `run_qc.py`.
  Do not put QC outputs at the top level of `$SCRATCH/results/`.

---

## Patch-level analysis limitation

Each 112×112 patch at 5x is scored independently. A patch from the intact
outer wall of a mid-stage duct and a patch from its disrupted interior get
independent pseudotime scores even though they belong to the same duct.
The pipeline scores tissue *content* at a fixed spatial scale, not the
biological state of larger anatomical structures.

This is not a bug — it's the level of analysis the pipeline operates at.
But results should be framed as "patch-level morphological similarity" not
"duct-level progression stage." Future extensions (slide-level aggregation
via attention pooling, or duct segmentation followed by per-structure
aggregation) would address this but are beyond current scope.