# Cancer Trajectory Atlas — Project State Document

**Written:** 2026-06-03 | **Last updated:** 2026-07-27  
**Purpose:** Durable reference for any Claude Code session. Read this before making any changes.  
**Status:** Smoke test passed (2026-06-04). `run_full_experiments.sh` (median cap, 20-root DPT) is the canonical full-suite job script — not yet confirmed run to completion on Narval (no results directories from it exist in this checkout; verify on cluster before assuming it ran). A PAGA connectivity check (`analysis/diffusion.py:compute_paga_topology`) is staged in the working tree but **still not committed** — see "Uncommitted changes" below. This update pass (2026-06-22) re-verified every code reference in this document against the current source tree; two stale items were found and corrected (Issue 1, Issue 7 — see "Corrections made in this update" below).

---

## Scientific Context (read this first)

**Dataset:** 16 MCF7 H&E slides from 4 mice (6027, 6028, 6029, 6031), 2 anatomical regions (2M-1, 2M-2), 2 sides (4L, 4R). ~28–33k patches per slide at 5× magnification.

**Current result:** The pipeline produces a morphological pseudotime that orders patches from dense tumor regions (low PT) to stromal/acellular regions (high PT). LOO projection stability is mean Spearman ρ ≈ 0.83 across 15 slides, with one outlier at ρ = 0.48 (slide 6028-4L-2M-2_x5). Feature correlations range ρ 0.2–0.5, all significant against permutation null.

**Critical unresolved question:** Pseudotime may just be tracking gross tumor cellularity per patch, not a real morphological trajectory. The cellularity confound test (via `analysis/cellularity_confound.py`) needs to be run. Until then, "trajectory" framing is not defensible.

**Two-island UMAP problem:** RESOLVED by Harmony batch correction (section_number key). Root cause: 2M-1 and 2M-2 sections used different stain reagents AND sampled different anatomical regions — stain normalization alone cannot correct this. No-normalization + Harmony is the canonical pipeline configuration.

**Canonical pipeline configuration for all future runs:**  
`--stain-method none --harmony --harmony-key section_number`

---

## Corrections made in this update (2026-06-22)

Verified every file/line reference in this document against the current source tree. Two items were stale:

1. **Issue 1 was wrong.** It claimed `--root-cluster` does not exist as a CLI argument. It does (`run_all.py:766`) — but it is dead: `run_pipeline()` never passes it to `compute_dpt_multi_root()` (only `cfg.n_roots` is passed, see `run_all.py:551`). The flag is parsed, stored on `PipelineConfig.root_cluster`, and then ignored. Corrected below.
2. **Issue 7 was stale.** `jobs/recover_loo_phase_b.py` no longer exists. It was moved to `analysis/recover_loo.py` back in commit `0fa4880` ("refactor/deleted unused files, added smoke test for entire pipeline"). The misplacement this issue described has already been fixed. Marked resolved below.

Also confirmed unchanged/still accurate: all of Phase 0–7 data flow, the AnnData obs schema, the LOO architecture, the harmonypy 0.0.9 pin, the annotation directory discrepancy (Issue 5), and the cap_strategy default (`median`, since commit `1a8b471`).

---

## Pipeline At a Glance — Numbered Step List

Use this as the index; each step links to its detailed phase further down.

1. **Convert** NDPI → cropped PNG + `slide_dimensions.json` sidecar (`run_all.py --convert`, see Phase 0).
2. **Discover** slides: match PNG ↔ annotation file, resolve original dimensions (Phase 1).
3. **Extract patches, Pass 1**: full uncapped patch grid per slide, ROI/tissue filtering, stain normalization (Phase 2, Pass 1).
4. **Extract/load Phikon features** per slide, cached to disk, always uncapped (Phase 3).
5. **Compute cohort cap** (median/fixed/none) once all slides' Pass-1 counts are known (Phase 2, Pass 2, step 8 below).
6. **Sample patches** down to the active cap per slide, shuffle order (Phase 2, Pass 2, steps 9–11).
7. **Cluster**: StandardScaler → PCA (95% var) → optional Harmony (`section_number`) → UMAP (viz only) → Leiden (Phase 4).
8. **Build AnnData**, run `sc.pp.neighbors` + `sc.tl.diffmap` (Phase 5, step 1).
9. **PAGA topology check** (diagnostic only, uncommitted) — connected-components count on the Leiden cluster graph (Phase 5, step 2).
10. **Pre-compute nuclear density** (quick Otsu-based) for multi-root DPT candidate selection (Phase 5, step 3).
11. **Multi-root DPT**: run DPT from 20 lowest-density roots, median-aggregate → `pseudotime`, std → `pseudotime_std` (Phase 5, step 4).
12. **Morphological validation**: 6 features × Spearman ρ × permutation test → verdict (Phase 6).
13. **Save artifacts**: `adata_full.h5ad`, `results.csv`, `validation.json`, `projector/`, figures (Phase 7).
14. *(LOO only)* **Phase B projection**: KNN-project held-out slide onto the 15-slide manifold, compute paired ρ/Wasserstein/KS (LOO Architecture section).
15. *(Not yet run)* **Cellularity confound test** — the next required experiment before "trajectory" framing is defensible (`analysis/cellularity_confound.py`).

---

## Uncommitted Changes (working tree, as of 2026-06-18)

Not yet committed — `git diff` shows changes to `analysis/diffusion.py`, `run_all.py`, `utils/viz.py`, plus deletion of `data/MCF7_x5_cropped/.gitkeep` (empty placeholder, harmless — real data lives outside git on Narval).

**PAGA topology gate (new):** After `compute_diffusion_map()` and before multi-root DPT, the pipeline now runs PAGA on the Leiden clusters and reports the connected-component count of the thresholded cluster graph:
- `analysis/diffusion.py:compute_paga_topology()` — runs `sc.tl.paga(adata, groups="cluster")`, thresholds `connectivities` at 0.05 (hardcoded, no CLI flag yet), counts components via `scipy.sparse.csgraph.connected_components`. Returns `(n_components, adata)`.
- `utils/viz.py:plot_paga()` — saves `qc_paga_topology.png` (scanpy's PAGA graph plot).
- `utils/viz.py:plot_umap_section_cluster()` — saves `qc_umap_section_vs_cluster.png`, a side-by-side UMAP colored by `section_number` vs. by Leiden cluster, to visually check whether disconnected components track the section/batch divide.
- Wired into `run_all.py` (~line 533) right after `compute_diffusion_map`. It is currently diagnostic-only: it prints whether the manifold is a single component (DPT valid) or N disconnected components, but does **not** abort the run or change behavior — it's a QC signal, not a hard gate despite the name.

**Why this was added:** the two-island UMAP problem was resolved by Harmony, but there was no automated check that a *given* run's post-Harmony manifold is actually a single connected component before trusting DPT pseudotime. PAGA + connected-components gives a numeric, scriptable version of the by-eye UMAP check.

**Not yet done:** no CLI flag to override the 0.05 threshold; no actual abort/warning surfaced in `validation.json` or the run summary; not yet exercised against a real disconnected-graph case to confirm it correctly flags `qc/graph_connectivity.py`-style failures.

---

## Repository Map

```
cancer_trajectory_atlas/
├── run_all.py                  ← MAIN entry point: --convert + --run
├── run_individual.py           ← Per-slide standalone pseudotime
├── run_train_test.py           ← DEAD CODE (imports broken config.py)
├── pipeline_config.py          ← PipelineConfig dataclass
├── paths.json                  ← Deployment-time path configuration
├── NOTES.md                    ← Running scientific + engineering log (IMPORTANT)
│
├── data/
│   ├── slide_registry.py       ← KNOWN_NDPI_DIMENSIONS fallback table
│   ├── stain_normalization.py  ← Reinhard + Macenko normalizers
│   ├── annotations/            ← Raw QuPath GeoJSON exports
│   └── annotations_ratio/      ← Ratio-coordinate JSON (converted from GeoJSON)
│
├── features/
│   ├── patching.py             ← Patch extraction, ROI filtering, tissue QC
│   └── extractors.py           ← Phikon (HuggingFace) / ResNet feature extraction
│
├── analysis/
│   ├── clustering.py           ← StandardScaler, PCA, UMAP, Leiden/HDBSCAN/KMeans
│   ├── diffusion.py            ← AnnData builder, scanpy neighbor graph, diffusion map, DPT
│   ├── harmony.py              ← Harmony batch correction wrapper
│   ├── projector.py            ← AtlasProjector: save/load/project new slides
│   ├── loo_project.py          ← Phase B: project held-out slide, compute Spearman ρ
│   ├── loo_summary.py          ← Aggregate 16 LOO results into summary CSV + figure
│   ├── cellularity_confound.py ← Post-hoc cellularity confound test
│   └── slide_diagnostics.py   ← Five-hypothesis outlier slide diagnostic
│
├── validation/
│   ├── morphological_features.py ← 6 morphological features per patch
│   ├── correlations.py           ← Spearman, permutation test, verdict
│   └── annotations.py            ← PARTIALLY DEAD: has legacy mask-loading code
│
├── qc/
│   ├── run_qc.py               ← Master QC runner (4 steps)
│   ├── graph_connectivity.py   ← Step 1: k-NN graph connectivity check
│   ├── stain_qc.py             ← Step 2: stain normalization QC
│   ├── cluster_contact_sheet.py← Step 3: cluster patch contact sheets
│   └── pseudotime_by_slide.py  ← Step 4: per-slide/per-mouse pseudotime violins
│
├── visualize/
│   ├── interactive_overlay.py  ← Interactive Plotly HTML overlay per slide
│   └── export_patches.py       ← Export patches binned by pseudotime
│
├── utils/
│   ├── viz.py                  ← All matplotlib plotting functions
│   └── io.py                   ← save_pickle, save_json, save_atlas_artifacts
│
├── jobs/                       ← SLURM job scripts
│   ├── run_all_capped.sh       ← Canonical run: no-stain + Harmony + patch cap
│   ├── run_all_none_section.sh ← Per-section (2M-1 or 2M-2) run
│   ├── submit_loo_array.sh     ← Dispatch 16 LOO jobs as SLURM array
│   ├── run_loo_single.sh       ← Per-task LOO: Phase A (train) + Phase B (project)
│   ├── submit_qc.sh            ← QC diagnostics job
│   ├── run_slide_diagnostics.sh← Slide outlier diagnostic job
│   ├── run_cache_population.sh ← GPU job: populate features cache for all 16 slides
│   ├── recover_loo_phase_b.py  ← One-off Python recovery script (MISPLACED in jobs/)
│   └── ...other utility scripts
│
└── converters/
    ├── ndpi_to_img.py          ← NDPI → image converter
    ├── tiff_to_img.py          ← TIFF → image converter
    └── batch_convert.py        ← Convert QuPath GeoJSON → ratio-coordinate JSON
```

---

## End-to-End Pipeline Data Flow

Re-derived 2026-06-22 by reading every function called from `run_pipeline()` (`run_all.py:243-664`) start to finish. The console output from a real run prints 5 banners — `PHASE 1: Stain Normalization`, `PHASE 2: Patch Extraction & Feature Embedding`, `PHASE 3: Morphological Clustering`, `PHASE 4: Diffusion Pseudotime`, `PHASE 5: Morphological Feature Validation` (`run_all.py:277,289,477,513,573`). This document uses a finer **Phase 0–7** breakdown for documentation clarity — the mapping is: code's PHASE 1 = doc Phase 2 (stain norm is folded into patch extraction here); code's PHASE 2 = doc Phases 2–3; code's PHASE 3 = doc Phase 4; code's PHASE 4 = doc Phase 5; code's PHASE 5 = doc Phase 6. Doc Phase 0 (`--convert`) and Phase 1 (slide discovery) run before any of the code's numbered phases; doc Phase 7 (save) runs after PHASE 5.

### Phase 0: NDPI → PNG Conversion — `run_all.py:51-129` (`convert_ndpi_to_left_half_png`, only runs with `--convert`)

1. `ndpi_dir.glob("*.ndpi")` — sorted list of all NDPI files (`run_all.py:69`).
2. For each NDPI file: open with `openslide.OpenSlide`, read `dims = slide.level_dimensions[cfg.ndpi_level]` (default level 0 = full resolution) (`run_all.py:86-87`).
3. If the target PNG (`{stem}_x5.png`) already exists, **skip re-encoding** but still record its dimensions in the sidecar — `run_all.py:89-91`. This means re-running `--convert` is idempotent/incremental, not a full rebuild.
4. Otherwise: `slide.read_region((0,0), level, dims)` reads the **entire** NDPI (both side-by-side slide copies) into memory as RGB (`run_all.py:96`).
5. Optional uniform downscale: if `ndpi_scale != 1.0`, resize with `Image.Resampling.LANCZOS` (`run_all.py:99-102`).
6. **Crop to left half only**: `img.crop((0, 0, full_w // 2, full_h))` (`run_all.py:105`). NDPI files contain two copies of the same slide side by side; QuPath annotations were drawn on the left copy only, so the right copy is discarded here, permanently.
7. Save as PNG, `compress_level=6`, to `png_dir/{stem}_x5.png` (`run_all.py:108`).
8. Record `{original_full_width, original_full_height, cropped_width, cropped_height}` per file into an in-memory dict (`run_all.py:112-119`); `cropped_width = original_full_width // 2` always.
9. After all files: write the dict to `png_dir/slide_dimensions.json` (`run_all.py:122-124`). This sidecar is the primary source of truth for ratio-coordinate scaling in Phase 2 — see invariant below.

**Key invariant:** `original_full_width` = full NDPI width (both copies, pre-crop). `cropped_width = original_full_width // 2`. If `slide_dimensions.json` is ever missing or wrong, ratio-annotation scaling silently produces incorrect pixel coordinates (Issue 5).

### Phase 1: Slide Discovery — `run_all.py:149-240` (`discover_slides`, runs at the start of every `--run`)

1. List PNGs in `png_dir` (`run_all.py:159`). Abort if none found.
2. Try to load `png_dir/slide_dimensions.json` (`run_all.py:165-170`). If absent, fall back per-slide to `KNOWN_NDPI_DIMENSIONS` (`data/slide_registry.py`) via `_get_known_dimensions()` (`run_all.py:134-146`), scaled by `ndpi_scale`.
3. For each PNG, derive `stem` (e.g. `6027-4L-2M-1_x5`) and `base_stem` (`_x5` stripped) (`run_all.py:179-180`).
4. Match an annotation file by trying, in order: `{stem}.json`, `{base_stem}.json`, `{stem}.geojson`, `{base_stem}.geojson` inside `annotation_dir` — first match wins (`run_all.py:183-191`). No match → `annotation = None`, and that slide later runs with **no ROI filtering** (full-slide patching).
5. Attach dimensions: sidecar first, fallback table second (`run_all.py:196-203`).
6. Print a per-slide discovery table (✓/✗ annotation, dims or "NO DIMS") (`run_all.py:212-216`) and warn if any annotated slide is missing dimensions (`run_all.py:219-224`) — this is the first thing to check if ROI filtering silently produces wrong patch counts.
7. Apply `--slides` / `--slides-from-file` subset filter if given: hard-errors (`sys.exit(1)`) if any requested slide stem isn't found (`run_all.py:227-238`).
8. Returns a list of dicts: `{image, annotation, original_full_width, original_full_height}`, one per surviving slide.

### Phase 2: Stain Normalization + Patch Extraction — `run_all.py:276-474`, `features/patching.py`, `data/stain_normalization.py`

**2a. Build the stain normalizer once, against slide 0 (`run_all.py:281-286`).**
1. `reference_image = slides[0]["image"]` — i.e. whichever slide sorts first alphabetically by filename. This is hardcoded; `paths.json`'s `stain_reference` key is never read (Issue 9).
2. `build_normalizer(cfg.stain_method, reference_image)` (`data/stain_normalization.py:67-101`):
   - `"none"` → returns `None` (no-op downstream).
   - `"reinhard"` → fits a `ReinhardNormalizer`: converts the reference to LAB (`cv2.cvtColor`), masks to tissue pixels (`L < 230`, fallback to all pixels if that yields <1000 px), records per-channel mean/std (`data/stain_normalization.py:25-37`).
   - `"macenko"` / `"vahadane"` → delegates to `staintools.StainNormalizer`, fit on the reference image (`data/stain_normalization.py:95-97`). Requires `spams` (conda-only dependency, not in requirements.txt).
3. If a normalizer was built, the reference PNG is copied to `output_dir/stain_reference.png` for provenance (`run_all.py:285-286`).

**2b. Pass 1 — per-slide extraction loop, always uncapped (`run_all.py:313-396`).** For each slide `i`:
1. Load the PNG with PIL, convert to RGB numpy array (`run_all.py:320-321`).
2. `normalize_slide(img_arr, stain_normalizer, slide_name)` (`data/stain_normalization.py:104-117`) — applies the fitted normalizer's `.transform()`, or passes through unchanged if `None`. Catches and logs (not raises) any per-slide transform failure, returning the original array.
3. If an annotation was matched: `load_roi_polygons()` (`features/patching.py:49-179`) —
   a. Parses GeoJSON `FeatureCollection` or raw JSON-list/dict (`features/patching.py:116-121`).
   b. Each polygon ring is scaled: `x *= original_full_width`, `y *= original_full_height` when `coordinate_space="ratio"` (`features/patching.py:127-129`).
   c. Classification: a feature named `"Tumor"` or unclassified (`None`) → inclusion polygon; anything else (`Ignore*`, `Necrosis`, `Region*`, …) → exclusion polygon (`features/patching.py:144-146`).
   d. Polygons with holes are built as compound `matplotlib.path.Path` objects via `_make_path()` so inner rings are excluded correctly (`features/patching.py:35-46`).
   e. Any polygon whose centroid falls outside the cropped (left-half) region is discarded — this is what removes right-half duplicate-copy polygons, if any leaked into the annotation file (`features/patching.py:166-178`).
   If no annotation matched: `roi_polys = exclude_polys = None` → the whole slide is patched with no ROI restriction.
4. `get_patches_from_array()` (`features/patching.py:206-292`) — slides a `(patch_size=112, stride=96)` grid over the image:
   a. For each grid cell, compute the patch centre `(cx, cy) = (x + 56, y + 56)`.
   b. **ROI inclusion check** (if `roi_polygons` given): `_find_containing_roi(cx, cy, roi_polygons)` — centre-point-in-polygon test; reject if no containing polygon (`features/patching.py:249-253`).
   c. **Optional coverage check** (`min_roi_coverage`, default `None` = skip): samples a 3×3 grid at 1/4, 1/2, 3/4 offsets inside the patch and requires ≥ that fraction of the 9 points to fall inside *any* ROI polygon — catches patches that are mostly outside the annotation despite their centre being inside (`features/patching.py:190-201, 254-257`).
   d. **Exclusion check**: reject if the centre falls inside any `exclude_polygons` (Ignore/Necrosis regions) (`features/patching.py:260-263`).
   e. **White-pixel reject**: reject if >70% of pixels have all RGB channels > 220 (`_is_mostly_white`, `features/patching.py:13-18`).
   f. **HSV tissue check**: convert patch to HSV; reject unless ≥50% of pixels are both saturated (`S > 15`) and non-bright (`V < 230`) (`_has_tissue_hsv`, `features/patching.py:21-30`).
   g. Surviving patches and their `(x, y)` top-left coords are collected; per-slide rejection counts (ROI/coverage/exclude) are printed (`features/patching.py:279-287`).
5. If zero patches survive, the slide is skipped entirely with a warning (`run_all.py:355-357`) — it contributes nothing downstream, including to the LOO experiment if it's the held-out slide.
6. `orig_count = len(patches)` is recorded — this is the pre-cap count used for both the median-cap calculation and the feature-cache shape contract.

**2c. Feature cache interaction, still inside the Pass-1 loop (`run_all.py:361-396`).** If `--features-cache-dir` is set:
   - Cache hit (`{slide_name}_features.npy` exists): load it; **hard assert** `len(cached) == orig_count`, else raise `RuntimeError` naming the mismatch — this is the cache/extraction-settings contract (`run_all.py:368-376`).
   - Cache miss: lazily load the embedding model once (`load_model_components`, `features/extractors.py:85-90`) the first time any slide misses, then `extract_features_from_model()` and `np.save()` the result (`run_all.py:378-387`). The model is reused across slides in the same run — only one cold load per process.
   - The cache always stores features for **every uncapped patch** — capping happens later, in Pass 2, never before caching.

**2d. Pass 2 — cap calculation and sampling, after all slides are loaded (`run_all.py:402-456`).**
1. Slide processing order is shuffled with `np.random.default_rng(cfg.patch_sample_seed)` before sampling, to avoid any systematic slide-order bias (`run_all.py:404`).
2. Active cap is computed once for the whole cohort based on `cfg.cap_strategy` (`run_all.py:406-415`):
   - `"median"` (current default): `active_cap = int(median(orig_count across all slides in this run))`.
   - `"fixed"`: `active_cap = cfg.max_patches_per_slide` (CLI: `--fixed-cap` / `--max-patches-per-slide`, default 200).
   - `"none"`: `active_cap = None` (no cap at all — every patch kept).
3. The resolved cap is written to `output_dir/active_cap.txt` (`run_all.py:418-420`) — this is what LOO Phase B reads (`BASELINE_CAP`) so the held-out slide is sampled to the exact same count as the training manifold.
4. For each slide: `sample_patches(patches, coords, active_cap, seed, slide_name)` (`features/patching.py:295-309`) — if `len(patches) <= active_cap` (or cap is `None`/`0`), returns everything unchanged; otherwise shuffles the **full** patch index array using a per-slide seed (`base_seed XOR md5(slide_name)[:8] as int`) and takes the first `active_cap` indices. This randomizes which patches survive *and* their order — not a spatial/sorted subsample.
5. Per-slide bookkeeping is appended to `sampling_log_rows`: `patches_before_cap`, `patches_after_cap`, `was_capped`, `seed`, `pct_retained` (`run_all.py:438-445`) — saved later as `sampling_manifest.csv`.
6. If a slide was capped, the chosen indices are stashed in `sampling_indices[slide_name]` (`run_all.py:446-447`) — saved later as `.npy` files under `sampling/`, so the exact subsample is reproducible/inspectable without re-running.
7. Cached features (if any) are indexed by the same `selected_idx` to stay aligned with the sampled patches (`run_all.py:453-456`).
8. All slides' patches/coords/features are concatenated (`np.concatenate`) into single arrays; `slide_ids` (int array, one entry per patch) tags each patch's originating slide index (`run_all.py:449-460`).

**Annotation coordinate invariant:** Ratio annotations × `original_full_width` → full-NDPI pixel space. Since QuPath annotates the left half of the NDPI, x-values after scaling land in `[0, original_full_width/2] = [0, cropped_width]` — exactly the same range as patch `(x, y)` coords from `get_patches_from_array()`. **They are already in the same pixel space; no offset is needed.** Right-half polygons are discarded by the crop-centroid filter in `load_roi_polygons()` (`features/patching.py:166-178`).

### Phase 3: Feature Extraction — `features/extractors.py`

1. If the cache had hits for every slide, the concatenated cached arrays are used directly — **no model inference runs at all** (`run_all.py:469-471`).
2. Otherwise (`run_all.py:473`), `extract_features(all_patches, model_name=cfg.model)` (`features/extractors.py:38-82`):
   a. Picks device: `cuda` if available else `cpu` (`features/extractors.py:54`).
   b. `get_model()` loads either a torchvision ResNet backbone (children sliced to drop the FC head, `features/extractors.py:13-26`) or, for `"phikon"`/`"phikon-v2"`, `AutoModel.from_pretrained("owkin/{model_name}")` + `AutoImageProcessor` from HuggingFace (`features/extractors.py:28-33`). Production runs use `phikon` (768-dim CLS token output).
   c. Batches of 32 patches: for Phikon, each batch is processed through the HF image processor then the model; the feature vector is `outputs.last_hidden_state[:, 0]` — the CLS token (`features/extractors.py:65-70`). For ResNet, raw tensor normalization (`/255.0`) then a forward pass with global-pool squeeze (`features/extractors.py:71-78`).
   d. Returns `(N, 768)` float32 array for Phikon.
3. **Cache shape contract:** the on-disk cache always stores features for *every* extracted (uncapped) patch of a slide. The cap is applied only in Pass 2 after the cohort median is known, so a single cache build supports any `cap_strategy`/`active_cap` combination downstream, including all 16 LOO training runs reusing the same cache. A mismatch (`len(cached) != orig_count`) means patch-extraction settings (`patch_size`, `stride`, annotations, `min_roi_coverage`) changed since the cache was built, and raises `RuntimeError` rather than silently reusing stale features.

### Phase 4: Morphological Clustering — `analysis/clustering.py`, invoked at `run_all.py:481-510`

1. `fit_pca(features, variance_target=0.95)` (`analysis/clustering.py:32-54`): `StandardScaler().fit_transform()` then `PCA(n_components=0.95, random_state=42)` — keeps the minimum number of components explaining ≥95% variance (typically ~100-200 PCs for 768-dim Phikon features, logged at runtime).
2. **Optional Harmony** (`--harmony`): `apply_harmony(X_pca, slide_names, slide_ids, key=cfg.harmony_key)` (`analysis/harmony.py:46-121`):
   a. Builds per-patch batch labels from slide name parsing — `section_number` (`"2M-1"`/`"2M-2"`, last two hyphen tokens), `mouse_id` (first hyphen token), or `slide_id` (raw integer index) (`analysis/harmony.py:23-43`).
   b. Wraps `X_pca` in a throwaway `AnnData`, calls `sc.external.pp.harmony_integrate(adata_tmp, key="batch", nclust=10)` — `nclust=10` is deliberately lower than harmonypy's own default (`min(N/30, 100)`) because with only 2-4 batches, 100 internal k-means clusters over-fits and converges in ≤2 iterations, which trips the harmonypy 0.2.0 shape bug (hence the 0.0.9 pin, see "SLURM / Narval Environment").
   c. Hard shape-equality check: `X_corrected.shape != X_pca.shape` raises `ValueError` — guards against ever silently shipping a squeezed/corrupted batch-corrected embedding.
3. `X_embed = X_pca` (uncorrected) or the Harmony output, depending on `--harmony`. This is the representation everything downstream (UMAP, Leiden, DPT) actually uses.
4. `run_umap(X_embed, n_neighbors=30, min_dist=0.1, metric="cosine")` (`analysis/clustering.py:59-78`) — **visualization only**; UMAP coordinates are never fed into clustering or DPT.
5. `cluster(X_embed, method=cfg.clustering_method, resolution=cfg.leiden_resolution)` → dispatches to `cluster_leiden()` by default (`analysis/clustering.py:106-156`):
   a. Builds a k-NN graph in `X_embed` space (`sklearn.neighbors.kneighbors_graph`, `n_neighbors=15`, cosine metric).
   b. Converts k-NN distances to similarities via a Gaussian kernel with `sigma = median(distances)` (`analysis/clustering.py:133-137`).
   c. Builds an `igraph.Graph`, simplifies (`combine_edges="max"`), runs `leidenalg.find_partition()` with `RBConfigurationVertexPartition` at `resolution_parameter=cfg.leiden_resolution` (default 0.5) (`analysis/clustering.py:139-151`).
6. `check_slide_independence(cluster_labels, slide_ids, dominance_threshold=0.80)` (`analysis/clustering.py:221-257`) — flags any cluster where one slide contributes >80% of its members as a likely single-slide artifact rather than real shared morphology; result saved to `slide_independence.json`.
7. `get_cluster_centroids(X_embed, cluster_labels)` (`analysis/clustering.py:260-281`) — per cluster, computes the centroid in embedding space and the index of the single closest real patch to that centroid (used for the cluster-patch-grid figure and as the `AtlasProjector`'s nearest-centroid lookup).
8. If `--harmony` was used, **both** representations are persisted on the AnnData: `adata.obsm["X_pca_original"] = X_pca` and `adata.obsm["X_pca_harmony"] = X_embed` (`run_all.py:527-529`) — the `AtlasProjector`'s KNN regressor is trained on `X_pca_original` specifically so that projecting a brand-new slide (which only gets a raw, non-Harmony-corrected PCA) stays in the same input space the regressor was fit on.

### Phase 5: Diffusion Pseudotime — `analysis/diffusion.py`, invoked at `run_all.py:517-554`

1. `build_adata(X_embed, cluster_labels, slide_ids, X_umap)` (`analysis/diffusion.py:24-41`) — wraps the embedding into an `AnnData` with `X = X_embed`, `obs["cluster"]` / `obs["slide_id"]` as strings, and `obsm["X_umap"]` if available.
2. `adata.obs["mouse_id"]` and `adata.obs["section_number"]` are derived from slide-name string parsing (`run_all.py:520-525`, same logic as Harmony's batch-label parser, duplicated rather than shared).
3. `compute_diffusion_map(adata, n_neighbors=30, n_comps=10)` (`analysis/diffusion.py:44-59`):
   a. `sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")` — builds the k-NN graph in `X_embed` space, written to `adata.obsp["connectivities"]`/`["distances"]`.
   b. `sc.tl.diffmap(adata, n_comps=10)` — writes `adata.obsm["X_diffmap"]` (10 components).
4. **PAGA topology gate — diagnostic only, still uncommitted as of 2026-06-22** (`analysis/diffusion.py:202-241`, wired at `run_all.py:533-543`):
   a. Coerces `adata.obs["cluster"]` to a pandas categorical if it isn't already (`analysis/diffusion.py:221-222`) — PAGA requires categorical group labels.
   b. `sc.tl.paga(adata, groups="cluster")` populates `adata.uns["paga"]["connectivities"]`, a cluster-by-cluster connectivity matrix.
   c. Thresholds that matrix at `> 0.05` (hardcoded, no CLI flag), converts to a sparse adjacency, and runs `scipy.sparse.csgraph.connected_components()` to count components (`analysis/diffusion.py:227-231`).
   d. Prints whether the manifold is a single connected component (DPT trustworthy) or N disconnected ones (DPT will produce `inf`/clamped artifacts for the smaller components) — **printed only, does not abort or alter the run**.
   e. Saves two figures: `qc_paga_topology.png` (`utils/viz.py:plot_paga`, scanpy's native PAGA graph plot) and `qc_umap_section_vs_cluster.png` (`utils/viz.py:plot_umap_section_cluster`, side-by-side UMAP colored by `section_number` vs. by Leiden cluster — lets you check by eye whether disconnected components track the batch/section divide).
5. `compute_nuclear_density_quick(all_patches)` (`validation/morphological_features.py:159-180`) — a fast, root-selection-only density estimate: hematoxylin deconvolution (`rgb2hed`, H channel) + Otsu threshold + connected-component count, divided by patch area. Deliberately cheaper than the full Phase 6 feature suite since it only needs to *rank* patches, not produce final validated numbers. Per-patch failures are swallowed and left at `0.0` (`analysis/morphological_features.py:172-178` — silent, no count of failures is logged).
6. `compute_dpt_multi_root(adata, nuclear_density_quick, n_roots=20)` (`analysis/diffusion.py:146-199`):
   a. `root_candidates = argsort(nuclear_density)[:20]` — the 20 *lowest*-density patches (sparsest/most acellular regions) become root candidates.
   b. For each candidate: deep-copies the AnnData (`adata.copy()`), sets `adata_tmp.uns["iroot"] = candidate_idx`, runs `sc.tl.dpt(adata_tmp)`, reads off `dpt_pseudotime`. Any `inf` values (disconnected-component artifacts) are clamped to that run's max finite value before being stored (`analysis/diffusion.py:179-183`).
   c. Stacks all 20 runs into a `(20, N)` matrix; final `pseudotime = median(matrix, axis=0)`, then min-max normalized to `[0, 1]`; `pseudotime_std = std(matrix, axis=0)` is stored **un-normalized** as an uncertainty map (`analysis/diffusion.py:185-198`).
   d. Note: single-root selection (`choose_root_cell()` / `compute_dpt()`, `analysis/diffusion.py:62-117`) still exists in this module and is fully functional, but `run_pipeline()` never calls it — multi-root DPT fully replaced it as of commit `62fb25c` (2026-06-04). The `--root-cluster`/`--root-metric` CLI flags are vestigial (see Issue 1).
7. Pseudotime figures generated: `fig4_umap_pseudotime.png`, `fig4b_umap_pseudotime_std.png` (magma colormap), `fig5_pt_violins.png` (per-cluster), per-slide `spatial_pt_*.png`, and `diffusion_3d.png` if ≥3 diffmap components exist (`run_all.py:557-570`).

**Disconnected graph behavior:** if the k-NN graph (pre-Harmony, or post-Harmony if correction failed) has multiple components, patches in smaller components get `dpt_pseudotime = inf` from scanpy, clamped to the max finite value per multi-root iteration — this produces a degenerate near-binary pseudotime distribution. `qc/graph_connectivity.py` is the standalone diagnostic for this; the PAGA gate above is a newer, automatic version of the same check.

### Phase 6: Morphological Feature Validation — `validation/morphological_features.py`, `validation/correlations.py`, invoked at `run_all.py:577-597`

1. `compute_morphological_features(all_patches, use_stardist=cfg.use_stardist)` (`validation/morphological_features.py:185-245`) — per patch:
   a. Hematoxylin deconvolution (`rgb2hed`, H channel) (`validation/morphological_features.py:22-30`).
   b. Nuclear segmentation: Otsu threshold + binary opening (disk radius 1) + remove objects <20px + connected-component labeling (`_segment_nuclei_simple`, `validation/morphological_features.py:33-52`) — or StarDist (`owkin`'s `"2D_versatile_he"` pretrained model) if `--use-stardist`, with automatic fallback to Otsu if `stardist`/`tensorflow` aren't installed (`validation/morphological_features.py:55-72`).
   c. Six features computed per patch: `nuclear_density` (nuclei count / patch area), `mean_nuclear_area` (mean `regionprops` area), `nc_ratio` (nuclear pixels / cytoplasm pixels), `texture_entropy` (Shannon entropy of the GLCM on grayscale, averaged over distances 1/3/5), `h_intensity` (mean hematoxylin optical density), `packing_irregularity` (coefficient of variation of nearest-neighbor distances between nuclear centroids, via `scipy.spatial.KDTree`) (`validation/morphological_features.py:75-154`).
   d. Per-patch exceptions are caught and leave that patch's features at `0.0`; only the first 5 failures per run are printed (`validation/morphological_features.py:240-243`) — silent beyond that, so a run with many segmentation failures won't visibly warn you.
2. `run_full_validation(pseudotime, morph_features, cluster_labels, all_coords, n_permutations=1000)` (`validation/correlations.py:188-233`):
   a. `correlate_features_with_pseudotime()` — Spearman ρ per feature, with an interpretation label: `|rho|>0.4` strong, `>0.3` moderate, else weak (`validation/correlations.py:9-47`).
   b. `permutation_test()` — shuffles pseudotime 1000× (seed 42), recomputes `|rho|` each time per feature, reports the empirical p-value (`fraction of null >= |real rho|`) and the null's 95th percentile; `significant = perm_p < 0.05` (`validation/correlations.py:50-113`).
   c. `cluster_ordering_analysis()` — median/mean/std pseudotime per cluster, ranked (`validation/correlations.py:116-150`).
   d. `spatial_depth_correlation()` — a **secondary** check: Spearman ρ between pseudotime and distance-from-ROI-boundary (or distance-from-centroid if no polygon given) (`validation/correlations.py:153-183`). Flagged in its own print statement as only meaningful if morphological features *also* correlate — otherwise it indicates the trajectory is just tracking spatial position, not biology.
   e. **Verdict logic** (`validation/correlations.py:206-219`): counts `n_strong` (features with `|rho|>0.4`) and `n_sig` (features passing the permutation test). `n_strong>=2 and n_sig>=2` → `"POSITIVE"`; exactly one of either → `"CAUTIOUS"`; otherwise → `"NULL RESULT"`. Note this is a different/finer rule than the "SIGNIFICANT/TREND/NULL" labels previously documented here — the actual three verdict strings emitted by the code are `POSITIVE`, `CAUTIOUS`, and `NULL RESULT`.
3. Figures: `fig6_features_vs_pt.png` (scatter per feature with correlation), `fig7_permutation_nulls.png` (null distributions) (`run_all.py:589-597`).

### Phase 7: Save Artifacts — `run_all.py:599-647`, `utils/io.py`

Under `output_dir/`, in the order they're written:
1. `adata_full.h5ad` — the complete AnnData (`X`=embedding, all `obs` columns including pseudotime/pseudotime_std/morph features, `obsm` including UMAP/diffmap/PCA variants) (`run_all.py:606-607`).
2. `results.csv` — flat per-patch table: `x, y, slide_id, slide_name, cluster, pseudotime, pseudotime_std` + the 6 morphological feature columns (`run_all.py:609-621`).
3. `sampling_manifest.csv` + `sampling/{slide}_sample_idx.npy` — only written if any slide was actually capped (`run_all.py:623-632`).
4. `validation.json` — full `run_full_validation()` output via `io.save_json()`, which recursively converts numpy scalars/arrays to native Python types for JSON-safety (`run_all.py:634`, `utils/io.py:29-45`).
5. `scaler.pkl`, `pca.pkl`, `umap_reducer.pkl` — fitted sklearn objects, raw `pickle.dump()` (`run_all.py:637-639`, `utils/io.py:19-21`).
6. `slide_independence.json` — cluster/slide dominance breakdown (`run_all.py:641`).
7. `projector/` — `AtlasProjector.from_training(scaler, pca, umap_reducer, adata, centroids).save(...)` (`run_all.py:644-647`) — the artifact used for LOO projection and any future new-slide projection (see LOO Architecture section).
8. Console summary: elapsed minutes, total patches, cluster count, verdict string, and a 3-step "next steps" hint to inspect `fig2_cluster_patches.png` (`run_all.py:649-664`) — this hint is stale boilerplate left over from the single-root-DPT era; root selection is now fully automatic (see Issue 1), so steps 1-2 of that hint no longer apply.

---

## AnnData obs Column Reference

After a complete `run_all.py` run, `adata.obs` contains:

| Column | Type | Description |
|---|---|---|
| `cluster` | str | Leiden cluster label (integer as string: "0", "1", ...) |
| `slide_id` | str | Integer slide index (as string: "0"..."15") |
| `mouse_id` | str | Mouse identifier: "6027", "6028", "6029", "6031" |
| `section_number` | str | Section: "2M-1" or "2M-2" |
| `dpt_pseudotime` | float | Raw scanpy DPT (stored per-root internally; not in final obs) |
| `pseudotime` | float | Median of n_roots DPT runs, normalized to [0, 1] |
| `pseudotime_std` | float | Std of n_roots DPT runs (un-normalized; uncertainty map) |
| `nuclear_density` | float | Morphological feature |
| `mean_nuclear_area` | float | Morphological feature |
| `nc_ratio` | float | Morphological feature |
| `texture_entropy` | float | Morphological feature |
| `h_intensity` | float | Morphological feature |
| `packing_irregularity` | float | Morphological feature |

`adata.obsm` contains: `X_umap`, `X_diffmap`, optionally `X_pca_original`, `X_pca_harmony` (Harmony runs).

---

## Configuration System

### `pipeline_config.py` — PipelineConfig dataclass

The single config object passed through `run_all.py`. All CLI args are wired to this. Clean.

### `paths.json` — Deployment path config

Loaded by both `run_all.py` and `run_individual.py` to set default paths on Narval. Contains:
- `raw_ndpi`, `cropped_png`, `annotations`, `results`, `stain_reference`

**KNOWN DISCREPANCY:** `paths.json` points to `~/cancer_trajectory_atlas/data/annotations` (raw GeoJSON). Job scripts use `~/cancer_trajectory_atlas/data/annotations_ratio` (ratio JSON). Both directories exist. `load_roi_polygons()` handles both formats (detects `FeatureCollection` type). However, using raw GeoJSON may produce different polygon scaling if coordinates are absolute-pixel rather than ratio. **Use `annotations_ratio` for all production runs.**

### Job scripts — hardcoded paths

SLURM scripts in `jobs/` hardcode key paths. The `--annotation-dir` in production job scripts points to `~/cancer_trajectory_atlas/data/annotations_ratio`. This is the authoritative source for cluster runs.

### `KNOWN_NDPI_DIMENSIONS` — fallback dimension table

Used when `slide_dimensions.json` sidecar is absent (e.g., if PNG was generated externally). Stores full NDPI level-0 dimensions (both copies side-by-side). Applied with `ndpi_scale` factor.

---

## LOO Experiment Architecture

The leave-one-out experiment runs in two phases:

**Phase A (SLURM array, `submit_loo_array.sh`):**  
For each of 16 slides as held-out:
1. `run_all.py` on 15 training slides (features from cache, no GPU needed)
2. Saves projector to `loo_{slide}/projector/`

**Phase B (same job, `run_loo_single.sh` → `loo_project.py`):**  
1. Load projector
2. Load held-out slide features from cache (apply same cap if was_capped)
3. KNN project → pseudotime for held-out patches
4. Load in-manifold pseudotime from full-run `results.csv`
5. Compute Spearman ρ (paired patch-level), Wasserstein, KS (distribution-level)
6. Save `loo_result_{slide}.json` + distribution plot

**Patch ordering invariant (critical):** The feature cache stores patches in extraction order. `results.csv` also stores patches in extraction order. LOO Phase B compares cache index i against results.csv index i for the same slide. This pairing is only valid if both used identical extraction settings (same patch_size, stride, annotations, min_roi_coverage). **Any change to extraction settings invalidates the pairing and requires cache rebuild.**

**AtlasProjector projection path:**  
`raw_features (768-dim)` → scaler → pca → `X_pca_test` → KNN regressor (trained on `X_pca_original` of training set) → pseudotime

Harmony correction is NOT applied to the test slide during projection. The KNN regressor is trained on pre-Harmony PCA to ensure the projection input space is consistent with what the model was trained on.

---

## Known Issues and Tech Debt

### Issue 1: `--root-cluster` CLI argument exists but is dead code

**CORRECTED 2026-06-22** — this issue previously claimed the flag didn't exist at all. It does: `run_all.py:766` (`parser.add_argument("--root-cluster", type=int, default=None, help="Legacy arg; unused with multi-root DPT.")`), and it's threaded onto `PipelineConfig.root_cluster` at `run_all.py:826`. But `run_pipeline()` never reads `cfg.root_cluster` — the only DPT entry point called is `compute_dpt_multi_root(adata, nuclear_density_quick, n_roots=cfg.n_roots)` (`run_all.py:551`), which takes `n_roots`, not a root cluster. Single-root selection (`choose_root_cell()` / `compute_dpt()` in `analysis/diffusion.py:62-117`) is still in the codebase and still works if called directly, but nothing in the live `run_all.py` path calls it anymore — multi-root DPT (Issue 8 below) fully replaced single-root selection as of 2026-06-04.

**Impact:** None currently — there is no friction here because root selection is fully automatic (20 lowest-nuclear-density candidates, median-aggregated). The flag is just vestigial; passing `--root-cluster N` silently does nothing.

**Fix (optional cleanup):** Either remove the `--root-cluster` flag and the now-dead `root_cluster` field from `PipelineConfig`, or repurpose it as a seed/override for one of the 20 multi-root candidates if a use case for that emerges. Not urgent.

### Issue 2: `run_individual.py` uses module-level globals

`PNG_DIR`, `MODEL`, `PATCH_SIZE`, `STRIDE`, `LEIDEN_RESOLUTION`, `STAIN_NORMALIZATION`, `MIN_ROI_COVERAGE` are module-level globals set by `main()`. This is inconsistent with `run_all.py` which uses `PipelineConfig`. Makes it harder to call `run_one_slide()` from other scripts without going through the CLI.

**Impact:** Low. `run_individual.py` is a standalone diagnostic tool that only runs via CLI. But it's an architectural inconsistency that would trip up anyone importing from it.

### Issue 3: `run_train_test.py` is broken dead code

`run_train_test.py` imports `from . import config as default_config`, but `config.py` does not exist (the file is `pipeline_config.py`). This means `run_train_test.py` will fail on import. It also uses `from .validation.annotations import load_annotations` — the legacy label-assignment annotation function, not the ROI-polygon function the main pipeline uses.

**Impact:** None on current workflows. The module simply cannot run. It appears to be an early prototype that was never updated when the architecture changed.

### Issue 4: `validation/annotations.py` has two distinct purposes

The file contains:
- `load_annotations()` — assigns ground-truth class labels to patches (used by `run_train_test.py` only, which is dead)
- `_load_mask_annotations()` — legacy colored-mask label loading (never called by any live code)
- `_load_qupath_geojson()` — polygon-based label assignment

None of this is called by the main pipeline (`run_all.py`). The main pipeline uses `features/patching.py:load_roi_polygons()` for ROI filtering, which is a different function with a different purpose (filter patches to annotation regions, vs. assign classification labels).

**Impact:** Confusion about what annotation loading looks like in this codebase. A future developer might find `validation/annotations.py` and think it's the active annotation-loading code.

### Issue 5: Annotation directory discrepancy

Three annotation paths appear in the codebase:
- `paths.json`: `~/cancer_trajectory_atlas/data/annotations` (raw GeoJSON)
- Job scripts: `~/cancer_trajectory_atlas/data/annotations_ratio` (ratio JSON)
- `NOTES.md` says canonical is `$SCRATCH/data/annotations` (different location entirely)

`load_roi_polygons()` handles both GeoJSON and JSON formats correctly. But if the wrong directory is used, annotations might contain absolute-pixel coordinates instead of ratio coordinates, causing a coordinate scale mismatch (polygons would be scaled by `original_full_width` again → 2× or more inflated polygon coordinates).

**Fix:** Update `paths.json` to point to `annotations_ratio`. Document the NOTES.md path as the archive/sync source only.

### Issue 6: Feature cache / sampling cap interaction (resolved by two-pass design)

Pass 1 always loads full (uncapped) features. Pass 2 applies the cap via `sample_patches`.
The cache shape contract is enforced by `len(slide_feats) != orig_count` → RuntimeError.

The `cap_strategy` arg controls sampling mode (default changed to `'median'` as of commit `1a8b471`, "run with median patch cap"; was `'fixed'` previously):
- `'fixed'` (default): cap each slide at `max_patches_per_slide` (default 200, Vig et al.)
- `'median'`: computes cohort median `orig_count` after Pass 1, uses it as cap for all slides
- `'none'`: backward compat — uses `max_patches_per_slide` if set, else no cap

`sample_patches` now shuffles the full patch list then takes the first N (not `np.choice` + sort),
so patch order is randomised. Slide order is also shuffled before Pass 2 begins.

`sampling_manifest.csv` records per-slide: slide_id, patches_before_cap, patches_after_cap,
was_capped, seed, pct_retained. Total vs. target_total (default 3200) is logged to stdout.

**Impact:** The LOO experiment is sensitive to cache/extraction settings. A cache built in one run must match exactly the extraction settings of the LOO training runs. The patch ordering invariant depends on this.

### Issue 7: `jobs/recover_loo_phase_b.py` misplaced — RESOLVED

**RESOLVED, confirmed 2026-06-22.** This issue previously described `jobs/recover_loo_phase_b.py` (a Python module living in a shell-script directory) as tech debt. The file no longer exists at that path — it was moved to `analysis/recover_loo.py` in commit `0fa4880` ("refactor/deleted unused files, added smoke test for entire pipeline"). `jobs/` now contains only shell scripts and `jobs/check_annotations.py` (a separate, intentionally-placed diagnostic — leave as is). No action needed; kept here only so a future session doesn't re-open this as a live issue.

### Issue 8: Multi-root DPT — design and parameters (implemented 2026-06-04)

Single-root DPT is sensitive to root choice. Replaced with multi-root averaging:
- `n_roots=20` (default) lowest-nuclear-density patches selected as root candidates
- DPT run independently from each candidate; results stacked into (20, N) matrix
- `pseudotime = median(matrix, axis=0)`, normalized to [0, 1]
- `pseudotime_std = std(matrix, axis=0)`, stored un-normalized (uncertainty map)

Nuclear density is pre-computed just before Phase 4 via `compute_nuclear_density_quick()`
(hematoxylin deconvolution + Otsu, no full morphological feature suite). Phase 5 still
runs the full feature suite independently.

New figure: `fig4b_umap_pseudotime_std.png` — UMAP colored by pseudotime_std (magma colormap).

CLI args: `--n-roots 20`, `--root-metric cellularity` (note: `--root-metric` is also parsed but, like `--root-cluster`, not actually consumed by `compute_dpt_multi_root()` — the function signature only takes `n_roots`; ranking is always by nuclear density via `compute_nuclear_density_quick()`, there is no alternate metric implemented yet despite the `choices=["cellularity"]` CLI constraint implying one exists). `--root-cluster` is now a legacy no-op (see Issue 1).

### Issue 9: `paths.json` has unused `stain_reference` key

`paths.json` has `"stain_reference": "~/scratch/data/MCF7_x5_cropped/6028-4L-2M-2_x5.png"` but neither `run_all.py` nor `run_individual.py` read this key. The reference image for normalization is always `slides[0]` (first alphabetically). Misleading — suggests the reference is configurable via paths.json when it is not.

---

## Results Directory Convention (Current State)

Under `$SCRATCH/results/` (Narval):

| Directory | Contents | Status |
|---|---|---|
| `atlas_none_harmony` | no-stain + Harmony, 16 slides, uncapped | complete (old) |
| `atlas_macenko_harmony` | Macenko + Harmony, uncapped | complete (old) |
| `atlas_none_section1` | Section 2M-1 slides only | complete (old) |
| `atlas_none_section2` | Section 2M-2 slides only | complete (old) |
| `atlas_none_harmony_cap1900` | no-stain + Harmony + 1900-patch cap | complete (old) |
| `baseline/atlas_none_harmony_median` | no-stain + Harmony + 200-patch cap + 20-root DPT | **pending (run_full_experiments.sh)** |
| `baseline/atlas_macenko_harmony_median` | Macenko + Harmony + 200-patch cap + 20-root DPT | **pending (run_full_experiments.sh)** |
| `loo/loo_{slide}` × 16 | LOO 15-slide training + Phase B projection | **pending (run_full_experiments.sh)** |
| `loo_summary` | Aggregated LOO CSV + stability figure | not yet run |
| `individual_pseudotime_runs/` | Per-slide standalone pseudotime runs | complete (old) |
| `slide_diagnostics_{slide}/` | Five-hypothesis diagnostic for outlier slide | not yet run |
| `confound_analysis/` | Cellularity confound scatter + CSV | not yet run |

**Current pipeline parameters (run_full_experiments.sh):**
- `--cap-strategy median` (default as of commit `1a8b471`; cohort-median patch count per slide, computed dynamically from Pass 1 extraction and logged to `active_cap.txt` in the output dir — see `run_all.py` `_cap_file_val`)
- `--n-roots 20` (multi-root DPT averaging; lowest-nuclear-density candidates)
- `--stain-method none --harmony --harmony-key section_number` (canonical config)
- LOO Phase B reads `BASELINE_CAP` from Baseline A's `active_cap.txt` and passes it as `--max-patches-per-slide` so held-out slide sampling matches the median cap used to build the reference manifold (paired Spearman comparison requires equal patch counts)

---

## SLURM / Narval Environment

- **Account:** `def-lmarti46`
- **venv:** `~/envs/atlas/` (activate with `source ~/envs/atlas/bin/activate`)
- **Modules:** `StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph`
- **Phikon model cache:** `$SCRATCH/huggingface_cache` (offline mode: `TRANSFORMERS_OFFLINE=1`)
- **Feature cache:** `$SCRATCH/data/features_cache/` (16 × `{slide}_features.npy`)

**Critical dependency versions:**
- `harmonypy`: must stay on `0.0.9` — 0.2.0 has a shape-squeezing bug. See NOTES.md.
- `leidenalg` + `python-igraph`: NOT in requirements.txt, must be conda-installed.
- `spams` (for staintools Macenko): NOT in requirements.txt, must be conda-installed.

**SLURM BASH_SOURCE note:** Job scripts must use hardcoded absolute paths (`~/cancer_trajectory_atlas/jobs/...`) for sibling file references, not paths derived from `${BASH_SOURCE[0]}`, which resolves to the SLURM spool directory.

---

## Things That Work and Must Not Be Changed

1. **Scientific methodology:** clustering parameters, Harmony settings, Phikon usage, DPT parameters, morphological feature computation — all validated.
2. **Coordinate system:** ratio coords × original_full_width = full NDPI pixel space = cropped PNG pixel space for left-half polygons. Do not change this.
3. **Feature cache layout:** `{slide_name}_features.npy`, unsampled, full extraction. Cap applied at load time. LOO experiment depends on this contract.
4. **AtlasProjector Harmony invariant:** KNN regressor trained on `X_pca_original` (pre-Harmony). Do not change this without rebuilding all projectors.
5. **SLURM setup:** venv, module list, account, output log paths. Don't change without testing on Narval.
6. **`slide_dimensions.json` format:** both `run_all.py` and `run_individual.py` read it. Changing key names breaks both.

---

## Refactor Plan

Each item is scoped to a single session of work. Items are ordered: highest value / lowest risk first. Items marked SAFE can be done in any order. Items marked CAUTION need testing before merging.

---

### Item 1 — Add `--root-cluster` CLI argument [HIGH VALUE, LOW RISK]

**What:** Add `--root-cluster INT` to `run_all.py` (and `run_individual.py`). When provided, use it instead of hardcoding cluster "0". When absent, keep current auto-select with the "re-run" prompt.

**Why:** Currently every first run is wasted because root is always wrong. This is the single most annoying friction point.

**Files:** `run_all.py`, `pipeline_config.py`, `run_individual.py`

**Risk:** None to existing behavior if absent. With it present, user-specified cluster that doesn't exist raises ValueError (already handled in `choose_root_cell()`).

**Test:** Run with `--root-cluster 2` and verify `adata.obs["pseudotime"]` is not all-zero.

---

### Item 2 — Fix `paths.json` annotation path + document the annotation paths [HIGH VALUE, LOW RISK, SAFE]

**What:** Update `paths.json` `"annotations"` to point to `annotations_ratio`. Remove the unused `"stain_reference"` key. Add a comment block at the top of `paths.json` (or a README in `data/`) explaining the two annotation directories.

**Why:** The current `paths.json` points to raw GeoJSON (`annotations/`) while all job scripts use ratio JSON (`annotations_ratio/`). Running via `paths.json` defaults may silently use wrong annotations.

**Files:** `paths.json`, optionally `data/README.md` (new file, only if user wants it)

**Risk:** None — annotation loading handles both formats, but ratio JSON is what's been tested.

**Test:** Run `run_all.py --run` (with defaults from paths.json) and verify annotation counts match job script runs.

---

### Item 3 — Delete or archive `run_train_test.py` [MEDIUM VALUE, LOW RISK, SAFE]

**What:** Either delete `run_train_test.py` or move it to an `archive/` subdirectory.

**Why:** It imports `config.py` (which doesn't exist), so it cannot run. It uses the legacy annotation API and AtlasProjector — architecture it was built on has diverged. It is dead code that confuses codebase navigation.

**Risk:** None — it can't run anyway. Confirm no job scripts reference it before deleting.

**Test:** `grep -r "run_train_test" jobs/` to confirm no job scripts call it.

---

### Item 4 — Clean up `validation/annotations.py` [LOW VALUE, LOW RISK, SAFE]

**What:** Move legacy mask-loading code (`_load_mask_annotations`, `_get_filled_mask`) to `archive/` or delete. Keep only the minimal public interface if anything outside still calls it (check with grep first). Document that `features/patching.py:load_roi_polygons()` is the active annotation-loading function.

**Why:** The file has two conflicting purposes (label assignment vs. ROI filtering) and most of its code is never called. Creates confusion about which annotation-loading function is authoritative.

**Files:** `validation/annotations.py`

**Risk:** Low. `grep -r "from .validation.annotations" .` to find all callers. Currently only `run_train_test.py` (dead) calls it.

**Test:** `python -c "from cancer_trajectory_atlas.validation import annotations"` should import cleanly after.

---

### Item 5 — Move `jobs/recover_loo_phase_b.py` to `analysis/` [LOW VALUE, LOW RISK, SAFE]

**What:** Move `jobs/recover_loo_phase_b.py` → `analysis/recover_loo.py`. Update any job script or README that references its location.

**Why:** It's a Python analysis module sitting in a shell-script directory. Future Claude sessions will look in `analysis/` for LOO-related code and not find it.

**Risk:** None if no active job scripts reference its path. Check first with `grep -r "recover_loo" jobs/`.

**Test:** Can still run as `python -m cancer_trajectory_atlas.analysis.recover_loo` after move.

---

### Item 6 — Convert `run_individual.py` globals to a config object [MEDIUM VALUE, LOW RISK]

**What:** Replace the 8 module-level globals in `run_individual.py` with a small dataclass (or just reuse `PipelineConfig`). Pass the config into `run_one_slide()` instead of relying on globals.

**Why:** Inconsistency with `run_all.py` which uses `PipelineConfig`. Module-level globals make the code untestable and fragile if any future code tries to import and call `run_one_slide()` directly.

**Files:** `run_individual.py`

**Risk:** Only internal refactor — no external API changes. No downstream callers of `run_one_slide()` exist outside this file.

**Test:** Run `python -m cancer_trajectory_atlas.run_individual --slide 6027` and verify output is identical.

---

### Item 7 — Establish results directory naming convention [MEDIUM VALUE, MEDIUM RISK]

**What:** Define and document a naming convention for `$SCRATCH/results/`:
- `baseline/` — canonical reference runs (atlas_none_harmony, atlas_macenko_harmony, etc.)
- `ablation/` — parameter variation experiments
- `loo/` — leave-one-out runs (loo_{slide}_*, loo_summary)
- `diagnostic/` — QC and post-hoc analyses (slide_diagnostics_*, confound_analysis)
- `individual/` — per-slide standalone runs

**Why:** Flat naming makes it hard to understand experiment history. As more runs accumulate, clobbering risk increases.

**Risk:** MEDIUM — job scripts have hardcoded output paths like `$SCRATCH/results/loo_{slide}`. Changing the structure requires updating all job scripts AND any existing results references in `loo_slides.txt`, `FULL_RUN` env var, etc. Do not rename existing results — only apply the convention to new runs.

**Files:** `jobs/*.sh` (update default output paths), `NOTES.md` (add the convention)

**Test:** Verify that new LOO runs write to `$SCRATCH/results/loo/{slide}` and the array job's `FULL_RUN` env var points to the correct baseline directory.

---

### Item 8 — Document the coordinate system formally in `features/patching.py` [LOW VALUE, LOW RISK, SAFE]

**What:** Add a coordinate-system docblock to `load_roi_polygons()` explaining the three coordinate spaces and the invariant. No code change.

**Why:** The coordinate system is correct but undocumented. Any future developer (or future Claude session) touching annotation loading will be confused by `original_full_width` (which sounds like it should be the cropped width). The NOTES.md says a coordinate bug was fixed — that knowledge will be lost.

**The invariant to document:**
- Full NDPI space: width = NDPI level-0 × ndpi_scale (includes BOTH slide copies)
- Ratio annotations × original_full_width → full NDPI pixel space
- Patch coords from get_patches_from_array → cropped PNG pixel space (= left half of NDPI space)
- These spaces are IDENTICAL for left-half annotations because original_full_width/2 == cropped_width

**Files:** `features/patching.py`

**Risk:** Zero. Documentation only.

---

### Items NOT being done

- **Scientific changes:** clustering resolution, Harmony nclust, DPT parameters, morphological feature computation — these are validated and produce the paper's results.
- **Pipeline architecture rewrite:** The data flow in `run_all.py` is clear and linear. Extracting phases into separate scripts would fragment what is currently easy to trace end-to-end. Not worth it.
- **Config format change:** `PipelineConfig` is a clean dataclass. Moving to YAML/TOML adds complexity without benefit at this project scale.
- **requirements.txt completion:** leidenalg, spams are intentionally not in requirements.txt because they require conda (C extensions). Adding them would give a false sense of completeness.
- **Parallelizing patch extraction:** Not needed — extraction is fast on Narval, GPU is the bottleneck.

---

## Working Log

**2026-06-24:** Added a standalone batch-effect diagnostic, `analysis/plot_umap_by_section.py` (SLURM wrapper: `jobs/run_umap_by_section.sh`). Pure plotting only — loads an existing run's `adata_full.h5ad` (no re-embedding/clustering/Harmony/DPT) and plots `obsm['X_umap']` colored by `obs['section_number']`. Produces an overlaid scatter (draw order shuffled so neither section is painted on top) and a side-by-side split panel (same axis limits, one section per panel). Purpose: visually confirm whether Harmony actually interspersed 2M-1/2M-2 patches within shared clusters, or whether they remain spatially segregated — the latter would mean any PAGA edge between sections is cosmetic rather than evidence of a connected manifold. Confirmed via inspection that `obs['section_number']` (categorical, values `'2M-1'`/`'2M-2'`) and `obsm['X_umap']` exist as expected. Note: the saved artifact in this run dir is named `adata_full.h5ad`, not `adata.h5ad` (script originally assumed the latter; corrected after a run against `$SCRATCH/results/runs_paga/all_harmony/full/` failed with file-not-found).

---

**2026-06-24 (later):** Added a quantitative batch-mixing diagnostic — `analysis/batch_mixing.py` + `analysis/run_batch_mixing.py` (SLURM wrapper: `jobs/run_batch_mixing.sh`). Pure post-hoc analysis over `adata_full.h5ad` — no changes to `run_all.py`, `clustering.py`, or `harmony.py`, no rerun. Computes per-patch kNN section-purity (k=15, cosine — read via introspection from `cluster_leiden`'s actual defaults, not hardcoded) on `obsm['X_pca_original']` (pre-Harmony) and `obsm['X_pca_harmony']` (post-Harmony), the same representation that feeds the Leiden neighbor graph. Reports the cohort-mean same-section neighbor fraction against a prevalence-weighted chance baseline.

**Result on `$SCRATCH/results/runs_paga/all_harmony/full/adata_full.h5ad`** (18,944 patches; 2M-1: 8,872, 2M-2: 10,072; chance baseline 0.502):
- raw PCA (no correction): **0.9995** — essentially total segregation, as expected.
- post-Harmony: **0.9425** — still ~94% of each patch's 15 nearest neighbors share its section, vs. a chance baseline of ~0.50.

**Why this matters:** Harmony reduces segregation from near-total to ~94%, but 94% is nowhere close to the ~0.50 chance baseline that would indicate real intermixing. This quantitatively confirms what the UMAP-by-section diagnostic suggested visually: Harmony is under-correcting the section batch effect on the actual neighbor-graph representation (not just the UMAP projection used for visualization). Any PAGA connectivity between 2M-1/2M-2 clusters should be treated as a weak, borderline-cosmetic bridge rather than evidence of a well-mixed manifold.

**Gates:** This number is the deciding input for the Harmony-vs-scVI escalation decision — at 94% same-section purity, escalating batch correction (e.g., scVI integration in place of Harmony) is justified before trusting any cross-section DPT/pseudotime comparison.

---

**2026-06-25:** Added scVI post-processing job and analysis module.

**Job:** `jobs/run_post_processing_scvi.sh` — sbatch against `$SCRATCH/results/atlas_none_scvi`. Runs the existing `interactive_overlay` and `export_patches` tooling on the scVI run (same args as `run_post_processing.sh`: patch-size 112, n-per-bin 50), then calls the new `visualize.scvi_postprocess` module. All outputs land in:
- `$SCRATCH/results/atlas_none_scvi/overlays/` — per-slide interactive HTML pseudotime overlays (qualitative high-PT coherence check)
- `$SCRATCH/results/atlas_none_scvi/patch_export/` — patches binned low/mid/high PT (inspect `high_pseudotime/` to confirm coherent tissue, not scattered noise)
- `$SCRATCH/results/atlas_none_scvi/postprocess/` — scVI-specific figures + summary

**New module:** `visualize/scvi_postprocess.py` (read-only; never writes `adata_full.h5ad`). Produces:
- `umap_leiden_lowres.png` — UMAP coloured by Leiden re-clustered at resolution=0.4 on the stored neighbor graph (`obsp["connectivities"]`). Cluster count TBD after job runs; fill in below.
- `umap_by_section.png` + `umap_by_section_split.png` — section-mixing check in scVI latent UMAP (equivalent of the Harmony section-mixing figures produced in 2026-06-24 batch-mixing session).
- `morphology_correlations.csv` + `.png` — reloads existing `validation.json` (no recompute); adds `effect_size_flag` column.
- `SUMMARY.txt`

**Morphology table (from run output — fill in after job completes):**

| feature | rho | perm_p | effect_size_flag |
|---|---|---|---|
| h_intensity | -0.299 | [TBD] | significant |
| texture_entropy | -0.146 | [TBD] | significant |
| others | ~0.02–0.05 | [TBD] | significant_but_negligible or not_significant |

Effect-size flag logic: `significant_but_negligible` = perm_p < 0.05 AND \|rho\| < 0.05; `significant` = perm_p < 0.05 AND \|rho\| ≥ 0.05. The rho~0.02 features (if any pass the permutation test) are flagged rather than silently reported as "validated".

**leiden_lowres cluster count at resolution=0.4:** [fill after job runs — check `postprocess/SUMMARY.txt`]

**Interpretation note:** scVI closed the batch-mixing gap Harmony couldn't (section kNN purity 0.9425 with Harmony vs. near-chance with scVI). Pseudotime morphology correlations are modest — h_intensity and texture_entropy show meaningful signal (\|rho\| ≥ 0.1); remaining features are negligible. Whether the two meaningful features reflect a real trajectory or a residual cellularity confound is still open (cellularity confound test not yet run — see `analysis/cellularity_confound.py`).

---

## Session Handoff Checklist

Before any new Claude Code session, this document should answer:
- [ ] What is the canonical run configuration? → `--stain-method none --harmony --harmony-key section_number`
- [ ] What is the active annotation directory? → `data/annotations_ratio/`
- [ ] How does coordinate scaling work? → See "Annotation coordinate invariant" above
- [ ] What is the LOO patch ordering contract? → Cache stores unsampled; cap re-applied at load time; same extraction settings required for pairing to be valid
- [ ] What harmonypy version? → Must stay on 0.0.9
- [ ] What is the next experiment to run? → Cellularity confound test (`analysis/cellularity_confound.py`)
- [ ] Is `run_train_test.py` safe to edit? → No — it's broken dead code
- [ ] What is the default patch cap strategy now? → `median` (changed from `fixed` in commit `1a8b471`)
- [ ] Is the PAGA topology gate committed? → No — still staged in working tree only (`analysis/diffusion.py`, `run_all.py`, `utils/viz.py`) as of 2026-06-22. Commit it (or continue iterating) before relying on it in a fresh checkout.
- [ ] Does `--root-cluster` do anything? → No, it's parsed but never consumed by the live multi-root DPT path (Issue 1). Don't rely on it.
- [ ] Where is the LOO recovery script? → `analysis/recover_loo.py` (NOT `jobs/` — that was fixed in commit `0fa4880`; Issue 7 closed).
- [ ] Has `run_full_experiments.sh` actually completed on Narval? → Unconfirmed from this checkout (no local results artifacts from it). Check `$SCRATCH/results/baseline/` and `$SCRATCH/results/loo/` on the cluster before assuming any baseline/LOO numbers exist.

---

## Working Log

### 2026-07-27 — Holeyness validation module added

**Module:** `analysis/holeyness.py`  
**Job script:** `jobs/run_holeyness_validation.sh`

**Purpose:** Correlate duct-level pseudotime against QuPath-measured hole fraction (holey-ness) as an independent morphological validation of the pseudotime axis.

**Input files:**
- `$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt` — QuPath measurement export (tab-separated); Tumor rows only; contains `Object ID` (UUID), `holes_carnoys: hole %`, `holes_carnoys: hole area µm^2`, centroids in µm
- `~/cancer_trajectory_atlas/data/annotations_ratio/*.json` — per-slide Tumor polygon geometry in ratio coords; each feature carries the same QuPath UUID as `"id"` field
- `$SCRATCH/data/MCF7_x5_cropped/slide_dimensions.json` — pixel dimensions per slide (used for ratio→pipeline px conversion)
- `$SCRATCH/results/per_section/atlas_<section>/results.csv` — per-patch pseudotime, nuclear_density, packing_irregularity

**Assignment strategy:** Cross-file UUID join (confirmed Case A accuracy, not disc approximation).
The measurement export does not contain polygon geometry (ROI column = shape-type label only). However, `data/annotations_ratio/*.json` already contains the same QuPath Tumor polygon vertices — same UUIDs confirmed by spot-check on `6027-4L-2M-1`. Join on UUID → polygon + hole %. Ratio coords × `original_full_width` → pipeline pixel space. Right-half polygons excluded by centroid x > `cropped_width`. Patch centres assigned via `matplotlib.path.Path.contains_points()`.

**ndpi_scale = 0.5 confirmed:** `slide_dimensions.json.original_full_width` values are exactly half the `KNOWN_NDPI_DIMENSIONS` (img_dims.txt) fallback values for every slide.

**Extra slides dropped:** 8 non-pipeline slides (timepoints 4W/8W: 6037, 6069, 6070, 6096, 6097, 6099) present in the combined export but not in the pipeline.

**Outputs (per section):**
- `$SCRATCH/results/holeyness/<section>/holeyness_per_duct.csv`
- `$SCRATCH/results/holeyness/<section>/holeyness_validation.json`
- `$SCRATCH/results/holeyness/<section>/scatter_pt_vs_hole_pct.{pdf,png}`
- `$SCRATCH/results/holeyness/<section>/scatter_hole_pct_vs_nd.{pdf,png}`

**Statistics computed:** Spearman ρ(pseudotime, hole%), Spearman ρ(pseudotime, hole area), partial Spearman ρ(pseudotime, hole% | nuclear_density), independence checks ρ(nuclear_density, hole%) and ρ(packing_irregularity, hole%). Permutation test: 1000 shuffles of duct-level pseudotime; two-tailed p-value on |ρ|.

**Aggregation:** median patches→duct (default); mean available via `--aggregation mean`.

**Section-agnostic:** `--section`, `--results`, `--slide-list`, `--output-dir` are all parameterised. Run 2M-2 by editing the four variables at the top of `run_holeyness_validation.sh`.

**Numerical results:** pending first Narval run. Check `holeyness_validation.json` after `sbatch jobs/run_holeyness_validation.sh`.

---

### 2026-07-27 (later) — Holeyness v2: area-adjusted validation added

**Why:** Post-hoc analysis of `holeyness_per_duct.csv` (section 2M-1) found a
confounder the v1 run didn't control for: duct area correlates with pseudotime
(rho +0.43) and with hole_pct (rho +0.39); controlling for area drops the
headline rho(pseudotime, hole_pct) from 0.28 to 0.12. v1's independence check
only controlled for nuclear_density (rho with hole_pct only 0.05), which is why
this wasn't caught. Also: 571/2173 ducts (26%) were excluded for zero assigned
patches, non-randomly (smallest ducts), and the v1 permutation test shuffled all
1602 retained ducts globally despite them being nested within 8 slides. The v1
result is not defensible without these checks.

**Module (extended in place):** `analysis/holeyness.py` — a new `--v2` flag
switches `main()` into an extended validation path; v1 behavior is unchanged
when `--v2` is absent (same functions, same output files, same code path).

**Design note — zero-patch ducts:** v1 never persisted zero-patch ducts
(`holeyness_per_duct.csv` only contains ducts with ≥1 patch), so the exclusion-
bias check needed them recovered. v2 re-derives the full duct table (incl.
zero-patch ducts) by calling v1's existing, unmodified loader functions
(`parse_measurement_export`, `load_duct_polygons`, `build_duct_table`,
`assign_patches_to_ducts`) against the same raw inputs v1 used — CPU-only,
deterministic, not a rerun of the atlas pipeline (no features/clustering/DPT
involved). Before trusting any v2 number, `check_consistency_with_v1()` merges
the recomputed retained-duct table against v1's saved `holeyness_per_duct.csv`
on `object_id` and asserts hole_pct/pseudotime match to 1e-6 — this is written
into `holeyness_validation_v2.json["consistency_check"]` and the top of the
markdown report; the process does not proceed to report other numbers as
trustworthy if it fails.

**New functions in `analysis/holeyness.py`:** `_partial_spearman_multi`
(rank-transform + OLS-residual generalization of the existing single-control
`_partial_spearman`, verified numerically equivalent with one control),
`run_area_covariate_checks`, `run_within_slide_checks`,
`run_within_slide_permutation`, `run_exclusion_bias_check`,
`run_aggregation_sensitivity`, `run_patch_sampling_artifact_check`,
`check_consistency_with_v1`, `write_v2_report`, `write_v2_outputs`, plus 3 new
figure functions (`write_scatter_pt_vs_hole_by_area`, `write_scatter_pt_vs_area`,
`write_small_multiples_per_slide`).

**Job script:** `jobs/run_holeyness_validation_v2.sh` — same SBATCH/module/venv
conventions as v1, CPU-only, `--time=00:30:00 --mem=16G`. Reads v1's
`holeyness_per_duct.csv` as an additional (read-only) input; writes to
`$SCRATCH/results/holeyness/<SECTION>/v2_area_adjusted/` — a new subdirectory,
so `holeyness_per_duct.csv`, `holeyness_validation.json`, and the two v1 scatter
figures at `$SCRATCH/results/holeyness/2M-1/` are never touched.

**Outputs (per section, in `v2_area_adjusted/`):**
- `holeyness_validation_v2.json` — `consistency_check`, `primary_correlation`
  (median agg.), `area_covariate`, `within_slide`, `permutation` (`global` +
  `within_slide`), `exclusion_bias`, `aggregation_sensitivity`,
  `patch_sampling_artifact`.
- `holeyness_validation_v2.md` — one section per check with a one-line
  plain-language verdict each.
- `duct_table_full.csv` — every measured duct (incl. zero-patch), tagged with a
  `retained` bool column.
- `v2_scatter_pt_vs_hole_pct_by_area.{pdf,png}`,
  `v2_scatter_pt_vs_area.{pdf,png}`, `v2_small_multiples_per_slide.{pdf,png}`
  (300 dpi).

**Verified locally (synthetic data, no `$SCRATCH` access from this checkout):**
`python -m py_compile analysis/holeyness.py` passes; `--help` shows the new
`--v2`/`--v1-per-duct-csv`/`--min-ducts-per-slide` args; a synthetic-data smoke
test confirmed `_partial_spearman_multi` agrees with `_partial_spearman` on a
single control, the exclusion-bias split sums to the full duct table, the
consistency check both passes on matching data and correctly flags a deliberate
perturbation, and all 9 expected v2 output files are written with a
well-formed JSON schema and markdown report.

**Numerical results (pending first Narval run):** not yet available — no
`$SCRATCH` data exists on this checkout. Run `sbatch
jobs/run_holeyness_validation_v2.sh` on Narval (after v1 has already produced
`holeyness_per_duct.csv` for the section), then check
`v2_area_adjusted/holeyness_validation_v2.json` and `.md`. Fill in here once
available: the area-adjusted partial rho, which of the 6 checks passed/failed
their plain-language verdict, and whether the headline pseudotime/hole_pct
correlation should still be reported as validation evidence given the area
confound.

---

### 2026-07-27 (later still) — Holeyness v3: significance test on the area-adjusted partial

**Why:** v2 established rho(pseudotime, hole_pct) = 0.276 raw → 0.131 partial
controlling for duct area (0.158 controlling for area + nuclear_density), a 52%
reduction — but v2's permutation tests (global and within-slide shuffle) were both
run against the *raw* correlation, never the partial, so there was no significance
test on the number that actually matters. v2's aggregation-sensitivity sweep (mean
vs median, n_patches thresholds) was likewise raw-only, so it was unknown whether
the 0.131 partial signal depends on poorly-sampled ducts. Three slides
(6028-4R-2M-1_x5: 0.025, 6029-4L-2M-1_x5: 0.022, 6031-4L-2M-1_x5: -0.069) showed a
near-zero/negative area-adjusted partial with no investigation into why.

**Module (new file, not another flag on holeyness.py):**
`analysis/holeyness_v3_significance.py` — imports (does not modify)
`holeyness.py`'s existing loader/stat functions
(`parse_measurement_export`, `load_duct_polygons`, `build_duct_table`,
`assign_patches_to_ducts`, `aggregate_per_duct`, `_partial_spearman`,
`_partial_spearman_multi`, `_safe_spearman`, `_format_perm_p`). v1/v2's
already-validated code path in `holeyness.py` is completely untouched by this
work.

**Data sourcing:** primary source is v1's already-persisted
`holeyness_per_duct.csv` (median-aggregated) plus v2's already-computed
`holeyness_validation_v2.json` (reused for reference numbers and per-slide raw
figures, not recomputed) — no raw-input access needed for GAP 1 or the per-slide
investigation. The one exception: GAP 2's mean-aggregation sweep needs a table
that was never persisted (v2 computed it in-memory only), so it's re-derived via
the same raw-input loader chain v1/v2 used (measurement export + annotations +
slide dimensions + results.csv) — cheap, deterministic, not an atlas-pipeline
rerun. A `mean_table_consistency_check` compares the re-derivation's own
median-aggregated output against v1's saved CSV before trusting the new
mean-aggregation numbers alongside it; a `consistency_check` similarly compares
v3's own recompute of the raw/partial correlations against v2's saved reference
numbers.

**GAP 1 closed — permutation test on the area-adjusted partial:**
`run_partial_permutation_global`/`run_partial_permutation_within_slide` shuffle
pseudotime (globally, or within each slide group, same nesting-preserving logic
v2 used) and rebuild the null from the **partial** Spearman each iteration (via
`_partial_spearman`/`_partial_spearman_multi`), run separately for
controls=[area] and controls=[area, nuclear_density], each in both shuffle
flavors (4 permutation runs total).

**GAP 2 closed — aggregation/patch-count sensitivity on the adjusted partial:**
`run_partial_aggregation_sensitivity` computes `partial_rho_pt_hole_given_area`
at all-ducts and at n_patches ≥ 3/5/10/20, run once on the median table (direct
filter of v1's CSV) and once on the re-derived mean table.

**Per-slide investigation:** `run_per_slide_investigation` reports median duct
area, median n_patches, and fraction of single-patch ducts per slide (from v1's
CSV), paired with each slide's raw rho/p and area-adjusted partial pulled
directly from v2's JSON (not recomputed). Flags the three low-signal slides and
compares their mean duct-size/coverage stats against the other slides
(`flagged_vs_other_summary`) — descriptive only, no causal claim.

**Job script:** `jobs/run_holeyness_v3_significance.sh` — same SBATCH/module/venv
conventions as v1/v2, CPU-only, `--time=00:30:00 --mem=16G`. Reads v1's CSV and
v2's JSON as read-only inputs; writes only to
`$SCRATCH/results/holeyness/2M-1/v3_significance/` — v1 and v2 outputs are never
touched.

**Outputs:** `holeyness_validation_v3.json` (`consistency_check`,
`reference_from_v2`, `partial_permutation.{given_area,given_area_and_nd}.{global,within_slide}`,
`aggregation_sensitivity_partial.{median_aggregation,mean_aggregation,mean_table_consistency_check}`,
`per_slide_investigation`), `holeyness_validation_v3.md` (same per-check,
one-line-verdict format as v2), and `v3_per_slide_partial_bar.{pdf,png}` (300
dpi, horizontal bar chart of each slide's area-adjusted partial, annotated with
median duct area and fraction of single-patch ducts, flagged slides in red).

**Verified locally (synthetic data, no `$SCRATCH` access from this checkout):**
`python -m py_compile` and `--help` pass; a synthetic-data smoke test confirmed
the v2-consistency check both matches on a self-consistent v1/v2 pair and
correctly flags a deliberate perturbation, the permutation-test functions return
an `observed_partial_rho` that matches a directly-computed partial Spearman, and
— critically — the full raw-input re-derivation chain (fabricated measurement
export + ratio-annotation JSON + slide_dimensions.json + patch-level results.csv)
runs end-to-end through the real, unmodified `holeyness.py` loader functions
with 100% patch-to-duct assignment, producing non-empty median and mean
aggregated tables.

**Numerical results and per-slide findings (pending first Narval run):** not yet
available. Run `sbatch jobs/run_holeyness_v3_significance.sh` on Narval (after
v1 and v2 have already run for the section), then check
`v3_significance/holeyness_validation_v3.json` and `.md` — specifically confirm
`consistency_check.all_match` and
`aggregation_sensitivity_partial.mean_table_consistency_check.all_match` are
both `true` before trusting anything else. Fill in here once available: the
within-slide permutation p-value for the area-adjusted partial (this is the
number that determines whether the holeyness validation can be reported as
significant at all after area adjustment), whether the partial is stable or
decays under stricter n_patches thresholds, and whether the three flagged
slides show a structural (smaller duct area / worse patch coverage) explanation
for their weak signal or whether it looks like ordinary slide-to-slide noise.

---

### 2026-07-28 — Timepoint projection (4W/8W), Stage 1+2 only (staged, hard-gated)

**Why:** 8 additional annotated slides exist at timepoints (4W: 6069-4R, 6070-4L,
6070-4R, 6096-4R, 6097-4L, 6099-4L; 8W: 6037-4L, 6037-4R) different from the 16
single-timepoint pipeline slides, from different mice (6037/6069/6070/6096/6097/6099
vs. the pipeline's 6027/6028/6029/6031), present in the holeyness export but never
run through the pipeline. Timepoint is an external, image-independent progression
ground truth — stronger validation than the morphological proxies used so far — but
only if tested by **projecting onto the existing manifold, never retraining** (which
would invalidate every number already in the manuscript), at the **slide level, not
patch level** (patches within a slide are not independent; a patch-level test would
produce a spuriously tiny p-value through pseudoreplication), and with timepoint's
confound with mouse stated as a limitation everywhere, never adjusted away.

**Pre-specified primary test (written down before any pseudotime is looked at):**
slide-level median projected pseudotime, early (4W, n=6) vs. late (8W + pipeline
2M-1, n=2+8=10) — resolves to the task's own stated "n=6 vs n=10," confirming "2M"
means the 8 2M-1 slides specifically (the same cohort the manifold is built from and
Stage 2 compares against). Mann-Whitney U + rank-biserial effect size. Everything
else (3-level 4W/8W/2M ordinal trend, mouse-level aggregation, per-slide
distributions) is EXPLORATORY and must be labelled as such.

**This is a 4-stage task with an explicit hard gate after Stage 2 — only Stage 1 and
Stage 2 are implemented in this pass.** Stage 3 (GPU feature extraction + projection
via the saved `AtlasProjector`) and Stage 4 (timepoint analysis) are deferred until
Stage 2 produces real numbers on Narval and the user confirms proceeding — if Stage
2 shows a staining confound comparable to the known cross-section effect, Stage 3
should not run at all.

**Stage 1 — convert + inventory:** No new conversion code — `run_all.py --convert`
already exposes `--ndpi-dir`/`--png-dir`/`--ndpi-level`/`--ndpi-scale` as CLI flags
and writes its own `slide_dimensions.json` sidecar into whatever `--png-dir` is
given, so pointing it at a brand-new directory (`$SCRATCH/data/timepoint_x5_cropped`,
never `MCF7_x5_cropped`) automatically satisfies "separate directory, separate
sidecar" with zero code changes. New module `analysis/timepoint_inventory.py`
reports, per slide: dimensions from the new sidecar; an `ndpi_scale` cross-check
(re-reads the raw NDPI's level-0 dimensions directly and compares
`level0_width * ndpi_scale` against the sidecar's `original_full_width`); whether a
matching annotation exists in `data/annotations/` (computed, not assumed — none
exist for these 8 slides as of writing); and a **left-crop-assumption diagnostic**
that is explicitly partial, not a full verification — no annotation-based
cross-check is available for these new slides (unlike the pipeline's original
slides), so the script computes an HSV tissue-fraction (same `S>15, V<230`
criterion `features/patching.py`'s `_has_tissue_hsv` uses per-patch, reimplemented
at whole-image scale) for the kept left half vs. the discarded right half at the
NDPI's coarsest pyramid level, and flags any slide where the right half isn't
clearly near-empty for **manual visual review** rather than asserting the
assumption holds. New job script `jobs/run_timepoint_stage1_convert.sh` (CPU) and
new slide list `jobs/slides_timepoint.txt` (bare NDPI stems, e.g. `6069-4R-4W` — a
different convention from `jobs/slides_section1.txt`'s `_x5`-suffixed pipeline
slide_names, since Stage 1 needs the bare stem to look up raw NDPI/annotation
files).

**Stage 2 — stain batch check (HARD GATE):** New module
`analysis/timepoint_stage2_stain_check.py` compares the 8 new slides against the 8
existing 2M-1 pipeline slides (read-only — `$SCRATCH/data/MCF7_x5_cropped` is never
modified), slide-level values only (n=8 vs n=8). Per slide: tissue-masked (same
`L<230` LAB convention as `qc/stain_qc.py::_lab_stats`) RGB channel mean/median (R,
G, B), plus hematoxylin channel mean/median via `_deconvolve_hematoxylin` +
`compute_hematoxylin_intensity` **imported directly from
`validation/morphological_features.py`** — the exact function that produces the
pipeline's own `h_intensity` feature, so this check is directly comparable to the
existing cross-section h_intensity confound rather than a similar-but-different
reimplementation. For each of 9 measures: Mann-Whitney U + rank-biserial
`r = 2*U/(n1*n2) - 1`, matching the exact formula/sign convention already
established in `diagnostics/audit_feature_diagnostics.py`'s D3 cross-section check
(reimplemented locally as a 3-line formula match, not cross-imported — that script
is a standalone diagnostic, not a shared library). **Interpretation rule:**
`--known-confound-r` (default `0.71` — supplied for this task; confirmed by
exploration to not exist anywhere in this repo, i.e. a given premise, not
independently re-derived) is the bar; any hematoxylin measure with `|r| >= 0.71`
triggers a `hematoxylin_confounded=True` / **"STOP — do not proceed to Stage 3"**
verdict. All measures are also labelled against the project's existing general
effect-size thresholds (`D3_LARGE_EFFECT=0.3`, `D3_SMALL_EFFECT=0.1`, same module)
for context. New job script `jobs/run_timepoint_stage2_stain_check.sh` (CPU).

**Bug caught and fixed during local verification (not during review):** the first
version of Stage 2 unconditionally appended `_x5.png` to every slide-list entry when
building PNG paths, which would have double-suffixed `jobs/slides_section1.txt`'s
already-`_x5`-suffixed entries (e.g. `6027-4L-2M-1_x5` → looking for
`6027-4L-2M-1_x5_x5.png`). Fixed with a `_normalize_stem()`/`_png_path()` helper
that strips a trailing `_x5` before re-appending it, so both list conventions (bare
stem for the new timepoint list, `_x5`-suffixed for the existing pipeline list) are
handled uniformly. Caught by a synthetic end-to-end CLI test using a real
`_x5`-suffixed list, not by code review.

**Verified locally (no `$SCRATCH`/OpenSlide access from this checkout):**
`python -m py_compile` both new modules; both import cleanly against the real,
unmodified `validation/morphological_features.py` functions (after installing
`scikit-image`/`tqdm` locally, present in `requirements.txt` for Narval but not
previously installed in this dev checkout). Stage 1's pure-Python tissue-fraction
logic (`tissue_fraction_hsv`, `left_right_tissue_check`) verified against
constructed cases (blank-vs-tissue = no review flag; tissue-vs-tissue = correctly
flagged) — the OpenSlide-dependent per-slide function
(`ndpi_scale_and_crop_check`) is thin/isolated but NOT exercised locally (no NDPI
test fixture available off Narval). Stage 2 verified end-to-end against real
synthetic PNGs with scipy's actual `mannwhitneyu`: a matched-distribution case
correctly does NOT trigger the confound gate; a case with a deliberate large
hematoxylin shift injected into the new-slide group correctly triggers
`hematoxylin_confounded=True` with `|r|` at/near 1.0; a hand-computed
perfect-separation case confirms the rank-biserial formula. Full CLI smoke test
(subprocess invocation, real JSON/MD file output) also passed.

**Numerical results (pending first Narval run):** not yet available — no `$SCRATCH`
access from this checkout. Run `sbatch jobs/run_timepoint_stage1_convert.sh` (after
placing the 8 raw NDPI files at `$SCRATCH/data/timepoint_ndpi/` and confirming
`atlas_none_section1` — or whichever run is the current canonical 2M-1 manifold —
is still correct, needed later for Stage 3), review `stage1_inventory.md`, then
`sbatch jobs/run_timepoint_stage2_stain_check.sh`. **Stage 2's real
`hematoxylin_confounded` verdict must be reported back and confirmed before Stage
3/4 are even designed**, per the task's explicit hard gate — this entry will be
updated with those numbers and the go/no-go decision once available.
