# Cancer Trajectory Atlas — Pipeline Handoff Guide

**A function-by-function walkthrough, in the order the pipeline actually executes.**

Written 2026-08-12 for handoff. Every constant and behaviour here was checked against the
source. Where something is surprising, fragile, or a known trap, it is flagged inline
rather than buried at the end.

**Audited 2026-08-23.** Re-checked against the code and against findings from the
anchor-validation work of 17–21 August. **One diagnosis in the original was wrong and has
been corrected in place:** 2M-2's `pseudotime_std` anomaly was attributed to the non-finite
clamp firing on a disconnected diffusion graph. It is not that — the graph has one
component, the clamp never fired, and the cause is three sign-discordant roots. See Part 9
and Part 16 item 6. Corrections and post-August-12 additions are marked **RESOLVED**,
**ASYMMETRY** or dated inline.

### Which document to read

| | |
|---|---|
| **This document** | Tutorial. Follows execution order, explains *why*, flags traps as you meet them. Read it once, straight through, to learn the pipeline. |
| **`docs/PIPELINE.md`** | Terse reference. Same pipeline, every claim carrying a `file:line` citation, verified against implementation bodies rather than docstrings. Use it to check a specific fact fast. |
| **`docs/ANCHOR_VALIDATION_RECORD.md`** | The experimental record: what was tested about the pseudotime anchor, what held, what did not, and the error log. |

The two pipeline documents were written independently and then reconciled. Where they still
disagree, `PIPELINE.md` carries the line citation — check that first, then the source.

---

## How to read this document

Sections follow **execution order**, not alphabetical order. Part 1 is what happens to a
raw `.ndpi`; Part 11 is the last thing written to disk. Read straight through and you have
followed one patch from a glass slide to a row in `results.csv`.

Four kinds of callout appear:

> **WHY** — the reasoning behind a choice. This is the part not recoverable from the code.

> **TIES INTO** — where this output is consumed later.

> **TRAP** — something that has already caused confusion, or would.

> **OPEN** — a live uncertainty about the current results, not a code defect.

**Contents**

| Part | |
|---|---|
| 0 | Orientation |
| 1–3 | Raw NDPI → PNG · Annotations · Slide discovery |
| 4–7 | Stain normalization · Patch extraction · Embedding + cache · The cap |
| 8–9 | PCA, batch correction, clustering · Diffusion pseudotime |
| 10–11 | Morphological validation · Outputs and figures |
| 12 | Projection and leave-one-out |
| 13 | QC, visualization, diagnostics — module by module |
| 14 | Analysis branches — module by module |
| 15 | Which job script to use |
| 16 | Open scientific questions |
| 17 | Trap index |
| 18 | Running it, and debugging it |
| — | Appendix: parameter reference |

---

## Part 0 — Orientation

### What the project is trying to do

Sixteen H&E whole-slide images of MCF7 breast-cancer xenografts. Four mice
(6027 / 6028 / 6029 / 6031), two flanks each (4L / 4R), two serial sections each
(2M-1 / 2M-2).

> **THEY ARE 8 MATCHED PAIRS.** Added 2026-08-23. Every
> mouse-flank combination (a *gland*) contributes exactly one slide to each section:
> 6027/6028/6029/6031 x 4L/4R. The 16 slides are therefore **8 matched pairs, not 16
> independent samples**, verified from the per-duct tables by
> `analysis/gland_pairing_audit.py`.
>
> **Nothing in the pipeline knows this.** Slides are treated as independent throughout, which
> is correct for extraction and embedding but means any downstream statistical test has to
> supply the pairing itself. Two consequences are live: the between-section comparison was
> mis-specified (now corrected, `analysis/holeyness_paired_comparison.py`) and the full-atlas
> 16-slide LOO **leaks** — holding out one slide leaves its gland partner in training. The
> within-section bootstraps are unaffected, because within one section no two slides share a
> gland. See `docs/KNOWN_ISSUES.md` sections 1.1-1.3.

> **Before you change anything, read `docs/KNOWN_ISSUES.md` section 5.** It is a four-item
> ordered shortlist from the 2026-08-24 correctness audit, with the reasoning for the order.
> Sections 6 to 8 there are the full findings. Most are **latent**: real defects that no
> recorded run triggered, deliberately left unfixed and recorded so you do not have to
> rediscover them. Section 6 states for each one whether any reported number is affected.
>
> **It also enabled the cohort's strongest measurement.** Matched tissue across two fixations
> gives a within-gland fixation effect, free of every between-animal source of variation:
> PFA ducts are **1.64x larger in area** and carry **5.48x more hole area** than Carnoy's,
> 8/8 glands, exact sign-test p = 0.0078. The lumen collapses far harder than the duct
> (anisotropy 0.261, 8/8 glands, p = 0.0078). **Treat that as a result, not a caveat** — it
> is a quantitative measurement of what Carnoy's does to ductal architecture, made against a
> within-animal control, and it is the most concrete thing this project has produced.
>
> It does follow that **`hole_pct` is not fixation-invariant**: numerator and denominator
> shrink by different factors, so the quantity means something different in each section.
> Within-section validation is unaffected. The cross-section replication (0.276 and 0.191, no
> evidence they differ) still holds, because Spearman ignores monotone rescaling — but state
> it as *the correlation replicates despite `hole_pct` being systematically rescaled between
> conditions*, which arguably strengthens it. Full account in
> `docs/ANCHOR_VALIDATION_RECORD.md` section 3.12.

The hypothesis: **tumour morphology varies along a continuum, and that continuum can be
recovered without labels.** The pipeline cuts each slide into small patches, embeds them
with a pathology foundation model, builds a manifold, and fits a *pseudotime* — a
one-dimensional coordinate along the dominant axis of morphological variation.

Pseudotime is only meaningful if it tracks something interpretable, so the pipeline then
computes six hand-crafted morphological features per patch and asks whether they correlate
with it. **That correlation test is the actual result.** A null result is valid and the
code says so in its own verdict strings.

### The mental model in one paragraph

Slide → patches → 768-dim embeddings → PCA → k-NN graph → clusters + diffusion map →
pseudotime → correlate against morphology. Everything else in the repo is feeding that
chain, checking it, or asking whether a specific confound explains it.

### How to invoke anything

**Every entry point uses relative imports and must be run as a module from the directory
that *contains* the repo:**

```bash
cd ~                                  # parent of cancer_trajectory_atlas/
python -m cancer_trajectory_atlas.run_all --run ...
```

`python run_all.py` fails with `ImportError: attempted relative import with no known parent
package`. Two exceptions run as plain scripts: `converters/batch_convert.py` (from the repo
root) and `jobs/check_annotations.py` (from anywhere).

On the cluster, do not invoke `run_all` by hand — use `jobs/`, which carries the exact flag
sets that produced published results.

### Repo map

```
run_all.py              THE entry point. --convert and --run.
pipeline_config.py      PipelineConfig dataclass (parameter container).
paths.json              Default paths. Exactly 4 keys. Authoritative.

converters/             NDPI + annotation conversion
data/                   Slide registry, stain normalization, annotation directories
features/               Patch extraction, feature embedding
analysis/               Clustering, batch correction, diffusion, projection + ~25 analyses
validation/             Morphological features and correlation tests
utils/                  Save/load, all plotting
qc/                     Post-run quality control (4 checks + runner)
diagnostics/            Targeted investigations of specific suspicions
visualize/              Interactive overlays, patch export
figures/                Publication figures
jobs/                   SLURM scripts — the canonical run recipes
docs/, reports/         This file, and written analyses
archive/                Dead or superseded code. Never imported. Kept for history.
```

---

## Part 1 — Raw NDPI → cropped PNG

**Module:** `run_all.py` · **Function:** `convert_ndpi_to_left_half_png(cfg)`
**Invoked by:** `run_all.py --convert`

Each `.ndpi` is a Hamamatsu whole-slide pyramid, ~4 gigapixels at level 0. This function
opens each with `openslide`, reads the full level-0 region, and **keeps only the left half**.

> **WHY the left half.** Each NDPI contains two side-by-side copies of the same physical
> slide. The pathologist annotated only the left copy. The right is redundant and would
> double every downstream cost, so it is discarded at the earliest possible moment.

The crop is `img.crop((0, 0, full_w // 2, full_h))` — **horizontal only**. Height is never
reduced. That matters in Part 5.

It then writes a **dimensions sidecar**, `slide_dimensions.json`, into the PNG directory:

```json
"6027-4L-2M-1_x5.png": {
  "original_full_width": 96000,     // FULL NDPI width, both copies
  "original_full_height": 42240,
  "cropped_width": 48000,           // what the PNG actually is
  "cropped_height": 42240
}
```

Note the sidecar is written even for slides whose PNG already existed — the loop reads
dimensions before checking for the existing file.

> **WHY that matters.** The sidecar is the *only* record of the original full width once
> the PNG exists, and annotation coordinates are fractions of that full width. A partial
> re-run must not produce a partial sidecar.

**`data/slide_registry.py`** holds `KNOWN_NDPI_DIMENSIONS`, a hardcoded table of all 16
slides' level-0 dimensions, used by `_get_known_dimensions()` when no sidecar exists.

> **WHY a hardcoded fallback.** If PNGs are copied to a new machine without the sidecar,
> ratio annotations would silently misplace. A safety net, not the primary path.

**Also in `converters/`:** `ndpi_to_img.py` and `tiff_to_img.py` are standalone converters
not used by `run_all`. Utilities, kept.

---

## Part 2 — Annotations: GeoJSON → ratio JSON

**Module:** `converters/batch_convert.py` · **Functions:** `convert_coordinates()`, `main()`
**Invoked manually:** `cd ~/cancer_trajectory_atlas && python converters/batch_convert.py`

The pathologist draws ROIs in QuPath and exports GeoJSON with **absolute pixel coordinates
in full-NDPI space**. This script divides every coordinate by that slide's full dimensions
(from `converters/img_dims.txt`) to produce **ratios in [0, 1]**, rounded to 6 decimals.
`convert_coordinates` recurses, so Polygons, MultiPolygons and inner rings are all handled.

```
data/annotations/*.geojson   →   data/annotations_ratio/*.json
```

> **WHY ratios rather than pixels.** Ratios survive rescaling. If the pipeline is ever run
> at a different `--ndpi-scale`, pixel coordinates would silently point at the wrong tissue;
> ratios stay correct because they are re-multiplied by whatever the current full width is.

### ⚠ The round-trip invariant — the single most important thing in this document

```
batch_convert.py    DIVIDES    by converters/img_dims.txt
load_roi_polygons   MULTIPLIES by original_full_width (from slide_dimensions.json)
```

**If those two tables disagree, every ROI lands in the wrong place and there is no error** —
just wrong numbers. Verified 2026-08-12: `img_dims.txt` and `KNOWN_NDPI_DIMENSIONS` contain
the same 16 keys with identical values, zero mismatches.
`jobs/run_full_pipeline_handoff.sh` re-checks this at runtime and aborts on mismatch.

**If you add a slide, add it to BOTH tables.**

### The four annotation directories

| Directory | Contents | Role |
|---|---|---|
| `data/annotations/` | 16 `.geojson`, absolute pixels | Source from QuPath |
| `data/annotations_ratio/` | 16 `.json`, ratios | **What the pipeline reads** |
| `data/old_annotations/` | 16 `.json` | Superseded round, genuinely different content. Historical only. |
| `data/annot_check_test/` | `.png` | Sample QC overlays |

> **TRAP.** `data/annotations/` *used to hold* the ratio JSONs. Commit `f050e4a` replaced
> them with GeoJSON and moved the ratio files out. Several old job scripts still point at
> `data/annotations` — they were correct when they ran and their past results are not
> suspect, but re-running one today feeds absolute pixels to a loader that assumes ratios.
> Those are now in `archive/jobs/`. One live script,
> `jobs/run_individual_pseudotime.sh`, still has the stale path and carries a warning header.

### Visual QC — `jobs/check_annotations.py`

Draws every polygon as a coloured outline over a slide thumbnail:
**Tumor red · Ignore\* grey · Necrosis dark red · Region\* blue · other orange.**
Run as a plain script, not with `-m`.

> **Always eyeball these after regenerating annotations.** A coordinate bug is obvious in
> one glance and invisible in a CSV.

---

## Part 3 — Slide discovery

**Module:** `run_all.py` · **Functions:** `_load_default_paths()`, `_get_known_dimensions()`,
`discover_slides(cfg)`

`discover_slides` globs `*.png` from `--png-dir`, then per PNG:

1. **Matches an annotation** by trying four candidates in order — `<stem>.json`,
   `<base>.json`, `<stem>.geojson`, `<base>.geojson` — where `stem` is `6027-4L-2M-1_x5`
   and `base` strips `_x5`.
2. **Attaches original dimensions:** sidecar first, registry second.
3. **Applies `--slides` / `--slides-from-file`**, erroring on any requested slide not found.

It prints a per-slide table with `✓ ann` / `✗ ann` and warns loudly when an annotated slide
has no dimensions — that combination produces silently wrong ROIs.

> **TRAP.** The `.geojson` fallbacks in the candidate list are exactly why pointing
> `--annotation-dir` at `data/annotations` *appears* to work. It finds files. They are in
> the wrong coordinate space.

**`paths.json`** supplies defaults. Exactly four keys — `raw_ndpi`, `cropped_png`,
`annotations`, `results`. Any other key is ignored. `annotations` must point at the **ratio**
directory. `results` is a base; `run_all` appends `/atlas_full` for the default output.

---

## Part 4 — Stain normalization

**Module:** `data/stain_normalization.py`
**Functions:** `ReinhardNormalizer.fit/transform`, `build_normalizer()`, `normalize_slide()`

H&E staining varies between batches, scanners and days. Normalization tries to remove that
so morphology, not chemistry, drives the embedding.

- **`build_normalizer(method, ref_path)`** — `"none"` returns `None`; `"reinhard"` fits the
  in-file LAB normalizer; `"macenko"` / `"vahadane"` route to `staintools`. Raises
  `ValueError` on anything else. *(`vahadane` is reachable only by calling this directly —
  `run_all` restricts `--stain-method` to reinhard/macenko/none.)*
- **`ReinhardNormalizer.fit`** — computes per-channel LAB mean and std over **tissue pixels
  only** (`L < 230`), falling back to all pixels if fewer than 1000 qualify.
  > **WHY tissue-only.** Whole-slide images are mostly white background. Including it would
  > make the statistics measure how much glass is in frame.
- **`ReinhardNormalizer.transform`** — standardizes the source with its own tissue stats, then
  rescales to the reference's, clips to [0, 255], converts back to RGB.
- **`normalize_slide(arr, normalizer, name)`** — applies it, or returns the input unchanged
  when `normalizer is None`.

### Two behaviours you must know

> **TRAP 1 — the stain reference is whichever slide sorts first.** `run_all` passes
> `slides[0]["image"]`, *after* `--slides` filtering. There is no designated reference. So
> **running a subset can change the reference**, and with it the normalization applied to
> every slide in that run. Two runs over different subsets are not comparable under
> `reinhard` or `macenko`. The chosen reference is copied to
> `<output_dir>/stain_reference.png` so it is recoverable.

> **TRAP 2 — failure is silent and per-slide.** `normalize_slide` catches *any* exception,
> prints a warning, returns the **un-normalized** array. Nothing is persisted. A run can be
> half-normalized with only the SLURM log as evidence. Deliberately unlike the feature path,
> which encodes failure as NaN. `qc/stain_qc.py` exists partly to detect this.

**Neither trap affects current results:** every reference run uses `--stain-method none`.

> **WHY none.** `NOTES.md` records that no-normalization outperformed both alternatives
> here. Reinhard is the *code* default; no current job script uses it.

---

## Part 5 — Patch extraction

**Module:** `features/patching.py` — where annotations become geometry.

### `load_roi_polygons(...)` — the coordinate-system function

Reads ratio JSON, returns `(include_polys, exclude_polys)` as matplotlib `Path` objects in
**cropped-PNG pixel space**.

**Three coordinate spaces exist. Confusing them is the classic bug here:**

1. **Full-NDPI pixel space** — width includes *both* copies. This is `original_full_width`.
2. **Cropped-PNG pixel space** — `original_full_width // 2`. Patch `(x, y)` live here.
3. **Ratio space** — `[0, 1]` relative to *full-NDPI* dimensions.

The mapping:

```
polygon_x × original_full_width  →  full-NDPI pixels
```

Because QuPath annotated the left half, left-half ratio x-values lie in `[0, 0.5]`, so after
multiplication they land in `[0, original_full_width/2]` — **exactly cropped-PNG space.** No
offset needed. That coincidence is the whole trick, and it only holds because the crop is
horizontal and taken from x=0.

Other behaviour:

- **Classification:** `"Tumor"` or unclassified (no `properties.classification`) → inclusion.
  Anything else (`Ignore*`, `Necrosis`, `Region*`) → exclusion. Exact, case-sensitive match.
- **Holes:** inner rings become compound `Path`s via `_make_path()`, so a donut-shaped ROI
  correctly excludes its middle.
  > **ASYMMETRY worth knowing.** Patch extraction honours holes; the *duct* loader used by
  > the holeyness analyses does not. `holeyness.load_duct_polygons` reads
  > `geometry["coordinates"][0]` — the outer ring only. So the two halves of the codebase
  > disagree about what a hole is. That is deliberate on the duct side (`hole_pct` is a
  > fraction of the *outer* duct area, so the lumen must not be subtracted), but if you ever
  > compare an in-ROI patch count against a duct area, they are not measuring the same region.
- **Right-half discard:** polygons whose vertex-mean centroid has `cx > cropped_w` (or
  `cy > cropped_h`, which never fires since the crop is horizontal) are dropped — they belong
  to the duplicate copy. Runs only when both `cropped_w` and `original_full_width` are supplied.
- **Zero-distance / degenerate rings** with fewer than 3 points after scaling are skipped.

### `get_patches_from_array(...)` — the extraction loop

Slides a `patch_size=112` window with `stride=96` (**16 px overlap**) and keeps a patch only
if it survives five filters, **in this fixed order**:

| # | Filter | Rule | Default |
|---|---|---|---|
| 1 | ROI inclusion | Patch **centre** inside ≥1 inclusion polygon | — |
| 2 | ROI coverage | ≥ `min_roi_coverage` of a 3×3 grid inside any inclusion polygon | `None` = off |
| 3 | Exclusion | Centre inside any exclusion polygon → drop | — |
| 4 | White | >70% of pixels above 220 in all channels → drop | `white_thresh=220`, `white_frac=0.70` |
| 5 | Tissue | ≥50% of pixels with saturation >15 **and** value <230 | `sat=15`, `val=230`, `thresh=0.5` |

> **WHY two background filters.** The white test catches empty glass. The HSV test catches
> pale, washed-out tissue that is technically not white but carries no morphology. Either
> alone lets junk through.

> **WHY the centre point, not full containment.** Requiring the whole patch inside an ROI
> would discard every boundary patch and shrink small ROIs to nothing. `min_roi_coverage`
> exists for when you want the stricter test; only `run_new_annotations.sh` has ever used it
> (at 0.75).

> **TIES INTO** the printed tallies. Because the order is fixed and each filter `continue`s,
> the reported counts are **disjoint** — a patch rejected by the ROI filter is never also
> counted by the tissue filter. Do not add them expecting the total.

Helpers:
- **`_is_mostly_white`** — `np.all(patch > 220, axis=-1).mean() > 0.70`
- **`_has_tissue_hsv`** — converts to HSV, requires ≥50% of pixels both saturated and non-bright
- **`_find_containing_roi`** — returns the first containing polygon, or None
- **`_coverage_in_rois`** — fraction of a 3×3 interior grid (at ¼, ½, ¾ offsets) inside **any**
  inclusion polygon, not just the containing one
  > **WHY any.** Grid points landing in completely unannotated territory must count as
  > outside. A patch straddling two adjacent ROIs correctly counts as covered.

Returns `(patches, coords)` where `coords` are the **top-left** `(x, y)` in cropped-PNG space.

> **TIES INTO** `results.csv`, every spatial plot, the interactive overlays, and the alignment
> key `(slide_name, x, y)` used by all run-comparison tools.

### `sample_patches(patches, coords, max_n, base_seed, slide_name)`

Shuffles then truncates to `max_n`. `max_n = 0` or `None` means no cap. The seed is
`base_seed XOR md5(slide_name)`. Returns `(patches, coords, selected_idx)` — `selected_idx`
is `None` when no cap was applied.

> **WHY hash the slide name into the seed.** It makes each slide's subsample independent of
> the order slides were processed in, so adding or removing a slide does not change which
> patches the *others* contribute. Essential for leave-one-out to be a clean comparison.

> **WHY shuffle before truncating.** Patches are generated in raster order. Taking the first
> N without shuffling would keep only the top strip of tissue.

**`get_patches(image_path, ...)`** is a thin convenience wrapper that loads from disk and
delegates. Not used by `run_all`, which loads and normalizes the image itself.

---

## Part 6 — Feature embedding and the cache

**Module:** `features/extractors.py`

- **`get_model(name, device)`** → `(model, processor)`. Phikon → HuggingFace `owkin/phikon`
  plus its `AutoImageProcessor`. ResNet18/50/101 → torchvision with the final FC layer
  stripped (`Sequential(*children[:-1])`), processor `None`.
- **`extract_features(patches, model_name, batch_size=32)`** — loads a model and embeds. The
  **bulk path**, used when no cache is configured.
- **`load_model_components(name)`** + **`extract_features_from_model(...)`** — the same work
  split so weights load once and are reused across slides. The **cache path**.

For Phikon the embedding is the **CLS token of the last hidden state, 768-dim**. ResNet gives
pooled conv features (2048-dim for resnet50/101, 512 for resnet18).

> **WHY Phikon.** A vision transformer pretrained on histopathology, so its embeddings already
> encode tissue-level concepts. An ImageNet ResNet would need far more data to learn the same
> things. ResNet is kept as a sanity-check baseline; no published run uses it.

> **WHY the CLS token** rather than mean-pooling the patch tokens: it is the representation
> the model was trained to use as a whole-image summary.

> **WHY the two paths are interchangeable** — and therefore why the cache is legitimate:
> their inner loops are identical (same batch size, same preprocessing, same CLS slice), and
> both backbones are batch-composition independent at inference. Phikon is a ViT using
> LayerNorm; ResNet runs under `model.eval()`, so BatchNorm uses running statistics rather
> than batch statistics. Splitting a cohort into per-slide batches cannot change any
> individual patch's embedding. **If a backbone with batch-dependent inference is ever added,
> the cache silently becomes invalid.**

### The feature cache

With `--features-cache-dir`, embeddings are stored per slide as `<slide>_features.npy`. First
run pays GPU inference; later runs load from disk. This is what lets 16 leave-one-out runs
cost one inference pass instead of sixteen.

**The cache shape contract** — `run_all.py`, the only cache-read site:

```python
if len(slide_feats) != orig_count:
    raise RuntimeError("Feature cache size mismatch ...")
```

> **WHY it must never be weakened.** It is the only thing between a stale cache and silently
> wrong features. It catches any change to how many patches a slide yields: `--patch-size`,
> `--stride`, `--min-roi-coverage`, edited annotations, and usually `--stain-method` (because
> normalization runs *before* the tissue filters, changing which patches survive).

> **TRAP.** It compares **N but not D**, and the cache key is the slide name alone. So
> `--model resnet50` against a Phikon-populated cache **passes** and silently uses 768-dim
> Phikon features. Until the key encodes the model, **one cache directory serves exactly one
> `(model, patch_size, stride, min_roi_coverage, stain_method)` combination.** Every current
> job script uses phikon/112/96, so this holds today by convention, not enforcement.

> **Float caveat.** Cached-vs-cached comparison is exact; cached-vs-fresh can differ in the
> last ULP, because a different final-batch size may select a different cuDNN kernel. Relevant
> whenever you compare a fresh run against a cached reference.

---

## Part 7 — The two-pass structure and the patch cap

`run_all.run_pipeline` deliberately runs **two passes** over the slides.

**Pass 1** extracts patches and embeddings for every slide at **full, uncapped** size, and
caches them.
**Pass 2** shuffles slide order, computes the cap, samples, and concatenates.

> **WHY two passes.** The default `--cap-strategy median` caps every slide at the *cohort
> median* patch count — which cannot be known until every slide has been extracted. Caching
> uncapped features also means one cache file serves any subset; a cap baked into the cache
> would invalidate it whenever the slide list changed.

> **WHY shuffle slide order before sampling.** Prevents systematic ordering bias in how
> slides are concatenated. `slide_ids` stores the *original* index, so `slide_names[sid]`
> stays correct despite the shuffle.

Cap strategies:

| Strategy | Behaviour |
|---|---|
| `median` | **Default.** Cap at cohort median, computed after full extraction. |
| `fixed` | Cap at `--fixed-cap` (default 200). |
| `none` | No cap at all. `--fixed-cap` ignored entirely. |

> **WHY cap at all.** Slide patch counts vary several-fold. Uncapped, one large slide can
> contribute a quarter of all patches and dominate PCA and clustering. Slide-aware sampling
> after Vig et al.

Written out: `active_cap.txt` (the realised cap), `sampling/` (per-slide selected indices),
`sampling_manifest.csv` (before/after counts, seed, % retained).

> **TIES INTO** leave-one-out. A LOO job can reproduce the exact subsample from these files.

> **TRAP.** `--fixed-cap` is **inert unless `--cap-strategy fixed`**. Passing
> `--max-patches-per-slide 1900` alone does nothing — the run stays median-capped. An archived
> job script made exactly this mistake and its name still claims otherwise.

---

## Part 8 — PCA, batch correction, clustering

**Module:** `analysis/clustering.py`

### `fit_pca(features, variance_target=0.95)`

`StandardScaler` → `PCA(n_components=0.95, random_state=42)`. Returns `(scaler, pca, X_pca)`.

> **WHY standardize first.** PCA maximizes variance; without scaling, embedding dimensions
> with larger numeric range dominate for purely arithmetic reasons.

> **TRAP.** A *float* `n_components` means "however many components reach 95% variance", so
> **output width is data-dependent** — 261 components for section 2M-1, 241 for 2M-2. Any
> comparison of two runs component-by-component must check widths match first.

`max_components` is a second capping argument that `run_all` never supplies — dead in practice.

### Batch correction — optional, applied in PCA space

- **`analysis/harmony.py: apply_harmony(X_pca, slide_names, slide_ids, key, nclust=10)`** —
  iterative correction via `harmonypy`. Builds per-patch batch labels with `_batch_labels`,
  which supports `section_number` (2 groups), `slide_id` (16) and `mouse_id` (4).
  > **WHY `nclust=10`** rather than harmonypy's default 100: with 2–4 batches, 100 internal
  > clusters is over-parameterized and triggers premature convergence.
  > **WHY pinned to harmonypy 0.0.9.** 0.2.0's PyTorch backend returns `Z_corr` as a 1-D array
  > when it converges in ≤2 iterations. The function raises if output shape ever mismatches.
- **`analysis/scvi_integration.py: apply_scvi(...)`** — a VAE alternative, nonlinear, returning
  a 30-dim latent space plus the trained model.
  > **WHY three non-default settings.** `gene_likelihood="normal"`,
  > `use_observed_lib_size=False`, `log_variational=False`. scVI's defaults assume non-negative
  > count data: the library-size estimator takes `log(row sum)`, and the encoder applies
  > `log(1+x)`. PCA output is mean-centred and routinely negative, so both produce NaN on the
  > very first forward pass.

Selection logic: `cfg.batch_method or ("harmony" if cfg.use_harmony else "none")`. The result,
`X_embed`, is what everything downstream uses. When correction is active, the uncorrected PCA
is preserved in `adata.obsm["X_pca_original"]`.

> **TIES INTO** `AtlasProjector`, which must train on `X_pca_original` because it can only
> produce *uncorrected* PCA features for new data.

**Current reference runs use `--batch-method none`** — per-section runs avoid the
cross-section batch effect by construction rather than correcting for it.

### `cluster(X_embed, method, **kwargs)` → `cluster_leiden`

Builds a k-NN graph with **k=15, cosine**, converts distances to similarities with a Gaussian
kernel (`sigma` = median edge distance), hands it to igraph as **undirected**, simplifies
keeping max weight, runs Leiden `RBConfigurationVertexPartition` at `resolution=0.5`.

- The directed→undirected step means an edge survives if *either* endpoint listed the other.
- `sigma` is recomputed per run, so **similarity values are not comparable across runs** —
  only the resulting partition is.
- A distance of exactly 0 is an implicit zero in the sparse matrix and is therefore *dropped*,
  so identical patches end up disconnected rather than maximally connected. Never observed
  with Phikon features.

`cluster_hdbscan` and `cluster_kmeans` (silhouette search over k=3..10) exist; neither is used.

### `run_umap(X_embed)` — k=30, cosine, `min_dist=0.1`

> **DISPLAY ONLY.** Nothing numerical depends on the UMAP embedding. It feeds figures and the
> interactive overlays. A change here cannot alter a result. Returns `(None, None)` if
> umap-learn is missing, and every caller guards on that.

### ⚠ Three separate k-NN graphs — the thing that has caused the most confusion

| Consumer | k | Metric | Set where |
|---|---|---|---|
| **Leiden clusters** | **15** | **cosine** | `cluster_leiden` defaults |
| **UMAP** (display) | 30 | cosine | `run_umap` defaults |
| **Diffusion map → DPT, PAGA** | 30 | **euclidean** | scanpy default — no metric ever passed |

`run_all` passes neither `n_neighbors` nor `metric` to `cluster()` or `run_umap()`, so **the
module defaults are what every published run used, and no CLI flag reaches them.** Only
`--diffmap-neighbors` is configurable.

**Consequence:** cluster identity comes from cosine geometry at k=15; pseudotime comes from
euclidean geometry at k=30. Two patches can be cluster-mates yet far apart in diffusion space,
and vice versa. Neither is wrong — they answer different questions.

### `check_slide_independence(labels, slide_ids, dominance_threshold=0.80)`

Flags any cluster where >80% of patches come from one slide, returning a per-cluster
composition breakdown plus warnings.

> **WHY.** Such a cluster is almost certainly a scanning or staining artifact, not morphology.
> **This is the first sanity check to read in any new run.**

### `get_cluster_centroids(X, labels)`

Returns `{cluster: (centroid_vector, index_of_nearest_patch)}`, skipping the `-1` noise label.

> **TIES INTO** the `AtlasProjector`.
>
> **The `index_of_nearest_patch` half of that return value is NEVER READ.**
> `plot_cluster_patch_grid` takes `cluster_centroids.keys()` only and then shows
> `indices[:n_per_cluster]`, the first patches of each cluster in array order. So the
> patch-grid figure does **not** show the most representative patch per cluster, despite
> its title and despite what this document said before 2026-08-24. Array order is
> array order, which is slide-processing order then raster or (for capped slides) a
> seeded permutation within each slide. Each row is therefore a biased sample. Corrected in
> the function's docstring; the code was left as-is because changing it would alter every
> cluster grid produced so far.

---

## Part 9 — Diffusion pseudotime

**Module:** `analysis/diffusion.py`

`_require_scanpy()` guards every entry point with a loud, explicit dependency error —
scanpy/anndata/igraph failures on a compute node are otherwise cryptic.

### `build_adata(X_embed, cluster_labels, slide_ids, X_umap)`

Wraps everything in an `AnnData`. `adata.X` is `X_embed`; `obs` gets `cluster` and `slide_id`
as **strings** (scanpy needs categoricals). `run_all` then adds `mouse_id` and `section_number`
parsed from slide names.

### `compute_diffusion_map(adata, n_neighbors=30, n_comps=10)`

`sc.pp.neighbors(..., use_rep="X")` then `sc.tl.diffmap`.

> **The euclidean metric is not a choice — it is scanpy's default**, because no `metric`
> argument is passed. There is no flag for it; changing it means editing the line.

> **WHY `use_rep="X"`.** Forces scanpy to read `adata.X` directly rather than recomputing its
> own PCA, so the diffusion map sees exactly the matrix `build_adata` was handed —
> post-batch-correction when one is active.

> **WHY a diffusion map at all.** It models the data as a random walk on the k-NN graph.
> Diffusion distance is robust to noise and naturally captures continuous transitions, which is
> exactly what a morphological gradient is. Unlike PCA it is nonlinear; unlike UMAP it has a
> principled distance.

### `compute_paga_topology(adata, groups="cluster", threshold=0.05)` — the validity gate

Runs PAGA, thresholds cluster connectivities at 0.05, counts connected components with
`scipy.sparse.csgraph`.

> **WHY this gate exists.** DPT assigns a distance from a root along the manifold. If the
> manifold is **disconnected**, patches in another component are at infinite distance and
> pseudotime there is meaningless. One component = DPT is valid.

> **TRAP — it mixes both graphs.** It groups by Leiden labels (**cosine, k=15**) but computes
> connectivity from the diffusion graph (**euclidean, k=30**). "DPT is valid" therefore means
> *the euclidean-k30 manifold is connected between the cosine-k15 clusters.* It is not a
> statement about the clustering, and a disconnected result does not imply the clusters are
> non-separable. The 0.05 threshold is hardcoded and the component count depends on it.

### `compute_dpt_multi_root(adata, nuclear_density, n_roots=20)` — the production path

DPT needs a starting point. The rule, stated exactly:

```python
finite_idx = flatnonzero(isfinite(nuclear_density))
roots      = finite_idx[argsort(nuclear_density[finite_idx])][:n_roots]
```

The 20 patches with the **lowest measured nuclear density**. DPT runs once per root; the final
pseudotime is the **element-wise median** across the 20 runs, min-max normalized to [0, 1]. The
**standard deviation** across roots is kept as `pseudotime_std`.

> **WHY lowest nuclear density.** Low cellularity is the closest available proxy for "least
> advanced" tissue. It is an assumption, and it is the assumption the entire trajectory
> *direction* rests on. `analysis/root_sensitivity.py` exists to test it.

> **WHY 20 roots and a median.** A single root makes pseudotime hostage to one patch. The
> median over 20 is robust to a few bad roots, and the spread gives a free per-patch
> uncertainty estimate.

> **WHY mask non-finite first.** `argsort` happens to place NaN last, giving the right answer
> by accident. Relying on that is fragile, and the cost of being wrong is that the pseudotime
> *origin* is anchored on whichever patches crashed the segmenter.
> **This is NOT `argsort(nuclear_density)[:n_roots]`** — the two agree only when nothing
> failed. Some analysis modules still quote the simpler form; those citations are stale.

Persisted: `adata.uns['dpt_root_candidates']` (the actual roots, int64) and
`dpt_n_roots_excluded_nonfinite`.

> **WHY persist.** Without it the root set can only be reconstructed by applying a rule to an
> array that may not be the one the run used — unverifiable after the fact.
> **Do not remove these writes.**

> `n_roots` is silently clamped to the number of finite-density patches. Read
> `len(adata.uns['dpt_root_candidates'])` for the true count rather than assuming 20.

Infinite DPT values are clamped **per root**, to that root's own maximum finite value, *before*
aggregation — so partial disconnection biases the median rather than propagating `inf`.

> **RESOLVED 2026-08-21 — read this before trusting the two caveats that used to be here.**
>
> **Root ties — CONFIRMED, and worse than tie-breaking.** In 2M-2 all 20 roots have
> `nuclear_density` exactly `0.0`, drawn from 21 such patches, so which 20 win is decided
> arbitrarily by `argsort`. Later inspection added a harder fact: **none of those 20 roots
> lies inside any Tumor annotation** (`duct_id` is null for all of them). `0.0` density means
> either genuinely acellular tissue *or* a segmentation failure — the two are
> indistinguishable in the stored value. See `docs/ANCHOR_VALIDATION_RECORD.md` §4, error #2.
>
> **The clamp and `pseudotime_std` — the diagnosis below was WRONG.** This document
> originally attributed 2M-2's `pseudotime_std` anomaly to the non-finite clamp firing on a
> disconnected diffusion graph. That hypothesis was tested and **contradicted**:
>
> - the diffusion graph has **one connected component**, not several
>   (`n_graph_components: 1`, from `holeyness_roots.assert_roots_connected`);
> - **`n_roots_clamped` is 0** — the clamp never fired;
> - the spread traces instead to **3 of 20 roots ordering the manifold backwards** relative to
>   the median of the other 19 (leave-one-out Spearman < 0). Two of them sit at pseudotime
>   0.717 and 0.673 while the other 18 span 0.009–0.144. Dropping those three takes
>   `pseudotime_std` from **27.70% to 3.40%** of the axis range.
>
> The affine `std`-vs-`pseudotime` observation still stands as an *observation*; only the
> cause attributed to it was wrong. `diagnostics/dpt_clamping_check.py` remains the right tool
> to rule the clamp in or out — it is what ruled it out here.
>
> **Do not use `pseudotime_std` as an anchor-health check.** Repairing those 3 roots improved
> it 8.2-fold while moving the axis itself by rho **0.9621** — barely at all. A median across
> 20 roots is robust to 3 outliers; `std` is not, so it reads dispersion the median has
> already absorbed. Use **mean leave-one-out concordance across roots** instead (2M-1 scores
> 0.726, 2M-2 only 0.478). Keep `std` for what it is good at: a per-patch uncertainty map.
> Full reasoning in `docs/ANCHOR_VALIDATION_RECORD.md` §7.
>
> Also note `pseudotime` is min-max normalized to [0, 1] but `pseudotime_std` is stored **raw**,
> on the diffusion-distance scale, computed **before** that normalization. **The two are not on
> the same scale** and must not be plotted on a shared axis.

### `compute_dpt` / `choose_root_cell` — single-root, `run_individual.py` only

`choose_root_cell` picks the patch nearest a named cluster's centroid; `compute_dpt` runs
scanpy DPT from it and normalizes to [0, 1], clamping infinities to the max finite value with
a warning. **Not used by `run_all`.**

> **WHY the difference is deliberate.** A single slide has no cohort to rank densities against,
> so the cluster anchor is the right choice there. Do not "unify" the two paths. Note
> `compute_dpt` writes `dpt_pseudotime` (which the multi-root path does not) and does **not**
> write `pseudotime_std`.

> **The diffusion graph differs too, and this is easy to miss.** `run_individual.py` calls
> `compute_diffusion_map(adata, n_neighbors=min(30, len(features) - 1), n_comps=10)`
> (`run_individual.py:318`) — **k is adaptive**, capped by the slide's own patch count, not the
> fixed 30 the atlas path uses. On a slide yielding fewer than 31 patches the graph is
> materially different from anything `run_all` builds. Another reason its pseudotime is not
> comparable to a per-section result.

---

## Part 10 — Morphological validation

This is where the pipeline tries to falsify itself.

### `validation/morphological_features.py`

Seven features per patch, computed from **pixels**, not embeddings.

> **WHY hand-crafted features at all**, when we already have a 768-dim embedding: the
> embedding is uninterpretable. If pseudotime correlates with nuclear density, that is a claim
> a pathologist can evaluate. If it correlates with "Phikon dimension 412", it is not.

**Segmentation first:**
`_deconvolve_hematoxylin` (skimage `rgb2hed`, keep the H channel) →
`_segment_nuclei_simple` (Otsu threshold → `binary_opening(disk(1))` → `remove_small_objects(min_size=20)` → connected-component `label`).
`_segment_nuclei_stardist` is an optional higher-quality path that falls back to Otsu if
StarDist is unavailable — **never used in any run** (`--use-stardist` has never been passed).

| Feature | Function | Measures |
|---|---|---|
| `nuclear_density` | `compute_nuclear_density` | nuclei per unit area — crowding / proliferation |
| `mean_nuclear_area` | `compute_mean_nuclear_area` | mean nucleus size in px — atypia |
| `nc_ratio` | `compute_nc_ratio` | nuclear ÷ cytoplasm pixels |
| `texture_entropy` | `compute_texture_entropy` | GLCM Shannon entropy — disorganisation |
| `h_intensity` | `compute_hematoxylin_intensity` | mean H **within nuclei** — chromatin density |
| `h_intensity_wholepatch` | same, `mask=None` | legacy whole-patch value, kept for comparison |
| `packing_irregularity` | `compute_packing_irregularity` | CV of nearest-neighbour centroid distances |

**GLCM detail:** 4 angles (0, π/4, π/2, 3π/4) × 3 distances (1, 3, 5), quantized to 64 grey
levels, `symmetric=True, normed=True`. Entropy is computed **per (distance, angle) pair and the
12 scalars averaged** — *not* one entropy over a pooled GLCM.

> **WHY per-pair then average.** Shannon entropy is concave, so averaging the matrices first
> yields a systematically higher value (Jensen's inequality) — an offset unrelated to the
> measurement. **WHY multi-angle:** a single angle partly measures how the section happened to
> be mounted, not the tissue.

**`packing_irregularity`** needs ≥3 nuclei (a CV over nearest-neighbour distances is undefined
below that) and returns NaN otherwise. `compute_hematoxylin_intensity` returns NaN on an empty
mask.

### The missing-value convention — load-bearing

**Every function returns `np.nan` for "could not be measured", never `0.0`.**

> **WHY this is not pedantry.** DPT roots are the *lowest* nuclear-density patches. A failure
> encoded as `0.0` is not merely lost — it is **preferentially promoted to a root**, anchoring
> the pseudotime origin on whichever patches crashed the segmenter.

Two deliberate exceptions: `mean_nuclear_area` and `nc_ratio` return `0.0` for an empty mask,
because "no nuclei" genuinely means zero area and zero nuclear fraction. `h_intensity` does
not, because the mean of an empty selection is undefined.

Two further non-NaN sentinels, both self-excluding: `compute_nuclear_density` returns `0.0` if
`patch_area <= 0` (unreachable — patch_area is 12544), and `compute_nc_ratio` returns `+inf`
for a 100%-nuclear mask (`np.isfinite(inf)` is False, so the same filters exclude it).

> **Important distinction.** A *measured* `nuclear_density` of `0.0` — no nuclei segmented —
> is legitimate and is **not** a failure. It is also exactly what makes a patch a root
> candidate. See the OPEN note in Part 9.

**`compute_nuclear_density_quick(patches, return_diagnostics=False)`** is a fast
hematoxylin+Otsu-only pass used solely to rank root candidates before DPT. Returns float64, NaN
where extraction failed, plus a diagnostics dict.

`compute_morphological_features` initialises every feature array to NaN, fills per patch inside
one `try`, and counts/indexes failures. Diagnostics land in
`<output_dir>/feature_failures.json`, including per-feature NaN counts, exception types, the
GLCM configuration, and both `h_intensity` definitions.

> **TRAP.** The reported `n_empty_mask` / `n_lt3_nuclei` are derived by *subtracting*
> `n_failed`, which assumes a failed patch is NaN in *every* feature. `texture_entropy` is
> computed last, so a patch that dies there keeps six real values while still incrementing
> `n_failed` — under-counting, possibly negative. Exact whenever `n_failed == 0`, which is
> every run to date.

### `validation/correlations.py`

- **`correlate_features_with_pseudotime`** — Spearman per feature on the pairwise-finite subset,
  requiring ≥10 valid points. `|rho| > 0.4` "strong", `> 0.3` "moderate".
  > **WHY Spearman not Pearson.** Pseudotime is a monotone coordinate with no meaningful scale;
  > only rank agreement is interpretable.
- **`permutation_test(..., n_permutations=1000, seed=42)`** — shuffles pseudotime, rebuilds the
  null distribution of `|rho|`, reports an empirical p-value and the null's 95th percentile.
  > **WHY permutation rather than the analytic p.** With thousands of spatially autocorrelated
  > patches, the analytic p-value is wildly anti-conservative — everything is "significant".
  > **Note** the empirical form `mean(null >= |rho|)` can return exactly `0.0`, which means "no
  > permutation reached the observed value", i.e. p < 1/1000 — not p = 0.
- **`cluster_ordering_analysis`** — median/mean/std pseudotime per cluster, ranked.
  > **TIES INTO** the biological reading: if clusters occupy distinct, ordered pseudotime
  > ranges, the trajectory is passing through discrete morphological states.
- **`spatial_depth_correlation`** — secondary check.
  > **TRAP — do not cite this number.** `run_all` never passes `roi_polygon`, so the fallback
  > always runs: it measures distance from `coords.mean(axis=0)`, the mean of patch coordinates
  > **pooled across every slide**. Each slide has its own origin, so that point is in no slide
  > at all. It does not feed the verdict. The `roi_polygon` branch would be meaningful but also
  > hardcodes `coords + 56` (half of patch_size=112) as the patch centre.
- **`run_full_validation(...)`** — orchestrates the above and emits the headline verdict.
  `verdict_features` excludes `h_intensity_wholepatch` so one quantity does not vote twice.
  Non-finite correlations are excluded from the verdict rather than counted as weak, and the
  count is reported as `n_uncomputable_correlations`.

**The verdict rule:**

```
n_strong >= 2 and n_sig >= 2  →  POSITIVE
n_strong == 1 or  n_sig == 1  →  CAUTIOUS
otherwise                     →  NULL RESULT
```

> **TRAP — this is non-monotonic.** The middle test matches *exactly* one, so adding evidence
> can move the verdict backwards:
> `n_strong=0, n_sig=1` → CAUTIOUS, but `n_strong=0, n_sig=2` → NULL RESULT;
> `n_strong=1, n_sig=0` → CAUTIOUS, but `n_strong=2, n_sig=0` → NULL RESULT.
> With thousands of patches, permutation significance is cheap, so `n_sig` is usually large and
> `n_strong` is the discriminating term. **Treat the string as a coarse label**; read
> `summary.n_strong_correlations` and `n_significant_permutations` for anything that matters.

---

## Part 11 — Outputs and figures

**Modules:** `utils/io.py`, `utils/viz.py`

`save_json` converts numpy types and maps **non-finite floats to `null`**, with
`allow_nan=False` so anything that slips through raises loudly.

> **WHY.** `json.dump` defaults to `allow_nan=True`, emitting bare `NaN` / `Infinity` tokens
> that Python reads back happily but that are **not valid JSON** — `jq`, JavaScript and every
> strict parser reject them. Since features can legitimately be NaN, the default would silently
> produce unreadable artifacts.

`save_pickle` / `load_pickle` / `load_json` are thin wrappers. `save_atlas_artifacts` bundles a
reference atlas into one directory; `run_all` writes the same files individually instead.

### Per run directory

| File | Contents |
|---|---|
| `adata_full.h5ad` | Everything — `X_embed`, obs, uns (incl. root candidates), obsm (UMAP, diffmap). **The canonical output.** |
| `results.csv` | One row per patch: `x, y, slide_id, slide_name, cluster, pseudotime, pseudotime_std` + the 7 features. **Row `i` here is row `i` of `adata.obs`, by construction and by nothing else** — there is no key column and no assertion. 16 modules rely on it; see `KNOWN_ISSUES.md` §7.1. Also note `slide_id` is run-local, so join across runs on `slide_name` (§7.3). |
| `validation.json` | Correlations, permutation tests, cluster ordering, spatial secondary, verdict + summary counts |
| `feature_failures.json` | Extraction failure accounting for both passes |
| `slide_independence.json` | Per-cluster slide composition + dominance warnings |
| `sampling_manifest.csv`, `sampling/` | What was capped, and the exact indices kept |
| `active_cap.txt` | The realised cap |
| `scaler.pkl`, `pca.pkl`, `umap_reducer.pkl` | Fitted transforms |
| `projector/` | Saved `AtlasProjector` |
| `stain_reference.png` | Only when normalization was active |
| `figures/` | All plots below |

### The figures, and what each is for

| File | Function | What it tells you |
|---|---|---|
| `fig1_umap_clusters.png` | `plot_umap_clusters` | Overall structure. Two islands = a batch effect. |
| `qc_umap_by_slide.png` | `plot_umap_by_slide` | **Batch check.** Slides should intermix, not segregate. |
| `fig2_cluster_patches.png` | `plot_cluster_patch_grid` | The first N patches of each cluster **in array order**, not the nearest to the centroid. Useful for eyeballing a cluster, but not a representative sample. See the note in section 8. |
| `fig4_umap_pseudotime.png` | `plot_umap_pseudotime` | Does pseudotime vary smoothly across the manifold? |
| `fig4b_umap_pseudotime_std.png` | `plot_umap_pseudotime_std` | Where the 20 roots disagree. |
| `fig5_pt_violins.png` | `plot_pseudotime_violins` | Do clusters occupy distinct pseudotime ranges? |
| `fig6_features_vs_pt.png` | `plot_feature_vs_pseudotime` | **The result.** Each feature against pseudotime with its rho. |
| `fig7_permutation_nulls.png` | `plot_permutation_nulls` | Observed rho against its null distribution. |
| `qc_paga_topology.png` | `plot_paga` | Cluster connectivity graph — the DPT validity gate, visually. |
| `qc_umap_section_vs_cluster.png` | `plot_umap_section_cluster` | Do clusters map onto sections? If yes, the split is batch. |
| `diffusion_3d.png` | `plot_3d_manifold` | First three diffusion components. |
| `spatial_*.png` | `plot_spatial_clusters`, `plot_spatial_pseudotime` | Clusters/pseudotime laid back over slide geometry. |

`plot_test_projection` is used by the LOO path, not the main run.

> **All plotting is pure output.** No figure function affects a number.

---

## Part 12 — Projection and leave-one-out

**Module:** `analysis/projector.py` · **Class:** `AtlasProjector`

- **`from_training(scaler, pca, umap_reducer, adata_train, centroids)`** — captures the fitted
  transforms and trains a `KNeighborsRegressor` (k = `min(15, n−1)`, distance-weighted) mapping
  PCA coordinates → pseudotime.
- **`project(raw_features, slide_ids, method="knn")`** — applies the **training** scaler and PCA
  to new embeddings, then assigns clusters by nearest centroid (`NearestNeighbors`, euclidean)
  and pseudotime by KNN regression, clipped to [0, 1]. Also transforms into the training UMAP.
- **`_project_ingest`** — an alternative using `scanpy.tl.ingest`. Available, not the default.
- **`save(dir)` / `load(dir)`** — pickles the transforms, writes centroids as JSON plus metadata.

> **WHY a KNN regressor rather than re-running DPT.** Re-running DPT on new data would build a
> new manifold with a new root and a new arbitrary direction — the numbers would not be on the
> training scale. KNN transfers the *existing* coordinate.

> **WHY it trains on `X_pca_original`** when Harmony/scVI was used: `project()` can only produce
> *uncorrected* PCA features for new data, so centroid matching and KNN must live in that same
> space. The class also detects and repairs centroid-dimension mismatches arising from scVI's
> 30-dim latent space, recomputing centroids in PCA space.

### The leave-one-out experiment, in detail

**The question.** Does the trajectory *generalise*, or is it memorising this exact cohort? If a
held-out slide lands somewhere very different than it did when it helped build the manifold,
the trajectory is not a property of the tissue.

**Structure: 16 folds, two phases each.**

```
submit_loo_array.sh          SLURM array, one task per held-out slide
   └─ run_loo_single.sh      one fold
        ├─ Phase A: run_all --run --slides <the other 15>   → trains, saves projector/
        └─ Phase B: loo_project.py                          → projects the held-out slide
```

Phase A is an ordinary `run_all` invocation on 15 slides. Its only special output is
`projector/`. Phase B is where the comparison happens.

> **Phase B is skipped, not failed, if the reference full-run `results.csv` does not exist yet.**
> The script prints where the projector was saved and the exact command to re-run Phase B later.
> That decoupling exists because the reference run and the 16 folds can be submitted in either
> order.

### `loo_project.py` — the fold comparison

1. **`load_cached_features(cache_dir, slide)`** — reads `<slide>_features.npy`. Errors loudly if
   absent; it will not silently fall back to inference.
2. **Re-applies the training subsample.** This is the subtle, essential step:
   ```python
   raw_features, _, _ = sample_patches(
       raw_features, np.arange(len(raw_features)),
       args.max_patches_per_slide, args.patch_sample_seed, slide_name)
   ```
   > **WHY.** The cap and seed **must match what Phase A used**, or the held-out slide's patch
   > set differs from the one the comparison assumes. Passing `np.arange(...)` as the "coords"
   > argument is a trick to get the selected indices back out. `--max-patches-per-slide` and
   > `--patch-sample-seed` on this script exist purely to be kept in sync with training.
3. **`projector.project(raw_features, method="knn")`** — clusters by nearest centroid, pseudotime
   by KNN regression.
4. **`load_inmanifold_pseudotime(full_run_dir, slide)`** — pulls that slide's rows from the full
   16-slide run's `results.csv`, **in patch-extraction order**, which is the same order as the
   feature cache.
5. **Hard length check.** If the projected and in-manifold counts differ, it raises rather than
   comparing misaligned vectors:
   > `Patch count mismatch ... Cache and reference run must use identical extraction settings.`

**Metrics, and which one is the headline:**

| Metric | Kind | Role |
|---|---|---|
| **Spearman rho** | **Paired** — patch *i* vs patch *i* | **PRIMARY.** Only valid because both vectors are in the same patch order. |
| Wasserstein distance | Unpaired, distribution-level | Secondary |
| KS statistic + p | Unpaired, distribution-level | Secondary |

> **WHY paired beats unpaired here.** Two pseudotime distributions can have identical *shape*
> while individual patches are shuffled arbitrarily within it. Wasserstein would call that a
> perfect match. The paired Spearman would not.

Outputs per fold: `loo_result_<slide>.json`, `loo_projected_pt_<slide>.npy` (persisted so
downstream analyses can reuse the projection without re-running it), and
`loo_distribution_<slide>.png` — a paired scatter with a y=x reference line beside a KDE
overlay of the two distributions.

### `loo_summary.py` — aggregating the 16 folds

Collects every `loo_result_*.json`, writes `loo_summary.csv`, prints the mean rho, and produces
a horizontal bar chart sorted by rho with a threshold line at **0.5** (red below, blue above)
and a dotted line at the cohort mean.

It flags low-rho slides explicitly and tells you what to check:

> `→ Check whether these are predominantly from one section (2M-1 vs 2M-2).`
> `→ A strong negative rho means the pseudotime axis is flipped between runs.`

> **WHY a negative rho is informative rather than just bad.** DPT direction is arbitrary. If a
> fold's axis flipped, the *ordering* may be perfectly preserved while rho reads −0.9. That is a
> very different problem from rho ≈ 0, which means the ordering was genuinely lost.

> **WHY the feature cache is essential here.** 16 folds × full GPU inference would be
> prohibitive; with the cache it is one inference pass total. This is the single biggest reason
> the cache exists.

**`analysis/loo_summary_scvi.py`** is the scVI-flavoured variant (reachable only from a
commented-out line in `submit_loo_array_scvi.sh`). **`analysis/recover_loo.py`** is a one-off
recovery script for interrupted LOO runs. Neither is on a live path.

---

## Part 13 — QC, visualization, diagnostics

### `qc/` — run after any new atlas run

**`qc/run_qc.py`** is the master runner (`--run-dir`, `--slides-dir`, `--stain-method`,
optional `--steps`). It prepends the project root to `sys.path`, which is why its bare
`from qc.X import ...` imports resolve.

| Step | Module | What it checks |
|---|---|---|
| 1 | `graph_connectivity.py` · `check_graph_connectivity` | Is the scanpy neighbour graph **fully connected**? A disconnected graph makes `compute_dpt` assign `inf` to smaller components, which then get clamped — **this is the direct cause of the two-island pseudotime problem.** |
| 2 | `stain_qc.py` · `run_stain_qc` | Per slide: original vs normalized vs difference thumbnails, H/E/LAB-L histograms, and for Macenko the stain-vector **angle deviation** from the reference. **Explicitly flags silent normalization failures** (output array identical to input) — the ones `normalize_slide` swallows. |
| 3 | `cluster_contact_sheet.py` · `make_contact_sheets` | 5×5 grid of **randomly sampled** patches per cluster, cropped from the original PNGs. Complements `fig2_cluster_patches.png`, which shows only the centroid-nearest patch — random sampling exposes the full spread and reveals clusters driven by background, fat, or slide edges. |
| 4 | `pseudotime_by_slide.py` · `plot_pseudotime_by_slide` | Violin plots of pseudotime **per slide** (16) and **per mouse** (4). If pseudotime is batch-driven, slides occupy distinct ranges instead of overlapping. The main pipeline only stratifies by cluster. |

> **This is the highest-value 20 minutes you can spend on a new run.** Steps 1 and 4 between
> them catch most ways a result can be an artifact.

### `visualize/` — for looking at results

| Module | Produces |
|---|---|
| `interactive_overlay.py` | One standalone HTML per slide: the PNG embedded as background, patch pseudotime as a WebGL scatter overlay, with a colorscale dropdown. **The best tool for asking "does the trajectory follow anything anatomically real?"** |
| `export_patches.py` | Crops representative patches from the PNGs (using `results.csv` coordinates — no re-extraction), stratified into low/mid/high pseudotime bins. For eyeballing what each end of the axis actually looks like. |
| `interactive_plotly.py` | Self-contained interactive UMAP and 3D diffusion scatter HTML. |
| `scvi_postprocess.py` | scVI-run post-processing: fresh low-resolution Leiden on the stored graph, UMAP by section. Never writes `adata_full.h5ad`. |

### `diagnostics/` — targeted investigations

| Module | The suspicion it tests |
|---|---|
| `dpt_clamping_check.py` | Is the diffusion graph disconnected, and is the non-finite clamp firing? Motivated by `pseudotime_std` being an affine function of `pseudotime` in 2M-2 (R² ≈ 1) — which is the clamp's signature, not real uncertainty. |
| `inspect_root_patches.py` | What are the 20 root patches *actually images of*? Motivated by all 20 roots in 2M-2 having `nuclear_density` exactly 0.0, from a pool of 21 tied patches. |
| `audit_feature_diagnostics.py` | Read-only implementation of diagnostics D1–D4 plus a failure-rate cross-check from `reports/morphological_features_audit.md`. Operates only on existing `results.csv` files. |

### `figures/make_paper_figures.py`

Read-only publication figures: batch mixing across correction methods, LOO reproducibility per
section, cross-section morphological correlates. Re-runs nothing.

---

## Part 14 — Analysis branches, module by module

`analysis/` contains **53 modules** as of 2026-08-23 (it was ~25 when this document was first
written), each a self-contained investigation driven by its own `jobs/` script.
Treat them as a **lab notebook**, not as pipeline. Read the docstring and the matching report in
`reports/` before running one.

### Confound tests — "is the trajectory actually measuring something else?"

| Module | Question | Status / note |
|---|---|---|
| `cellularity_confound.py` | Is pseudotime just a cellularity meter? Computes a PC1 cellularity proxy, then partial Spearman controlling for `nuclear_density`, with a seeded permutation null. | Fully NaN-safe. Per-feature verdicts: `SURVIVES` / `collapses` / `UNCOMPUTABLE`. Its `_decision_gate` helper compares **signed** rho with no isfinite guard — print-only, never persisted. |
| `batch_mixing.py` + `run_batch_mixing.py` | How well-mixed is `section_number` in the embedding? kNN section-purity against a prevalence-weighted chance baseline. Scores `X_pca_original`, `X_pca_harmony`, `X_scvi` side by side. | Pure post-hoc; re-runs nothing. Note `get_pipeline_k_and_metric()` reads k and metric via `inspect.signature(cluster_leiden)` rather than hardcoding them — so it cannot drift from the live pipeline even if those defaults change. |
| `sign_flip_check.py` | Are the two sections' axes *opposed*, or merely *oriented* oppositely? | **Read this before interpreting cross-section sign disagreements.** It enforces the constraint that sign agreement after a flip is **arithmetic, not evidence** — what counts is (i) agreement in sign *and approximate magnitude*, and (ii) whether features non-directional in one section *stay* non-directional, since a flip can neither manufacture nor destroy a relationship. |
| `crop_calibration.py` | Is the left/right HSV tissue-fraction diagnostic even calibrated? Runs it on the known-good 16 slides to establish a baseline. | A hard gate before any re-conversion. |

### Robustness — "would a different choice have given a different answer?"

| Module | Question | Recorded finding |
|---|---|---|
| `root_sensitivity.py` | Is the pseudotime axis an artifact of its own root rule? `nuclear_density` is simultaneously the root selector, a validation feature, and the covariate partialled out in the confound test — circular by construction. | Recorded finding (cited in `pseudotime_std_analysis.py`, not re-derived here): random 20-root sets reproduce production pseudotime at \|rho\| **0.78–0.89**. So the *ordering* is robust to root choice; only the *orientation* is root-determined. Confirm against the module's own output before quoting. |
| `pseudotime_std_analysis.py` | How much do the 20 roots disagree, per patch? A stable aggregate ordering says nothing about individual patches. | Carries the load-bearing scale warning: `pseudotime` is normalized to [0,1], `pseudotime_std` is raw and pre-normalization. |
| `slide_diagnostics.py` | Why does one slide project poorly? Tests five hypotheses: H1 patch count, H2 cluster composition, H3 Phikon-space outlier, H4 isolated in UMAP, H5 unusual annotation pattern. | ⚠ H5 is fed GeoJSON via a stale `ANN_DIR`, so its `total_area` arm is in px² not ratio² and is confounded by slide size (1.9× spread). `n_polygons` is unaffected. |
| `v2_comparison.py` | Did the four feature fixes change any number? | Self-validating: it refuses to report a reconstructed baseline root set unless the reconstruction first reproduces v2's *stored* roots. |
| `cross_section_compare.py` | Do 2M-1 and 2M-2 replicate? A feature "replicates" if both sections agree in sign **and** both have \|rho\| ≥ `REPLICATE_RHO_THRESHOLD` = **0.1**. | Read-only over two `validation.json` files → `cross_section_comparison.csv`. Note 0.1 is a low bar — read the rhos, not just the flag. |
| `v3_regression_check.py` | Did a code change alter behaviour? Six exact-equality checks plus a patch-alignment precondition. | Built for the 2026-08 cleanup; **PASSED** with `max_abs_diff = 0.0`. |

### Holeyness — a duct-area confound investigation

**The question.** Ducts in these tumours contain lumens — holes. Does a duct's *holeyness*
track its pseudotime? If so, that is an interpretable, duct-level morphological correlate,
stronger evidence than a patch-level correlation.

**Why it needed five modules** is the instructive part, and worth walking her through: each
version found a problem with the previous one's *method*, not its arithmetic.

#### How duct-level data is assembled (`holeyness.py`)

This is the only analysis that joins pipeline output to an external QuPath export.

```
data/annotations_ratio/<slide>.json          Tumor polygon geometry, keyed by QuPath UUID
combined_matched_measurements.txt            hole % and hole area, keyed by the same Object ID
        └── join on UUID ──> polygon + hole measurement per duct
results.csv                                  patch (x, y) + pseudotime
        └── patch-centre-in-polygon ──> patches assigned to ducts
        └── aggregate (median) ──> one pseudotime per duct
```

Key functions: `parse_measurement_export`, `load_duct_polygons`, `build_duct_table`,
`assign_patches_to_ducts`, `aggregate_per_duct`, `run_correlations`, `run_permutation_test`.

> **WHY a UUID join rather than spatial matching.** QuPath assigns each annotation a stable
> Object ID that appears in both the GeoJSON and the measurement export. Matching on geometry
> instead would be fragile and would silently mis-pair near-identical ducts.

#### The version chain

| Version | What it added | What it found |
|---|---|---|
| **v1** | Base correlation, median per-duct aggregation, permutation test on the raw correlation | rho(pseudotime, hole_pct) ≈ **0.276** |
| **v2** (`--v2`) | Duct-**area** covariate checks, within-slide/nested permutations, exclusion-bias check on zero-patch ducts, aggregation sensitivity, patch-sampling artifact check | **The raw 0.276 is substantially confounded by duct area.** Area-adjusted partial = **0.131**; controlling area + nuclear_density = 0.158 |
| **v3** | A permutation test **on the partial** — v2 had only ever permuted the *raw* correlation. Plus investigation of 3 slides with near-zero/negative area-adjusted partial | Closed the significance gap v2 left open |
| **v3b** | Tested **patch count per duct** (not area) as the discriminator between those 3 slides and the other 5; within-slide undersampling test | All 3 flagged slides sit at `median_n_patches = 2`; every other slide is 3 or 4 |
| **final** | Consolidation and correction of four methodological problems | See below |

> **A detail worth admiring:** v3b's **step 0** resolves an unexplained ~0.000118 mismatch
> between v3's recomputed partial and v2's saved reference value *before trusting anything
> downstream*. That is the right instinct — a small numerical discrepancy you cannot explain is
> a reason to stop, not to round.

#### What `holeyness_final.py` corrects

Four problems accumulated across v1–v3b. It fixes them in **one new module** rather than
patching each in place, leaving v1–v3b frozen as provenance.

1. **Circular group selection.** v3/v3b's "3 flagged slides" were chosen *by looking at* their
   weak area-adjusted partials in v2, then tested for properties **on the same data**. Replaced
   — not patched — with a non-circular check across all 8 slides with no subsetting (Task A).
2. **No pre-specified vs exploratory separation.** ~60+ correlations were computed across v1–v3b
   with no labelling and no multiplicity correction. Every reported quantity is now tagged
   **PRIMARY** or **EXPLORATORY**, and the exploratory ones are counted (Task C).
3. **The estimand was never stated.** **571 of 2173 ducts (26%)** were excluded for containing
   zero patches under the centre-in-polygon rule — and v2 had already shown those excluded ducts
   are systematically *smaller and less holey*. So the correlation does not describe "ducts"; it
   describes a biased subpopulation. The population is now stated explicitly (Task B).
4. **Sign/magnitude conflation.** v3b's "3/3 slides strengthened" verdict used **absolute
   magnitude**, counting a slide moving from −0.069 to −0.202 as "strengthened" when it had
   moved *away* from the cohort's positive signal. Corrected with direction preserved (Task E).

**Two further design choices in `final`:**

- **Raw and area-adjusted partial are reported as CO-PRIMARY** (Task D), rather than picking
  one. > **WHY.** Whether duct area is a *confounder* (adjust for it) or a *mediator* (do not —
  bigger ducts may be holier *because* they are further along) is unresolved biology. Picking
  one silently picks an answer.
- **Optional Task F** re-derives patch-to-duct assignment under an **area-overlap** rule
  (`OVERLAP_MIN_FRACTION_DEFAULT = 0.25`) instead of centre-in-polygon, to test whether the 26%
  exclusion bias can be *addressed* rather than merely documented.

> **Read only `holeyness_final.py`.** The earlier versions are kept for provenance and are
> imported by it, never edited. Per project memory the final numbers were pending a Narval run —
> **confirm they exist before quoting anything.**

### Timepoint — a cancelled branch, and why

**The question.** A separate cohort of slides exists at *different timepoints* (4W, 7W, 8W, 12W),
from different mice, never run through the pipeline. If projected pseudotime tracked weeks, that
would be the strongest possible validation — an external, biologically meaningful axis.

**Status: CANCELLED.** Two independent problems, either of which is disqualifying. This section
exists so nobody revives it without knowing both.

#### The stage chain

| Stage | Module | Role |
|---|---|---|
| 1 | `timepoint_inventory.py` | Convert + inventory the first 8 slides |
| 1b | `timepoint_convert_nocrop.py` | Re-convert **without cropping** (see below) |
| 2 | `timepoint_stage2_stain_check.py` | Stain batch check vs the 2M-1 slides — **hard gate** |
| 2a | `stage2_reference_threshold.py` | Establish the *correct* threshold for that gate |
| A | `timepoint_cohort_inventory{,_v2}.py` | Corrected cohort inventory (29 usable slides) |
| B | `timepoint_stain_homogeneity{,_v2}.py` | **Within-cohort** stain gate, full resolution |
| C | *(conversion of remaining slides)* | 22 of 24 succeeded |
| D | `timepoint_projection.py` | Feature extraction + projection |
| E | `timepoint_diagnostic.py` | Does pseudotime track weeks independently of stain? |
| F | `timepoint_roi_mismatch.py` | **The finding that ends it** |

#### Three methodological corrections worth learning from

**1. The crop assumption did not transfer.** `run_all --convert` hardcodes a left-half crop —
correct for the original 16, where the duplicate right half is *annotation-confirmed*. Per-slide
QuPath inspection found the timepoint batch is **not uniform**: some slides duplicate across both
halves, some do not. No blanket rule is safe.

> **Resolution:** convert **without** cropping and keep full-width coordinates, making the crop
> decision **reversible** — any slide can be filtered to `x < width/2` post-hoc once confirmed.
> Cropping during conversion is irreversible; the pixels are gone. This is a good general
> instinct: when you cannot make a decision correctly, make it *later*.

**2. The stain gate was comparing the wrong unit.** Stage 2 originally used
`--known-confound-r 0.71`, borrowed from the D3 diagnostic. But D3 computed rank-biserial on
**per-patch** `h_intensity` (8244 vs 10072 patches), while Stage 2 compares **per-slide**
summaries. Those are not the same quantity — patch-level carries within-slide *plus*
between-slide variance; slide-level carries only between-slide.

> `stage2_reference_threshold.py` exists solely to recompute the project's own known
> cross-section confound (2M-1 vs 2M-2) **at the slide level**, using the identical
> tissue-masked method, so the two are guaranteed comparable.

**3. A prior Stage 2 run was declared VOID for background dilution.** It masked tissue with
`qc/stain_qc.py`'s LAB `L < 230` rule, which is looser than what the pipeline actually uses to
decide "is this tissue". Full-width timepoint PNGs contain **two** mounted pieces plus the gap
and margins; left-cropped originals contain **one**. Systematically different background
fractions would make the comparison measure *how much background is in frame*, not stain
chemistry.

> **Fixed** by rebuilding the mask from `features/patching.py`'s real criteria — `_has_tissue_hsv`
> and `_is_mostly_white`. The lesson generalises: a QC mask and a pipeline mask must be the same
> mask, or you are measuring the difference between them.

**Data issues Stage A v2 surfaced:** `6041-4L-12W` and `6069-4R-4W` both fail `read_region` with
`OpenSlideError("Restart marker not found")` — corrupted, permanently excluded, not retried.
`60997-4L-4W-2` has level-0 dimensions and file size *identical* to `6097-4L-4W` (86400×49280,
602.7 MB) — a near-certain duplicate rescan. It is verified and reported explicitly by
`check_duplicate_60997`, never silently dropped.

#### Problem one — Stage B v2: the stain gate FAILED

**State the finding precisely, because the loose version is wrong:**

> The RGB channels are **mostly NOT separating** by timepoint — negligible-to-small in most
> comparisons. The measure driving FAIL is specifically **hematoxylin intensity**, which
> separates in every adequately-powered pairwise comparison and shows a **monotonic trend with
> weeks**.

> **Do not describe this as "broad staining differences." It is not one.** It is narrow and
> hematoxylin-specific, and it is consistent with **either** a reagent-side confound **or**
> genuine cellularity change with tumour age. **This cohort cannot distinguish those two.**

That ambiguity is the whole difficulty. If it is reagent drift, the timepoint signal is an
artifact. If it is real cellularity change, it is exactly the biology you want. Nothing in the
data resolves it.

No correction was applied; the gate stands. Stages D and E ran anyway, explicitly as
**non-blocking diagnostics** — not steps toward a claimed positive result.

#### Problem two — Stage F: the projection was 100% extrapolation

This is the finding that ends the branch, and it is independent of the stain issue.

> Stage D projected all 29 slides and **every single one** returned
> `frac_beyond_training_p99 = 1.0` — total extrapolation, with **no discrimination between
> slides**. Training's own median first-neighbour distance is 19.3 (p99 = 25.5); every projected
> slide sits uniformly at **32–40**.

So Stage E's pseudotime-versus-weeks correlation was computed **entirely outside the training
manifold's support**. It is uninterpretable as it stands — not weak, not marginal:
uninterpretable.

**The hypothesis Stage F tests:** the 29 timepoint slides have **no annotations**, so Stage D
patched them **whole-slide**, while the training manifold was built **exclusively from annotated
tumour ROI patches**. Stroma, necrosis and other non-tumour tissue are therefore present in the
projection set but were never in training.

> That is a patch-**composition** mismatch, distinct from — and possibly larger than — the
> hematoxylin confound. Per project memory, Stage F was written 2026-08-05 and its **numbers were
> still pending**. Check whether it has run before drawing any conclusion.

#### If you ever revive this

The blocking requirements, in order:

1. **Annotate the timepoint slides**, or otherwise restrict projection to tumour ROI. Without
   this, Stage F's mismatch stands and nothing downstream is interpretable.
2. **Disambiguate hematoxylin: reagent drift vs real cellularity change.** Probably needs
   staining controls or a same-batch restain, not more analysis.
3. Only then re-run Stages D and E.

> `timepoint_diagnostic.py`'s own docstring is unusually firm about this, and it is right:
> *no output may be interpreted or shown to anyone as a timepoint result until the PI's biology
> question and the stain-versus-cellularity disambiguation are resolved.*

### Eccentricity — the 2026-08-11 rejection was itself overturned

`eccentricity_check.py` was read on 2026-08-11 as showing the axis is an *eccentricity*
measure (atypical in any direction) rather than a trajectory, on the strength of
`rho(PT, diffusion-map centroid distance)` ≈ 0.80 against `rho(PT, DC1)` ≈ 0.50.

**That reading does not survive.** The centroid-distance correlation is **partly true by
construction** — DPT pseudotime *is* a diffusion distance from its roots, so a high value
there is definitional and cannot be evidence. In the spaces where DPT's construction does
*not* force the answer, the picture reverses: morphological eccentricity is only 0.15–0.28
(weaker still within slides), and **0 of 6 features show both tails enriched** among
late-pseudotime patches, with 4–5 unidirectional. That is the trajectory signature, not the
eccentricity one.

Both "live concerns" it left behind are now characterised:

- **Slide-dominated late tail — real, and quantified.** The late decile is 3.3–3.6× one
  slide against 12.5% uniform. `eccentricity_within_slide.py` shows the cohort late
  *subclusters* largely **are** slides (Cramér's V 0.81 / 0.54), so the "opposing features"
  verdict is a batch split rather than two late phenotypes. Within slides the eccentricity
  signature is **absent** on every axis tested.
- **Sections pointing opposite ways — an anchor artifact.** It tracked the duct-size
  extremity of the root set, not biology. See `docs/ANCHOR_VALIDATION_RECORD.md` §3.2.

### Anchor validation — added 17–21 August 2026

Nine modules written after this document's first draft. They exist to answer one question:
**is the pseudotime axis a property of the tissue, or of where the roots were placed?**
Full narrative in `docs/ANCHOR_VALIDATION_RECORD.md`; this table is the index.

| Module | Question | Headline |
|---|---|---|
| `anchor_area_control.py` | Is `rho(pt, duct area)` a duct-size artifact? | **Yes.** Size-matched anchors that ignore hole % reproduce it in full. All 20 root ducts sit below the median eligible duct (p ≈ 1e-6). |
| `export_anchor_axis.py` | Persist an alternative-anchor axis as a readable run dir | Supports `area_stratified`, `area_matched_surrogate`, `v2_repaired`. Gated against the values it must reproduce. |
| `eccentricity_within_slide.py` | Is the late structure biology or one slide? | Late subclusters largely **are** slides; within slides the eccentricity signature is absent. |
| `duct_white_fraction.py` | Does `hole_pct` track actual white space? | **Yes** — rho 0.92 (2M-1) / 0.79 (2M-2) by direct polygon rasterisation. The annotation is sound; the *root rule* built on it was not. |
| `holeyroot_duct_checks.py` | Nesting, intervals, and what `hole_pct` measures | Effects hold within slides, 8/8. Intervals are narrow, not wide. `texture_entropy` is **not** an independent validator. |
| `holeyness_asymmetry.py` | Why does validation differ by section? | The premise was wrong — the circulated 2M-2 "0.020" was a different quantity. Real value +0.1906, 7/8 slides. |
| `holeyness_section_comparison.py` | Do the sections' correlations actually differ? | Exact test over all C(16,8)=12,870 relabellings: **no evidence they do**. |
| `holeyness_repaired_sensitivity.py` | Does repairing 2M-2's roots fix the divergence? | **No** — a pre-declared mechanistic hypothesis, refuted. Reported as such. |
| `holeyroot_compare.py` | Holeyness-rooted vs density-rooted axis | Its headline discriminator turned out to be the duct-size artifact above. |

> **Read `ANCHOR_VALIDATION_RECORD.md` §5 before quoting any correlation from this family.**
> It lists twelve statistics that are circular, superseded, artifactual or null. The
> surviving independent validators for 2M-2 are **`mean_nuclear_area` and `nc_ratio`** —
> that is the whole list.

---

## Part 15 — Which job script to use

| Goal | Script |
|---|---|
| **Full run from raw NDPI (handoff demo)** | `jobs/run_full_pipeline_handoff.sh` |
| **The canonical reference config** | `jobs/run_per_section_v2.sh` |
| Baseline per-section + LOO + cross-section | `jobs/run_per_section.sh` |
| Populate the feature cache | `jobs/run_cache_population.sh` |
| Convert NDPIs only | `jobs/convert_ndpi.sh` — **corrected 2026-08-25**, now `--ndpi-dir $SCRATCH/data/ndpi --ndpi-level 0 --ndpi-scale 0.5`. Verify any conversion with `jobs/verify_conversion_smoke.sh`. |
| Quick end-to-end smoke test | `jobs/run_smoke_test.sh` |
| Post-run QC | `jobs/submit_qc.sh` |
| Verify a code change changed nothing | `jobs/run_per_section_v3_regression.sh` + `analysis/v3_regression_check.py` |
| PAGA variant suite (6 variants) | `jobs/submit_paga_runs.sh` → `run_cache_prepop.sh` → `run_paga_variant.sh` |
| LOO array | `jobs/submit_loo_array.sh` → `run_loo_single.sh` |

**Reference outputs that must stay reproducible:**
`$SCRATCH/results/per_section_v2/atlas_2M-1/` and `atlas_2M-2/`.

`archive/jobs/` holds eight superseded scripts. Kept for history, **must not be run** — six
point at the pre-migration annotation directory.

---

## Part 16 — Open scientific questions

These are **not** code defects. They are live uncertainties about what the current results mean,
and they are the most useful thing to pick up.

**1. The two sections disagree, and it is not only the arbitrary DPT direction.**
After flipping 2M-2's axis, four of six features reconcile in sign — but `h_intensity` does not
(+0.117 in 2M-1 vs −0.405 flipped in 2M-2), and it is 2M-2's strongest correlate.
`sign_flip_check.py` is the rigorous version of this comparison; its constraint is that sign
agreement after a flip is vacuous on its own, so judge by **magnitude** and by whether
non-directional features stay non-directional.

**2. `h_intensity` and `h_intensity_wholepatch` are near-swapped between sections.**

| | masked (`h_intensity`) | whole-patch |
|---|---|---|
| 2M-1 | +0.117 | **+0.399** |
| 2M-2 | **+0.405** | +0.039 |

In 2M-1 the signal lives in overall patch darkness — staining and density. In 2M-2 it lives in
nuclei-masked chromatin. Fix 1c was built to tell these apart, and it is reporting that they
behave in opposite ways across sections.

**3. 2M-2's POSITIVE verdict is knife-edge.** Its two qualifying features are `nc_ratio` at
0.4006 and `h_intensity` at 0.4053, against a 0.4 threshold. `nc_ratio` clears by 0.0006. Given
the non-monotonic verdict rule, do not put weight on the word "POSITIVE".

**4. The confound analysis disagrees by section.** Every feature `SURVIVES` partialling out
nuclear density in 2M-2; every feature `collapses` in 2M-1. Same code, same seed — a real
difference needing an explanation.

**5. The 2M-2 root set is a tie — CONFIRMED, and worse.** All 20 roots have
`nuclear_density` exactly 0.0, from 21 such patches, so which 20 win is `argsort`
tie-breaking. Later work added: **none of the 20 lies inside any Tumor annotation**, and
`0.0` cannot be told apart from a segmentation failure. See
`docs/ANCHOR_VALIDATION_RECORD.md` §4, error #2.

**6. 2M-2's `pseudotime_std` — RESOLVED 2026-08-21, and it was NOT a clamp artifact.**
The diffusion graph has one connected component and `n_roots_clamped` is 0, so the clamp
never fired. The spread came from **3 of 20 roots ordering the manifold backwards**;
dropping them takes `pseudotime_std` from 27.70% to 3.40% of range while moving the axis
by only rho 0.9621. Full account in Part 9 and in `docs/ANCHOR_VALIDATION_RECORD.md` §7.

**7. Section 4's disagreement is now partly explained, and item 4 above needs re-reading.**
The "every feature SURVIVES in 2M-2, every feature collapses in 2M-1" asymmetry rests on
partialling out `nuclear_density` — which is *also* the root selector, so that analysis was
never independent of the anchor. Under a size-neutral anchor `nuclear_density`'s own
correlation goes to +0.101 (2M-1) vs −0.244 (2M-2): the sections disagree in *sign*.

**8. `rho(duct area, pseudotime)` diverges between sections and nobody knows why.**
+0.4325 (2M-1) vs −0.0844 (2M-2), differing at the smallest p an exact 12,870-way
slide-relabelling test can produce. It is **anchor-rule-dependent** but **not**
root-repair-dependent, and **not** explained by duct-size/density geometry —
`rho(area, nuclear_density)` is +0.389 vs +0.342, near-identical. Genuinely open.

---

## Part 17 — Trap index

Roughly by severity.

1. **`data/annotations` vs `data/annotations_ratio`** — the pipeline needs *ratio*. Feeding
   GeoJSON produces no error, just no in-ROI patches.
2. **`img_dims.txt` must match `KNOWN_NDPI_DIMENSIONS`** — the round-trip invariant. Add new
   slides to both.
3. **One feature cache = one parameter combination.** The guard checks N, not D; the key is the
   slide name alone.
4. **Three k-NN graphs**, different k *and* metric. The PAGA gate mixes two of them.
5. **`--fixed-cap` is inert** unless `--cap-strategy fixed`.
6. **Stain reference = first slide in sorted order**, after subsetting. Subsets are not
   comparable under reinhard/macenko.
7. **Stain normalization failure is silent.** Grep the log for "Stain normalization failed", or
   run `qc/stain_qc.py`.
8. **The verdict string is non-monotonic.** Read the counts.
9. **`spatial_depth_secondary` is meaningless** as computed.
10. **`pseudotime` and `pseudotime_std` are on different scales.** Normalized vs raw.
11. **`--root-cluster` / `--root-metric` on `run_all` are no-ops.** `run_individual`'s
    `--root-cluster` is live.
12. **PCA width is data-dependent.** Check widths match before comparing runs component-wise.
13. **`run_individual.py` results are not comparable** with atlas results — no cap, no cache,
    per-slide PCA basis, single-root DPT, and a default root that is an arbitrary Leiden label.
14. **The cohort was converted at `--ndpi-scale 0.5`, not 1.0.** `jobs/convert_ndpi.sh`
    passed 1.0 and pointed at a non-existent NDPI directory, so it could not regenerate
    the dataset; **both fixed 2026-08-25**. Scale 1.0 gives twice the linear resolution and
    roughly 4x the patch count. Verified by reproducing two reference PNGs bit-identically
    at level 0 / scale 0.5 (`jobs/verify_conversion_smoke.sh`, job 1648162). The "x5" in
    `MCF7_x5_cropped` is therefore meaningful, not vestigial labelling.

---

## Part 18 — Running it, and debugging it

### The full run

```bash
cd ~/cancer_trajectory_atlas
sbatch jobs/run_full_pipeline_handoff.sh
```

That single job performs: NDPI → PNG, GeoJSON → ratio annotations (in an isolated staging
directory, so the repo is never written), annotation QC overlays, the dimension-invariant check,
and the per-section pipeline with a **new** feature cache. Everything lands under one fresh
`$SCRATCH/handoff/<run_id>/`. It refuses to start if any output path would fall inside an
existing results tree, and it is **resumable** — re-submit with the same `RUN_ID` and completed
stages are skipped.

### Reading a new result, in priority order

1. **`feature_failures.json`** — should be 0 and 0.
2. **`slide_independence.json`** — no cluster >80% one slide.
3. **The PAGA line in the log** — "SINGLE component" means DPT is valid.
4. **`annotation_qc/*.png`** — polygons must sit on tissue.
5. **`validation.json` → `summary.n_strong_correlations`** — not the verdict string.
6. **`figures/fig6_features_vs_pt.png`** — the actual result.

### If something looks wrong

| Symptom | First thing to check |
|---|---|
| "0 slides discovered" | Slide stems vs `--slides` list; PNG filenames |
| "0 with annotations" | Is `--annotation-dir` the **ratio** directory? |
| Few or no patches survive | Annotation coordinate space; run `check_annotations.py` |
| `RuntimeError: Feature cache size mismatch` | Extraction settings changed. Delete the stale `.npy` or use a fresh cache dir. |
| Two-island UMAP | `qc/graph_connectivity.py`, then `qc_umap_section_vs_cluster.png` |
| Pseudotime looks binary | Disconnected graph → `diagnostics/dpt_clamping_check.py` |
| `pseudotime_std` ∝ `pseudotime` | Same — the clamp is firing |
| A cluster looks like junk | `qc/cluster_contact_sheet.py` |
| Slides occupy distinct pseudotime ranges | Batch effect → `qc/pseudotime_by_slide.py`, `analysis/batch_mixing.py` |
| Sections disagree in sign | `analysis/sign_flip_check.py` — and read its constraint first |
| Suspect it's all just cellularity | `analysis/cellularity_confound.py` |
| Suspect the root rule drives it | `analysis/root_sensitivity.py` |

---

## Appendix — Parameter reference (as used in reference runs)

| Parameter | Value | Set by |
|---|---|---|
| model | `phikon` (768-dim CLS) | CLI |
| patch_size / stride | 112 / 96 (16 px overlap) | CLI |
| tissue filters | white 220 / 0.70; HSV sat 15, val 230, frac 0.50 | code default |
| stain_method | `none` | CLI |
| batch_method | `none` | CLI |
| cap_strategy | `median` | CLI |
| patch_sample_seed | 42 | code default |
| min_roi_coverage | `None` (centre-point only) | code default |
| PCA variance target | 0.95 → 241–261 components | code default |
| Leiden | **k=15, cosine**, resolution 0.5 | k/metric code default; resolution CLI |
| UMAP | k=30, cosine, min_dist 0.1 | code default |
| Diffusion map | **k=30, euclidean**, 10 components | k from CLI; metric is scanpy's default |
| PAGA threshold | 0.05 | code default |
| n_roots | 20 | CLI |
| GLCM | 4 angles × 3 distances (1,3,5), 64 levels | code constant |
| Nuclear segmentation | Otsu → opening `disk(1)` → drop <20 px | code default |
| n_permutations | 1000 (200 for LOO folds, 10 for smoke) | CLI |
| Correlation thresholds | strong \|rho\| > 0.4, moderate > 0.3 | code constant |
| Slide dominance warning | 0.80 | code default |
| Projector KNN | k = min(15, n−1), distance-weighted | code default |

**Never overridden by any job script:** `--diffmap-comps`, `--target-total`, `--use-stardist`,
all `--scvi-*`, `--root-cluster`, `--root-metric`.

---

*Further reading in this repo: `NOTES.md` (design decisions and scientific status),
`reports/codebase_inventory.md` (module reachability and parameter provenance),
`reports/morphological_features_audit.md`, `PROJECT_STATE.md` (working log),
`archive/README.md` (what was removed and why).*
