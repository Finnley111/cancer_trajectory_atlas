# PIPELINE.md — what the atlas pipeline actually does

**Written against the code as of 2026-08-22.** Every claim here is traceable to a file
and line. Where a prior document disagrees with the code, the code wins and the
discrepancy is called out.

**Relationship to `docs/PIPELINE_HANDOFF.md`:** that document is a longer tutorial-style
walkthrough written 2026-08-12. This one is a shorter reference written independently
against the current tree, and it corrects two things that document and the surrounding
notes get wrong (§7). Where the two disagree, prefer this one and check the cited line.

---

## 1. Purpose and scope

The pipeline takes whole-slide H&E images of MCF7 xenograft tissue and produces, for
every image patch, a scalar **pseudotime** — a position on a one-dimensional axis derived
from the geometry of a learned feature manifold. It then correlates that axis against six
independently computed morphological descriptors.

**What it computes.** A patch-level ordering, per run, together with per-patch uncertainty,
cluster labels, a UMAP embedding for display, and a validation suite of correlations and
permutation tests.

**What it does not claim.**

- Not a measurement of real time. Nothing in the data is longitudinal; all 16 slides are
  one timepoint.
- Not comparable across runs. Pseudotime is min-max normalised per run and the patch cap
  is the cohort median, so absolute values mean nothing between runs (§5.3).
- Not a directed trajectory by construction. The manifold fixes the ordering; the root
  rule only fixes which end is called zero (§5.4).
- Not causally interpretable with respect to fixation. Fixation is perfectly collinear
  with section in this cohort.
- `hole_pct` is **not fixation-invariant**, because Carnoy's deforms ductal
  architecture anisotropically — the lumen collapses to roughly a quarter of what
  the duct does (5.48× vs 1.64×, 8/8 glands). That is a **result** in its own
  right (`docs/ANCHOR_VALIDATION_RECORD.md` §3.12), not merely a limitation.
  Within-section validation is unaffected, and the cross-section replication still
  holds — Spearman ignores monotone rescaling — but it should be stated as *the
  correlation replicates despite `hole_pct` being systematically rescaled between
  conditions*. See `docs/KNOWN_ISSUES.md` §2.1.

---

## 2. Data model

### 2.1 Three coordinate spaces

Source: `features/patching.py:66-90` (`load_roi_polygons` docstring).

| # | Space | Definition |
|---|---|---|
| 1 | **Full-NDPI pixel** | width = NDPI level-0 dimension. Includes **both** slide copies side by side. Stored as `original_full_width`. |
| 2 | **Cropped-PNG pixel** | width = `original_full_width // 2`, left half only. **Patch `(x, y)` coordinates live here.** |
| 3 | **Ratio** | coordinates in `[0, 1]` relative to full-NDPI dimensions. QuPath annotates the left half, so left-half annotation x-values fall in `[0, 0.5]`. |

**The invariant that relates them.** With `coordinate_space="ratio"`, polygon x is
multiplied by `original_full_width`, landing left-half annotations in
`[0, original_full_width/2]` — which *equals* `[0, cropped_width]`. That is the same
space as patch coordinates, so **no further offset is applied**
(`features/patching.py:83-87`).

Right-half polygons are discarded. The discard runs only when both `cropped_w` and
`original_full_width` are supplied, and drops any polygon whose vertex-mean centroid has
`cx > cropped_w` or `cy > cropped_h`. The y-condition never fires in practice because the
crop is horizontal only (`cropped_h == original_full_height`), but it is there
(`patching.py:88-94`).

**Consequence, and the failure mode it produces.** Feeding absolute-pixel GeoJSON to a
function expecting ratio coordinates multiplies every coordinate by
`original_full_width` a second time and puts every ROI off-canvas — few or no in-ROI
patches survive. This is exactly why eight first-generation job scripts were archived
(`archive/README.md`).

### 2.2 The feature-cache contract

Source: `run_all.py:363-390`, guard at `396-403`.

- The cache stores **full, uncapped** features. The cap is applied in Pass 2, after the
  cohort median is known. Sampling before caching would bake one cap into the file.
- Cache key is **the slide name alone**: `<slide_name>_features.npy`.
- The guard compares **N only**, not D. It raises `RuntimeError` when the cached row
  count differs from the freshly extracted patch count.

**What the guard catches:** anything changing how many patches a slide yields —
`--patch-size`, `--stride`, `--min-roi-coverage`, edited annotations, and usually
`--stain-method` (normalisation runs before the tissue filters, so it changes which
patches survive).

**What it does not catch, and this is load-bearing:**

- A **different model**. `--model resnet50` against a Phikon-populated cache passes (both
  have N rows) and silently uses 768-dim Phikon features for a run labelled resnet50.
- A `--stain-method` change that leaves the patch count unchanged while changing every
  pixel.

**Therefore: one cache directory must serve exactly one
`(model, patch_size, stride, min_roi_coverage, stain_method)` combination.** All current
job scripts share `$SCRATCH/data/features_cache` and all pass `--model phikon
--patch-size 112 --stride 96 --stain-method none`, so this holds today. It is an
unenforced convention, not a checked constraint.

### 2.3 Slides → glands — the 16 slides are 8 matched pairs

**Every mouse-flank combination (a *gland*) contributes exactly one slide to each section:**
6027 / 6028 / 6029 / 6031 × 4L / 4R. So the 16 slides are **8 matched pairs**, not 16
independent samples. Verified empirically from the per-duct tables by
`analysis/gland_pairing_audit.py`, which refuses to proceed on an unbalanced design.

**Nothing in `run_all.py` knows this.** The pipeline treats slides as independent throughout —
`slide_ids`, the per-slide cap, the feature cache, LOO. That is fine for extraction and
embedding, where slides genuinely are processed independently, but it means **any downstream
statistical test must supply the pairing itself.**

Consequences, and where they bite:

| analysis | correct unit | status |
|---|---|---|
| between-section comparison | **gland** (8 pairs, 2⁸ = 256 sign flips) | corrected 2026-08-23 |
| within-section bootstrap | slide | unaffected — no two slides in a section share a gland |
| full-atlas 16-slide LOO | **gland** | **leaks** — holding out a slide leaves its partner in training |

See `docs/KNOWN_ISSUES.md` §1.1–1.3.

### 2.4 Patches → slides → ducts

- Patches carry `slide_name` and `(x, y)` in cropped-PNG space. `slide_ids` indexes into
  a `slide_names` list.
- **Patch coordinates are not in `adata.obs`.** `obs` accumulates `cluster` and `slide_id`
  (`diffusion.py:35-36`), `mouse_id` and `section_number` (`run_all.py:572-573`),
  `pseudotime` and `pseudotime_std` (`diffusion.py:372-374`), and the morphological
  features (`run_all.py:672-673`) — but never `x`, `y` or `slide_name`. Those go only to
  `results.csv` (`run_all.py:728-739`), built from the same arrays in the same block, so
  the two are in identical row order **by construction**.
- **That row-order invariant is unchecked, and 16 modules depend on it.** There is no
  shared key column and no assertion anywhere. Only `analysis/anchor_area_control.py`
  verifies it (`_verify_row_alignment`, line 167), and only because the assumption
  already crashed job 1200392. A future change to either writer misaligns every
  dependent module **with no error**. See `KNOWN_ISSUES.md` §7.1, which is item 2 on
  the fix shortlist.
- **`slide_id` integers are run-local.** They are positions in `slide_names`, itself a
  sorted, `--slides`-filtered glob, so two runs over different subsets assign different
  integers to the same slide. `results.csv` carries both name and id; `adata.obs` carries
  only the id. **Cross-run comparisons must join on `slide_name`.** See §7.3 there.
- Duct assignment is **patch centre inside Tumor polygon**
  (`analysis/holeyness.py:305`). This excludes ~26% of ducts (2M-1) and ~22% (2M-2),
  systematically the smallest. An area-overlap alternative exists
  (`analysis/holeyness_roots.py:99`) but is **not** what any published number used.

---

## 3. Stage by stage

Phase banners below are the ones the code prints (`run_all.py:278, 290, 505, 553, 656`).
Parameter values are as invoked by the reference run `jobs/run_per_section_v2.sh:179-195`.

### Stage 0 — NDPI → cropped PNG (`--convert` only)

| | |
|---|---|
| Module | `run_all.convert_ndpi_to_left_half_png` (`run_all.py:51`) |
| Input | `.ndpi` files |
| Output | left-half PNGs + `slide_dimensions.json` (`run_all.py:122`) |
| Parameters | `--ndpi-level 0`, **`--ndpi-scale 0.5`** (`jobs/convert_ndpi.sh`). **Corrected 2026-08-25:** this table previously said scale 1.0, which is what the script passed but NOT what produced the reference PNGs. Scale 0.5 reproduces them bit-identically (job 1648162); scale 1.0 gives twice the linear resolution and ~4x the patch count. See `reports/codebase_inventory.md` §3.1. |

**Why:** the NDPIs contain two copies of the same slide side by side; annotations were
drawn on the left, so the right is discarded (`run_all.py:54-55`).

**Note on "x5".** The magnification appears only in the *directory name*
`MCF7_x5_cropped`. This code reads level 0 at scale 1.0 and applies **no downsampling**.
Any x5 reduction happened before these files reached the pipeline.

### Stage 1 — Stain normalisation (PHASE 1)

| | |
|---|---|
| Module | `data/stain_normalization.py` |
| Parameter | `--stain-method none` **(reference run)** |
| Config default | `reinhard` — deliberately overridden |

**Why the override matters:** with `none`, this stage is a pass-through. Every published
per-section number was produced without stain normalisation, despite the config default
saying otherwise. When a method *is* active, `<output_dir>/stain_reference.png` is written
(`run_all.py:286`) so the reference is recoverable after the fact.

### Stage 2 — Patch extraction and embedding (PHASE 2)

Two passes. **Pass 1** extracts and caches; **Pass 2** applies the cap and assembles.

**Extraction** — `features/patching.py:get_patches_from_array`

| Parameter | Value | Source |
|---|---|---|
| `patch_size` | 112 | CLI |
| `stride` | 96 | CLI (patches overlap by 16 px) |
| `min_roi_coverage` | `None` → centre-point test only | argparse default |

Tissue filters, applied in order at `patching.py:317, 324`:

| Filter | Threshold | Source |
|---|---|---|
| white rejection | `white_thresh=220`, `white_frac=0.70` | **function defaults** (`patching.py:21-22`) |
| HSV tissue | `sat_thresh=15`, `val_thresh=230`, `tissue_threshold=0.5` | **function defaults** (`patching.py:29-31`) |

**None of these five values is in `PipelineConfig` or on the CLI.** They are function
defaults that have never been varied. Reported as fact, not justified.

**Embedding** — `features/extractors.py`. Phikon returns the **CLS token of the last
hidden state, 768-dim** (`extractors.py:28`). Output `(N, 768)` float32.

**The cap** — `run_all.py:433-435`:

```python
active_cap = int(np.median([d["orig_count"] for d in slide_data]))
```

Cohort median, computed after full extraction across all slides in the run, written to
`active_cap.txt`. **This is why patch counts and therefore the PCA basis depend on which
slides are in the run** — a fact that blocks cross-run comparison (§5.3).

`max_patches_per_slide=200` is **inert** under `cap_strategy="median"`. `target_total=3200`
is informational only — `pipeline_config.py` states it is "logged; never used in sampling
logic."

### Stage 3 — PCA, batch correction, clustering, UMAP (PHASE 3)

**PCA** — `analysis/clustering.py:fit_pca`, called at `run_all.py:508` as
`fit_pca(features, variance_target=0.95)`. The 0.95 is **hardcoded at the single call
site**, not configurable. sklearn interprets a float `n_components` as "retain this much
cumulative variance", so the output width is data-determined: **261 components for 2M-1,
241 for 2M-2**.

Output `X_pca` → `(N, ~250)`.

**Batch correction** — `--batch-method none` in the reference run. When Harmony or scVI
runs, it operates on the post-PCA matrix and the **pre-correction PCA is preserved in
`adata.obsm["X_pca_original"]`**, which matters for the projector (§5.7).

**Clustering** — `cluster_leiden` (`clustering.py:177`), k=15, cosine, resolution 0.5.
k and metric are **function defaults not plumbed to the CLI**; only resolution is.

**UMAP** — `run_umap` (`clustering.py:117`), k=30, cosine, `min_dist=0.1`, all function
defaults. **Display only** (§5.2).

### Stage 4 — Diffusion pseudotime (PHASE 4)

**Diffusion map** — `analysis/diffusion.py:compute_diffusion_map`, `n_neighbors=30`,
`n_comps=10`. The graph is built by `sc.pp.neighbors(adata, n_neighbors=..., use_rep="X")`
at `diffusion.py:82` — **with no `metric=` argument**, so scanpy supplies `euclidean`.

**PAGA gate** — `compute_paga_topology` (`diffusion.py:380`). Groups by Leiden labels
(cosine k=15) but computes connectivity on the diffusion graph (euclidean k=30). So
"single component → DPT is valid" means precisely: *the euclidean k=30 manifold is
connected between the cosine k=15 clusters*. It says nothing about cluster separability
(`diffusion.py:389-398`).

**Multi-root DPT** — `compute_dpt_multi_root`, `n_roots=20`.

Root rule, stated exactly. The pseudocode below is the docstring's own summary
(`diffusion.py:219-223`); the implementation it describes is at `diffusion.py:301-316`
and was read separately to confirm it matches:

```
finite_idx = flatnonzero(isfinite(nuclear_density))
roots      = finite_idx[argsort(nuclear_density[finite_idx])][:n_roots]
```

This is **not** the same as `argsort(nuclear_density)[:20]`; the two agree only when every
patch has a measured density. Several analysis modules quote the simpler form. Anything
needing the true root set should read `adata.uns['dpt_root_candidates']` (written at
`diffusion.py:325`).

**`n_roots` is silently clamped** to the number of finite-density patches
(`n_roots = min(n_roots, finite_idx.size)`, `diffusion.py:314`), so the realised root
count can be lower than the CLI value with no error and no warning. Read
`len(adata.uns['dpt_root_candidates'])` for the true count rather than assuming the CLI
value took effect.

Aggregation (`diffusion.py:364-373`): each root runs its own `sc.tl.dpt`; non-finite
values are clamped per-root to that root's max finite value; the median across roots
becomes `pseudotime` after min-max normalisation; the **standard deviation across roots
becomes `pseudotime_std` and is stored unnormalised** (§5.6).

### Stage 5 — Morphological validation (PHASE 5)

| | |
|---|---|
| Module | `validation/morphological_features.py`, `validation/correlations.py` |
| Parameters | `--n-permutations 1000`, `use_stardist=False` (Otsu segmentation) |

Six verdict features (`run_all.py:693-696`): `nuclear_density`, `mean_nuclear_area`,
`nc_ratio`, `texture_entropy`, `h_intensity`, `packing_irregularity`. A seventh,
`h_intensity_wholepatch`, is computed and reported but **does not vote**
(`run_all.py:691-692`).

**This stage runs after Stage 4** — see §5.3.

### Stage 6 — Outputs and projector

`AtlasProjector.from_training(scaler, pca, umap_reducer, adata, centroids)` at
`run_all.py:763`. Artifact list in `reports/codebase_inventory.md` §4.

---

## 4. End-to-end worked example

```bash
sbatch jobs/run_per_section_v2.sh
```

**Reads:** `$SCRATCH/data/MCF7_x5_cropped/*.png`,
`~/cancer_trajectory_atlas/data/annotations_ratio/*.json`,
`$SCRATCH/data/features_cache/*_features.npy`.

**Runs:** `run_all.py --run` twice, once per section, each over 8 slides.

**Writes:** `$SCRATCH/results/per_section_v2/atlas_2M-{1,2}/` — `results.csv`,
`adata_full.h5ad`, `validation.json`, `projector/`, `figures/`, `scaler.pkl`, `pca.pkl`,
`umap_reducer.pkl`, `active_cap.txt`, `feature_failures.json`, `sampling_manifest.csv`,
`slide_independence.json`.

**Resources requested** (`jobs/run_per_section_v2.sh` SBATCH block): 8 h walltime, 8 CPUs,
64 GB, account `def-lmarti46`. That is the *request*, not the observed runtime — actual
walltime was not recovered for this document and should be read from the job's `sacct`
record rather than guessed.

**Scale:** 8,244 patches (2M-1) and 10,072 patches (2M-2).

---

## 5. Eight non-obvious facts

Each independently verified against source for this document. **Two of the eight, as
stated in the brief that requested this file, were inaccurate** — see §5.1 and §5.7.

### 5.1 There are THREE k-NN graphs, not two

The brief that requested this document says "TWO distinct kNN graphs". **The code builds
three** (`analysis/clustering.py:5-17`):

| Consumer | Built by | k | Metric | Set where |
|---|---|---|---|---|
| Leiden clusters | sklearn `kneighbors_graph` | 15 | **cosine** | `cluster_leiden` defaults |
| UMAP (display) | `umap.UMAP` | 30 | **cosine** | `run_umap` defaults |
| Diffusion map → DPT, PAGA | scanpy `sc.pp.neighbors` | 30 | **euclidean** | `compute_diffusion_map` |

All three run over the *same* matrix (`X_embed`) and none derives from another. The
euclidean metric is **not a choice** — it is scanpy's default, because
`compute_diffusion_map` passes no `metric` (`diffusion.py:80-82`). There is no CLI flag
for it; changing it means editing that line.

**Consequence:** clusters and pseudotime describe different geometries. Two patches can be
cluster-mates yet far apart in diffusion space. Neither is wrong; they answer different
questions.

### 5.2 UMAP is visualisation only

`X_umap` is passed to `build_adata` and stored in `adata.obsm["X_umap"]`
(`diffusion.py:38-39`), then used only by `viz.plot_umap_*` (`run_all.py:542-543, 640-641`)
and the interactive overlays. **No clustering, pseudotime, or validation result depends on
it.** `umap_reducer.pkl` is saved and the projector can transform new points into UMAP
space for display, but nothing numerical reads it.

### 5.3 Morphological features are computed post hoc

PHASE 5 (`run_all.py:656`) runs **after** PHASE 4 (`run_all.py:553`). The six descriptors
are computed on patch pixels once pseudotime already exists, and **never enter the
manifold** — the manifold is built from Phikon embeddings alone. This is what makes the
feature correlations a validation rather than a circularity.

With one exception, which is §5.4.

### 5.4 The root rule is circular, and this is real

`nuclear_density` occupies three roles simultaneously:

1. **Root selector** — `diffusion.py:219-223` picks the 20 lowest-density patches.
2. **Validation feature** — it is in `verdict_features` (`run_all.py:694`).
3. **Confound covariate** — `analysis/cellularity_confound.py:211`
   `analyze_run_nuclear_density` partials it out of the other five.

So the axis is partly *defined* by a quantity it is later validated against, and by the
quantity used to adjust that validation. **Neither the reported `nuclear_density`
correlation nor the "collapses under cellularity adjustment" finding is independent of the
anchor.**

Mitigating evidence, for calibration rather than dismissal: 25 uniformly random 20-root
sets reproduce the production axis at |rho| 0.78–0.89, and the holeyness-anchored axis
agrees with the density-anchored one at rho 0.9476 in 2M-1 despite 19 of 20 roots
differing. The manifold, not the root rule, is doing most of the work.

### 5.5 The axis direction is unanchored

Because random root sets reproduce the ordering, the root rule effectively fixes only
**which end is called zero**, not the sequence. A root-rule change is expected to alter
orientation and root membership, not ordering.

### 5.6 `pseudotime_std` is stored unnormalised

`diffusion.py:364-373`:

```python
pseudotime_std = np.std(pt_matrix, axis=0)          # raw diffusion-distance scale
...
adata.obs["pseudotime"]     = (pseudotime_median - pt_min) / (pt_max - pt_min)
adata.obs["pseudotime_std"] = pseudotime_std        # NOT rescaled
```

`pseudotime` is on `[0, 1]`; `pseudotime_std` is on the **raw diffusion-distance scale**.
Comparing them directly, or reading std as a fraction of the axis, requires dividing by
the raw pre-normalisation range — which is printed (`diffusion.py:375`) but **never
persisted**. Recovering it after the fact means parsing the SLURM log.

**Do not use `pseudotime_std` as an anchor-health check.** It reads dispersion the median
has already absorbed: in 2M-2 it read 27.70% of range, fell to 3.40% after dropping three
discordant roots, yet the axis itself moved by rho 0.9621 — barely at all. Use **mean
leave-one-out concordance across roots** instead, and keep std for what it is good at,
a per-patch uncertainty map. See `docs/ANCHOR_VALIDATION_RECORD.md` §7.

### 5.7 `AtlasProjector` fits its KNN on the pre-correction PCA — and here is why

`analysis/projector.py:46-51`:

```python
# project() applies scaler+PCA to get raw PCA features, so both centroid
# matching and KNN pseudotime must be trained in that same space.
if "X_pca_original" in adata_train.obsm:
    train_pca = adata_train.obsm["X_pca_original"]
else:
    train_pca = adata_train.X
```

**The reason is that `project()` cannot reproduce the correction.** It applies
`scaler_` then `pca_` to new data (`projector.py:105-106`) and has no way to apply
Harmony's or scVI's learned correction to points the correction never saw. So the
regressor must be trained in the space projection can actually produce.

**A correction to how this is usually stated.** It is conditional, not unconditional: the
pre-correction PCA is used *only when a correction was applied*. In the reference run
(`--batch-method none`) there is no `X_pca_original`, so `train_pca = adata_train.X`,
which **is** the uncorrected PCA. For every published per-section run the two branches
coincide and nothing is being discarded.

### 5.8 `run_individual.py` is a separate pipeline

It uses **`IndividualConfig`** (`run_individual.py:77-88`), not `PipelineConfig`.

The table below was verified against the **implementation**, not its docstring. The
three "none/no" rows are absence claims, confirmed by grepping `run_individual.py` for
`sample_patches`, `features_cache`, `max_patches`, `cap_strategy`,
`run_full_validation` and `compute_dpt_multi_root`: **zero occurrences of any of them in
code**. The single match is the string `compute_dpt_multi_root` inside a docstring
comment at line 31.

| | `run_all.py` | `run_individual.py` |
|---|---|---|
| PCA | cohort-wide, once | **per slide**, inside the per-slide loop (`run_individual.py:304`) |
| Patch cap | cohort median | **none** — `sample_patches` never called |
| Feature cache | yes | **no** — calls `extract_features` directly (`run_individual.py:302`) |
| DPT | 20 density roots, median-aggregated | **single cluster-anchored root** via `compute_dpt` (`run_individual.py:332`) |
| Diffusion-map k | fixed 30 | **`min(30, n_patches - 1)`** — adaptive (`run_individual.py:318`) |
| Validation suite | yes | **no** — `run_full_validation` never called |

**Its pseudotime is not comparable across slides, nor to any per-section result.** Note
also that its `--root-cluster` is **live**, backed by `IndividualConfig.root_cluster` —
unlike `PipelineConfig.root_cluster`, which is a vestigial no-op (§6).

---

## 6. Vestigial parameters

`PipelineConfig.root_cluster` and `PipelineConfig.root_metric` are populated by
`run_all.py` from `--root-cluster` / `--root-metric` and then **never read**. Nothing on
the `--run` path consumes either. They predate multi-root DPT: single-root DPT chose its
origin as the patch nearest a named cluster's centroid, and `compute_dpt_multi_root`
replaced that with the density rule. Setting them changes nothing.

The dataclass comment warns that **field order must not be rearranged**, because these are
dataclass fields and their order defines the positional `__init__` signature.

---

## 7. Where this document corrects prior claims

| Source | Claim | Reality |
|---|---|---|
| Brief requesting this doc | "TWO distinct kNN graphs" | **Three.** UMAP builds its own (k=30, cosine). §5.1 |
| Brief requesting this doc | AtlasProjector fits on pre-correction PCA (unconditional) | **Conditional** on a correction having been applied. In the reference run the branches coincide. §5.7 |
| `PROJECT_STATE.md:79, 391` | `run_train_test.py` exists, imports a broken `config.py` | File does not exist. |
| `PROJECT_STATE.md:400, 572` | `validation/annotations.py` provides `load_annotations()` | File does not exist. |
| Various notes | `paths.json` points at `data/annotations` | Points at `data/annotations_ratio`. |

`PROJECT_STATE.md` (90 KB, last modified 2026-07-28) predates the 2026-08-12 cleanup and
should not be used as a reference for tree structure.

---

## 8. Environment

From `jobs/run_per_section_v2.sh:153-158`:

```bash
module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Compute nodes have no internet, so **Phikon weights must already be in `$HF_HOME`**;
`TRANSFORMERS_OFFLINE=1` makes a missing cache fail loudly rather than hang on a network
call.

| Location | Contents |
|---|---|
| `$SCRATCH/data/MCF7_x5_cropped/` | cropped PNGs + `slide_dimensions.json` |
| `$SCRATCH/data/features_cache/` | per-slide Phikon features — **one config only**, §2.2 |
| `$SCRATCH/results/` | all run outputs |
| `~/cancer_trajectory_atlas/data/annotations_ratio/` | ratio-coordinate annotations (**authoritative**) |
| `$SCRATCH/huggingface_cache/` | offline model weights |

Scheduler is SLURM, account `def-lmarti46`.
