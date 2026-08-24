# Codebase Inventory — Phase 0

**Date:** 2026-08-22
**Scope:** read-only analysis. No source file was created, modified, moved, or archived
by this phase. The one file operation performed was preserving the previous inventory
(see §0).
**Authority rule:** the code is authoritative. Where any prior document disagrees with
what the code does, the discrepancy is reported in §6 rather than silently resolved.

---

## 0. Read this first — a prior cleanup already ran

This is not a clean-slate inventory. A multi-phase cleanup executed on **2026-08-12** and
left durable artifacts in the tree. Several premises in the current brief are already
resolved, and re-doing that work would be wasted effort.

| Prior artifact | State |
|---|---|
| `reports/codebase_inventory.md` | Existed, dated 2026-08-12, covering 64 real modules. **Superseded by this document**; the original is preserved verbatim at `archive/reports/codebase_inventory_2026-08-12.md`. |
| `archive/` + `archive/README.md` | Exists. 10 files archived across that pass's Phases 1, 5 and 7, each with a written rationale. |
| `docs/PIPELINE_HANDOFF.md` | Exists, 79 KB, 12 parts. **This is substantially the Phase 1 deliverable already** — see §7. |
| `docs/ANCHOR_VALIDATION_RECORD.md` | Exists, written 2026-08-21. Covers the analysis-branch experiments, an error log, an untrusted-statistics list, parameter provenance and a roadmap. Overlaps Phase 2's scope. |

Note the prior pass used **different phase numbers** from this brief. Its Phase 5 was
diffusion (this brief's Phase 7); its Phase 7 was job scripts (this brief's Phase 10).
When reading `archive/README.md`, treat its phase numbers as belonging to that pass.

**Premises in the current brief that are already false or already fixed:**

1. *"`run_train_test.py` is documented as importing a non-existent `config.py`; confirm
   and find others."* — **The file does not exist.** It is absent from the working tree
   and was deleted before the 2026-08-12 pass began. `config.py` has never existed. The
   claim survives only in `PROJECT_STATE.md`, which is stale on this point. Phase 10's
   instruction to archive it cannot be carried out. Its orphaned companions
   `train_test_config.json` and `example_config.json` remain in the tree and are read by
   nothing (§4.3).
2. *"`paths.json` `annotations` points at `data/annotations` while job scripts use
   `data/annotations_ratio`."* — **Already fixed.** `paths.json` now reads
   `"annotations": "~/cancer_trajectory_atlas/data/annotations_ratio"`.
3. *"`paths.json` `stain_reference` is unread."* — **The key no longer exists.**
   `paths.json` has exactly four keys: `raw_ndpi`, `cropped_png`, `annotations`,
   `results`. All four are read by `run_all._load_default_paths()` (`run_all.py:29`).
4. *"Archive the dead mask-loading code in `validation/annotations.py`."* (Phase 8) —
   **`validation/annotations.py` does not exist.** `validation/` contains only
   `__init__.py`, `correlations.py` and `morphological_features.py`. `validation/__init__.py:17`
   records that its consumer was `run_train_test.py`, "itself since removed."

**Consequence for the plan:** Phase 3 is largely a no-op, Phase 8's archival target is
gone, and Phase 10's archival target is gone. I recommend confirming how you want those
phases scoped before we reach them.

---

## 1. Method

Reachability was computed mechanically, not by eye.

1. Every `*.py` outside `.git/`, `__pycache__/` and `archive/` was parsed with `ast` and
   its internal imports resolved (relative imports against the module's own package;
   absolute `cancer_trajectory_atlas.X` imports normalised to `X`). Imports **inside
   function bodies** were captured — this matters, because `run_all.run_pipeline()` does
   all of its pipeline imports lazily.
2. Entry points were extracted from `jobs/*.sh`, distinguishing live lines from
   commented-out ones. An initial regex missed invocations of the form
   `python "$REPO/converters/batch_convert.py"`; those were found by a second sweep and
   folded in. **Two modules were misclassified as unreachable until that second sweep** —
   worth knowing if this analysis is ever repeated.
3. Transitive closures were taken separately from `run_all`, from `run_individual`, and
   from the job seeds, so the classes can be told apart.

**Totals:** 92 `.py` files. 10 are package `__init__.py` (all empty or a single
docstring; none re-export anything), leaving **82 real modules** including the two entry
points. Up from 64 at the 2026-08-12 inventory — the growth is entirely in
`analysis/`, from the anchor-validation work.

---

## 2. Reachability map

### 2.1 Class A — reachable from `run_all.py` (18 modules)

The pipeline proper.

| Module | Role |
|---|---|
| `run_all.py` | entry point, `--convert` and `--run` |
| `pipeline_config.py` | `PipelineConfig` dataclass |
| `data/slide_registry.py` | `KNOWN_NDPI_DIMENSIONS` fallback table |
| `data/stain_normalization.py` | stain normalisation (reinhard / macenko / none) |
| `features/patching.py` | ROI polygons, patch extraction, tissue filters, sampling |
| `features/extractors.py` | Phikon / ResNet50 embedding |
| `analysis/clustering.py` | PCA, Leiden, UMAP, slide-independence check |
| `analysis/harmony.py` | Harmony batch correction |
| `analysis/scvi_integration.py` | scVI batch correction |
| `analysis/diffusion.py` | diffusion map, PAGA, multi-root DPT |
| `analysis/holeyness.py` | imported for duct loaders used by holeyness rooting |
| `analysis/holeyness_roots.py` | `root_source="holeyness"` root selection |
| `analysis/projector.py` | `AtlasProjector` |
| `validation/morphological_features.py` | the six descriptors |
| `validation/correlations.py` | validation suite |
| `utils/io.py`, `utils/viz.py`, `utils/__init__.py` | JSON/pickle IO, figures |

`analysis/holeyness.py` and `analysis/holeyness_roots.py` are on this path **only** when
`--root-source holeyness` is passed. The reference `per_section_v2` run does not pass it,
so neither executed in that run, but both are imported.

### 2.2 Class B — reachable only from a job script (54 modules)

Invoked directly by one or more `jobs/*.sh`, never imported by `run_all`. This is the
analysis and diagnostics estate: the `holeyness*` family (12 modules), the `timepoint*`
family (10), the anchor-validation modules (`anchor_area_control`, `export_anchor_axis`,
`eccentricity_*`, `holeyroot_*`, `duct_white_fraction`, `root_sensitivity`, …), plus
`diagnostics/` (4), `qc/run_qc.py`, `figures/make_paper_figures.py` and `visualize/` (4).

Two of these are reachable only via `$REPO`-path invocation inside
`jobs/run_full_pipeline_handoff.sh`: `converters/batch_convert.py` (line 320) and
`jobs/check_annotations.py` (line 399).

### 2.3 Class C — reachable only via another module (5)

Never invoked directly; imported by a Class B module.

| Module | Imported by |
|---|---|
| `analysis/batch_mixing.py` | `analysis/run_batch_mixing.py` |
| `qc/cluster_contact_sheet.py` | `qc/run_qc.py` |
| `qc/graph_connectivity.py` | `qc/run_qc.py` |
| `qc/pseudotime_by_slide.py` | `qc/run_qc.py` |
| `qc/stain_qc.py` | `qc/run_qc.py` |

### 2.4 Class D — unreachable (4)

**All four import cleanly.** There are no broken imports anywhere in the tree — the
condition the brief expected to find does not exist.

| Module | Status |
|---|---|
| `analysis/loo_summary_scvi.py` | Imports OK. Referenced only in a **commented-out** line of `jobs/submit_loo_array_scvi.sh:30` and in its own docstring. |
| `analysis/recover_loo.py` | Imports OK. Referenced only in its own docstring usage example. A recovery utility for interrupted LOO arrays. |
| `converters/ndpi_to_img.py` | Imports OK. Standalone CLI; superseded on the pipeline path by `run_all.py --convert`, which does left-half cropping the converter does not. |
| `converters/tiff_to_img.py` | Imports OK. Standalone CLI. No TIFF input exists in the current data. |

These are archive candidates for Phase 10, but note `recover_loo.py` is an operational
recovery tool — archiving it removes a rescue path for a long array job. Recommend
keeping it and documenting rather than archiving. **No action taken in this phase.**

---

## 3. Parameter provenance

Reference run = `per_section_v2` (`jobs/run_per_section_v2.sh:179-195`). The CLI it
issues is reproduced verbatim in §3.3.

**Legend for "Source":**
**CLI** = explicitly passed by the reference job · **cfg-default** = `PipelineConfig`
default, never overridden · **argparse-default** = `run_all.py` default, never overridden
· **fn-default** = Python function default, never plumbed to the CLI · **lib-default** =
third-party library default, never set by this codebase.

### 3.1 Parameters the brief asked about

| Parameter | Value in `pipeline_config.py` | Job override? | Value in `per_section_v2` | Source | Deliberate? |
|---|---|---|---|---|---|
| `patch_size` | 112 | yes, `--patch-size 112` | 112 | CLI | **deliberate** (matches config) |
| `stride` | 96 | yes, `--stride 96` | 96 | CLI | **deliberate** |
| `ndpi_scale` | 1.0 | n/a (`--convert` only) | 1.0 | cfg-default | conversion used `--ndpi-scale 1.0` explicitly (`jobs/convert_ndpi.sh:26`) |
| `ndpi_level` | 0 | n/a (`--convert` only) | 0 | cfg-default | `jobs/convert_ndpi.sh:25` passes 0 explicitly |
| magnification | — | — | — | — | **Not a pipeline parameter.** "x5" appears only in the *directory name* `MCF7_x5_cropped`. Conversion reads NDPI level 0 at scale 1.0; no downsampling is applied by this code. |
| `white_thresh` | not in config | no | 220 | **fn-default** (`features/patching.py:21,230`) | **DEFAULT, not a choice** |
| `white_frac` | not in config | no | 0.70 | **fn-default** (`patching.py:22,231`) | **DEFAULT, not a choice** |
| `sat_thresh` (saturation) | not in config | no | 15 | **fn-default** (`patching.py:29,227`) | **DEFAULT, not a choice** |
| `val_thresh` (value) | not in config | no | 230 | **fn-default** (`patching.py:30,228`) | **DEFAULT, not a choice** |
| `tissue_threshold` (fraction) | not in config | no | 0.5 | **fn-default** (`patching.py:31,229`) | **DEFAULT, not a choice** |
| PCA variance target | not in config | no | 0.95 | **fn-default**, hardcoded at the one call site (`run_all.py:508`) | **DEFAULT, not a choice** |
| Leiden k | not in config | no | 15 | **fn-default** (`clustering.py:179`) | **DEFAULT, not a choice** — not plumbed to CLI |
| Leiden metric | not in config | no | cosine | **fn-default** (`clustering.py:181`) | **DEFAULT, not a choice** |
| `leiden_resolution` | 0.5 | yes, `--leiden-resolution 0.5` | 0.5 | CLI | **deliberate.** Function default is 1.0 and is never used. |
| diffmap k | `diffmap_neighbors = 30` | **no** | 30 | argparse/cfg-default | value chosen in config, but **not passed by the reference job** |
| diffmap metric | not in config | no | euclidean | **lib-default** (scanpy) | **DEFAULT, not a choice.** `sc.pp.neighbors` is called with no `metric=` (`diffusion.py:82`). No CLI flag exists. |
| diffmap `n_comps` | `diffmap_comps = 10` | no | 10 | argparse/cfg-default | |
| `n_roots` | 20 | yes, `--n-roots 20` | 20 | CLI | **deliberate** |
| `cap_strategy` | `"median"` | yes, `--cap-strategy median` | median | CLI | **deliberate** |
| `stain_method` | `"reinhard"` | yes, `--stain-method none` | **none** | CLI | **deliberate override of the config default** |
| harmony `nclust` | not in config | no | n/a — Harmony not run | **fn-default** 10 (`harmony.py:51`) | inert in the reference run (`--batch-method none`) |

### 3.2 Additional parameters worth recording

| Parameter | Config | Reference run | Note |
|---|---|---|---|
| `model` | `phikon` | `--model phikon` | deliberate |
| `n_permutations` | 1000 | `--n-permutations 1000` | deliberate |
| `max_patches_per_slide` | 200 | not passed → 200 | **inert**: `cap_strategy="median"` ignores it |
| `target_total` | 3200 | not passed | **informational only**; config comment states it is "logged; never used in sampling logic" |
| `patch_sample_seed` | 42 | not passed → 42 | argparse-default |
| `min_roi_coverage` | `None` | not passed → `None` | centre-point check only |
| `use_stardist` | `False` | not passed | Otsu segmentation used |
| `batch_method` | `None` | `--batch-method none` | deliberate |
| `root_source` | `"cellularity"` | not passed | density rooting |
| `root_cluster`, `root_metric` | `None`, `"cellularity"` | not passed | **vestigial no-ops**, already documented as such in `pipeline_config.py` |
| UMAP k / metric / min_dist | — | 30 / cosine / 0.1 | fn-defaults; **display only**, nothing downstream reads the embedding |

### 3.3 The reference invocation, verbatim

```
python -m cancer_trajectory_atlas.run_all --run \
  --png-dir "$PNG_DIR" --annotation-dir "$ANN_DIR" --output-dir "$OUT_DIR" \
  --stain-method none --batch-method none --model phikon \
  --patch-size 112 --stride 96 \
  --clustering-method leiden --leiden-resolution 0.5 \
  --n-roots 20 --n-permutations 1000 \
  --features-cache-dir "$CACHE_DIR" --cap-strategy median --slides "$SLIDES_CSV"
```

`ANN_DIR="$HOME/cancer_trajectory_atlas/data/annotations_ratio"` (line 82).

### 3.4 Summary — what was a default rather than a choice

**Eleven** operative values were never deliberately chosen. Reporting only what the code
shows, with no inference about intent:

- Five tissue-filter thresholds (`white_thresh` 220, `white_frac` 0.70, `sat_thresh` 15,
  `val_thresh` 230, `tissue_threshold` 0.5) — function defaults, not in config, not on
  the CLI.
- PCA variance target 0.95 — hardcoded at its single call site.
- Leiden k=15 and metric=cosine — function defaults, not plumbed to the CLI.
- Diffusion-map metric=euclidean — **scanpy's** default; the call passes no `metric`.
- UMAP k=30, metric=cosine, min_dist=0.1 — function defaults (display only).
- Harmony `nclust`=10 — function default (inert in the reference run).

The diffusion-map **k=30** is a middle case: chosen in `PipelineConfig` and exposed as
`--diffmap-neighbors`, but not passed by the reference job, so the config default is what
ran. It is the one parameter here with an external benchmark — see
`docs/ANCHOR_VALIDATION_RECORD.md` §6 on the k≥50 plateau.

---

## 4. Output artifacts

### 4.1 Written by `run_all.py --run`, per run directory

| Artifact | Line | Consumed by |
|---|---|---|
| `results.csv` | 730 | **74 files.** The most-read artifact in the repo. |
| `adata_full.h5ad` | 716 | 39 files |
| `validation.json` | 744 | 20 files |
| `projector/` | 764 | 13 files (LOO) |
| `holeyness_roots.json` | 628 | 11 files (only written when `--root-source holeyness`) |
| `feature_failures.json` | 678 | 9 files |
| `scaler.pkl`, `pca.pkl` | 747-748 | 5 files each |
| `umap_reducer.pkl` | 749 | 3 files |
| `active_cap.txt` | 446 | 3 files |
| `figures/` | 251 | figure consumers |
| `scvi/` | 752 | only when scVI is the batch method |
| `sampling/*.npy` | 738-742 | written only when sampling occurred |

### 4.2 Orphaned artifacts — nothing reads them

| Artifact | Line | Assessment |
|---|---|---|
| `sampling_manifest.csv` | 735 | **Zero consumers.** Provenance record of the per-slide cap. Recommend keeping — it documents which patches were sampled, which is not recoverable otherwise — but it is genuinely unread. |
| `slide_independence.json` | 758 | **Zero consumers.** The cluster/slide-dominance check. Its *console output* is read by humans; the JSON is not read by code. |

Both are diagnostics whose value is archival rather than programmatic. **No action
proposed** — deleting a provenance record to satisfy a "nothing reads it" criterion would
be a net loss.

### 4.3 Orphaned config files at repo root

`train_test_config.json` and `example_config.json` are read by nothing. They are
leftovers from the deleted `run_train_test.py` workflow. Archive candidates for Phase 10.

---

## 5. Duplicate and near-duplicate logic

Identified only. **Nothing merged in this phase.**

### 5.1 Partial Spearman — four implementations, numerically identical

| Implementation | Method |
|---|---|
| `analysis/holeyness.py:384` `_partial_spearman` | algebraic three-correlation formula |
| `analysis/cellularity_confound.py:188` `partial_spearman` | algebraic (docstring claims parity with the above) |
| `analysis/holeyness.py:481` `_partial_spearman_multi` | rank-transform → OLS residualise → Pearson |
| `analysis/holeyness_v3_significance.py:89` `_partial_rho` | dispatches to the two above |

**Verified numerically** on n=500 with a common confounder: all four agree to
**3.1e-17**. Repeated with heavy ties (values rounded to 1 dp): still agree to
**7.6e-17**. The algebraic and rank-residual forms are equivalent here in practice, not
merely in theory.

**Consolidation is viable in Phase 9** on the numerical-identity criterion the brief sets.
Caveat: `_partial_spearman_multi` is the only one accepting >1 control, so it must be the
survivor, and `analysis/root_sensitivity.py:895` `partial_with_permutation` wraps a
partial correlation with permutation machinery and is *not* a duplicate.

### 5.2 Safe-Spearman helpers — nine copies, two behaviours

Nine modules define a private `_safe_rho` / `_rho` / `_safe_spearman`. Seven **hardcode a
minimum n of 10**; two (`holeyroot_duct_checks.py`, `holeyness_asymmetry.py`)
**parameterise it**.

That divergence is deliberate and load-bearing: with 8 slides, a hardcoded `n >= 10`
guard silently returns `NaN` for every between-slide correlation, which reads in output
as "not computed" rather than "n is small". Any consolidation must keep the parameterised
form, or it will reintroduce that bug.

### 5.3 Patch-to-duct assignment — two rules, deliberately distinct

| Implementation | Rule |
|---|---|
| `analysis/holeyness.py:286` `assign_patches_to_ducts` | patch **centre** inside polygon |
| `analysis/holeyness_roots.py:98` `assign_patches_to_ducts_overlap` | **area overlap** ≥ 25% of the patch |

**Not a duplicate — two different estimands.** The centre rule excludes ~26% of ducts
(systematically the smallest); the overlap rule exists to recover them. Every published
number uses the centre rule. **These must not be merged.** A third copy of the overlap
logic lives inside `holeyness_final.py`'s Task F; `holeyness_roots.py:120` states it was
lifted from there, so that pair *is* a genuine duplicate.

### 5.4 Other repetition

- **JSON default-encoders**: `_json_default` is redefined in at least 8 analysis modules,
  each handling numpy scalars/arrays identically. Low-risk consolidation, low payoff.
- **Slide-clustered bootstrap**: implemented independently in
  `holeyroot_duct_checks.py` and `holeyness_section_comparison.py`. Same resampling
  scheme; not verified numerically identical.
- **Duct-table loading**: `load_slide_list` / `parse_measurement_export` /
  `load_duct_polygons` / `build_duct_table` are correctly imported from `holeyness.py` by
  every downstream consumer rather than reimplemented. **No duplication here** — worth
  stating, since it is the pattern the rest should follow.

---

## 6. Code-vs-documentation discrepancies

Code wins in every row.

| Document | Claim | What the code shows |
|---|---|---|
| `PROJECT_STATE.md:79,391` | `run_train_test.py` exists and imports a broken `config.py` | File absent from the tree. Stale. |
| `PROJECT_STATE.md:400,572` | `validation/annotations.py` provides `load_annotations()` | `validation/annotations.py` does not exist. |
| Current brief | `paths.json` `annotations` → `data/annotations` | Points at `data/annotations_ratio`. Already fixed. |
| Current brief | `paths.json` has an unread `stain_reference` key | No such key. All four keys are read. |
| Current brief | `validation/annotations.py` holds dead mask-loading code | File does not exist. |

`PROJECT_STATE.md` (90 KB, last modified 2026-07-28) predates the 2026-08-12 cleanup and
is stale in at least the two rows above. It should not be used as a reference for tree
structure.

---

## 7. Bearing on the remaining phases

Reported so the plan can be adjusted before work starts, not discovered mid-phase.

| Phase | Finding |
|---|---|
| **1** (write `docs/PIPELINE.md`) | `docs/PIPELINE_HANDOFF.md` already exists at 79 KB and covers the required structure — coordinate spaces and the round-trip invariant (Part 2), per-stage inputs/outputs (Parts 1–11), the three k-NN graphs (Part 8), UMAP as display-only, the missing-value convention (Part 10), projection and LOO (Part 12). **Recommend: audit and update it rather than writing a second document.** Writing `PIPELINE.md` alongside it creates two documents that will diverge. |
| **2** (`KNOWN_ISSUES` / `FUTURE_WORK` / `ANALYST_ERRORS`) | None of the three exist. `docs/ANCHOR_VALIDATION_RECORD.md` §4 (error log), §5 (untrusted statistics) and §9 (roadmap) already cover perhaps half the requested content and can be drawn on. Genuine work remains. |
| **3** (conversion/slide discovery) | **Near no-op.** Both named issues are already fixed. |
| **4** (patching docblock) | `load_roi_polygons` already carries a substantial coordinate-system docstring (`patching.py:80-119`). Verify sufficiency rather than assume absence. |
| **5** (cache contract) | Real work; not yet examined in detail. |
| **6** (clustering/harmony/scVI) | Real work. Note `clustering.py`'s module docstring already documents the multi-graph situation thoroughly. |
| **7** (`--root-cluster` / `--root-metric` dead flags) | Already marked vestigial in `pipeline_config.py` with an extensive comment block. Decision needed: remove, or leave as documented no-ops. The config comment warns that `run_individual.py` has its **own live** `--root-cluster` that must not be touched. |
| **8** (archive `validation/annotations.py`) | **Target does not exist.** Phase reduces to whatever else validation needs. |
| **9** (consolidate duplicates) | Viable for §5.1 (verified identical). Must not touch §5.3. Must preserve the parameterised guard in §5.2. |
| **10** (archive `run_train_test.py`, review jobs) | **Primary target does not exist.** Job-script review is still real: 79 scripts, and the v3/timepoint families are likely superseded. Orphaned root configs (§4.3) are candidates. |

---

## 8. Files touched by this phase

| File | Action |
|---|---|
| `reports/codebase_inventory.md` | **Modified** — replaced with this document |
| `archive/reports/codebase_inventory_2026-08-12.md` | **Created** — verbatim copy of the prior inventory, preserved before supersession |

No source file was created, modified, moved, or archived.
