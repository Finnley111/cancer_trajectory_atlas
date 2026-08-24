# Codebase Inventory — Phase 0

**Date:** 2026-08-12
**Repo state:** `main` @ `c0562d1`, working tree clean at time of inventory.
**Scope:** read-only. No source file was created, modified, moved, or archived.

---

## 1. Method

Reachability was computed mechanically, not by eye:

1. Every `*.py` outside `.git/` and `__pycache__/` was parsed with `ast` and its
   internal imports resolved (relative imports resolved against the module's own
   package; absolute `cancer_trajectory_atlas.X` imports normalised to `X`).
   Imports inside function bodies were captured, which matters because
   `run_all.run_pipeline()` does all of its pipeline imports lazily.
2. Entry points ("seeds") were extracted from `jobs/*.sh` by scanning for both
   `python -m cancer_trajectory_atlas.<mod>` and `python <path>/cancer_trajectory_atlas/<path>.py`,
   distinguishing live lines from commented-out ones.
3. The transitive closure of imports was taken from `run_all` and from the job seeds
   separately, so "reachable from `run_all`" and "reachable only from a job script"
   can be told apart.

**Totals:** 75 `.py` files — 11 of which are package `__init__.py` (all empty or a
single line; none re-export anything), leaving **64 real modules**.

---

## 2. Reachability map

### 2.1 Class A — reachable from `run_all.py` (16 entries, incl. 2 package inits)

These are the pipeline proper. Everything here executes on the `--run` path unless noted.

| Module | Role | Note |
|---|---|---|
| `run_all.py` | entry point | `--convert` and `--run` |
| `pipeline_config.py` | `PipelineConfig` dataclass | module-level import |
| `data/slide_registry.py` | `KNOWN_NDPI_DIMENSIONS` fallback table | module-level import; used by `_get_known_dimensions` |
| `data/stain_normalization.py` | `build_normalizer`, `normalize_slide` | |
| `features/patching.py` | `get_patches_from_array`, `load_roi_polygons`, `sample_patches` | |
| `features/extractors.py` | `extract_features`, `load_model_components`, `extract_features_from_model` | latter two only on the cache-miss path |
| `analysis/clustering.py` | `fit_pca`, `run_umap`, `cluster`, `check_slide_independence`, `get_cluster_centroids` | |
| `analysis/harmony.py` | `apply_harmony` | imported only when `effective_batch_method == "harmony"` |
| `analysis/scvi_integration.py` | `apply_scvi` | imported only when `effective_batch_method == "scvi"` |
| `analysis/diffusion.py` | `build_adata`, `compute_diffusion_map`, `compute_dpt_multi_root`, `compute_paga_topology` | see §2.5 for dead functions inside it |
| `analysis/projector.py` | `AtlasProjector` | saved at end of every run |
| `validation/morphological_features.py` | `compute_morphological_features`, `compute_nuclear_density_quick` | |
| `validation/correlations.py` | `run_full_validation` | |
| `utils/io.py` | `save_json`, `save_pickle` | |
| `utils/viz.py` | all figure writers | |
| `utils/__init__.py` | package init | |

`--convert`-only dependencies: none internal. `convert_ndpi_to_left_half_png()` uses
only `openslide` + `PIL`, plus `PipelineConfig`.

### 2.2 Class B — reachable only from a job script (30 seeds + 5 transitive)

Seeds invoked directly by `jobs/*.sh`:

| Seed module | Invoked by |
|---|---|
| `run_individual.py` | `run_individual_pseudotime.sh`, `run_new_annotations.sh` |
| `analysis/loo_project.py` | `run_full_experiments.sh`, `run_loo_single.sh`, `run_loo_single_scvi.sh`, `run_paga_variant.sh`, `run_per_section.sh`, `run_smoke_test.sh` |
| `analysis/loo_summary.py` | `run_paga_variant.sh`, `run_per_section.sh` |
| `analysis/cross_section_compare.py` | `run_per_section.sh` |
| `analysis/run_batch_mixing.py` → `analysis/batch_mixing.py` | `run_batch_mixing.sh`, `run_per_section.sh` |
| `analysis/plot_umap_by_section.py` | `run_umap_by_section.sh` (script path, not `-m`) |
| `analysis/cellularity_confound.py` | `run_cellularity_confound.sh` |
| `analysis/crop_calibration.py` | `run_crop_calibration.sh` |
| `analysis/pseudotime_std_analysis.py` | `run_pseudotime_std_analysis.sh` |
| `analysis/sign_flip_check.py` | `run_sign_flip_check.sh` |
| `analysis/slide_diagnostics.py` | `run_slide_diagnostics.sh` |
| `analysis/stage2_reference_threshold.py` | `run_stage2_reference_threshold.sh` |
| `analysis/v2_comparison.py` | `run_v2_comparison.sh` |
| `analysis/eccentricity_check.py` | `run_eccentricity_check.sh` |
| `analysis/root_sensitivity.py` | `run_root_sensitivity.sh` |
| `analysis/holeyness.py` | `run_holeyness_validation.sh`, `run_holeyness_validation_v2.sh` |
| `analysis/holeyness_final.py` | `run_holeyness_final.sh` |
| `analysis/holeyness_v3_significance.py` | `run_holeyness_v3_significance.sh` |
| `analysis/holeyness_v3b_patch_count_check.py` | `run_holeyness_v3b_patch_count_check.sh` |
| `analysis/timepoint_inventory.py` | `run_timepoint_stage1_convert.sh` |
| `analysis/timepoint_cohort_inventory.py` | `run_timepoint_cohort_inventory.sh` |
| `analysis/timepoint_cohort_inventory_v2.py` | `run_timepoint_cohort_inventory_v2.sh` |
| `analysis/timepoint_convert_nocrop.py` | `run_timepoint_convert_nocrop.sh`, `run_timepoint_convert_stageC.sh` |
| `analysis/timepoint_stage2_stain_check.py` | `run_timepoint_stage2_stain_check.sh` |
| `analysis/timepoint_stain_homogeneity.py` | `run_timepoint_stain_homogeneity.sh` |
| `analysis/timepoint_stain_homogeneity_v2.py` | `run_timepoint_stain_homogeneity_v2.sh` |
| `analysis/timepoint_projection.py` | `run_timepoint_stageD_projection.sh` |
| `analysis/timepoint_diagnostic.py` | `run_timepoint_stageE_diagnostic.sh` |
| `analysis/timepoint_roi_mismatch.py` | `run_timepoint_stageF_roi_mismatch.sh` |
| `diagnostics/audit_feature_diagnostics.py` | `run_feature_diagnostics.sh` |
| `diagnostics/dpt_clamping_check.py` | `run_dpt_clamping_check.sh` |
| `diagnostics/inspect_root_patches.py` | `run_inspect_root_patches.sh` |
| `figures/make_paper_figures.py` | `run_paper_figures.sh` |
| `qc/run_qc.py` → `qc/graph_connectivity.py`, `qc/stain_qc.py`, `qc/cluster_contact_sheet.py`, `qc/pseudotime_by_slide.py` | `submit_qc.sh` |
| `visualize/interactive_overlay.py` | 8 job scripts |
| `visualize/export_patches.py` | 8 job scripts |
| `visualize/interactive_plotly.py` | `run_post_processing_scvi.sh` |
| `visualize/scvi_postprocess.py` | `run_post_processing_scvi.sh` |
| `jobs/check_annotations.py` | `submit_annotation_check.sh` (script path) |

Note on `qc/run_qc.py`: it uses bare `from qc.X import …` rather than relative or
package-qualified imports, but it prepends both the project root and its parent to
`sys.path` at lines 33–36, so `python -m cancer_trajectory_atlas.qc.run_qc` resolves.
Reachable and functional; not a broken import.

### 2.3 Class C — unreachable from any entry point (5 real modules)

| Module | Compiles? | Internal imports resolve? | Any reference anywhere? |
|---|---|---|---|
| `converters/batch_convert.py` | yes | none (stdlib only) | `PROJECT_STATE.md` only |
| `converters/ndpi_to_img.py` | yes | none | `PROJECT_STATE.md`, `README.md` only |
| `converters/tiff_to_img.py` | yes | none | `PROJECT_STATE.md`, `README.md` only |
| `analysis/recover_loo.py` | yes | yes (`analysis.projector`, `analysis.clustering`, both exist) | `PROJECT_STATE.md` only |
| `analysis/loo_summary_scvi.py` | yes | yes (`utils.io.save_json`, `validation.correlations.correlate_features_with_pseudotime`, both exist) | referenced only in a **commented-out** line of `jobs/submit_loo_array_scvi.sh:30` |

**No unreachable module fails to import.** Every one of the five compiles cleanly and
every internal import it makes resolves to an existing symbol. There is no equivalent
of the "imports a non-existent `config.py`" failure described in the brief — see §4.

The 11 `__init__.py` files also show as unreachable in the graph because nothing
imports the packages by bare name; they are structural and must stay.

### 2.4 `converters/batch_convert.py` is not really dead — it is a manual tool

It reads `./data/annotations/*.geojson` + `./converters/img_dims.txt` and writes
`./data/annotations_ratio/*.json`, i.e. it is the **producer** of the ratio annotations
the pipeline consumes. It uses hardcoded relative paths, so it must be run from the
repo root as a plain script, which is why it never appears in `jobs/`. Archiving it
would remove the only in-repo record of how `data/annotations_ratio/` was generated.

**Maintainer confirms this script is correct and works as intended.** Keep in place;
the only gap is that nothing in the repo says it must be run from the repo root, and
`jobs/submit_annotation_check.sh` calls a differently-named module for the same job
(§3.1). Both are documentation-level items for Phase 1.

### 2.5 Dead code *inside* reachable modules

- `analysis/diffusion.py:run_diffusion_pseudotime()` — **zero callers** anywhere in the
  repo (`.py`, `.sh`, `.md`). Fully dead.
- `analysis/diffusion.py:compute_dpt()` and `choose_root_cell()` — **not** dead, but not
  on the `run_all` path either. Their only caller is `run_individual.py:296` /
  `:220`. Any Phase 5 change to them changes `run_individual`, not the atlas pipeline.

---

## 3. Broken references found

### 3.1 `jobs/submit_annotation_check.sh:37` invokes a module that does not exist

```
python -m cancer_trajectory_atlas.converters.geojson_to_ratio_json …
```

`converters/geojson_to_ratio_json.py` is not in the repo and is not in git history under
that name. The functionality lives in `converters/batch_convert.py` (§2.4), but with a
different CLI (no arguments — hardcoded relative paths), so the call site cannot simply
be renamed. `NOTES.md:195` repeats the same wrong filename.

**This script is broken as written and will exit non-zero at that line.** Reported, not
touched — repairing it is a Phase 1 decision.

### 3.2 Stale directory listings in docs

`PROJECT_STATE.md:132` still lists `jobs/recover_loo_phase_b.py`, which no longer
exists (moved to `analysis/recover_loo.py` in `0fa4880`). `PROJECT_STATE.md` itself
notes this as resolved at line 439, so the tree diagram at line 132 simply was not
updated. Cosmetic; noted for whoever documents this pass.

---

## 4. Premises in the brief that no longer hold

Four items the brief instructs later phases to act on have **already been done**, in
commit `0fa4880` ("refactor/deleted unused files, added smoke test") and `d265574`.
Flagging them now so Phases 1, 2, 6 and 7 can be re-scoped rather than executed blind.

| Brief says | Actual state |
|---|---|
| Phase 0/6/7: `run_train_test.py` exists and imports a non-existent `config.py` | **File does not exist.** It is gitignored (`.gitignore:12`) and absent from the working tree. `config.py` has never existed. Its companions `train_test_config.json` and `example_config.json` are still present but untracked and read by nothing. |
| Phase 6/7: `validation/annotations.py` contains legacy mask-loading code | **File does not exist.** Deleted in `0fa4880` (231 lines removed). `validation/` now holds only `correlations.py` and `morphological_features.py`. `features/patching.py:load_roi_polygons` is already the sole annotation path. |
| Phase 1: `paths.json` `"annotations"` points at `data/annotations`, and has an unused `"stain_reference"` key | **Neither is true.** `paths.json` has exactly four keys and `"annotations"` already reads `~/cancer_trajectory_atlas/data/annotations_ratio`. There is no `"stain_reference"` key. (`stain_reference.png` is written *into the output dir* by `run_all.py:286`, which is unrelated.) |
| Phase 2: add the coordinate-system docblock to `load_roi_polygons()` | **Already present**, `features/patching.py:59–95`, and it documents exactly the three spaces and the `ratio × original_full_width` invariant the brief describes. |
| Phase 7: `run_individual.py` uses module-level globals | **No longer true.** It was rewritten in `0fa4880` (253 lines changed) and now uses its own `IndividualConfig` dataclass (`run_individual.py:40–52`) plus a `_load_default_paths()` reading `paths.json`. It does *not* use `PipelineConfig` — that is a real inconsistency, but a different one than described. |

What *is* live from the brief: the dead `--root-cluster` / `--root-metric` flags
(Phase 5, confirmed below), the cache shape contract (Phase 3, confirmed enforced),
the two-kNN-graph documentation gap (Phase 4), and job-script supersession (Phase 7).

---

## 5. Annotation directories — both are legitimate

Not a discrepancy to fix. There is a producer/consumer relationship:

| Directory | Contents | Role |
|---|---|---|
| `data/annotations/` | 16 `.geojson`, QuPath export, **absolute pixel coords** | source of truth from the annotator |
| `data/annotations_ratio/` | 16 `.json`, **ratio coords in [0,1]** | derived by `converters/batch_convert.py`; what the pipeline consumes |
| `data/old_annotations/` | 16 `.json` | superseded ratio annotations, predates the current set |
| `data/annot_check_test/` | `.png` overlays | output of `jobs/check_annotations.py` |
| `annotations_ratio/` (repo root) | **empty** | stray empty directory, untracked (git does not track empty dirs) |

**`data/annotations_ratio/` is authoritative for the pipeline** and `paths.json` already
says so. Nine job scripts still point `--annotation-dir` at `data/annotations`:

- `run_all_none.sh`, `run_all_macenko.sh`, `run_all_reinhard.sh`, `run_all_none_section.sh`
- `submit_harmony.sh`, `submit_harmony_macenko.sh`, `submit_harmony_none.sh`
- `run_individual_pseudotime.sh`
- `run_timepoint_stage1_convert.sh` (`ANNOTATION_DIR`)

### 5.1 Those scripts were correct when they ran — the directory changed under them

Confirmed from git history, and matches the maintainer's account:

| Commit | `data/annotations/` contained |
|---|---|
| `4e29a80` (initial release) | 16 **`.json`** — ratio coordinates |
| `f050e4a` ("new annotations") | 16 `.json` **deleted**, 16 **`.geojson`** added |
| `4f66418` ("new annotations added") | GeoJSONs updated in place |

So `data/annotations` was the ratio-annotation directory for the whole early period.
The ratio files later moved to `data/annotations_ratio/` and `data/annotations/` was
repurposed as the QuPath GeoJSON source. **The nine scripts above were not wrong at the
time they were submitted; results produced by them are not retroactively suspect.**

What is true is that they are **stale today**: re-running any of them as written would
feed absolute-pixel GeoJSON to a loader that is always called with
`coordinate_space="ratio"` (`run_all.py:334`), and `discover_slides()`
(`run_all.py:183–188`) matches `*.geojson` happily. Every polygon coordinate would be
multiplied by `original_full_width` a second time and land far off-canvas, yielding few
or no in-ROI patches.

This makes them archive candidates on grounds of supersession (Phase 7), not repair
candidates. Nothing changed here.

### 5.2 One live case: `run_slide_diagnostics.sh` — confirmed hazard

`run_slide_diagnostics.sh:26` sets `ANN_DIR=$SCRATCH/data/annotations`, which the
maintainer confirms **contains GeoJSON**. This one does not go through
`load_roi_polygons`; `analysis/slide_diagnostics.py:98` reads the raw JSON itself and
shoelace-integrates each polygon into a field it names `total_area_ratio` (line 113),
used by the H5 check (`investigate_annotations`, line 399).

Fed GeoJSON, that number is in **pixel², not ratio²**, and the H5 check compares it
across slides with a ±2σ z-score (line 424). Slide areas in `KNOWN_NDPI_DIMENSIONS`
span 2.50e9 px² (`6029-4R-2M-1`) to 4.73e9 px² (`6028-4L-2M-1`) — a **1.9× spread**. Two
slides with identical *fractional* annotated area therefore differ by up to 1.9× in the
quantity H5 actually compares. The `n_polygons` half of H5 is coordinate-independent and
unaffected.

Practical consequence: the `total_area` arm of H5 is confounded by slide size. The LOO
target `6028-4L-2M-2` (3.53e9 px², mid-range) is unlikely to have been *spuriously*
flagged, but a real area anomaly could equally have been masked. **The recorded H5
verdict from the existing `slide_diagnostics` run should be re-read on Narval before
being cited.** I could not check it from here — no `slide_diagnostics` output exists in
the local tree.

Not changed. Fixing it means either pointing `ANN_DIR` at ratio JSON or normalising by
`original_full_width × original_full_height` inside `load_annotations` — both change
H5's output, so both are your call, not a behaviour-preserving cleanup.

---

## 6. Parameter provenance: `pipeline_config.py` vs. what `jobs/*.sh` actually passes

### 6.1 First, a structural point

`PipelineConfig`'s field defaults are **not** what a `run_all` invocation uses. `run_all.py`
constructs `PipelineConfig(...)` at line 865 passing *every* field explicitly from `argparse`,
so the **argparse defaults are the operative defaults** for the pipeline. The dataclass
defaults only bind for callers that construct `PipelineConfig` directly — and there are
none: `run_individual.py` uses its own `IndividualConfig`.

I compared all 24 overlapping fields. **Today the two sets of defaults agree exactly** —
no divergence. That is worth stating in Methods, because it is not structurally
guaranteed; it is currently true by coincidence of maintenance.

### 6.2 Override matrix

"Default" below means the argparse default in `run_all.py`.

| Parameter | Default | Overridden by a job script? |
|---|---|---|
| `--model` | `phikon` | Passed explicitly by every run script, always as `phikon`. Never actually differs. |
| `--patch-size` | `112` | Same — always passed, always `112`. |
| `--stride` | `96` | Same — always passed, always `96`. |
| `--clustering-method` | `leiden` | Same — always passed, always `leiden`. |
| `--leiden-resolution` | `0.5` | Effectively always `0.5`. Only `run_smoke_test.sh` differs (`0.3`); `run_per_section*.sh` pass `$LEIDEN_RES` which is set to `0.5`. |
| `--stain-method` | `reinhard` | **Genuinely varies**: `none` (per-section, LOO, scVI, cache), `macenko`, `reinhard`. The default is *not* what the reference runs use. |
| `--n-permutations` | `1000` | `1000` for full runs, `200` for LOO folds, `10` for smoke. |
| `--n-roots` | `20` | Passed as `20` by per-section / paga / full-experiments / cache-prepop / LOO-scVI. Everything else takes the default, also `20`. |
| `--cap-strategy` | `median` | Passed as `median` by per-section, paga, full-experiments, cache-prepop; `fixed` by smoke. **Not passed at all** by `run_all_none/macenko/reinhard/none_section`, `submit_harmony*`, `run_all_capped`, `run_cache_population`, `run_loo_single*`, `run_new_annotations`, `run_scvi` — those silently get `median`. |
| `--fixed-cap` / `--max-patches-per-slide` | `200` | Passed by `run_all_capped.sh` (`$MAX_PATCHES`, default 1900) and `run_smoke_test.sh` (50). **Inert everywhere else** — only read when `cap_strategy == "fixed"`. |
| `--patch-sample-seed` | `42` | Passed by `run_all_capped.sh` (42), `run_loo_single*.sh` (`${SAMPLE_SEED:-42}`). Always resolves to 42 in practice. |
| `--harmony` / `--harmony-key` | off / `section_number` | `--harmony` set by `run_all_capped`, `run_cache_population`, `run_cache_prepop`, `run_full_experiments`, `run_loo_single`, `run_new_annotations`, `run_smoke_test` (ref run only), `submit_harmony*`. Key always `section_number`. |
| `--batch-method` | `None` (falls back to `--harmony`) | Only `run_per_section.sh` / `run_per_section_v2.sh` (`none`), `run_scvi.sh` / `run_loo_single_scvi.sh` (`scvi`). |
| `--diffmap-neighbors` | `30` | Only `run_smoke_test.sh` (`10`). |
| `--min-roi-coverage` | `None` | Only `run_new_annotations.sh` (`0.75`). |
| `--features-cache-dir` | `None` | Set by 10 scripts, always `$SCRATCH/data/features_cache`. |
| `--ndpi-level` / `--ndpi-scale` | `0` / `1.0` | Passed by `convert_ndpi.sh` (at the defaults) and `run_timepoint_stage1_convert.sh` (via vars). |
| `--diffmap-comps` | `10` | **Never overridden by any script.** |
| `--target-total` | `3200` | **Never overridden.** Informational only — logged at `run_all.py:463`, never used in sampling. |
| `--use-stardist` | off | **Never overridden.** StarDist has never been enabled by any job script. |
| `--scvi-n-latent/-layers/-hidden/-max-epochs` | `30`/`2`/`128`/`400` | **Never overridden**, including by `run_scvi.sh`. |
| `--root-cluster` | `None` | **Never overridden — and never read.** See §6.4. |
| `--root-metric` | `cellularity` | **Never overridden — and never read.** See §6.4. |

### 6.3 Three divergences between documented and actual behaviour

These feed straight into Methods and all three are documentation bugs, not code bugs.
I have not changed them.

1. **`pipeline_config.py:64` says `'fixed'` is the default cap strategy.** Line 71 sets
   `cap_strategy: str = "median"`. The `(default)` annotation is attached to the wrong
   option. The pipeline has been capping at the cohort median.

2. **`pipeline_config.py:66` says `'none'` means "use `max_patches_per_slide` if set
   (backward compat), else no cap".** `run_all.py:413–415` sets `active_cap = None`
   unconditionally for `'none'`; `max_patches_per_slide` is ignored entirely. The
   backward-compat behaviour described does not exist.

3. **`jobs/run_all_capped.sh` does not do what its name says.** It passes
   `--max-patches-per-slide "$MAX_PATCHES"` (default 1900) but never passes
   `--cap-strategy fixed`, so `cap_strategy` defaults to `median` and `MAX_PATCHES` is
   dead. Any result produced by that script is median-capped, not capped at 1900. If
   anything in the writeup cites a 1900-patch cap, it is wrong.

### 6.4 Confirmed: `--root-cluster` and `--root-metric` are dead on the `run_all` path

`run_all.py:896,898` store them on `PipelineConfig`. Grepping every read of
`cfg.root_cluster` / `cfg.root_metric`: **there are none** in `run_all.py` or in anything
it calls. `compute_dpt_multi_root` takes `n_roots` and a precomputed nuclear-density
vector; the root rule is hardcoded as `argsort(nuclear_density)[:n_roots]`
(`analysis/diffusion.py:165`). The argparse help for `--root-cluster` already reads
"Legacy arg; unused with multi-root DPT"; `--root-metric` has no such warning and its
`choices=["cellularity"]` makes it look load-bearing when it is not.

Caveat for Phase 5: `run_individual.py` has its own, **live** `--root-cluster` flag
(`run_individual.py:285–296`) feeding `diffusion.compute_dpt`. Removing the dataclass
fields must not touch `IndividualConfig.root_cluster`.

### 6.5 Confirmed: the cache shape contract is intact

`run_all.py:369–376` raises `RuntimeError` when `len(slide_feats) != orig_count`. There
is exactly one cache-read site (`run_all.py:367`) and the check guards it directly, with
no early-return or `try`/`except` around it. The check runs in Pass 1, before sampling,
so it compares full uncapped counts on both sides — which is the correct comparison
given the cache stores uncapped features (`run_all.py:361–362`). No path bypasses it.
Full verification is Phase 3's job; this is a first-pass confirmation only.

### 6.6 Confirmed: `dpt_root_candidates` persistence is intact

`analysis/diffusion.py:195` writes `adata.uns["dpt_root_candidates"]` as `int64`.
`analysis/v2_comparison.py:112` reads it back. Present in the current tree.

---

## 7. Reference configuration — what `per_section_v2` actually ran

Resolved from `jobs/run_per_section_v2.sh` (vars at lines 77–83, invocation at 179):

```
--run
--png-dir          $SCRATCH/data/MCF7_x5_cropped
--annotation-dir   $HOME/cancer_trajectory_atlas/data/annotations_ratio
--output-dir       $SCRATCH/results/per_section_v2/atlas_<SECTION>
--stain-method     none
--batch-method     none
--model            phikon
--patch-size       112
--stride           96
--clustering-method leiden
--leiden-resolution 0.5
--n-roots          20
--n-permutations   1000
--features-cache-dir $SCRATCH/data/features_cache
--cap-strategy     median
--slides           <per-section CSV>
```

Everything not listed takes the argparse default — notably `diffmap-neighbors=30`,
`diffmap-comps=10`, `patch-sample-seed=42`, `min-roi-coverage=None`, `use-stardist=off`,
`target-total=3200`. This is the exact flag set Phase 8's regression script must
reproduce.

---

## 8. Open items — resolved by the maintainer, 2026-08-12

All four items originally flagged as undetermined are now settled.

1. **`$SCRATCH/data/annotations`** — **contains GeoJSON.** Confirmed. This makes
   `run_slide_diagnostics.sh` a live hazard for the `total_area` arm of check H5; see
   §5.2 for the full analysis and what needs re-reading on Narval.
2. **Whether the `data/annotations`-pointing scripts were ever run post-migration** —
   moot. `data/annotations` *held the ratio JSONs* when those scripts were written; the
   ratio files were later moved to `data/annotations_ratio/`. Git history confirms
   (§5.1). Those scripts were correct at the time; they are stale now. No past result
   is invalidated.
3. **`train_test_config.json` / `example_config.json`** — **not kept deliberately.**
   Leftovers from the deleted `run_train_test.py` workflow. Archive candidates for
   Phase 7. Nothing to be deleted at any point in this cleanup.
4. **`data/old_annotations/`** — **genuinely different content**, based on older
   un-updated annotations, not a duplicate of `data/annotations_ratio/`. Keep as
   historical record; do not archive, do not delete.

### 8.1 Still unverified — but nothing blocking

- The recorded **H5 verdict** from the existing `slide_diagnostics` run (§5.2). Needs a
  look at the Narval output directory. Does not block Phases 1–8; `slide_diagnostics`
  is an analysis branch, outside this cleanup's scope.
- Nothing else. **`converters/img_dims.txt` was checked and is consistent.**

### 8.2 Verified: the two dimension tables agree

`converters/img_dims.txt` (the divisor `batch_convert.py` uses to produce ratio
coordinates) and `data/slide_registry.py:KNOWN_NDPI_DIMENSIONS` (the multiplier
`run_all.py` uses to turn them back into pixels) are **the same 16 keys with identical
values, zero mismatches**. Checked programmatically.

This matters more than its size suggests: it is the round-trip invariant behind the
whole annotation path. `batch_convert` divides by `img_dims.txt`; `load_roi_polygons`
multiplies by `original_full_width`, which comes from `slide_dimensions.json` if present
and `KNOWN_NDPI_DIMENSIONS` otherwise. Because the two tables match, the round trip is
exact and the fallback path is equivalent to the sidecar path. Worth stating in Methods.

---

## 9. Recommended re-scoping before Phase 1

Given §4, three later phases are now mostly or entirely empty:

- **Phase 2** (coordinate docblock) — already done. Suggest converting it to a
  *verification* step: confirm the existing docblock matches the code, and stop.
- **Phase 6** (archive `validation/annotations.py` dead functions) — the file is gone.
  What remains is auditing `validation/correlations.py` and
  `analysis/cellularity_confound.py`, which the brief did not otherwise specify.
- **Phase 7** (archive `run_train_test.py`) — the file is gone. The job-script
  supersession half of Phase 7 is still fully live and is the larger piece.

Phase 1's stated task also dissolves (`paths.json` is already correct), but it gains two
real items in its place: the broken `submit_annotation_check.sh` reference (§3.1) and the
`data/annotations` vs `data/annotations_ratio` job-script hazard (§5). Both are
decisions for you, not behaviour-preserving edits.
