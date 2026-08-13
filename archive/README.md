# Archive

Code that is no longer reachable, no longer runnable, or superseded by a later version.
Nothing here is deleted — it is kept so the history stays inspectable and so anything
archived by mistake can be restored.

**Relative structure is preserved.** A file archived from `jobs/` lives in
`archive/jobs/`, so its original location is always recoverable from its path here.

Nothing in this directory is on the pipeline's execution path. No job script, module, or
entry point in the live tree imports or invokes anything here. If something here turns
out to still be needed, move it back rather than copying it.

---

## Contents

### `jobs/submit_annotation_check.sh` — archived 2026-08-12, Phase 1

Two-step SLURM script: convert QuPath GeoJSON annotations to ratio coordinates, then
draw polygon overlays on slide thumbnails for visual QC.

**Why archived:** Step 1 (line 37) invokes
`python -m cancer_trajectory_atlas.converters.geojson_to_ratio_json`, a module that does
not exist in the repo and has never existed under that name in git history. The script
therefore fails at its first command under `set -euo pipefail` and cannot complete. The
conversion work it was meant to do is now done by `converters/batch_convert.py`, which
takes no arguments and hardcodes its paths, so the call site could not simply be renamed.

**What was still working, and where it went:** Step 2 (line 47) called
`jobs/check_annotations.py` to render the overlay thumbnails, and this script was its
only caller. That step was *not* broken. `jobs/check_annotations.py` remains in the live
tree, and its invocation is now documented in `NOTES.md` → *Annotation directory* →
*Visual QC of annotations*, so the recipe is not lost with this script.

**Dangling reference cleaned up:** `jobs/submit_new_annotations_all.sh` listed this
script as a prerequisite ("run submit_annotation_check.sh first"). That comment now
points at `converters/batch_convert.py` instead.

### `jobs/` first-generation atlas run scripts — archived 2026-08-12, Phase 7

Eight SLURM scripts, all added 2026-05-13 to 2026-05-28, superseded by the per-section
workflow (`jobs/run_per_section.sh`, `jobs/run_per_section_v2.sh`).

**Six of them cannot run correctly today.** They pass
`--annotation-dir ~/cancer_trajectory_atlas/data/annotations`, which held the
ratio-coordinate JSON when they were written. Commit `f050e4a` replaced that directory's
contents with QuPath GeoJSON and moved the ratio files to `data/annotations_ratio/`.
`load_roi_polygons` is always called with `coordinate_space="ratio"`, so feeding it
absolute-pixel GeoJSON multiplies every coordinate by `original_full_width` a second
time and puts every ROI off-canvas — few or no in-ROI patches would survive.

**Results produced by these scripts at the time are NOT suspect.** They were correct
when submitted; only the directory beneath them changed. See `NOTES.md` →
*Annotation directory* → *History*.

| Script | Why archived |
|---|---|
| `run_all_none.sh` | First-generation full-atlas run, one script per stain method, no Harmony, no patch cap. Stale annotation dir. |
| `run_all_macenko.sh` | Same generation, Macenko variant. Stale annotation dir. |
| `run_all_reinhard.sh` | Same generation, Reinhard variant. Stale annotation dir, and Reinhard was abandoned — `NOTES.md` records no-norm as the default. |
| `run_all_none_section.sh` | Single-section runs via `--slides-from-file`. Superseded by `run_per_section.sh` (2026-06-27), which runs both sections with within-section LOO and a cross-section replication check. Stale annotation dir. |
| `submit_harmony.sh` | Parameterised stain × harmony-key runner. Superseded by `run_cache_population.sh` / `run_full_experiments.sh`. Stale annotation dir. |
| `submit_harmony_none.sh` | Hardcoded specialisation of `submit_harmony.sh`. Stale annotation dir. |
| `submit_harmony_macenko.sh` | Hardcoded specialisation of `submit_harmony.sh`. Stale annotation dir. |
| `run_all_capped.sh` | **Actively misleading.** Passes `--max-patches-per-slide` (default 1900) but never `--cap-strategy fixed`, so `cap_strategy` fell through to `median` and the cap value was inert. Any output it produced was median-capped, not 1900-capped. |

**Kept in `jobs/` despite looking similar**, because both are load-bearing:
`run_cache_population.sh` is named as the remedy in `run_all.py`'s cache-mismatch
`RuntimeError` and in `run_per_section_v2.sh`'s cache gate; `run_per_section.sh` is the
baseline that `run_per_section_v2.sh` reproduces and is referenced by four live scripts.

**Dangling reference cleaned up:** `jobs/submit_qc.sh` said "Match the env setup from
`run_all_macenko.sh`"; the module list is now inlined there instead.

### `analysis/diffusion_run_diffusion_pseudotime.py` — archived 2026-08-12, Phase 5

The `run_diffusion_pseudotime` function, extracted verbatim from
`analysis/diffusion.py`. A single-root convenience wrapper chaining
`build_adata` → `compute_diffusion_map` → `compute_dpt`.

**Why archived:** dead. Its only caller was the top-level `run_train_test.py`,
deleted before this cleanup began — the sole surviving reference anywhere in the repo
was a stale `__pycache__/run_train_test.cpython-310.pyc` still holding the symbol name.
Zero callers in any `.py`, `.sh`, or `.md`.

**Superseded by:** `compute_dpt_multi_root`, which the atlas pipeline uses — 20
density-ranked roots, median-aggregated, rather than one cluster-anchored root.

**Not importable as-is.** It is a verbatim record so the removal stays reversible; to
restore it, paste the function back into `analysis/diffusion.py`. The three functions
it calls are all still there. Removing it changed no behaviour: `analysis/diffusion.py`
is otherwise byte-identical, verified by AST comparison.
