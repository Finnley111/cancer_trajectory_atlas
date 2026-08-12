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
