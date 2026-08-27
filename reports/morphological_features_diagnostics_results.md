# Morphological Feature Diagnostics Results

> **STATUS 2026-08-26: never populated.** This file is still the empty placeholder it
> started as. The job below has not been run, so none of the D1-D4 / FC diagnostics in
> `reports/morphological_features_audit.md` were ever measured against real data.
>
> Note the input paths below point at `per_section` (**v1**). The current reference tree
> is `per_section_v2`, which is what the Task 1 fixes produced. Point the job there
> instead, or the diagnostics will describe the superseded run.
>
> **This file is generated at runtime** by `diagnostics/audit_feature_diagnostics.py`.  
> Run `sbatch jobs/run_feature_diagnostics.sh` on Narval to populate it with real values from the per-section results.csv files.
>
> Input files (on Narval):
> - `$SCRATCH/results/per_section/atlas_2M-1/results.csv`
> - `$SCRATCH/results/per_section/atlas_2M-2/results.csv`
>
> See `reports/morphological_features_audit.md` for the audit findings this script validates (D1–D4, FC).
