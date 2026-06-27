"""Compare per-section Phase-6 morphology correlations across 2M-1 and 2M-2.

Read-only: loads validation.json from two per-section run directories and writes
cross_section_comparison.csv recording rho, p-value, and replication status for
each morphological feature.  A feature "replicates" if both sections show the same
sign and |rho| >= REPLICATE_RHO_THRESHOLD.

Usage:
    python -m cancer_trajectory_atlas.analysis.cross_section_compare \\
        --run-dir-2m1  $SCRATCH/results/per_section/atlas_2M-1 \\
        --run-dir-2m2  $SCRATCH/results/per_section/atlas_2M-2 \\
        --output-dir   $SCRATCH/results/per_section
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPLICATE_RHO_THRESHOLD = 0.1


def _load_feature_correlations(run_dir: Path) -> dict:
    vpath = run_dir / "validation.json"
    if not vpath.exists():
        print(f"ERROR: validation.json not found in {run_dir}")
        sys.exit(1)
    with open(vpath) as f:
        v = json.load(f)
    if "feature_correlations" not in v:
        print(f"ERROR: 'feature_correlations' key missing in {vpath}")
        sys.exit(1)
    return v["feature_correlations"]


def main():
    parser = argparse.ArgumentParser(
        description="Cross-section morphology replication check (read-only)."
    )
    parser.add_argument("--run-dir-2m1", type=Path, required=True,
                        help="Per-section run dir for 2M-1 (contains validation.json)")
    parser.add_argument("--run-dir-2m2", type=Path, required=True,
                        help="Per-section run dir for 2M-2 (contains validation.json)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination for cross_section_comparison.csv")
    parser.add_argument("--rho-threshold", type=float, default=REPLICATE_RHO_THRESHOLD,
                        help=f"Minimum |rho| in both sections to call replication "
                             f"(default: {REPLICATE_RHO_THRESHOLD})")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresh = args.rho_threshold

    corr_1 = _load_feature_correlations(args.run_dir_2m1)
    corr_2 = _load_feature_correlations(args.run_dir_2m2)

    features = sorted(set(corr_1) | set(corr_2))
    rows = []
    for feat in features:
        c1 = corr_1.get(feat, {})
        c2 = corr_2.get(feat, {})
        rho1 = c1.get("rho", float("nan"))
        p1   = c1.get("p_value", float("nan"))
        rho2 = c2.get("rho", float("nan"))
        p2   = c2.get("p_value", float("nan"))

        if math.isnan(rho1) or math.isnan(rho2):
            same_sign  = False
            replicated = False
        else:
            same_sign  = (rho1 * rho2) > 0
            replicated = same_sign and (abs(rho1) >= thresh) and (abs(rho2) >= thresh)

        rows.append({
            "feature":    feat,
            "rho_2m1":   rho1,
            "p_2m1":     p1,
            "rho_2m2":   rho2,
            "p_2m2":     p2,
            "same_sign":  same_sign,
            "replicated": replicated,
        })

    df = pd.DataFrame(rows)
    out_csv = args.output_dir / "cross_section_comparison.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Saved: {out_csv}")
    print()

    print(f"  {'Feature':<28s}  {'rho 2M-1':>8s}  {'rho 2M-2':>8s}  Status")
    print("  " + "-" * 60)
    for _, r in df.iterrows():
        if r["replicated"]:
            status = "REPLICATED"
        elif r["same_sign"]:
            status = "same sign (weak)"
        else:
            status = "---"
        print(f"  {r['feature']:<28s}  {r['rho_2m1']:>+8.3f}  {r['rho_2m2']:>+8.3f}  {status}")

    print()
    n_rep = int(df["replicated"].sum())
    n_ss  = int(df["same_sign"].sum())
    print(f"Replicating (same sign, |rho| >= {thresh} in both): {n_rep}/{len(df)}")
    print(f"Same sign (any magnitude):                           {n_ss}/{len(df)}")


if __name__ == "__main__":
    main()
