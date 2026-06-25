"""CLI runner for the batch-mixing diagnostic.

Loads an existing run's adata_full.h5ad and writes batch_mixing.json to the
same run directory. Pure read of saved data — no re-embedding, no Harmony
rerun, no clustering/DPT, no feature rebuild.

Usage (run from the repo's parent directory, e.g. ~ on Narval):
    python -m cancer_trajectory_atlas.analysis.run_batch_mixing <run_dir>

<run_dir> must contain adata_full.h5ad.
"""

import sys
from pathlib import Path

from .batch_mixing import compute_batch_mixing_report
from ..utils.io import save_json


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_batch_mixing.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    h5ad_path = run_dir / "adata_full.h5ad"
    if not h5ad_path.exists():
        print(f"ERROR: {h5ad_path} does not exist.")
        sys.exit(1)

    report = compute_batch_mixing_report(h5ad_path)

    out_path = run_dir / "batch_mixing.json"
    save_json(report, out_path)

    harmony_str = f"{report['harmony']:.3f}" if report["harmony"] is not None else "N/A"
    print(
        f"Batch mixing (k={report['k_used']}, n={report['n_patches']}): "
        f"raw_pca={report['raw_pca']:.3f} harmony={harmony_str} "
        f"chance_baseline={report['chance_baseline']:.3f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
