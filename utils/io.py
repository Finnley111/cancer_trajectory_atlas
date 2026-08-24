"""Persistence helpers for atlas artifacts.

What is actually used, as of 2026-08-24:

  ``save_json``    the only widely-used function here. run_all.py writes
                   validation.json, feature_failures.json, slide_independence.json
                   and holeyness_roots.json through it, and three analysis modules
                   import it directly.
  ``save_pickle``  run_all.py only, for scaler.pkl / pca.pkl / umap_reducer.pkl.
  ``load_pickle``  no callers.
  ``load_json``    no callers. Analysis modules that read JSON define their own
                   loader with a file-specific error message rather than using it.

``save_atlas_artifacts`` has no callers either, and describes an output layout
(``reference_atlas/``) that no current run produces. See its docstring.

Read ``save_json`` before adding a caller. Its NaN handling is deliberate and is
the reason a feature that is legitimately missing survives a round trip.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict


def save_pickle(obj: Any, path: str):
    """Pickle ``obj`` to ``path``, overwriting it.

    Used for fitted sklearn objects (StandardScaler, PCA) and the UMAP reducer,
    none of which have a stable non-pickle serialization.

    Assumes the parent directory exists; it does not create one. Unpickling
    requires the same library versions that wrote the file, so these artifacts
    are reproducible only within one environment. Nothing on the ``--run`` path
    reads them back, so a version skew shows up when projecting, not when
    running.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Return the object pickled at ``path``.

    No current caller. Kept as the counterpart to ``save_pickle`` for anyone
    loading a saved scaler or PCA.

    Unpickling executes code from the file, so use it only on artifacts this
    pipeline wrote.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: Dict, path: str):
    """Save dict to JSON, converting numpy types automatically.

    Non-finite floats become null. json.dump defaults to allow_nan=True, which
    emits bare NaN / Infinity tokens: Python reads those back happily, but they
    are NOT valid JSON, so jq, JavaScript and every strict parser reject the
    file. Since features can now legitimately be nan, that would silently produce
    unreadable artifacts. null round-trips to None, which is explicit and valid.
    """
    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return v if np.isfinite(v) else None
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return _convert(obj.tolist())
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, np.integer) else k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        # allow_nan=False so any non-finite that slips past _convert raises
        # loudly instead of writing an invalid file.
        json.dump(_convert(data), f, indent=2, allow_nan=False)


def load_json(path: str) -> Dict:
    """Return the parsed contents of the JSON file at ``path``.

    No current caller. Analysis modules that read JSON define a local loader that
    names which file failed and why it was wanted, which produces a far better
    message than a bare FileNotFoundError when a job script points at the wrong
    run directory.

    Round-trips ``save_json`` output: the nulls that function writes for
    non-finite floats come back as None.
    """
    with open(path) as f:
        return json.load(f)


def save_atlas_artifacts(
    output_dir: str,
    scaler,
    pca,
    umap_reducer,
    adata_train,
    cluster_centroids: Dict,
    metadata: Dict,
    stain_reference_path: str = None,
):
    """Write a complete reference-atlas bundle to ``output_dir``.

    NO CALLERS. Nothing in the repo invokes this, and no recorded run produced
    the layout below. ``run_all.py`` writes the same pickles directly into its
    own output directory instead, without the h5ad, the centroids or the
    metadata. Treat this as a design that was not adopted rather than as a
    description of any run directory you will find on disk.

    The "pipeline document Section 9.1" the previous docstring cited does not
    exist in this repo. The layout it describes:

        reference_atlas/
            scaler.pkl
            pca.pkl
            umap_reducer.pkl
            cluster_centroids.npy
            adata_train.h5ad
            stain_reference.png
            metadata.json

    Creates ``output_dir`` including parents. ``adata_train`` and
    ``stain_reference_path`` are optional and skipped when None; the rest are
    required. ``cluster_centroids`` maps a cluster id to a sequence whose first
    element is the centroid vector, and the saved matrix is ordered by sorted
    cluster id, so the row order is recoverable only by re-sorting those keys.
    Nothing records the id order alongside the matrix.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_pickle(scaler, out / "scaler.pkl")
    save_pickle(pca, out / "pca.pkl")
    save_pickle(umap_reducer, out / "umap_reducer.pkl")

    # Sorting the ids fixes the row order. The .npy holds no labels, so this sort
    # is the only thing that makes row i recoverable as a cluster.
    centroid_ids = sorted(cluster_centroids.keys())
    centroid_matrix = np.array([cluster_centroids[c][0] for c in centroid_ids])
    np.save(out / "cluster_centroids.npy", centroid_matrix)

    if adata_train is not None:
        adata_train.write(out / "adata_train.h5ad")

    if stain_reference_path:
        # Imported here rather than at module scope: this is the only use, and
        # the function is unreachable in practice.
        import shutil
        shutil.copy2(stain_reference_path, out / "stain_reference.png")

    save_json(metadata, out / "metadata.json")

    print(f"  Atlas artifacts saved to: {out}")
