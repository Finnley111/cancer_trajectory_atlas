"""
I/O Utilities — Save / Load Atlas Artifacts

Handles persistence of the reference atlas components:
  - scaler.pkl, pca.pkl, umap_reducer.pkl
  - adata_train.h5ad
  - cluster_centroids.npy
  - stain_reference.png
  - metadata.json
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict


def save_pickle(obj: Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
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
    """
    Save all reference atlas files to disk.

    Output structure matches pipeline document Section 9.1:
        reference_atlas/
            scaler.pkl
            pca.pkl
            umap_reducer.pkl
            cluster_centroids.npy
            adata_train.h5ad
            stain_reference.png
            metadata.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_pickle(scaler, out / "scaler.pkl")
    save_pickle(pca, out / "pca.pkl")
    save_pickle(umap_reducer, out / "umap_reducer.pkl")

    # Cluster centroids as numpy array
    centroid_ids = sorted(cluster_centroids.keys())
    centroid_matrix = np.array([cluster_centroids[c][0] for c in centroid_ids])
    np.save(out / "cluster_centroids.npy", centroid_matrix)

    # AnnData
    if adata_train is not None:
        adata_train.write(out / "adata_train.h5ad")

    # Copy stain reference
    if stain_reference_path:
        import shutil
        shutil.copy2(stain_reference_path, out / "stain_reference.png")

    # Metadata
    save_json(metadata, out / "metadata.json")

    print(f"  Atlas artifacts saved to: {out}")
