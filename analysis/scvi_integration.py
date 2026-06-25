"""
scVI batch correction for phikon PCA features (continuous-data variant).

Alternative to Harmony (see analysis/harmony.py) for removing slide-level /
section-level batch effects from the PCA embedding before clustering and DPT.
Unlike Harmony, scVI learns a nonlinear latent space via a VAE.

Input is the SAME post-PCA matrix Harmony receives (continuous, already
standardized+PCA'd Phikon features) — NOT a counts matrix. scVI is
configured with gene_likelihood="normal" for this reason.

use_observed_lib_size=False: scVI's default architecture estimates a
per-cell library size as log(sum of X), which assumes non-negative
count-like data. PCA components are mean-centered and routinely negative,
so the default would feed log() a negative/near-zero value and produce
NaNs. Setting use_observed_lib_size=False makes the model learn library
size as a latent variable instead of computing it from the (here,
meaningless) row sum — the standard fix for applying SCVI to continuous,
non-count inputs.

Batch key: section_number only ("2M-1" vs "2M-2") — this backend exists
specifically to test whether a more expressive nonlinear correction closes
the residual batch-mixing gap Harmony leaves (see PROJECT_STATE.md Working
Log, 2026-06-24: Harmony k-NN batch purity 0.9425 vs 0.502 chance baseline).
"""

import numpy as np

from .harmony import _batch_labels


def apply_scvi(
    X_pca: np.ndarray,
    slide_names: list,
    slide_ids: np.ndarray,
    key: str = "section_number",
    n_latent: int = 30,
    n_layers: int = 2,
    n_hidden: int = 128,
    max_epochs: int = 400,
):
    """
    Apply scVI batch correction to PCA features.

    Args:
        X_pca:       (N, k) raw PCA features from fit_pca().
        slide_names: List of slide name strings, one entry per slide,
                     indexed by the integer values in slide_ids.
        slide_ids:   (N,) int array mapping each patch to its slide index.
        key:         Batch grouping key (always "section_number" for scVI).
        n_latent:    Dimensionality of the scVI latent space.
        n_layers:    Number of hidden layers in the encoder/decoder.
        n_hidden:    Number of nodes per hidden layer.
        max_epochs:  Maximum training epochs.

    Returns:
        (X_scvi, model): (N, n_latent) corrected embedding and the trained
        scvi.model.SCVI instance (for saving to disk by the caller).
    """
    try:
        import anndata as ad
        import scvi
    except ImportError as e:
        raise ImportError(
            "scvi-tools and anndata are required for scVI batch correction. "
            "pip install scvi-tools"
        ) from e

    batch = _batch_labels(slide_names, slide_ids, key)
    unique_batches, batch_counts = np.unique(batch, return_counts=True)

    print(f"  scVI key='{key}': {len(unique_batches)} batches")
    for b, c in sorted(zip(unique_batches.tolist(), batch_counts.tolist())):
        print(f"    {b}: {c} patches")

    adata_tmp = ad.AnnData(X=X_pca.astype(np.float32))
    adata_tmp.obs["batch"] = batch

    scvi.model.SCVI.setup_anndata(adata_tmp, batch_key="batch")

    print(
        f"  Training scVI (n_latent={n_latent}, n_layers={n_layers}, "
        f"n_hidden={n_hidden}, max_epochs={max_epochs}, "
        f"gene_likelihood=normal, use_observed_lib_size=False)..."
    )
    model = scvi.model.SCVI(
        adata_tmp,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        gene_likelihood="normal",
        use_observed_lib_size=False,
    )
    model.train(max_epochs=max_epochs)

    X_scvi = model.get_latent_representation()
    print(f"  Done. Output shape: {X_scvi.shape}")

    return X_scvi.astype(X_pca.dtype), model
