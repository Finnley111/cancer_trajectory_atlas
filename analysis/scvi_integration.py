"""scVI batch correction for Phikon PCA features, continuous-data variant.

Alternative to Harmony (analysis/harmony.py) for removing slide-level or
section-level batch effects from the PCA embedding before clustering and DPT.
Where Harmony fits a linear correction, scVI learns a nonlinear latent space
through a VAE.

NOT ON THE PRODUCTION PATH. Every recorded run passes ``--batch-method none``.

Also note the OUTPUT DIMENSION DIFFERS from Harmony's. Harmony returns the input
shape; this returns (N, n_latent), 30 by default. Anything downstream that
assumes the batch-correction step is shape-preserving is wrong for this backend.

Input is the SAME post-PCA matrix Harmony receives: continuous, already
standardized and PCA'd Phikon activations, NOT a counts matrix. Everything
awkward below follows from that one mismatch. scVI is written for
non-negative integer counts, and this feeds it neither.

Three settings depart from scVI's defaults, all for the same reason. Each was
required to make training run at all, not tuned for quality:

- gene_likelihood="normal": the default is a negative binomial, a distribution
  over non-negative integers. PCA components are real-valued and signed, so the
  default likelihood cannot represent them.

- use_observed_lib_size=False: the default estimates a per-cell library
  size as log(sum of X), which assumes non-negative count-like data. PCA
  components are mean-centered and routinely negative, so the default
  would feed log() a negative/near-zero value. This makes the model learn
  library size as a latent variable instead of computing it from the
  (here, meaningless) row sum.
- log_variational=False: the encoder applies log(1+x) to its input by
  default, again assuming non-negative counts. For x <= -1 (common in
  standardized PCA output) this produces NaN/-inf in the very first
  encoder forward pass, before any training even happens. Disabling it
  feeds the encoder the raw continuous values directly.

Batch key: section_number only ("2M-1" vs "2M-2"). This backend exists to test
whether a more expressive nonlinear correction closes the residual batch-mixing
gap Harmony leaves (PROJECT_STATE.md Working Log, 2026-06-24: Harmony k-NN batch
purity 0.9425 against a 0.502 chance baseline).

Read the result cautiously. A VAE with enough capacity can mix the batches by
discarding the variation that distinguishes them, and improved mixing is
therefore not by itself evidence that biology survived.
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
    """Apply scVI batch correction to PCA features.

    Args:
        X_pca: (N, k) raw PCA features from ``fit_pca()``.
        slide_names: slide name strings, one per slide, indexed by the integers
            in ``slide_ids``.
        slide_ids: (N,) int array mapping each patch to its slide index.
        key: batch grouping key, always "section_number" for this backend.
        n_latent: latent dimensionality, and the WIDTH OF THE RETURNED ARRAY.
        n_layers: hidden layers in the encoder and decoder.
        n_hidden: nodes per hidden layer.
        max_epochs: training epochs. scVI may stop earlier on its own
            early-stopping criterion, so this is a ceiling.

    Returns:
        ``(X_scvi, model)``. The embedding is (N, n_latent), NOT (N, k), and row
        order matches the input. The trained ``scvi.model.SCVI`` is returned so
        the caller can persist it; this function saves nothing.

    NOT DETERMINISTIC unless the caller seeds scVI beforehand. Training involves
    random initialisation and stochastic minibatching, so two runs on identical
    input give different embeddings. That is the main reason this backend is
    unsuitable for the bit-identical regression comparison the Harmony and
    no-correction paths support.

    Raises ImportError when scvi-tools or anndata is missing.
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
        f"gene_likelihood=normal, use_observed_lib_size=False, "
        f"log_variational=False)..."
    )
    model = scvi.model.SCVI(
        adata_tmp,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        gene_likelihood="normal",
        use_observed_lib_size=False,
        # Left at False for the reason given in the module docstring: the
        # encoder's default log(1+x) transform assumes non-negative counts, and
        # standardized PCA output routinely goes below -1, which produces NaN in
        # the first forward pass before any training happens.
        log_variational=False,
    )
    model.train(max_epochs=max_epochs)

    X_scvi = model.get_latent_representation()
    print(f"  Done. Output shape: {X_scvi.shape}")

    return X_scvi.astype(X_pca.dtype), model
