"""Harmony batch correction for Phikon PCA features.

Wraps ``scanpy.external.pp.harmony_integrate`` (itself a wrapper over harmonypy)
to remove slide-level batch effects from the PCA embedding before clustering and
DPT.

NOT ON THE PRODUCTION PATH. Every recorded run passes ``--batch-method none``, so
nothing here executed in the reference outputs. Batch correction removes variance
attributable to the grouping key, and when the biological contrast of interest is
confounded with that key it removes the signal along with the artifact. The two
sections differ by fixation, which is exactly such a confound.

Supported batch keys:
  section_number  "2M-1" vs "2M-2", 2 groups. Appropriate when the section split
                  dominates, as diagnosed by QC pseudotime violins showing PT
                  around 0.8 for every 2M-1 slide and around 0.05 for every 2M-2
                  slide.
  slide_id        one batch per slide, 16 groups. The most aggressive option, and
                  the one most likely to remove real between-slide biology.
  mouse_id        one batch per mouse, 4 groups (6027/6028/6029/6031).

Expected slide name format: "6027-4L-2M-1_x5"
  mouse_id:       "6027", the first hyphen-separated token with _x5 stripped
  section_number: "2M-1", the last two tokens with _x5 stripped

That parsing is positional and unvalidated. A slide named outside this convention
produces a wrong batch label rather than an error. See ``_batch_labels``.
"""

import numpy as np


def _batch_labels(slide_names: list, slide_ids: np.ndarray, key: str) -> np.ndarray:
    """Expand per-slide identity into a per-patch batch label array.

    Returns an (N,) array of strings aligned with ``slide_ids``, so it is
    per-PATCH, not per-slide. Harmony needs one label per row of the embedding.

    Assumes ``slide_ids`` holds valid indices into ``slide_names``. An
    out-of-range id raises IndexError rather than producing a wrong label.

    Raises ValueError on an unrecognised key rather than defaulting.

    TRAP: the name parsing below is positional and validates nothing. A slide
    whose name has fewer than two hyphen-separated tokens yields a nonsense
    section label, and one with extra tokens silently takes the wrong two. Both
    produce plausible-looking batches that correct against the wrong grouping.
    """

    def _mouse(name):
        # "6027-4L-2M-1_x5" -> "6027"
        return name.replace("_x5", "").split("-")[0]

    def _section(name):
        # ["6027", "4L", "2M", "1"] -> "2M-1", taken from the END so the
        # mouse and flank tokens do not shift the result.
        parts = name.replace("_x5", "").split("-")
        return f"{parts[-2]}-{parts[-1]}"

    if key == "slide_id":
        return slide_ids.astype(str)
    elif key == "mouse_id":
        return np.array([_mouse(slide_names[sid]) for sid in slide_ids])
    elif key == "section_number":
        return np.array([_section(slide_names[sid]) for sid in slide_ids])
    else:
        raise ValueError(
            f"Unknown harmony key: '{key}'. "
            f"Choose from: slide_id, section_number, mouse_id"
        )


def apply_harmony(
    X_pca: np.ndarray,
    slide_names: list,
    slide_ids: np.ndarray,
    key: str = "section_number",
    nclust: int = 10,
) -> np.ndarray:
    """Apply Harmony batch correction to PCA features.

    Removes variance attributable to ``key`` so clustering and DPT reflect
    morphology rather than slide of origin. That is also the risk: whatever is
    confounded with the key goes with it.

    Args:
        X_pca: (N, k) raw PCA features from ``fit_pca()``.
        slide_names: slide name strings, one per slide, indexed by the integers
            in ``slide_ids``.
        slide_ids: (N,) int array mapping each patch to its slide index.
        key: batch grouping, one of "section_number", "slide_id", "mouse_id".
        nclust: internal K-means cluster count for Harmony. 10 is a deliberate
            departure from harmonypy's default of 100, which is
            over-parameterized for 2 to 4 batches and converges immediately.
            Not exposed on the CLI.

    Returns:
        (N, k) corrected embedding, same shape and dtype as ``X_pca``. Row order
        is preserved, so it stays aligned with everything else keyed on patch
        index.

    Raises ImportError if scanpy, anndata or harmonypy is missing, and ValueError
    if the backend returns a differently shaped array. That shape check is not
    defensive padding: harmonypy 0.2.0's PyTorch backend really does return a 1-D
    array on fast convergence, and without the check it would propagate as a
    silently wrong embedding.
    """
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            "scanpy and anndata are required for Harmony correction. "
            "pip install scanpy anndata"
        ) from e

    try:
        # Imported only to fail early with a useful message. scanpy's own error
        # for a missing harmonypy arrives mid-run and names its internals.
        import harmonypy  # noqa: F401
    except ImportError:
        raise ImportError(
            "harmonypy is required for Harmony batch correction.\n"
            "  pip install harmonypy\n"
            "Then re-run with --harmony."
        )

    batch = _batch_labels(slide_names, slide_ids, key)
    unique_batches, batch_counts = np.unique(batch, return_counts=True)

    print(f"  Harmony key='{key}': {len(unique_batches)} batches")
    for b, c in sorted(zip(unique_batches.tolist(), batch_counts.tolist())):
        print(f"    {b}: {c} patches")

    # AnnData exists here only as a carrier: harmony_integrate has no array API
    # and insists on reading obsm["X_pca"] and writing obsm["X_pca_harmony"].
    # X is set as well as obsm because AnnData requires it to infer n_obs.
    adata_tmp = ad.AnnData(X=X_pca.astype(np.float32))
    adata_tmp.obsm["X_pca"] = X_pca.astype(np.float32)
    adata_tmp.obs["batch"] = batch

    print(f"  Running harmony_integrate (nclust={nclust})...")
    # PIN: harmonypy 0.0.9, which is pure numpy, so no GPU path and no
    # shape-squeezing bug. 0.2.0 was deliberately downgraded: its PyTorch backend
    # returns Z_corr as a 1-D array when convergence takes 2 iterations or fewer,
    # on both CPU and CUDA. Few batches converge that fast, which is exactly this
    # cohort's situation. The shape check after this call catches it if the pin
    # is ever lost.
    sc.external.pp.harmony_integrate(
        adata_tmp, key="batch", nclust=nclust,
    )

    X_corrected = adata_tmp.obsm["X_pca_harmony"]
    print(f"  Done. Output shape: {X_corrected.shape}")

    if X_corrected.shape != X_pca.shape:
        raise ValueError(
            f"Harmony output shape {X_corrected.shape} != input shape {X_pca.shape}. "
            f"Backend returned a squeezed array — check harmonypy version."
        )

    return X_corrected.astype(X_pca.dtype)
