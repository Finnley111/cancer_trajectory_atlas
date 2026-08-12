"""Validation suite: morphological features and their correlation with pseudotime.

Contents
--------
``morphological_features.py``
    Computes the seven interpretable descriptors from raw patch pixels
    (nuclear density, mean nuclear area, NC ratio, texture entropy, hematoxylin
    intensity masked + whole-patch, packing irregularity).
``correlations.py``
    Spearman correlations, permutation tests, cluster ordering, and the headline
    verdict.

ANNOTATION LOADING IS NOT IN THIS PACKAGE
-----------------------------------------
There used to be a ``validation/annotations.py`` holding legacy mask-loading
helpers. It was deleted in commit ``0fa4880`` and is not coming back; its only
consumer was the top-level ``run_train_test.py``, itself since removed.

**The single active annotation path is**
``features/patching.py:load_roi_polygons`` — it reads the ratio-coordinate JSON
in ``data/annotations_ratio/``, splits polygons into inclusion/exclusion sets,
and maps them into cropped-PNG pixel space. Patch-level ROI filtering then
happens in ``features/patching.py:get_patches_from_array``.

If you are looking for "where do annotations enter the pipeline", it is there,
not here. Nothing in ``validation/`` reads an annotation file; this package only
ever sees patch arrays that ROI filtering has already selected.
"""
