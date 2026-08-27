"""Shared NDPI dimension registry.

Level-0 pixel dimensions for all 16 MCF7 slides, as ``(width, height)``. Used as
a fallback when slide_dimensions.json, written by ``--convert``, is absent. Both
run_all.py and run_individual.py import from here so the values are never
duplicated.

These are openslide LEVEL-0 widths, covering both side-by-side copies of the
slide at full scanner resolution.

TRAP: the cropped PNG is a QUARTER of the width recorded here, not a half. The
cohort was converted with ``--ndpi-scale 0.5``, so 96000 becomes 48000, and the
left-half crop then halves it again to 24000. The invariant
``cropped_width == original_full_width // 2`` still holds, but
``original_full_width`` there means the SCALED full width (48000), not the
level-0 width in this table (96000).

That matters because ``_get_known_dimensions`` (``run_all.py:142``) multiplies
these values by ``cfg.ndpi_scale``. The fallback therefore reproduces the
recorded ``slide_dimensions.json`` ONLY when ``--ndpi-scale 0.5`` is passed. At
the config default of 1.0 it returns twice the correct width, every ratio
annotation is scaled by twice the right number, and every ROI lands off-canvas.

This is latent, not live: ``slide_dimensions.json`` exists in
``$SCRATCH/data/MCF7_x5_cropped`` and takes precedence, so the fallback has not
been exercised. It would bite on a fresh namespace where the sidecar is missing
and the scale is left at its default.

Hardcoding is a fallback, not the source of truth. slide_dimensions.json is
preferred whenever it exists, because it records what was actually converted. A
slide missing from this table and lacking that JSON cannot be processed, so
adding a slide to the cohort means adding it here too.

The keys are the 8 matched gland pairs: mouse 6027/6028/6029/6031, flank 4L/4R,
section 2M-1 (Carnoy's-fixed) or 2M-2 (PFA-fixed). Every gland appears exactly
once per section.
"""

KNOWN_NDPI_DIMENSIONS = {
    "6027-4L-2M-1": (96000, 42240),
    "6027-4L-2M-2": (94080, 45056),
    "6027-4R-2M-1": (86400, 38016),
    "6027-4R-2M-2": (94080, 45056),
    "6028-4L-2M-1": (96000, 49280),
    "6028-4L-2M-2": (86400, 40832),
    "6028-4R-2M-1": (80640, 35200),
    "6028-4R-2M-2": (86400, 38016),
    "6029-4L-2M-1": (78720, 30976),
    "6029-4L-2M-2": (74880, 32384),
    "6029-4R-2M-1": (71040, 35200),
    "6029-4R-2M-2": (76800, 32384),
    "6031-4L-2M-1": (82560, 46464),
    "6031-4L-2M-2": (94080, 46464),
    "6031-4R-2M-1": (94080, 38016),
    "6031-4R-2M-2": (78720, 35200),
}
