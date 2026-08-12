"""Stain normalization helpers for slide preprocessing.

Two behaviours here are load-bearing for interpreting any run's results, and
neither is obvious from the call site. Both are described precisely on the
functions below; summarised here because they belong in Methods:

1. The stain reference is **whichever slide sorts first**, not a chosen reference
   slide. Changing the slide subset can change the reference, and therefore
   changes the normalization applied to every other slide. See
   :func:`build_normalizer`.

2. Normalization failure is **silent and per-slide**. A slide whose transform
   raises is passed through un-normalized with only a stdout warning; nothing is
   persisted. See :func:`normalize_slide`.

Neither affects the current reference outputs (``per_section_v2``), which run
with ``--stain-method none`` — ``build_normalizer`` returns ``None`` and
``normalize_slide`` returns its input unchanged on the first line.
"""

import numpy as np
from pathlib import Path


# Reinhard normalizer

class ReinhardNormalizer:
    """
    Reinhard color normalization in LAB space.

    Transfers the mean and standard deviation of each LAB channel from
    a reference image to the target image.  Simple, fast, and doesn't
    require any problematic native libraries.

    Reference:
        Reinhard et al., "Color Transfer between Images", IEEE CG&A 2001.
    """

    def __init__(self):
        self.ref_means = None
        self.ref_stds = None

    def fit(self, reference_rgb: np.ndarray):
        """Compute LAB channel statistics from a reference image."""
        import cv2
        lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)

        # Only compute stats from tissue pixels.
        tissue_mask = lab[:, :, 0] < 230
        if tissue_mask.sum() < 1000:
            # Fall back to all pixels if tissue detection fails.
            tissue_mask = np.ones(lab.shape[:2], dtype=bool)

        self.ref_means = np.array([lab[:, :, c][tissue_mask].mean() for c in range(3)])
        self.ref_stds = np.array([lab[:, :, c][tissue_mask].std() + 1e-6 for c in range(3)])

    def transform(self, image_rgb: np.ndarray) -> np.ndarray:
        """Normalize an image to match the reference color distribution."""
        import cv2

        if self.ref_means is None:
            raise RuntimeError("Call fit() with a reference image first.")

        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)

        # Compute stats from tissue pixels of the source image.
        tissue_mask = lab[:, :, 0] < 230
        if tissue_mask.sum() < 1000:
            tissue_mask = np.ones(lab.shape[:2], dtype=bool)

        src_means = np.array([lab[:, :, c][tissue_mask].mean() for c in range(3)])
        src_stds = np.array([lab[:, :, c][tissue_mask].std() + 1e-6 for c in range(3)])

        # Normalize the source, then shift to the reference distribution.
        for c in range(3):
            lab[:, :, c] = (lab[:, :, c] - src_means[c]) / src_stds[c]
            lab[:, :, c] = lab[:, :, c] * self.ref_stds[c] + self.ref_means[c]

        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# Public API

def build_normalizer(method: str, reference_image_path: str):
    """Build a stain normalizer fitted to a reference slide.

    Reference slide selection
    -------------------------
    ``run_all.py`` passes ``slides[0]["image"]`` — the **first slide in sorted
    PNG-filename order**, after any ``--slides`` / ``--slides-from-file`` filter
    has been applied. There is no designated reference slide.

    The consequence is that **running a subset can change the reference**, and
    with it the normalization applied to every slide in that run. Two runs over
    different subsets are therefore not directly comparable under ``reinhard``
    or ``macenko``. ``run_all.py`` copies the chosen reference to
    ``<output_dir>/stain_reference.png`` so it can be recovered after the fact.

    Methods
    -------
    ``"none"`` returns ``None`` (no normalization, the current default for all
    reference runs). ``"reinhard"`` uses the in-file :class:`ReinhardNormalizer`.
    ``"macenko"`` and ``"vahadane"`` both route to ``staintools``; note that
    ``vahadane`` is reachable only by calling this function directly, since
    ``run_all.py`` restricts ``--stain-method`` to reinhard/macenko/none.

    Raises ``ValueError`` on an unrecognised method and ``ImportError`` if
    ``staintools`` is missing for the methods that need it.
    """
    method = (method or "none").lower()
    if method == "none":
        return None

    print(f"  Building {method.title()} normalizer from: {reference_image_path}")

    if method == "reinhard":
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        ref_img = np.array(Image.open(reference_image_path).convert("RGB"))
        normalizer = ReinhardNormalizer()
        normalizer.fit(ref_img)
        print("  Normalizer ready.")
        return normalizer

    # Macenko and Vahadane use staintools.
    if method in ("macenko", "vahadane"):
        try:
            import staintools
        except ImportError as exc:
            raise ImportError(
                "staintools is required for Macenko/Vahadane normalization.  "
                "pip install staintools\n"
                "If spams won't install, use method='reinhard' instead."
            ) from exc

        target = staintools.read_image(str(reference_image_path))
        normalizer = staintools.StainNormalizer(method=method)
        normalizer.fit(target)
        print("  Normalizer ready.")
        return normalizer

    raise ValueError(f"Unsupported stain normalization method: {method}")


def normalize_slide(image_array: np.ndarray, normalizer, slide_name: str = "") -> np.ndarray:
    """Apply stain normalization to one slide image array.

    Failure semantics — read before trusting a normalized run
    --------------------------------------------------------
    Any exception raised by ``normalizer.transform`` is caught and the **original,
    un-normalized array is returned**. A warning goes to stdout and nothing else
    is recorded: no sentinel, no counter, no entry in ``feature_failures.json``.

    A run can therefore be silently half-normalized, and after the fact the only
    evidence is the SLURM log. This is deliberately *unlike* the feature-extraction
    path, which encodes failure as NaN and persists a diagnostic. If you are
    interpreting a ``reinhard`` or ``macenko`` run, grep its log for
    "Stain normalization failed" before drawing conclusions.

    ``normalizer is None`` (i.e. ``--stain-method none``) returns the input
    immediately and is not a failure path.
    """
    if normalizer is None:
        return image_array

    try:
        normalized = np.asarray(normalizer.transform(image_array))
        if normalized.dtype != np.uint8:
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        print(f"    Applied stain normalization to {slide_name}")
        return normalized
    except Exception as exc:
        print(f"    WARNING: Stain normalization failed for {slide_name}: {exc}")
        return image_array