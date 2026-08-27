# Morphological Features Audit
**File audited:** `validation/morphological_features.py`  
**Date:** 2026-07-24  
**Author:** code review (no pipeline changes made)  
**Purpose:** Assess methodological soundness of the six features that underpin every quantitative claim currently in the manuscript draft. No source files were modified.

---

## STATUS AS OF 2026-08-26 — read this before acting on anything below

**This audit describes the code as it stood on 2026-07-24, BEFORE the Task 1
fixes. Four of its eight ranked findings have since been fixed.** The text below is
preserved verbatim as the record of what was found and why; it is NOT a description
of the current pipeline.

| # | Finding | Status now |
|---|---|---|
| 1 | Fixed `rgb2hed` matrix across differently-stained sections | **LIVE** — `KNOWN_ISSUES.md` §3.7 |
| 2 | `h_intensity` over whole patch, not nuclei | **FIXED** (Fix 1c). Masked to `labeled_mask > 0` (`validation/morphological_features.py:141`). The old definition survives as the separate feature `h_intensity_wholepatch`, reported but not voting. |
| 3 | Silent failure zeroing beyond patch index 4 | **FIXED** (Fix 1a). Failures yield `np.nan`, are counted, and are written to `feature_failures.json`. Both reference sections record **zero** failures, confirmed by the Tier 1 gate. |
| 4 | `nc_ratio` returns `inf` for fully-nuclear patches | **LIVE** — unchanged at `validation/morphological_features.py:145`. `save_json` converts non-finite to `null`, so an `inf` reaches `validation.json` as `null` and `results.csv` as `inf`. |
| 5 | `packing_irregularity` returns 0.0 below 3 nuclei | **PARTLY FIXED** (Fix 1d). It now returns `np.nan`, not 0.0, so it is no longer pulled toward the density floor. The structural confound remains: the patches where it is missing are exactly the low-density ones. `KNOWN_ISSUES.md` §3.8. |
| 6 | Single-angle GLCM for `texture_entropy` | **FIXED** (Fix 1b). Now 4 angles x 3 distances, averaged over the 12 scalars rather than over a pooled GLCM. |
| 7 | Per-patch Otsu not comparable across patches | **LIVE** — inherent to patch-level processing, unchanged. |
| 8 | StarDist path unused | **LIVE**, and worse than recorded here: StarDist is not in `requirements.txt`, so `--use-stardist` silently falls back to Otsu. `KNOWN_ISSUES.md` §6.9. |

**The four fixed findings are why `per_section_v2` exists.** `jobs/run_per_section_v2.sh`
re-ran both sections with them applied, and that tree is the current reference.

**Companion file.** `reports/morphological_features_diagnostics_results.md` was meant to
hold the measured diagnostics validating these findings. It has never been populated.

---

---

## Pipeline Invariant — Confirmed

`compute_morphological_features(all_patches, ...)` is called at `run_all.py:668`, inside PHASE 5 ("Morphological Feature Validation"). By that point in `run_all.py`:

- PCA has been fitted and applied (`run_all.py:517`)
- Harmony / scVI correction has been applied (`run_all.py:523-530`)
- UMAP has been computed (line 502)
- Leiden clustering has been computed (line 504)
- DPT pseudotime has been computed (lines 517–565)

Features are computed on `all_patches`, the same post-cap patch array used to build the manifold — but **after** all manifold construction is complete. The features are then attached to `adata.obs` (line 596–597) and saved to `validation.json` and `results.csv`. They are never fed back into clustering or DPT. The held-out-of-manifold invariant is confirmed correct.

---

## Per-Feature Verdicts

### 1. Hematoxylin Deconvolution (`rgb2hed`)

**Verdict: CHOICE — defensible but methodologically risky given the known stain discordance between sections.**

`_deconvolve_hematoxylin` calls `skimage.color.rgb2hed(patch_rgb)` and returns `hed[:, :, 0]`. The skimage implementation uses the **fixed Ruifrok & Johnston (2001) stain matrix**, derived from bright-field H&E images at standard staining conditions. This matrix is hardcoded and is not estimated from the slide or the section.

**Why this matters for this specific paper:** The project's own notes (`NOTES.md`) confirm that "2M-1 and 2M-2 sections were stained with different reagents." The two-island UMAP problem was attributed in part to this. A fixed stain matrix applied to sections stained with genuinely different dye chemistries will produce systematically biased H-channel values: the same tissue imaged under different staining conditions will produce different H values not because of biology, but because the stain absorption spectrum differs from the assumed matrix.

**Direct consequence:** Every feature derived from the H channel — `nuclear_density`, `mean_nuclear_area`, `nc_ratio`, `h_intensity`, and (via Otsu thresholding) indirectly `packing_irregularity` — is computed on a channel whose absolute scale is not comparable between the two sections. This provides an alternative explanation for any cross-section discordance observed in validation.json results and the cross-section replication comparison.

**Would fixing it change the reported numbers?** Yes, potentially substantially for `h_intensity` (which is a direct mean of the H channel) and moderately for density/area metrics. It would not necessarily strengthen or weaken the within-section correlations, but could change the *direction* of cross-section replication for `h_intensity` specifically.

**Nature of finding:** CHOICE (not a bug in isolation), but **directly relevant to the paper's central finding** that some features do not replicate across sections. The fixed stain matrix is a confound for any cross-section comparison of H-derived features that has not been reported or acknowledged in the draft.

---

### 2. Nuclear Segmentation (Otsu + binary_opening + remove_small_objects)

**Verdict: CHOICE — defensible for exploratory work; two sub-issues worth distinguishing.**

**2a. Per-patch Otsu threshold** (`_segment_nuclei_simple`, line 45)

`threshold_otsu(h_channel)` is computed independently on each 112×112-pixel patch. This means the threshold adapts to the local H distribution of that patch.

*Benefit:* Patches with few nuclei can still segment the nuclei present, rather than falling below a global threshold calibrated to dense regions. This prevents systematic under-counting in sparse patches.

*Risk:* The threshold is not biologically meaningful across patches. A patch of pure background (no nuclei) will still have Otsu find a threshold, potentially classifying high-H background pixels as "nuclear." A patch of very dense, uniform nuclear tissue will threshold some nuclear pixels as background (Otsu maximizes inter-class variance, so it will artificially split any unimodal distribution). In the extreme case, a truly featureless patch produces a degenerate all-True or all-False binary depending on float precision, which is the precondition for the `nc_ratio` inf case discussed below.

*Assessment:* Per-patch Otsu is standard in patch-based digital pathology pipelines at low magnification (5×). The inconsistency across patches introduces noise but not systematic directional bias. This is a reasonable default.

**2b. `disk(1)` opening and `min_size=20` object removal**

At 5× magnification, a 112×112-pixel patch covers a large tissue area. Nuclei at this magnification are small — approximately 5–15 pixels in diameter, corresponding to areas of roughly 20–175 pixels. The `min_size=20` threshold removes objects with area < 20 pixels, which corresponds to the very smallest legitimate nuclei. Objects below this threshold are either noise or very small lymphocytes. This is defensible but the choice of 20 is not derived from the actual cell size distribution at this magnification; it is an arbitrary round number.

The `disk(1)` morphological opening removes 1-pixel-wide protrusions. Given that Otsu on the H channel typically produces a noisy binary at 5× magnification, this is minimal cleaning and is standard practice.

**2c. StarDist path**

The `_segment_nuclei_stardist` function exists and falls back to Otsu if not installed. It uses the `"2D_versatile_he"` pretrained model (Weigert et al., 2020), which was trained on diverse H&E data and produces instance-level segmentation rather than binary thresholding. StarDist would substantially improve nuclear instance counts — it handles overlapping nuclei, does not rely on an H-channel Otsu threshold, and produces more accurate nuclear areas. The cost is ~1–2 seconds per patch and a tensorflow/stardist dependency. At ~10,000 patches per section, this is ~3–6 hours of segmentation per run — feasible but significant.

**Would switching to StarDist change reported numbers?** Yes. Nuclear counts would change, affecting `nuclear_density`, `mean_nuclear_area`, `nc_ratio`, and `packing_irregularity`. These are the features the manuscript uses as evidence. This would be a substantial re-experiment, not a robustness check, and should not be done without careful comparison against current numbers.

---

### 3. `nuclear_density` and `mean_nuclear_area`

**Verdict: SOUND AS IMPLEMENTED, with one edge-case note.**

```python
def compute_nuclear_density(labeled_mask, patch_area):
    n_nuclei = labeled_mask.max()
    return n_nuclei / patch_area if patch_area > 0 else 0.0

def compute_mean_nuclear_area(labeled_mask):
    props = regionprops(labeled_mask)
    if len(props) == 0:
        return 0.0
```

**Zero-nucleus case:** `labeled_mask.max()` returns 0 when no nuclei are present (all-background mask), correctly yielding `density = 0.0`. `compute_mean_nuclear_area` with `len(props) == 0` returns `0.0` — a different semantic (zero area vs. undefined area), but consistent with density = 0.0 for empty patches. No division-by-zero risk.

**`labeled_mask.max()` as nucleus count:** `skimage.measure.label()` assigns contiguous labels 1..N on the post-cleaned binary image, so `.max()` equals the actual nucleus count. This is correct.

**`patch_area > 0` guard:** `patch_area = patch_h * patch_w = 112 * 112 = 12544`. This is always positive for the fixed patch size. The guard is dead code but harmless.

---

### 4. `nc_ratio` — Potential `inf` Values

**Verdict: POTENTIAL CORRECTNESS ISSUE — `inf` values are produced and are silently excluded from correlations, but the exclusion is unlogged and its scale is unknown.**

```python
def compute_nc_ratio(labeled_mask):
    nuclear_pixels = (labeled_mask > 0).sum()
    total_pixels = labeled_mask.size
    cytoplasm_pixels = total_pixels - nuclear_pixels
    if cytoplasm_pixels == 0:
        return float("inf")
    return nuclear_pixels / cytoplasm_pixels
```

When `cytoplasm_pixels == 0`, `float("inf")` is stored in the numpy features array (which holds float64 and can represent inf). This value then appears in `results.csv` and `adata.obs`.

**How downstream code handles it:** Both `correlations.py` and `cellularity_confound.py` apply `valid = np.isfinite(values) & np.isfinite(pseudotime)` before calling `spearmanr`. So `inf` nc_ratio patches are **excluded** from the correlation, not ranked as maximum. This is the correct mathematical behavior for Spearman rank correlation — treating inf as maximum rank would be wrong, and the code avoids this.

**The unresolved question:** How many patches actually hit this path? `cytoplasm_pixels == 0` requires the per-patch Otsu threshold to fall below the minimum H value in the patch, causing all pixels to be labeled nuclear. With per-patch Otsu on real H&E patches at 5×, this is possible in high-cellularity patches where the H distribution is nearly unimodal. It is not possible to determine the count without reading `results.csv` from the Narval runs — but it is determinable without recomputation (see diagnostic recommendations below).

**If the count is non-trivial:** The excluded patches are not a random subset — they are by definition the highest-density patches (the ones where Otsu collapses). Excluding them from the nc_ratio correlation would systematically underrepresent the dense end of the nc_ratio distribution, pulling the measured rho toward zero. This would make the nc_ratio correlation appear weaker than it is.

---

### 5. `texture_entropy` — Single-Angle GLCM

**Verdict: CHOICE — single-angle GLCM is a meaningful limitation for tissue texture; the quantization level is reasonable.**

```python
glcm = graycomatrix(patch_q, distances=[d], angles=[0], levels=64, symmetric=True, normed=True)
```

`angles=[0]` computes the GLCM only for **horizontal pixel pairs** (0° direction). For tissue texture capturing spatial disorganization — which has no preferred axis in H&E histology — the standard approach averages the GLCM entropy over four angles: 0°, 45°, 90°, 135° (i.e., `angles=[0, π/4, π/2, 3π/4]`). A single-angle GLCM will:

- Be sensitive to horizontally-oriented textures (e.g., linear staining gradients in the scan direction)
- Underweight vertically- or diagonally-oriented tissue structure
- Introduce orientation-dependent variance across patches from the same tissue type

This is a genuine methodological limitation. However, the compute cost is trivial (adding three angles to the GLCM call changes nothing else) and would likely increase the entropy values slightly (averaging over more directions typically gives a higher entropy). Whether it would change the *ranking* of patches, and thus the Spearman rho, depends on how much cross-patch variance is currently attributable to scan-direction artifacts.

**64-level quantization:** `patch_q = patch_gray // 4` maps 0–255 to 0–63. This is a common and defensible choice. Too many levels makes the GLCM sparse (many zeros); too few loses texture resolution. 64 levels at 112×112 pixels gives ~200 pixel pairs per gray-level pair in the horizontal direction, which is adequate.

**Would fixing the single-angle issue change reported numbers?** Likely a modest change in absolute entropy values (systematic upward shift), but the Spearman rho with pseudotime would change only if the current single-angle values are systematically correlated with scan direction in a way that pseudotime is not. This is a low-priority check but worth noting in a methods section.

---

### 6. `h_intensity` — Whole-Patch vs. Nuclear-Region Mean

**Verdict: CHOICE — but the current implementation is a proxy for nuclear density, not a measure of chromatin density per nucleus. The docstring is misleading.**

```python
def compute_hematoxylin_intensity(h_channel):
    """Mean optical density in the hematoxylin channel."""
    return float(np.mean(h_channel))
```

This computes the mean H-channel value over all 12,544 pixels in the patch — including background, stroma, cytoplasm, and nuclei alike. The docstring on the module header calls this "chromatin density," but:

- Background pixels contribute near-zero H values, diluting the mean.
- Patches with more nuclei have a higher fraction of high-H pixels, so the mean increases with nuclear density.
- `h_intensity` as implemented is therefore **mathematically dependent on `nuclear_density`**, in addition to whatever patch-to-patch variation in actual chromatin intensity exists.

The more standard interpretation of "chromatin density" or "hematoxylin intensity" in quantitative histology computes mean H within nuclear regions only (i.e., masked to `labeled_mask > 0`). That measure would be independent of nuclear density and would reflect the actual staining intensity per nucleus — an indicator of chromatin organization, mitotic activity, or stain uptake variation.

**Direct consequence for the cellularity confound analysis:** The confound analysis tests whether features survive partial Spearman controlling for `nuclear_density`. For the current `h_intensity` (whole-patch mean), part of the signal is mechanistically explained by density — the confound analysis partially compensates for this, but the confound is baked into the feature itself rather than being an incidental correlation. If `h_intensity` were computed mask-only, its partial rho would be a purer measure of chromatin density, and the survival/collapse verdict from the confound analysis could differ.

**Would fixing it change reported numbers?** Yes, substantially for `h_intensity` itself (the value for a sparse patch would go up, since low-H background is no longer diluting the mean; the value for a dense patch would also change because background pixels are removed). The direction and magnitude of the rho with pseudotime could change. This is a high-priority point to report in the manuscript's methods limitations section.

---

### 7. `packing_irregularity` — Confound with `nuclear_density` (Untested)

**Verdict: POTENTIAL CORRECTNESS ISSUE — the fallback to 0.0 for low-density patches creates a structural correlation with nuclear_density that has not been tested in the cellularity confound analysis.**

```python
def compute_packing_irregularity(labeled_mask):
    props = regionprops(labeled_mask)
    if len(props) < 3:
        return 0.0
    centroids = np.array([p.centroid for p in props])
    tree = KDTree(centroids)
    distances, _ = tree.query(centroids, k=2)
    nn_dists = distances[:, 1]
    mean_dist = nn_dists.mean()
    if mean_dist < 1e-10:
        return 0.0
    return float(nn_dists.std() / mean_dist)
```

**The structural issue:** Any patch with 0, 1, or 2 segmented nuclei returns `packing_irregularity = 0.0`. These are by definition the lowest-density patches. The result is that `packing_irregularity` is pinned at 0.0 for an entire range of the `nuclear_density` distribution (specifically, the low end), creating a mechanical floor in the packing_irregularity vs. density relationship that is not biological.

**Consequences:**
1. The Spearman correlation of `packing_irregularity` with pseudotime includes a structural component: patches that map to "acellular" pseudotime regions have both low density AND `packing_irregularity = 0.0`, not because cells are tightly packed but because the fallback fires.
2. The cellularity confound analysis (`cellularity_confound.py`) tests `mean_nuclear_area`, `nc_ratio`, `texture_entropy`, `h_intensity`, and `packing_irregularity` against the `nuclear_density` confound. `packing_irregularity` IS included in this test, which is correct in principle.
3. However, the partial Spearman formula for packing_irregularity controlling for nuclear_density will correctly detect that some of packing_irregularity's correlation with pseudotime is driven by the density floor, and would attenuate the partial rho relative to the raw rho. Whether packing_irregularity "survives" (|partial_rho| ≥ 0.1) after controlling for density is an empirical question already answered by the confound analysis — but the *reason* for any collapse is partly the 0.0 fallback, not the biology.

**Additional note — coefficient of variation with k=2:** With exactly 3 nuclei, `nn_dists` has 3 values; the CV has a large variance with so few observations. CV estimates from small samples are unreliable. No minimum sample size beyond the ≥3 threshold is enforced.

**Would flagging this change reported numbers?** The reported numbers (rho values) were already computed with this implementation and reflect its behavior. The point is that the interpretation of packing_irregularity as a measure of spatial disorder is partially compromised at low density, and this should be acknowledged in the methods or supplementary text.

---

### 8. Failure Handling — Silent Zeroing Beyond Patch 4

**Verdict: BUG in logging; potential bias depending on failure rate, which is unknown.**

```python
except Exception as exc:
    if i < 5:
        print(f"  WARNING: Feature extraction failed for patch {i}: {exc}")
```

**The logging bug:** The warning is printed only for `i < 5`, meaning patches 0–4. If failures begin at patch 5 or later — which is the typical case, since early patches tend to be from the first slide processed and may be representative — they produce no output at all. A run could have 500 silently-failed patches with no indication in the job log.

**The bias risk:** Failed patches leave all six features at 0.0 (the array initialization value). A 0.0 in any feature is indistinguishable from a genuinely zero-valued observation (e.g., an empty background patch). Consider:

- `nuclear_density = 0.0`: could be a failure OR a background patch with no nuclei. Both are legitimate zero values; no bias unless failures are concentrated in particular tissue regions.
- `h_intensity = 0.0`: in the H channel (from `rgb2hed`), values of 0.0 correspond to no hematoxylin absorption, which genuinely occurs in acellular regions. A failure-induced 0.0 mimics an acellular patch.
- `texture_entropy = 0.0`: this would require a perfectly uniform gray patch. Genuine 0.0 is extremely rare (it would require all pixels to have the same grayscale value). **A failure-induced 0.0 in texture_entropy is anomalous and potentially identifiable.**

If the failure rate is >1% and failures are not uniformly distributed across tissue types, the 0.0 values could shift the measured correlations. If failures cluster in high-density (complex, crowded) patches — which are computationally harder to segment — the 0.0 floor injection would push all six feature values toward the low end for the high-density regime, attenuating rho estimates.

**Determinability without recomputation:** The failure count IS determinable from existing run artifacts on Narval without recomputing anything. See diagnostics section below.

---

## Ranked Summary

### Ranked by severity

| Rank | Finding | Type | Feature(s) | Expected impact on reported numbers |
|---|---|---|---|---|
| 1 | Fixed `rgb2hed` stain matrix applied across sections stained with different reagents | CHOICE with cross-section impact | h_intensity, nuclear_density, nc_ratio, mean_nuclear_area, packing_irregularity | Potentially large for h_intensity; moderate for density-derived features. Directly relevant to cross-section discordance finding. |
| 2 | `h_intensity` computed over whole patch, not nuclear regions only — mechanically correlated with density | CHOICE, misleading docstring | h_intensity | Moderate to large. Changes the biological interpretation and the partial rho in the confound analysis. |
| 3 | Silent failure zeroing beyond patch index 4 — failure count and distribution unknown | BUG in logging; potential bias | All six features | Unknown until diagnostic is run. Could range from negligible to significant depending on failure rate. |
| 4 | `nc_ratio` produces `inf` for fully-nuclear patches; these are silently excluded from correlations; count unknown | POTENTIAL BUG (incomplete) | nc_ratio | Low if count is small (< 0.1% of patches); moderate if count is in the hundreds. |
| 5 | `packing_irregularity` returns 0.0 for patches with < 3 nuclei, structurally confounding it with nuclear_density | CHOICE with interpretation consequences | packing_irregularity | Modest — the confound analysis already tests this feature against density, and the 0.0 floor would reduce its partial rho. But the confound is structural, not incidental. |
| 6 | Single-angle GLCM (0°) for texture_entropy misses non-horizontal texture patterns | CHOICE | texture_entropy | Likely modest — systematic upward shift in entropy values but rho ranking may be similar. |
| 7 | Per-patch Otsu threshold is not comparable across patches of different cellularity | CHOICE | nuclear_density, nc_ratio, mean_nuclear_area, packing_irregularity | Adds noise but not directional bias; inherent to patch-level processing. |
| 8 | StarDist path unused — Otsu segmentation at 5× magnification is noisier than instance segmentation | CHOICE | All segmentation-derived features | Would change absolute numbers substantially. Out of scope to change without a dedicated re-experiment. |

---

## Recommended Next Steps

### (a) Read-only diagnostics — do now, no recomputation required

These checks can be run against existing `results.csv` files on Narval. They are diagnostic only and do not require rerunning the pipeline.

**D1. Count failure-zeroed patches in `texture_entropy`.** In `results.csv`, a genuine `texture_entropy = 0.0` requires a perfectly uniform grayscale patch. Any `texture_entropy == 0.0` entry in the CSV is almost certainly a failed patch (not a genuine zero), because even a nearly-uniform patch has some GLCM entropy. Count `(results_csv["texture_entropy"] == 0.0).sum()` for each per-section run. If this count is > ~50 (>0.5% of ~10,000 patches), the failure rate is non-trivial and the zeroed patches should be masked from all correlations.

```python
# Read-only diagnostic — run on login node or in a short interactive session
import pandas as pd
df = pd.read_csv("$SCRATCH/results/per_section/atlas_2M-1/results.csv")
print("texture_entropy == 0.0:", (df["texture_entropy"] == 0.0).sum(), "of", len(df))
print("nuclear_density == 0.0:", (df["nuclear_density"] == 0.0).sum(), "of", len(df))
print("nc_ratio is inf:", (~df["nc_ratio"].apply(lambda x: x != float('inf'))).sum())
# Or equivalently:
import numpy as np
print("nc_ratio is inf:", np.isinf(df["nc_ratio"].values).sum())
```

**D2. Count `nc_ratio == inf` patches.** From the same `results.csv`, check `np.isinf(df["nc_ratio"])`. If this is > 0.1% of patches, note the count in the manuscript methods section.

**D3. Check cross-section texture_entropy value distributions.** Plot or describe the 5th/median/95th percentile of each feature split by section (2M-1 vs. 2M-2) from `results.csv`. A large mean shift in `h_intensity` between sections with similar tissue would be evidence that the fixed stain matrix is introducing bias rather than biology.

**D4. Check packing_irregularity == 0.0 count.** Count `(df["packing_irregularity"] == 0.0).sum()`. The sum of patches with < 3 nuclei (true zero) plus failed patches (silent zero). Can be cross-checked against nuclear_density: patches with `packing_irregularity == 0.0` but `nuclear_density > 0` are likely failed patches (packing cannot be zero if nuclei are present and well-spread).

---

### (b) Follow-up experiments — only after reading current numbers, comparison against manuscript values required

These changes would invalidate numbers already written into the manuscript and must be treated as new experimental conditions compared against the current baseline.

**E1. Restrict `h_intensity` to nuclear regions.** Change `compute_hematoxylin_intensity` to compute `np.mean(h_channel[labeled_mask > 0])` instead of the whole-patch mean (with a fallback to 0.0 for empty masks). Re-run Phase 5 only (no manifold rebuild needed). Compare rho values with pseudotime and partial rho in confound analysis. This is the single change most likely to meaningfully alter a reported number and most clearly corrects a methodological mismatch between the docstring and implementation.

**E2. Multi-angle GLCM for `texture_entropy`.** Change `angles=[0]` to `angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]` in `compute_texture_entropy`. Re-run Phase 5 only. This is low-risk (values shift up uniformly) and easily described as "isotropic GLCM entropy" in methods. Compare rho values.

**E3. Quantify fixed-stain-matrix bias across sections.** Before any deconvolution change (which would be a large experiment), check whether per-section means of the raw R, G, B channels in `all_patches` differ substantially between 2M-1 and 2M-2. This is a read-only diagnostic on the cached `.npy` feature arrays or existing `results.csv` data. If a strong systematic shift is found, it should be disclosed in the manuscript as a known limitation of the h_intensity and density features for cross-section comparison.

**E4. StarDist segmentation sensitivity check.** Run StarDist on a held-out subset (e.g., 200 randomly-sampled patches from each section) and compare nuclear counts and areas from Otsu vs. StarDist. If counts differ by < 20%, the Otsu results are stable enough. If counts differ by > 50% on average, the reported nuclear_density and related features warrant a caveat. This is a read-only experiment on existing patch arrays.

---

## Notes for the Manuscript

The following points should be documented in the methods or supplementary material of the paper draft regardless of whether the code is changed:

1. **Stain matrix limitation:** Features derived from hematoxylin channel deconvolution (`h_intensity`, `nuclear_density`, `nc_ratio`, `mean_nuclear_area`, `packing_irregularity`) use a fixed Ruifrok–Johnston stain matrix, which is not calibrated to the specific reagents used in each section. Cross-section differences in these features should be interpreted with this caveat.

2. **`h_intensity` is a whole-patch mean:** The feature reflects both nuclear density and per-nucleus chromatin staining intensity. It is not a pure measure of chromatin organization.

3. **`packing_irregularity` floor:** Patches with fewer than 3 segmented nuclei are assigned `packing_irregularity = 0.0`. This occurs for a non-negligible fraction of low-density patches, and the zero is treated as a true observation in all correlations.

4. **Single-angle GLCM:** `texture_entropy` is computed from a 0°-direction GLCM only (horizontal pixel pairs). Multi-angle averaging was not used.

5. **Silent failure handling:** Patches for which feature extraction fails have all features set to 0.0. The failure rate is not reported in the current implementation beyond the first five occurrences. Diagnostic D1 above is needed to bound this.
