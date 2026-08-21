# Pseudotime anchor validation — complete experimental record and handoff

*Self-contained. Written for someone with no prior context who has to either continue this
work or defend it. Every number below comes from a completed run; nothing is projected.*

---

## 1. Setting

**Data.** 16 whole-slide H&E images of MCF7 xenograft tissue at 5× downsampling, split into two
tissue **sections** of 8 slides each: **2M-1** (Carnoy's-fixed) and **2M-2** (PFA-fixed).

**Pipeline.** 112 px patches → six morphological descriptors per patch → PCA → k-NN graph →
diffusion map → diffusion pseudotime (DPT) from 20 root patches, median-aggregated across roots,
min–max normalised to [0,1].

- 2M-1: 8,244 patches, PCA width 261
- 2M-2: 10,072 patches, PCA width 241

**Duct annotations.** A pathologist annotated "Tumor" polygons in QuPath, each carrying
`hole_pct` — the percent of duct area that is lumen. 2,173 ducts in 2M-1, 1,749 in 2M-2. Patches
join to ducts by **patch centre inside polygon**, which leaves 571 (26%) and 389 (22%) ducts with
no assigned patch — systematically the smallest.

**The claim under test.** That pseudotime orders patches along a progression, and that duct area
and the morphological features validate that ordering.

---

## 2. How the question arose

Two problems motivated everything.

**The root rule was circular.** Production DPT selects the 20 patches with the lowest
`nuclear_density`. But `nuclear_density` is simultaneously (i) the root selector, (ii) one of the
six features pseudotime is *validated against*, and (iii) the covariate partialled out in the
cellularity-confound analysis. The axis was partly defined by a quantity it was later validated
against.

**An earlier analysis suggested the axis wasn't a trajectory.** A root-sensitivity study found
`rho(PT, diffusion-map centroid distance)` = 0.808 / 0.802 versus `rho(PT, DC1)` = 0.543 / 0.467,
and that 25 uniformly random 20-root sets reproduced production pseudotime at |rho| 0.78–0.89. The
reading at the time: the axis measures how *unusual* a patch is, not how far *along* — and the
manifold, not the root rule, fixes it.

**The intervention.** A "holeyroot" run re-anchored DPT on the expert `hole_pct` instead, on the
grounds that a hand annotation is not derived from the pipeline's own pixels. Direction: low hole %
= early.

---

## 3. Experiments and results

### 3.1 Re-reading the holeyroot comparison (Phase 2)

Phase 2 nominated two non-circular tests — `rho(pt, duct area)` and `rho(pt, nuclear_density)` —
and reported that 2M-2's area correlation moved from −0.084 to +0.249, calling it "the sharpest
single discriminator available."

Two things its raw JSONs showed that the summary did not:

**(a) The realised anchor is duct-size-extreme.** Root duct area median 7,763 µm² against an
eligible-duct median of 33,807 µm²; the *largest* root duct (30,822) is still below the eligible
median. All 20 root ducts below the median has probability 0.5²⁰ ≈ 1e-6 under any size-blind rule.
And although the rule advertises "bottom decile," a round-robin taking the 20 lowest from a
136–160-duct pool makes the realised anchor the bottom **1–3%**.

**(b) The v2 baseline in 2M-2 was degenerate.** All 20 roots had `nuclear_density` exactly 0.0 and
`duct_id: null` — not one inside any Tumor annotation. Two (adjacent patches on the same slide) sat
at pseudotime 0.717 and 0.673 while the other 18 spanned 0.009–0.144.

### 3.2 `anchor_area_control` — is `rho(pt, area)` a duct-size artifact?

Re-runs only `sc.tl.dpt` from alternative root sets on the **frozen** graph and diffusion map.
Consistency gate: re-running from the stored roots reproduced the stored axis at **rho = 1.000000**
in both sections.

**Task A — anchor extremity.** 20/20 root ducts below the eligible median in both sections
(p = 1.9e-06); root ducts 3.44× and 4.35× smaller than typical.

**Task B — area-matched surrogate (25 draws).** 20 ducts size-matched to the real roots but chosen
*without reference to hole %*:

| | observed | surrogate null median | null range | frac ≥ as extreme |
|---|---|---|---|---|
| 2M-1 `rho(pt, area)` | +0.4182 | **+0.4366** | [+0.363, +0.483] | 0.76 |
| 2M-2 `rho(pt, area)` | +0.2490 | **+0.2347** | [+0.199, +0.309] | 0.32 |

The observed values sit inside their nulls, at or below the median. The 2M-2 surrogates were drawn
from ducts with hole % median 10.6–19.1 against the real anchor's 1.80 — an order of magnitude
holier — and reproduced the same correlation. Against the random-duct baseline, duct size accounts
for **110%** of the 2M-1 gap and **94%** of the 2M-2 gap.

**Task C — area-stratified anchor.** Lowest hole % *within* each of 5 equal-count area strata:

| | `rho(pt, area)` | random-duct baseline | `rho(pt, hole %)` | `rho(pt, nd)` |
|---|---|---|---|---|
| 2M-1 | +0.4182 → **+0.2664** | +0.2289 | +0.3851 → +0.3864 | +0.2556 → +0.1014 |
| 2M-2 | +0.2490 → **+0.0205** | +0.0038 | +0.3196 → +0.2881 | +0.0203 → **−0.2439** |

Removing the size extremity returns `rho(pt, area)` to *exactly* the random-anchor baseline in both
sections while leaving the hole correlation intact.

**Task E — v2 root repair.** 0/20 discordant in 2M-1; 3/20 in 2M-2, whose removal took
`pseudotime_std` from 27.70% to 3.40% of range.

**Conclusion.** `rho(pt, duct area)` is not the non-circular test Phase 2 treated it as. Phase 2's
cross-section reconciliation dies with it: under the size-neutral anchor `nuclear_density` is
**+0.101 (2M-1) vs −0.244 (2M-2)** — the sections disagree in sign again.

### 3.3 `holeyroot_duct_checks` — nesting, uncertainty, and what hole % measures

**Nesting is not a problem.** Within-slide medians reproduce the pooled values with 8/8 slides
carrying the sign. The Simpson's-paradox risk did not materialise.

**Cluster bootstrap intervals are narrow, not wide**, because the slides agree:

| run | quantity | point | 95% CI | frac draws opposite sign |
|---|---|---|---|---|
| holeyroot 2M-2 | `pt_area` | +0.2490 | [+0.161, +0.323] | 0.000 |
| v2 2M-2 | `pt_area` | −0.0843 | **[−0.200, +0.025]** | 0.071 |
| holeyroot 2M-2 | `pt_nd` | +0.0203 | **[−0.080, +0.098]** | 0.331 |

**v2 2M-2's −0.084 was never distinguishable from zero.** Phase 2's premise — "the sections
disagree about the mediator" — was itself unfounded.

**Task 3 — hole % vs the pipeline's own optics** (this later turned out to be misleading; see §3.7):

| feature | 2M-1 | 2M-2 |
|---|---|---|
| `h_intensity_wholepatch` | +0.080 | −0.271 |
| `texture_entropy` | +0.377 | +0.335 |

`texture_entropy` correlates with the anchor variable at ~+0.35 in **both** sections, so it is not
an independent validator of a hole-anchored axis.

### 3.4 `eccentricity_check` across three axes

Tests the "it's eccentricity not a trajectory" claim in spaces where DPT's construction does not
force the answer, explicitly excluding the diffusion-space correlation as definitional.

| measure | density-rooted | holeyroot | area-stratified |
|---|---|---|---|
| **2M-1** `morph_mean_abs_z` | 0.0881 | 0.1539 | 0.1909 |
| **2M-1** `morph_mean_signed_z` | 0.3022 | 0.2548 | 0.1990 |
| **2M-1** verdict | DIRECTIONAL | DIRECTIONAL | **ECCENTRICITY IN EMBEDDING ONLY** |
| **2M-2** `morph_mean_abs_z` | 0.2392 | 0.2452 | 0.2835 |
| **2M-2** verdict | DIRECTIONAL (5/6) | DIRECTIONAL (4/6) | DIRECTIONAL (4/6) |

**Bidirectional enrichment: 0 of 6 features on every axis, in both sections**, with 4–5
unidirectional.

Three findings:

1. **The eccentricity headline does not survive.** It rested on a measure that is partly true by
   construction. PCA-space eccentricity is real (0.50–0.66, surviving within slides at 0.42–0.64),
   so about half the axis is radial in the representation space — but it does not become
   morphological eccentricity, and the discriminating test is clean.
2. **`nuclear_density` drops out of the directional set exactly when the size extremity is
   removed** — present on the density-rooted axis (circular) and holeyroot (size artifact), absent
   on area-stratified. Patch-level agreement with §3.2's duct-level +0.020 → −0.244.
3. **2M-1's trajectory verdict does not survive de-sizing.** Signed and unsigned morphological
   terms converge monotonically across the three axes (gap 0.214 → 0.101 → 0.008).

### 3.5 `eccentricity_within_slide` — is the late structure biology or one slide?

`eccentricity_check` ran its enrichment and subclustering on the **global** top decile, which is
majority one slide.

| | 2M-1 holeyroot | 2M-1 area-strat | 2M-2 holeyroot | 2M-2 area-strat |
|---|---|---|---|---|
| Cramér's V (late subcluster × slide) | 0.815 | 0.812 | 0.535 | 0.536 |
| largest cluster's top-slide share | 74.9% | 71.6% | 84.8% | 84.9% |
| bidirectional in a majority of slides | 0/6 | 0/6 | 0/6 | 0/6 |

**The cohort late subclusters largely *are* slides.** `eccentricity_check`'s opposing-feature
verdict is a batch split, not two late phenotypes, and V is identical to three decimals across
axes — so the late tail's slide composition is a property of the manifold, not of the roots.

**Within slides the eccentricity signature is absent**, on both axes in both sections. 2M-1 *loses*
directional structure when de-sized (`nuclear_density` 6/8 → 4/8 slides unidirectional); 2M-2
*gains* it (`nc_ratio` 7/8 → 8/8). One residual signal: `mean_nuclear_area` has opposing late
subclusters in 4 of 7 slides on the area-stratified 2M-2 axis, against a baseline rate of 0–1/7.

### 3.6 `export_anchor_axis` — persisting derived axes

The area-stratified pseudotime existed only as summary correlations. This rebuilds it
deterministically (~20 DPT calls per section on the frozen graph) and writes a derived run
directory. Gated two ways: the source axis must reproduce at rho ≥ 0.999, and the rebuilt
correlations must match the recorded Task C values. Independent confirmation: the derived axis's
eccentricity values (0.7893 / 0.7721) match Task F to four decimals.

Later extended with a `v2_repaired` anchor — see §3.9.

### 3.7 `duct_white_fraction` — does the annotation track the pixels?

§3.3 tested the anchor variable through **patch-derived proxies**. This measures it directly: every
Tumor polygon rasterised against its slide PNG, white = mean RGB > threshold, no patch assignment,
**every duct included** — including the zero-patch ducts no earlier analysis could see.

| section | ducts measured | rho(hole %, white) | 95% CI | within-slide | partial given area |
|---|---|---|---|---|---|
| 2M-1 | 2162/2173 | **+0.9202** | [+0.901, +0.935] | +0.917 (8/8) | +0.9022 |
| 2M-2 | 1746/1749 | **+0.7930** | [+0.763, +0.832] | +0.788 (8/8) | +0.7658 |

**The annotation is sound, and 2M-1 is the stronger section** — the opposite of what the proxy
suggested. Zero-patch ducts correlate just as well (+0.9078 vs +0.9157; +0.8306 vs +0.7753), ruling
out the exclusion bias.

**Why the proxy failed.** Measured white is a median of **0.95%** of duct area in 2M-1 and 4.92% in
2M-2. Inside a 112 px patch, the hole contributes ~1% of pixels, so `h_intensity_wholepatch` is
dominated by tissue staining. The signal was real and simply far below patch-level stain variation.

**Threshold sweep — both sections peak at 190:**

| threshold | 2M-1 | 2M-2 |
|---|---|---|
| **190** | **+0.9763** | **+0.9553** |
| 200 | +0.9728 | +0.9142 |
| 220 (pre-specified) | +0.9202 | +0.7930 |
| 240 | +0.7801 | +0.5228 |

**Calibration.** Median per-duct difference (measured − annotated), percentage points: 2M-1 +0.75
at 190 and −2.53 at 220; 2M-2 +6.98 and −11.35. Zero-crossings at ≈197 and ≈201.

So the correlation-optimal threshold (190) and the bias-zeroing threshold (~197–201) nearly
coincide, in both sections, within a 90-unit sweep. Four independent selections in a ~10-unit
window. **At the annotator's own cut-off the annotation is correct in both rank and magnitude.**
220 was an inherited convention that understated the agreement.

### 3.8 `holeyness_asymmetry` and `holeyness_section_comparison`

**The premise of the asymmetry investigation was wrong.** 2M-2's circulated "0.020" was
`rho(pt, nuclear_density)` on the holeyroot axis. Recomputed: **+0.1906 with 7/8 slides positive**,
not 0.020 with 5/8.

Both candidate explanations were ruled out. Rank compression: `rank_sd_ratio` = 1.0000 in *both*
sections, >99% distinct values, and 2M-2's raw spread is *larger* (IQR ratio 2.74). Annotation
behaviour: `rho(area, hole_pct)` = +0.386 vs +0.361, 8/8 slides positive in both, sanity check
`rho(area, hole_area)` +0.862 / +0.921.

**The full correlation table** (slide-clustered bootstrap CIs, 2,000 resamples; within-slide
permutation p, 5,000 shuffles):

| section | raw | \| area | \| area + nd |
|---|---|---|---|
| 2M-1 | +0.2763 [0.235, 0.333] | +0.1315 [0.077, 0.192] | +0.1580 [0.102, 0.227] |
| 2M-2 | +0.1906 [0.105, 0.275] | +0.2379 [0.174, 0.308] | +0.1809 [0.104, 0.258] |

**All six exclude zero. All six permutation p < 0.0002.** The validation holds in both sections
under every estimand. A consistency gate reproduced the on-record 2M-1 adjusted value of 0.131 to
+0.1315, and the rank-residual and algebraic partial implementations agreed to 1e-16.

**Exact permutation over all C(16,8) = 12,870 slide relabellings:**

| quantity | observed difference | exact p | min detectable difference |
|---|---|---|---|
| raw | +0.0857 | 0.461 | 0.250 |
| adjusted for area | −0.1065 | 0.330 | 0.205 |
| **`rho(area, pseudotime)`** | **+0.5168** | **0.000155** | 0.353 |
| raw, within-slide normal scores | +0.0631 | 0.313 | 0.133 |

**No evidence the sections' holeyness correlations differ**, including on the best-powered variant
(MDD 0.133), whose null is exchangeable with its observed value. The one decisive divergence is
`rho(area, pseudotime)`, at the smallest p the design can produce.

The raw-vs-adjusted crossover is arithmetic, not biology — both values reproduce exactly from the
partial-correlation formula, and the only differing input is the *sign* of `rho(area, pseudotime)`.

An exchangeability check came out favourably: null spread is largest at 4/4 (sd 0.113) and shrinks
toward section-pure splits (0.085 at 7/1), so the pooled null is *wider* than a composition-matched
one and "no evidence" is robust.

### 3.9 The repaired axis — a pre-declared mechanistic test that failed

**Hypothesis, recorded before any number:** the `rho(area, pseudotime)` divergence is an artifact of
2M-2's degenerate root set. Tested by repairing the axis — keeping the nuclear-density anchor,
dropping only roots whose leave-one-out Spearman against the median of the others is negative — so
`hole_pct` stays external. Three predictions recorded in advance.

| | before | after | change |
|---|---|---|---|
| 2M-2 `pseudotime_std` | 27.70% | 3.40% | **8.2× reduction** |
| 2M-2 `rho(area, pt)` | −0.0844 | −0.0670 | **+0.017** |
| between-section difference | +0.5168 | +0.4994 | −0.017 |
| exact p | 0.0001554 | **0.0001554** | none — still the floor |

**The hypothesis is refuted.** The repair moved the quantity of interest by 3.4% of the gap it was
meant to explain. Prediction (b) held; (c) failed decisively (0.4994 vs a 0.3528 threshold); (a)
"held" only on the letter of a movement-direction rule at magnitude 0.017 and carries no evidential
weight — **score this 1 of 3, not 2 of 3.**

**Why it failed.** `rho(repaired axis, all-roots axis) = 0.9621`. Dropping three backwards roots
barely changed the axis, because the aggregation is a *median* across 20 roots and a median is
robust to three outliers. The 27.7% `pseudotime_std` was measuring **uncertainty, not bias**.

**The leave-one-out values show more than three bad apples.** Mean LOO concordance: **2M-1 0.726,
2M-2 0.478**, with a fourth 2M-2 root at 0.031 just above the drop threshold. Post-repair, 2M-2
still averages ~0.61. The rule removes only the sign-flipped tail. It also identified patch **1246**,
a third discordant root not visible from outlying pseudotime values.

**Negative control clean.** 2M-1 deltas exactly 0.0000; a gate confirmed the two run trees its
primary and repaired arms derive from carry identical pseudotime (max abs difference 0, Spearman
1.000000).

Small movements went the *wrong* way for a "2M-2 is broken" story: every 2M-2 holeyness correlation
fell slightly (raw 0.1906 → 0.1730) and every between-section difference rose slightly (raw 0.0857
→ 0.1033). All far from significance.

### 3.10 The geometric explanation — ruled out

`rho(area_um2, nuclear_density)` = **+0.3894 (2M-1) vs +0.3422 (2M-2)**. Nearly identical. In both
sections, bigger ducts have higher nuclear density to about the same degree.

So it is **not** the case that low-density patches sit in small ducts in one section and not the
other. The obvious mechanical explanation for the area divergence does not hold.

*Caveat:* this is duct-level and aggregated, while the anchor operates at patch level on the 20
lowest-density patches specifically. A duct-level average could hide a difference in where those
extreme patches sit. The remaining cheap check is the duct-size distribution of the 20 root patches
per section.

---

## 4. Error log

**This is the most valuable section of this document.** Each of these cost real time and would cost
the next person the same.

| # | Error | What it was | How it surfaced | Cost |
|---|---|---|---|---|
| 1 | **`nuclear_density` circularity** | The root selector was simultaneously a validation feature and the cellularity-confound covariate. The axis was partly defined by what it was validated against. | Noticed during root-sensitivity review | Every `nuclear_density` correlation before the fix is uninterpretable as validation |
| 2 | **Silent exception handler in density** | `compute_nuclear_density_quick` returns 0.0 both for genuinely acellular tissue *and* for any patch whose segmentation threw. Segmentation failures were indistinguishable from real measurements. | Traced from 2M-2's 20 roots all sitting at exactly 0.0, none inside a Tumor annotation | Produced the degenerate 2M-2 anchor; ~4 experiments' worth of investigation |
| 3 | **`h_intensity` definitional bug** | Computed over the whole patch rather than masked to nuclei. The legacy definition survives as `h_intensity_wholepatch`. v1 and v2 runs differ *only* in this feature. | Found on feature audit | Any pre-fix `h_intensity` number is a different quantity from the post-fix one under the same name |
| 4 | **Circular flagged-slide selection** | Three "low-signal" slides were selected *by looking at* their weak partial correlations, then tested for properties on that same data. | Caught in the v1–v3b consolidation pass | Invalidated the v3/v3b per-slide inference |
| 5 | **The 0.020 that was a different quantity** | A circulated 2M-2 "holeyness correlation of 0.020" was `rho(pt, nuclear_density)` on the *holeyroot* axis, not `rho(pt, hole_pct)` on v2. The real value is +0.1906 with 7/8 slides positive. | Reconciliation gate in the asymmetry diagnostic | An entire diagnostic was designed to explain an asymmetry that may not exist |
| 6 | **`_safe_rho`'s n ≥ 10 guard on 8 slides** | A shared helper silently returned NaN for every between-slide correlation, which reads in output as "not computed" rather than "n is small." | Synthetic test asserting a known Simpson's paradox | Caught before first real run |
| 7 | **Operator-precedence blanking a verdict** | `a + b + c if p is not None else ""` binds as `(a + b + c) if …`, erasing an entire verdict string whenever one optional field was missing. | Synthetic test | Caught before first real run |
| 8 | **Coordinates assumed to be in `adata.obs`** | Patch x/y live only in `results.csv`; `obs` never carries them. A module built on the wrong assumption crashed on first submission. | Job failure, then reading `run_all.py` | One failed job; fixed with a row-alignment verifier |

**Pattern worth internalising:** five of these eight are *the same kind of error* — a number that
means something other than its name suggests. Density that means "or the segmenter crashed."
`h_intensity` that means two different things across runs. A correlation labelled by the wrong
variable. The defence that worked was **always recomputing from the raw table and reconciling
against the circulated figure**, and it caught #5 immediately.

---

## 5. Statistics that should not be trusted

| Statistic | Status | Why |
|---|---|---|
| Any pre-fix `h_intensity` value | **Superseded** | Different definition; compare only within a run generation |
| `rho(pt, nuclear_density)` on the density-rooted axis | **Circular** | Density selected the roots |
| `rho(pt, hole_pct)` on holeyroot or area-stratified axes | **Circular** | Holeyness selected the roots |
| `rho(pt, duct area)` on any holeyness-anchored axis | **Artifact** | Fully reproduced by size-matched anchors ignoring hole % |
| `texture_entropy` as a validator of a hole-anchored axis | **Not independent** | rho with hole % ≈ +0.35 in both sections |
| `h_intensity_wholepatch` as a validator in 2M-2 | **Not independent** | −0.271 with hole %, partial −0.456 |
| v2 2M-2 `rho(pt, area)` = −0.084 | **Indistinguishable from zero** | CI [−0.200, +0.025], 5/8 slides |
| holeyroot 2M-2 `rho(pt, nuclear_density)` = +0.020 | **Null** | CI [−0.080, +0.098], 33% of draws opposite sign |
| `rho(PT, diffusion-map centroid distance)` | **Definitional** | DPT *is* a diffusion distance; excludable as evidence |
| v3/v3b per-slide "flagged slide" inference | **Invalid** | Circular selection (error #4) |
| `pseudotime_std` as an anchor-health check | **Misleading** | See §7 |
| Anything computed on the `v2_repaired` axis | **Sensitivity only** | Rule applied after the discordance was observed |

**Surviving independent validators for 2M-2:** `mean_nuclear_area` and `nc_ratio`. That is the
whole list.

---

## 6. Parameter provenance

| Parameter | Value | Chosen or default? | Note |
|---|---|---|---|
| Diffusion graph k | **30** | Chosen | **Below the k ≥ 50 plateau reported by Vig. Revisit — see §8.** |
| Diffusion graph metric | euclidean | **scanpy default** | No CLI flag exists; not a deliberate choice |
| Leiden graph | k=15, cosine | Chosen | A *different* graph from the diffusion one |
| DPT roots | 20 | Chosen | Median-aggregated |
| Patch size | 112 px at 5× | Chosen | |
| Patch→duct rule | centre-in-polygon | Chosen | Excludes 26% / 22% of ducts, systematically the smallest |
| PCA width | 261 / 241 | Data-determined | Variance threshold |
| `--aggregation` | median | Default | Same in both sections' primary runs |
| `--n-permutations` | 1000 | Explicit (2M-1), default (2M-2) | Same value either way |
| `--seed` | 42 | Default | |
| `WHITE_THRESH` | 220 | **Inherited convention** | From root-sheet thumbnails. The annotation actually matches ~190–200 — see §3.7 |
| Holeyness pool percentile | P10 | Chosen | **Realised anchor is the bottom 1–3%, not the bottom decile** |
| Eccentricity tail fraction | 0.10 | Chosen | |
| Enrichment threshold | 1.5× | Convention | |
| Directional threshold | \|rho\| ≥ 0.15 | Convention | The "3 features agree" count is sensitive to it |
| `MIN_LATE_PER_SLIDE` | 60 | Chosen | Inherited guard would have fitted k=4 to a 41-patch tail |
| Bootstrap resamples | 2000 | Chosen | Slide is the unit |
| Permutation shuffles | 5000 | Chosen | Reported as `< 1/n`, never 0 |

---

## 7. QC guidance — use LOO concordance, not `pseudotime_std`

These answer different questions and only one is a health check.

**`pseudotime_std`** is, for each patch, the spread of its pseudotime across the 20 individual root
runs. It is a **spread statistic**. It is genuinely useful for saying *this patch's position on the
axis is uncertain* — an uncertainty map for interpretation.

**It failed as a health check, and §3.9 is the proof.** It read 27.7% in 2M-2 and flagged a problem.
The repair dropped it to 3.4%, an 8.2× improvement. And the axis barely moved:
`rho(repaired, all-roots) = 0.9621`. Std was measuring dispersion the median had already absorbed.
It flagged something that did not matter for the output.

**Mean LOO concordance** asks: for each root, how well does the axis *it* produces agree with the
consensus of the other 19? That is a **coherence statistic**, and it caught what std missed — 2M-2's
mean is **0.478** against 2M-1's **0.726**, with a fourth root at 0.031 sitting just above the drop
threshold. That is the finding that 2M-2's anchor is *broadly weakly conditioned*, not three bad
apples. Std never surfaced it.

**Recommendation.** When asking "is this anchor sound?", look at mean LOO concordance across roots.
Keep `pseudotime_std`, but use it for what it is good at — which individual patches have uncertain
positions — not for whether the anchor is healthy.

---

## 8. Where things stand, and what is still open

**Settled.**

- The hole % annotation is a valid, consistently applied measurement with a real pixel referent in
  both sections (rho 0.92 / 0.79 at the pre-specified threshold, 0.98 / 0.96 at the annotator's own,
  unbiased in magnitude near threshold 200).
- The external validation is **positive in both sections under every estimand**, with no evidence
  the sections differ.
- The defect was the **root rule**, not the annotation. The bottom-1–3% rule is duct-size-extreme,
  and that extremity — not holeyness — produced Phase 2's headline result.
- The eccentricity concern is dead outside the diffusion map, and the late-subcluster split is a
  slide effect on every axis tested.
- With the size extremity removed, 2M-2 carries a morphological trajectory (`nc_ratio` 8/8 slides,
  `mean_nuclear_area` 7/8) and 2M-1 does not.

**Open.**

- **Why does `rho(area, pseudotime)` diverge between sections?** +0.4325 vs −0.0844, exact p at the
  design floor. It is anchor-dependent (density +0.517 / repaired +0.499 / holeyroot +0.169 /
  area-stratified +0.246), **not** root-repair-dependent, and **not** explained by duct-size/density
  geometry (§3.10). This is a genuinely odd result and should be reported as an open question.
- **Is the late tail's single-slide concentration (3.3–3.6×) a problem?** Characterised, not fixed.
- **`mean_nuclear_area`'s opposing late subclusters in 4/7 slides** on the de-sized 2M-2 axis —
  real sub-structure or noise at n=7?

---

## 9. Roadmap, with reasoning

**More slides.** Eight per section is the binding constraint on nearly everything above. Minimum
detectable differences run 0.13–0.35; the exact permutation test cannot go below p = 1.55e-4; every
cluster-bootstrap interval and every "no evidence of a difference" inherits n=8. No amount of
analytical care substitutes for this.

**Bridge samples — and block availability is time-sensitive.** Fixation is perfectly collinear with
section: every Carnoy's slide is 2M-1, every PFA slide is 2M-2. No analysis of this cohort can
separate fixation chemistry from anatomical region. Serial sections from a single block, split
across both fixations and stained in one run, are the only way. **Blocks degrade and get consumed —
this is the item on the list with a deadline.**

**Holeyness anchoring, once a second external ground truth exists.** Anchoring on hole % makes
`rho(pt, hole_pct)` circular and destroys the only external validator currently available. That
trade is only worth making when something else can validate the axis. Note that **a learned model
does not substitute for an independent measurement** — a model trained on these images inherits
their biases and cannot serve as external ground truth.

**Learned nuclear segmenter.** Fixes error #2 at source. The current density function returns 0.0
for both acellular tissue and segmentation failure, which is what produced the degenerate 2M-2
anchor. A segmenter that reports failure as failure would have made that visible immediately
instead of after four experiments.

**Per-section stain estimation.** The two sections differ substantially in stain characteristics
and no normalisation is applied. `duct_white_fraction`'s limitations section notes that a
systematically paler slide reads as holier; the within-slide permutation nulls control for it, but
the pooled numbers do not.

**Revisit k.** k=30 sits below the k ≥ 50 plateau. The diffusion graph is the object that fixes the
axis's ordering — random 20-root sets reproduce production pseudotime at |rho| 0.78–0.89 — so the
graph, not the root rule, is where the axis actually comes from. A parameter that important should
not be below a known plateau by default.

---

## Appendix — modules and outputs

| module | purpose | output |
|---|---|---|
| `analysis/anchor_area_control.py` | Tasks A–F: is `rho(pt, area)` a size artifact? | `holeyroot_experiment/anchor_area_control/` |
| `analysis/holeyroot_duct_checks.py` | Nesting, cluster bootstrap, hole-vs-optics | `holeyroot_experiment/duct_checks/` |
| `analysis/eccentricity_check.py` | Trajectory vs eccentricity (pre-existing) | `eccentricity*/` |
| `analysis/eccentricity_within_slide.py` | Within-slide de-confounding, Cramér's V | `eccentricity_within_slide*/` |
| `analysis/export_anchor_axis.py` | Persist derived axes (`area_stratified`, `area_matched_surrogate`, `v2_repaired`) | `holeyroot_experiment/anchor_axes/<anchor>/` |
| `analysis/duct_white_fraction.py` | Annotation vs pixels, rasterised | `holeyness/duct_white_fraction/` |
| `analysis/holeyness_asymmetry.py` | Why does validation differ by section? | `holeyness_asymmetry_diagnostic/` |
| `analysis/holeyness_section_comparison.py` | Four-cell table + exact C(16,8) test | `holeyness_section_comparison/` |
| `analysis/holeyness_repaired_sensitivity.py` | Repaired-axis sensitivity, side by side | `holeyness_repaired_sensitivity/` |

All are read-only on existing run trees; none re-embeds or re-runs the pipeline. Alternative anchors
re-run only `sc.tl.dpt` on the stored graph and diffusion map. Every module was verified against
synthetic data with known ground truth before its first real run, and several real defects were
caught that way (errors #6, #7 above).
