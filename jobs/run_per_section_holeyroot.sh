#!/bin/bash
# PHASE 2 — holey-ness-rooted pseudotime, BOTH sections.
#
# Identical to jobs/run_per_section_v2.sh in every respect except ROOT SELECTION:
# same cached Phikon features, --stain-method none, --batch-method none,
# --cap-strategy median, --n-roots 20, Leiden k=15 cosine (module default),
# diffusion k=30 euclidean / 10 comps, same per-root inf-clamping, same median
# aggregation, same min-max normalisation.
#
# ── THE ROOT RULE, FIXED HERE AND NOT TUNED ─────────────────────────────────
#   percentile          P10 of the PER-DUCT hole % distribution (matches v3a).
#                       A percentile, not the strict minimum, so the anchor is
#                       not a handful of extreme ducts.
#   patch -> duct       centre-in-polygon, via holeyness.assign_patches_to_ducts
#                       reused unmodified. Matches v3a exactly, which is what
#                       makes 2M-1's new run comparable to its old one.
#   no-duct patches     EXCLUDED from root candidacy. NOT assigned hole % = 0 —
#                       that would make them the lowest-holeyness patches in the
#                       cohort and therefore the PREFERRED roots, which is the
#                       precise failure this anchor exists to avoid.
#   min patches/duct    1
#   max roots/duct      1 (one root per duct, so 20 roots span 20 ducts)
#   tie-break           object UUID string order. ⚠ EXPLICITLY NOT
#                       nuclear_density: removing that variable from root
#                       selection is the entire point of this experiment, and
#                       breaking ties on it would quietly reinstate the
#                       circularity.
#   degenerate pool     HARD ERROR. If every duct at/below the threshold shares
#                       one hole %, "lowest holeyness" orders nothing and the
#                       arbitrary tie-break picks the roots. The correct response
#                       is to widen the percentile, never to fall back silently.
#                       Verified locally for 2M-2: P10 = 6.05, 178 ducts strictly
#                       below, 0 ties. Not degenerate.
#   topology            Roots are asserted to occupy ONE connected component of
#                       the k=30 euclidean DPT graph; the job aborts otherwise.
#                       Automatic via run_all's holeyness path.
#
# ── ⚠ THE TWO SECTIONS READ DIFFERENT EXPORT FILES ──────────────────────────
#   2M-1: the original merged TSV, holes_carnoys: prefix (Carnoy's-fixed).
#   2M-2: the converted TSV from the per-slide GeoJSON, whose header was renamed
#         holes_pfa: -> holes_carnoys: so holeyness.py can read it. THE HEADER
#         MISSTATES THE FIXATIVE for 2M-2; values are untouched. See the
#         .provenance.json sidecar beside that file.
#   Carnoy's and PFA differ in shrinkage, so hole % distributions may differ
#   between sections for FIXATION reasons alone, independent of biology. That is
#   a live confound for any cross-section claim this run produces.
#
# ── WHAT IS AND IS NOT EVIDENCE ─────────────────────────────────────────────
#   rho(pseudotime, hole_pct) will rise under this anchor. That is PARTLY
#   CIRCULAR — the anchor IS holeyness — and is NOT evidence the anchor is
#   better. The non-circular tests are rho(pseudotime, duct area) and
#   rho(pseudotime, nuclear_density); neither is used in root selection here.
#
# READS (READ-ONLY): $SCRATCH/data/features_cache (never written; aborts if
#                    incomplete), MCF7_x5_cropped, data/annotations_ratio,
#                    both holeyness exports
# WRITES (NEW ONLY): $SCRATCH/results/per_section_holeyroot/atlas_2M-{1,2}
#   per_section/, per_section_v2/, holeyness/ are NEVER written.
#
# WALLTIME/MEMORY: inherited from jobs/run_per_section_v2.sh (8h/64G/8cpu), which
#   itself documents those as an upper bound carried over from a larger job, not
#   a measurement. No sacct record was recoverable. Substitute real numbers after
#   the first run:
#     sacct -X --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,State --name=holeyroot
#
# Run both sections as independent parallel jobs:
#     sbatch --export=ONLY_SECTION=2M-1 jobs/run_per_section_holeyroot.sh
#     sbatch --export=ONLY_SECTION=2M-2 jobs/run_per_section_holeyroot.sh
# or omit ONLY_SECTION to run them sequentially in one job.

#SBATCH --account=def-lmarti46
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=holeyroot
#SBATCH --output=logs/holeyroot-%j.out

set -euo pipefail
mkdir -p logs

REPO="$HOME/cancer_trajectory_atlas"

LEIDEN_RES=0.5
N_ROOTS=20
N_PERMUTATIONS=1000
PATCH_SIZE=112
STRIDE=96

PERCENTILE=10
MIN_PATCHES_PER_DUCT=1
MAX_ROOTS_PER_DUCT=1
# Pre-specified primary = centre. 'overlap' exists for a documented follow-up and
# is deliberately NOT the default: it is v3d's rule, which has not been run, and
# using it would break "identical to v2 except the root rule".
ASSIGNMENT="${HOLEYNESS_ASSIGNMENT:-centre}"

CACHE_DIR="$SCRATCH/data/features_cache"
PNG_DIR="$SCRATCH/data/MCF7_x5_cropped"
ANN_DIR="$REPO/data/annotations_ratio"
SLIDE_DIMS="$PNG_DIR/slide_dimensions.json"
OUT_BASE="$SCRATCH/results/per_section_holeyroot"

# PROTECTED — never written here.
PROTECTED=(
    "$SCRATCH/results/per_section"
    "$SCRATCH/results/per_section_v2"
    "$SCRATCH/results/holeyness"
)

# The two sections read DIFFERENT export files. See the header.
EXPORT_2M1="$SCRATCH/data/holeyness/raw/combined_matched_measurements.txt"
EXPORT_2M2="$SCRATCH/data/holeyness/2M-2_converted/2M-2_measurements_COLUMN_RENAMED_holes_pfa_to_holes_carnoys.tsv"

SLIDES_2M_1=(
    6027-4L-2M-1_x5  6027-4R-2M-1_x5  6028-4L-2M-1_x5  6028-4R-2M-1_x5
    6029-4L-2M-1_x5  6029-4R-2M-1_x5  6031-4L-2M-1_x5  6031-4R-2M-1_x5
)
SLIDES_2M_2=(
    6027-4L-2M-2_x5  6027-4R-2M-2_x5  6028-4L-2M-2_x5  6028-4R-2M-2_x5
    6029-4L-2M-2_x5  6029-4R-2M-2_x5  6031-4L-2M-2_x5  6031-4R-2M-2_x5
)

if [ -n "${ONLY_SECTION:-}" ]; then SECTIONS=("$ONLY_SECTION"); else SECTIONS=("2M-1" "2M-2"); fi

echo "============================================================================"
echo "  PHASE 2 — holey-ness-rooted pseudotime"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Sections   : ${SECTIONS[*]}"
echo "  Output base: $OUT_BASE   (NEW)"
echo "  Root rule  : P${PERCENTILE} of per-duct hole %, assignment=${ASSIGNMENT},"
echo "               max ${MAX_ROOTS_PER_DUCT} root/duct, tie-break = UUID (NOT nuclear_density)"
echo "  Degenerate pool -> HARD ERROR (no --holeyness-allow-degenerate-pool)"
echo "============================================================================"
echo ""
echo "  rho(pseudotime, hole_pct) rising is PARTLY CIRCULAR and is NOT evidence."
echo "  The non-circular tests are rho(pt, duct area) and rho(pt, nuclear_density)."
echo "============================================================================"

for SECTION in "${SECTIONS[@]}"; do
    OUT_DIR="$OUT_BASE/atlas_${SECTION}"
    for p in "${PROTECTED[@]}"; do
        case "$OUT_DIR" in
            "$p"|"$p"/*) echo "ERROR: '$OUT_DIR' is inside protected tree '$p'."; exit 1;;
        esac
    done
    if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
        echo "ERROR: $OUT_DIR exists and is non-empty. Refusing to overwrite."
        exit 1
    fi
done

echo ""
echo "=== Pre-run checks ==="
MISSING=0
for p in "$CACHE_DIR" "$PNG_DIR" "$ANN_DIR" "$SLIDE_DIMS"; do
    echo -n "  $p : "; if [ -e "$p" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
done
for SECTION in "${SECTIONS[@]}"; do
    if [ "$SECTION" = "2M-1" ]; then E="$EXPORT_2M1"; S=("${SLIDES_2M_1[@]}");
                              else E="$EXPORT_2M2"; S=("${SLIDES_2M_2[@]}"); fi
    echo -n "  export $SECTION : "; if [ -e "$E" ]; then echo "ok"; else echo "NOT FOUND"; MISSING=1; fi
    for SLIDE in "${S[@]}"; do
        [ -f "$CACHE_DIR/${SLIDE}_features.npy" ] || {
            echo "  ERROR: feature cache miss for $SLIDE"; MISSING=1; }
    done
done
[ "$MISSING" -eq 0 ] || {
    echo "ERROR: missing inputs. This job is CPU-only and will not fall back to"
    echo "       GPU inference; populate the cache with run_cache_population.sh."
    exit 1; }
echo "  Phikon cache complete; export files present."

CACHE_BEFORE=$(find "$CACHE_DIR" -name '*_features.npy' -printf '%f %s\n' | sort | md5sum | cut -d' ' -f1)
echo "  Production cache fingerprint (pre-run): $CACHE_BEFORE"
echo "============================================"

module load StdEnv/2023 python/3.11 gcc opencv openslide openblas hdf5 igraph
source ~/envs/atlas/bin/activate
export HF_HOME=$SCRATCH/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
mkdir -p "$OUT_BASE"
cd ~

for SECTION in "${SECTIONS[@]}"; do
    echo ""
    echo "============================================"
    echo "  Section: $SECTION"
    echo "============================================"
    if [ "$SECTION" = "2M-1" ]; then
        EXPORT="$EXPORT_2M1"; SECTION_SLIDES=("${SLIDES_2M_1[@]}")
    else
        EXPORT="$EXPORT_2M2"; SECTION_SLIDES=("${SLIDES_2M_2[@]}")
        echo "  NOTE: this export's header says holes_carnoys: but the values are"
        echo "        PFA measurements. See its .provenance.json."
    fi
    SLIDES_CSV=$(IFS=,; echo "${SECTION_SLIDES[*]}")
    OUT_DIR="$OUT_BASE/atlas_${SECTION}"
    mkdir -p "$OUT_DIR"

    python -m cancer_trajectory_atlas.run_all \
        --run \
        --png-dir                      "$PNG_DIR" \
        --annotation-dir               "$ANN_DIR" \
        --output-dir                   "$OUT_DIR" \
        --stain-method                 none \
        --batch-method                 none \
        --model                        phikon \
        --patch-size                   "$PATCH_SIZE" \
        --stride                       "$STRIDE" \
        --clustering-method            leiden \
        --leiden-resolution            "$LEIDEN_RES" \
        --n-roots                      "$N_ROOTS" \
        --n-permutations               "$N_PERMUTATIONS" \
        --features-cache-dir           "$CACHE_DIR" \
        --cap-strategy                 median \
        --slides                       "$SLIDES_CSV" \
        --root-source                  holeyness \
        --holeyness-export             "$EXPORT" \
        --holeyness-slide-dims         "$SLIDE_DIMS" \
        --holeyness-assignment         "$ASSIGNMENT" \
        --holeyness-percentile          "$PERCENTILE" \
        --holeyness-min-patches         "$MIN_PATCHES_PER_DUCT" \
        --holeyness-max-roots-per-duct  "$MAX_ROOTS_PER_DUCT"

    echo ""
    echo "  --- root selection, section $SECTION ---"
    python - "$OUT_DIR" <<'PY'
import json, sys
try:
    d = json.load(open(f"{sys.argv[1]}/holeyness_roots.json"))
except Exception as e:
    sys.exit(f"  (holeyness_roots.json unreadable: {e})")
c = d["counts"]
print(f"  patches in no duct   : {c['n_patches_no_duct']}/{c['n_patches_total']} "
      f"({100*c['frac_patches_no_duct']:.1f}%)  [EXCLUDED from candidacy]")
print(f"  ducts with 0 patches : {c['n_ducts_with_zero_patches']}/{c['n_ducts_in_table']}")
print(f"  candidate pool       : {c['n_ducts_in_candidate_pool']} ducts "
      f"({c['n_pool_ducts_strictly_below_threshold']} strictly below threshold)")
print(f"  hole% P{c['percentile']:g} threshold : {c['hole_pct_threshold']:.4f}")
print(f"  degenerate pool      : {c['pool_is_degenerate_all_at_threshold']}")
print(f"  distinct ducts among roots: {c['n_distinct_ducts_among_roots']}/20")
t = d.get("topology", {})
if t:
    print(f"  graph components     : {t['n_graph_components']}   "
          f"spanned by roots: {t['n_components_spanned_by_roots']}")
    print(f"  root tightness vs random: {t['tightness_ratio_vs_random']:.3f}")
PY

    echo ""
    echo "  --- extraction failures ---"
    python -c "
import json,sys
d=json.load(open('$OUT_DIR/feature_failures.json'))
print('  quick n_failed =', d['nuclear_density_quick']['n_failed'])
print('  full  n_failed =', d['morphological_features']['n_failed'])" \
        2>/dev/null || echo "  (feature_failures.json not readable)"
done

CACHE_AFTER=$(find "$CACHE_DIR" -name '*_features.npy' -printf '%f %s\n' | sort | md5sum | cut -d' ' -f1)
echo ""
if [ "$CACHE_BEFORE" != "$CACHE_AFTER" ]; then
    echo "  WARNING: the production cache CHANGED during this run. It should not have."
else
    echo "  Production cache unchanged, as expected."
fi

echo ""
echo "============================================================================"
echo "  PHASE 2 RUNS COMPLETE — $OUT_BASE"
echo "  Next: jobs/run_holeyroot_compare.sh, then jobs/run_holeyroot_root_inspection.sh"
echo ""
echo "  Look for 'CLAMPING FIRED' above. 2M-2's v2 pseudotime_std was reportedly"
echo "  27.7% of range; if the clamp is firing under the new anchor too, that is"
echo "  where to look first."
echo "============================================================================"
