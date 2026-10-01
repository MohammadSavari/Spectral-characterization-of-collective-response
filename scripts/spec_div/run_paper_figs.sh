#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --job-name=paper_figs
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/paper_figs_%j.out
#
# Regenerate the publication figures (Figs. 2, S1, S2) under
# output/spec_div/paper_figures/ -- the versions that go into the paper, as opposed
# to the annotated working figures under output/spec_div/figs_*/. Reads the CSVs in
# nets/spec_div/. Submit from the repo root:
#
#     mkdir -p logs
#     sbatch scripts/spec_div/run_paper_figs.sh
#
# Four runs: {ws, mhk} x {unnormalised, normalised}. Each executes an analyser
# notebook once, on one family only, under three environment variables read in
# cell 1:
#
#   SPEC_DIV_FAMILY=ws|mhk|both   which family to load. A single family means
#                                 FAMS has one entry, every per-family panel
#                                 collapses to one column with its own colour
#                                 scale, and the cross-family comparison cell
#                                 skips itself via its len(FAMS) >= 2 guard --
#                                 so no statistic is computed across families.
#                                 Default "both" reproduces figs/ and figs_norm/.
#
#   SPEC_DIV_PAPER_FIGS=1         suppress fig.suptitle() and every
#                                 ax.set_title(), which duplicate the journal
#                                 caption, and draw a bold panel letter above
#                                 each panel instead (Fig. 1's style). Redirects
#                                 output to output/spec_div/paper_figures/.
#
#   SPEC_DIV_PANEL_OFFSET=second  continue the panel-letter sequence instead of
#                                 restarting it. Each figure is used in the paper
#                                 as one half of a two-family figure, so WS is
#                                 lettered (a), (b), ... and MHK picks up where
#                                 it left off. The offset is the figure's own
#                                 panel count, so a two-panel figure gives WS
#                                 (a),(b) and MHK (c),(d), while the four-panel
#                                 control gives (a)-(d) and (e)-(h). Without
#                                 this, MHK would restart at (a) and collide
#                                 with the caption's own (a)/(b) for the two
#                                 halves.
#
# The notebooks are executed through a throwaway copy (_paperfig_*.ipynb, deleted
# after each run) so the shared sources keep the saved outputs they were
# committed with rather than being overwritten with single-family state.
#
# Figure 1 of the paper is NOT produced here -- it comes from plot_top_pair.py,
# which is a plain script:
#     python scripts/spec_div/plot_top_pair.py --family ws
#
# Runtime is roughly 10-15 min per run; the memory request is driven by the
# ~130 MB eigenvalue/H2 CSVs and the wide pivot the notebook builds from them.
# Never run this on the login node.

set -uo pipefail

source <path-to-venv>/bin/activate
module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10
export MPLBACKEND=Agg

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "$REPO/logs"
# nbconvert executes a notebook with its own directory as cwd, which is what the
# notebooks' ../../nets and ../../output defaults are relative to.
cd "$REPO/scripts/spec_div"
PF="$REPO/output/spec_div/paper_figures"

rc=0
for FAM in ws mhk; do
    # ws is the first half of every paired figure, mhk the second.
    OFFSET=""
    [ "$FAM" = "mhk" ] && OFFSET="second"

    for NB in analyzer_corr_gen analyzer_corr_gen_norm; do
        OUT="_paperfig_${NB}_${FAM}.ipynb"
        cp "${NB}.ipynb" "$OUT"
        echo "=========== ${NB} ${FAM} (panel offset='${OFFSET}') ==========="
        SPEC_DIV_FAMILY="$FAM" SPEC_DIV_PAPER_FIGS=1 SPEC_DIV_PANEL_OFFSET="$OFFSET" \
            jupyter nbconvert --to notebook --execute --inplace \
                --ExecutePreprocessor.timeout=7200 "$OUT" 2>&1 | tail -6
        status=${PIPESTATUS[0]}
        echo "exit=${status} for ${NB} ${FAM}"
        [ "$status" -ne 0 ] && rc=1
        rm -f "$OUT"
    done
done

echo "--- $PF/ ---"
for d in "$PF"/*/; do
    [ -d "$d" ] && echo "  $d $(ls "$d"/*.pdf 2>/dev/null | wc -l) pdf"
done

# Panel letters, as a check that the offset landed: expect (a)(b) / (c)(d) for the
# two-panel figures and (a)-(d) / (e)-(h) for matched_eigenvalue_effect.
if command -v pdftotext >/dev/null 2>&1; then
    echo "--- panel letters ---"
    for f in ws/eigenvalue_frequency_response mhk/eigenvalue_frequency_response \
             ws/matched_eigenvalue_effect mhk/matched_eigenvalue_effect; do
        [ -f "$PF/$f.pdf" ] || continue
        printf '  %-42s' "$f"
        pdftotext -q "$PF/$f.pdf" - 2>/dev/null \
            | grep -oE '^\([a-h]\)$' | tr '\n' ' '
        echo
    done
fi

echo "ALL_PAPER_FIGS_EXIT=${rc}"
exit "$rc"
