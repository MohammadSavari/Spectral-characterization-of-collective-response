#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --job-name=figures
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/figures_%j.out
#
# Regenerates Figs. 1, 3, S3, S4 and S5 from nets/ into output/. Figs. 2, S1 and
# S2 come from scripts/spec_div/run_paper_figs.sh. Submit from the repo root:
#
#     mkdir -p logs
#     sbatch scripts/figures/run_figures.sh
#
# Outputs:
#   output/spec_div/top_pair_ws.pdf                                   Fig. 1
#   output/figures/WS_16_240_lfc_lazy_metro_bc_portrait_CRgSK_eps0.2.pdf  Fig. 3
#   output/figures/MHK_16_240_lfc_lazy_metro_bc_portrait_CRgSK_eps0.2.pdf Fig. S3
#   output/figures/WS_BC_supplement.pdf                               Fig. S4
#   output/figures/MHK_BC_supplement.pdf                              Fig. S5
# Figures_CRgSK.ipynb also writes the eps = 0.05 / 0.3 portraits and the landscape
# variants, which are not paper figures.

set -uo pipefail

source <path-to-venv>/bin/activate
module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10
export MPLBACKEND=Agg
# Fig. 1 re-diagonalises its two graphs; the published figure was rendered with 4
# BLAS threads, and a different thread count changes the eigenvalues in the last
# bits, enough to move panel (b)'s automatic y-limit by under a pixel.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p logs output/figures/executed

rc=0
python scripts/spec_div/plot_top_pair.py --family ws || rc=1

# Executed copy goes to output/; the notebook resolves the repo root itself.
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 \
    --output-dir output/figures/executed scripts/figures/Figures_CRgSK.ipynb || rc=1

python scripts/figures/gen_bc_supplement.py || rc=1

echo "ALL_FIGURES_EXIT=${rc}"
exit "$rc"
