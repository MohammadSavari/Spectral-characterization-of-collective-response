# `scripts/figures/` - consensus-model figures (Figs. 3, S3, S4, S5)

| File | Produces |
|---|---|
| `Figures_CRgSK.ipynb` | Figs. 3 (WS) and S3 (MHK): collective response and Spearman correlation with C, R_g, S, K for the LFC, lazy random-walk, Metropolis–Hastings and bounded-confidence (ε = 0.2) models, plus the normalised and unnormalised eigenvalue histograms. Also writes the ε = 0.05 / 0.3 portraits and landscape variants (not paper figures). |
| `gen_bc_supplement.py` | Figs. S4 (WS) and S5 (MHK): the bounded-confidence rows at ε = 0.05 (row 1) and ε = 0.3 (row 2). |
| `run_figures.sh` | SLURM job running Fig. 1 (`../spec_div/plot_top_pair.py`), the notebook and `gen_bc_supplement.py`. |

Run from the repo root (`sbatch scripts/figures/run_figures.sh`). The notebook and
script find the repo root themselves (or take `REPO_ROOT`), read `nets/LFC/240/`,
and write to `output/figures/`. Run from the shipped data, every figure is
pixel-identical to the published one. `run_figures.sh` pins 4 BLAS threads because
Fig. 1 re-diagonalises its graphs, and another thread count shifts panel (b)'s
automatic y-limit by under a pixel (the data and printed values are unaffected).

What is pooled: rows 1–3 (the three linear models) use all 100 k = 16 seeds per
family; the bounded-confidence row uses the rewiring sweep of seed 1, since each
frequency of that nonlinear model needs a time-domain simulation per leader. The
histograms use the seed-1 graphs at the four marked rewiring probabilities.
