# `scripts/` - generation and figure pipeline

Everything that produces `nets/` (the data) and `output/` (regenerated figures).
None of it is needed just to look at the data: `bash reassemble.sh all` from the
repo root unpacks the shipped `nets/` directly.

| Folder | Produces | Figures |
|---|---|---|
| [`spec_div/`](spec_div/README.md) | matched graph pairs, their H² gains, the pair CSVs; Figs. 2/S1/S2 | 1, 2, S1, S2 |
| [`LFC/`](LFC/README.md) | WS/MHK ensembles with LFC/lazy/Metropolis gains and spectra; the bounded-confidence sweep; per-seed CSVs | (data for 3, S3–S5) |
| [`figures/`](figures/README.md) | the consensus-model figures from `nets/LFC/` | 3, S3, S4, S5 |

## Run order

Spectral divergence (Figs. 1, 2, S1, S2):

1. `spec_div/run_spec_div.sh` + `spec_div/stop_at_target.sh`: generate 700 matched pairs per seed, 20 seeds per family
2. `spec_div/run_gain_all.sh`: add the all-node collective response H² to every pair graph
3. `spec_div/spec_div_csv_gen.py`: flatten the graphs to the eigenvalue and H² CSVs
4. `spec_div/run_paper_figs.sh` (Figs. 2, S1, S2) and `spec_div/plot_top_pair.py` (Fig. 1)

Consensus models (Figs. 3, S3–S5):

1. `LFC/submit_diff_weights.sh`: generate the WS/MHK ensembles and their LFC, lazy and Metropolis gains
2. `LFC/submit_bc_rms_leaders_*.sh`: bounded-confidence sweeps on the k = 16, seed-1 graphs (and WS seed 2)
3. `LFC/submit_gen_seed_csvs.sh`: extract the per-seed props / gains / bc_gains CSVs
4. `figures/run_figures.sh`: Figs. 1, 3, S3, S4, S5

Each step reads only what the previous step wrote under `nets/`, so any step can
start from the shipped data instead of regenerating its inputs.

## Common prerequisites

- **Run from the repository root.** Every wrapper resolves paths relative to it
  (`$SLURM_SUBMIT_DIR` under SLURM), reads from `nets/` and writes figures to `output/`.
- `mkdir -p logs` before submitting; SLURM writes job logs there.
- The SLURM headers carry placeholders: replace `<your-account>`, `<your-email>`
  and `<path-to-venv>` with your own values (or pass `--account=...` on the
  `sbatch` command line).
- Environment, in this order:

  ```bash
  source <path-to-venv>/bin/activate
  module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10
  ```

  Off-cluster, the conda environment in the top-level README replaces both.
- **Never run the generators, gain computations or notebooks on a login node.**
  Use `sbatch` or an interactive allocation (`salloc`).
- Large array jobs: check queue headroom first (`squeue -u $USER -r | tail -n +2 | wc -l`).
  The `submit_*.sh` helpers throttle themselves below a 1000-task cap and are resumable.
