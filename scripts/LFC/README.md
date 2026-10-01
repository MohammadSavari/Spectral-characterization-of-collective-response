# `scripts/LFC/` - WS/MHK ensembles and consensus-model responses

Builds the Watts–Strogatz (`ws`) and Holme–Kim (`mhk`) network ensembles at
N = 240 over a 100-point rewiring sweep `p = linspace(0.001, 1, 100)`, computes the
collective response of four consensus models on every graph, and extracts the
per-seed tables that Figs. 3 and S3–S5 read.

## Contents

| File | Purpose |
|---|---|
| `LFC_net_gen_diff_weights.py` | One graph per `(model_type, k, p, seed)`: Laplacian (LFC), Metropolis–Hastings and lazy-random-walk gains with every node as leader, plus normalised and unnormalised Laplacian spectra, stored on the `.gt`. |
| `run_diff_weights.sh` / `submit_diff_weights.sh` | SLURM array + resumable, queue-throttled submitter. Seed 1 sweeps k ∈ {2, 4, …, 32} × 100 p; seeds 2–100 sweep k = 16 × 100 p. |
| `bc_rms_sweep.py` | Bounded-confidence model: drives one leader, converges each frequency, and stores the per-leader collective RMS on the `.gt` as `bc_rms_eps{ε}`. Resumable per (leader, frequency). |
| `run_bc_rms_leaders_*.sh` / `submit_bc_rms_leaders_*.sh` | One task per (graph, leader): 100 graphs × 240 leaders = 24,000 tasks per sweep. |
| `LFC_diffw_props_csv_gen.py` | Extracts `<k>_seed<N>_props.csv`, `_gains.csv` and (where BC data exists) `_bc_gains.csv` beside each seed directory. |
| `run_gen_seed_csvs.sh` / `submit_gen_seed_csvs.sh` | SLURM wrapper for the extraction (ws and mhk as a 2-task array). |
| `lfc_data_loader.py` | Pooling helpers used by the figure notebook (`pooled_props`, `pooled_gains_all`, `find_graph_for_p`). |

## Bounded-confidence sweeps

All on the k = 16 graphs, every node as leader:

| Wrapper | Graphs | ε | Frequencies |
|---|---|---|---|
| `run_bc_rms_leaders_ws16.sh` | `ws/16_seed1` | 0.2 | 60, ω ∈ [10⁻³, 1] |
| `run_bc_rms_leaders_mhk16.sh` | `mhk/16_seed1` | 0.2 | 60, ω ∈ [10⁻³, 1] |
| `run_bc_rms_leaders_ws16_seed2.sh` | `ws/16_seed2` | 0.2 | 60, ω ∈ [10⁻³, 1] |
| `run_bc_rms_leaders_{ws16,mhk16}_eps005.sh` | `16_seed1` | 0.05 | 40, ω ∈ [10⁻³, 10^−0.699] |
| `run_bc_rms_leaders_{ws16,mhk16}_eps03.sh` | `16_seed1` | 0.3 | 40, ω ∈ [10⁻³, 10^−0.699] |

The ε = 0.05 / 0.3 grids must stay exactly as written (`--fmax -0.699`, not
`log10(0.2)`): `bc_rms_sweep.py` refuses to append to a graph whose stored grid differs.

## Usage (from the repo root)

```bash
mkdir -p logs
MODEL_TYPE=ws  bash scripts/LFC/submit_diff_weights.sh       # 11,500 tasks per family
MODEL_TYPE=mhk bash scripts/LFC/submit_diff_weights.sh
bash scripts/LFC/submit_bc_rms_leaders_ws16.sh               # one per sweep in the table
FORCE=1 bash scripts/LFC/submit_gen_seed_csvs.sh             # FORCE re-reads BC data from the .gt
```

The `submit_*` helpers keep the queue below 900 tasks, split arrays above the
cluster's 10,000-task `MaxArraySize`, and record progress in a `.submit_*` file in
the repo root, so a plain re-run resumes.

## What ships

All 463 per-seed CSVs (every k and seed, both families) and the k = 16 graphs
that carry bounded-confidence results: `ws/16_seed1`, `ws/16_seed2` and
`mhk/16_seed1` (100 graphs each). These are also the graphs Figs. 3/S3 read for
their eigenvalue histograms. The remaining 22,700 graphs (6.7 GB WS, 6.8 GB MHK) are
not shipped; every one of them is summarised in the shipped CSVs, which are what the
figures pool. The MHK generator is a verbatim copy of `net_functions.mhk_network`
in the companion Scale-Free-Social-Contagion repository (2m = round(N·k) exactly,
degree-preferential attachment).

## Regenerating the unshipped graphs

A graph is a deterministic function of `(model_type, k, p, seed)`, with
`p = numpy.linspace(0.001, 1, 100)[p_idx]`. Regenerating it reproduces the original
**edge for edge** (checked against the originals for both families, k = 8 and 16,
seeds 1, 3 and 57), together with its gains and spectra. Only the filename differs:
it embeds the SLURM job and task IDs (`<ID>_<jobid>_<taskid>.gt`). The generator
does not run the bounded-confidence model, so a regenerated graph carries no
`bc_rms_*` data; that exists only for the shipped seeds.

Run everything from the repo root on a compute node (about 6 min per graph,
dominated by the gain computation).

**One seed at k = 16** (seeds 2–100): one array task per `p_idx`:

```bash
mkdir -p logs
sbatch --array=0-99 --export=ALL,SEED=57,MODEL_TYPE=ws scripts/LFC/run_diff_weights.sh
```

**Seed 1 at every k**: 16 k × 100 p, where task `t` is `k = 2·(t // 100 + 1)`, `p_idx = t % 100`:

```bash
sbatch --array=0-1599 --export=ALL,SEED=1,MODEL_TYPE=mhk scripts/LFC/run_diff_weights.sh
```

**Every seed** (11,500 tasks per family, queue-throttled and resumable):
`MODEL_TYPE=ws bash scripts/LFC/submit_diff_weights.sh` (pass a start seed as `$1`).

**A single graph**:

```bash
P=$(python -c "import numpy as np; print(np.linspace(0.001, 1, 100)[42])")
python scripts/LFC/LFC_net_gen_diff_weights.py --seed 57 --model_type ws --k 16 --p "$P"
```

Output goes to `nets/LFC/240/<model_type>/<k>_seed<seed>/`, beside the shipped CSVs.
Do not regenerate seeds whose graphs you already unpacked (`ws/16_seed1`, `ws/16_seed2`,
`mhk/16_seed1`): the copies would sit next to the originals under different names, and
`LFC_diffw_props_csv_gen.py --force` would then count each graph twice.

To check a regenerated seed against the release, extract its tables into a scratch
tree and compare them with the shipped CSVs of the same name, matching rows on `p`
(the `ID` column comes from the filename, so it differs):

```bash
mkdir -p /tmp/check/LFC/240/ws
cp -r nets/LFC/240/ws/16_seed57 /tmp/check/LFC/240/ws/
python scripts/LFC/LFC_diffw_props_csv_gen.py --network ws --nets-root /tmp/check --lfc-flat-root /tmp/check/flat
# compare /tmp/check/LFC/240/ws/16_seed57_{props,gains}.csv with nets/LFC/240/ws/16_seed57_{props,gains}.csv
```
