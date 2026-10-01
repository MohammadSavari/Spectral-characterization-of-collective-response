# `scripts/spec_div/` - matched-pair spectral divergence (Figs. 1, 2, S1, S2)

Graph pairs `(G1, G2)` that agree on λ₂, λ_N **and edge count** but are made as
different as possible in the rest of the Laplacian spectrum, so any difference in
their collective response H² has to come from the intermediate eigenvalues.
References are drawn from the two network families studied in the paper,
Watts–Strogatz (`ws`) and Holme–Kim (`mhk`), at N = 240 and ⟨k⟩ = 16.

## Contents

| File | Purpose |
|---|---|
| `spectral_divergence_gen.py` | Generator: reference graph, then simulated annealing over edge relocations toward maximal bulk divergence under the λ₂/λ_N/m constraints. |
| `run_spec_div.sh` | SLURM array wrapper, one task per seed; `FAMILY` selects ws or mhk. |
| `stop_at_target.sh` | Watcher that cancels each seed's task once it reaches `TARGET` (700) pairs. |
| `gain_laplacian_all_gen.py` / `run_gain_all.sh` | Adds the all-node collective response H²(ω) to every pair graph. |
| `spec_div_csv_gen.py` | Flattens a family's graphs to `all_graph_eigenvalues_{family}.csv` and `all_graphs_h2_data_{family}.csv`. |
| `analyzer_corr_gen.ipynb` | Correlation analysis on the unnormalised Laplacian `L = D − A` (Figs. 2, S2). |
| `analyzer_corr_gen_norm.ipynb` | The same on the normalised Laplacian `I − D^(−1/2) A D^(−1/2)` (Fig. S1). |
| `run_paper_figs.sh` | Executes both notebooks once per family in publication mode → `output/spec_div/paper_figures/`. |
| `plot_top_pair.py` | Fig. 1: the pair with the largest bulk divergence in a family. |
| `rebuild_pairs_index.py` | Rebuilds every `{family}_seed_N/pairs_index.csv` from `pair_properties_per_pair.csv` (see "Provenance" below). |
| `checks/check_mhk_equiv.py`, `checks/check_equiv.sh` | Asserts the copied MHK generator is edge-for-edge identical to the companion repository's `net_functions.mhk_network`. |

## Usage (from the repo root)

```bash
mkdir -p logs

# 1. generate (one array per family, 20 seeds each), then cap every seed at 700 pairs
FAMILY=ws  sbatch --array=1-20 scripts/spec_div/run_spec_div.sh
FAMILY=mhk sbatch --array=1-20 scripts/spec_div/run_spec_div.sh
FAMILY=ws  ARRAY=<ws jobid>  sbatch scripts/spec_div/stop_at_target.sh
FAMILY=mhk ARRAY=<mhk jobid> sbatch scripts/spec_div/stop_at_target.sh

# 2. collective-response gains
FAMILY=ws  sbatch --array=1-20 scripts/spec_div/run_gain_all.sh
FAMILY=mhk sbatch --array=1-20 scripts/spec_div/run_gain_all.sh

# 3. CSVs (on a compute node); the normalised spectrum is a second eigenvalue CSV
for fam in ws mhk; do
  python scripts/spec_div/spec_div_csv_gen.py --family $fam --root nets/spec_div --out-dir nets/spec_div
  python scripts/spec_div/spec_div_csv_gen.py --family $fam --root nets/spec_div --out-dir nets/spec_div \
      --eig-prop eig_laplacian_norm --eig-name all_graph_eigenvalues_norm_$fam.csv --no-h2
done

# 4. figures
sbatch scripts/spec_div/run_paper_figs.sh                 # Figs. 2, S1, S2
python scripts/spec_div/plot_top_pair.py --family ws      # Fig. 1 (compute node)
```

Seed directories are `nets/spec_div/{family}_seed_{N}/`. If a seed ends short of
700 pairs, top it up with `RESUME=1` rather than resubmitting plain (which wipes
the directory): `FAMILY=mhk RESUME=1 sbatch --array=3,7 scripts/spec_div/run_spec_div.sh`.
Before step 1, `checks/check_equiv.sh` should report `EQUIV_CHECK_RESULT=PASS`.

## Seeding

`--seed` indexes the run and names the output directory. The randomness of each
reference is derived from its **index**, not from the position of a shared stream:

```
p        = P_GRID[ default_rng(SeedSequence([seed, ref_index])).integers(100) ]
ref_seed = default_rng(SeedSequence([seed, ref_index, FAMILY_ID[family]])).integers(2**31)
```

so reference *i* is the same graph however many annealing steps preceded it, on
any node, and `(base_type, k, p, ref_seed)` (recorded on every `.gt` and in
`pairs_index.csv`) rebuilds any reference exactly. The `ref` column of
`pairs_index.csv` is 1-based (`ref_index = ref - 1`); with that, the formula above
reproduces the recorded `(p, ref_seed)` of all 28,000 pairs. `p` is drawn
family-independently, so `ws_seed_N`'s and `mhk_seed_N`'s reference *i* sit at the
same `p`. Do not change `--budget`, `--rtol` or `--min-w1`: both families were
produced at the values in `run_spec_div.sh`.

## What ships, and provenance

The release ships the tables (`all_graph_eigenvalues{,_norm}_{ws,mhk}.csv`,
`all_graphs_h2_data_{ws,mhk}.csv`, `pair_properties/`, 40 `pairs_index.csv`) and
the two graphs of each family's top pair (Fig. 1). The graphs of the other 27,998
pairs (55,996 `.gt` files, 11.5 GB) are not shipped. Every number the figures use
(both spectra and H² of every G1 and G2) is in the shipped CSVs; see below for
what can be rebuilt.

`pairs_index.csv` is the generator's per-seed matching record. The original
files were not retained; `pair_properties/pair_props_gen.py` had joined all of them
row for row into `pair_properties_per_pair.csv` (renaming `base_type` to `family`
and adding `mean_degree`), and `rebuild_pairs_index.py` inverts that join exactly.
The rebuilt files select the same Fig. 1 pairs (ws seed 5 pair 263, mhk seed 8
pair 376) and reproduce Fig. 1.

## Regenerating the unshipped pair graphs

**Reference graphs G1: exactly.** Each `pair_<N>_G1.gt` is the reference built from
its `pairs_index.csv` row, and rebuilding it gives the original edge for edge
(checked against the originals for both families). On a compute node, from the repo root:

```python
import csv, sys
sys.path.insert(0, 'scripts/spec_div')
from spectral_divergence_gen import reference_graph

row = next(r for r in csv.DictReader(open('nets/spec_div/ws_seed_12/pairs_index.csv'))
           if r['pair'] == '40')
G1 = reference_graph(int(row['n']), int(row['k']), row['base_type'],
                     float(row['p']), int(row['ref_seed']))      # networkx.Graph
```

To write it in the release's `.gt` format with its spectra and H², use `save_gt`
as `save_pair` does in `spectral_divergence_gen.py`.

**Annealed partners G2: statistically only.** `anneal()` runs on a fixed wall-clock
budget (`--budget 120` s per reference), and its temperature schedule follows elapsed
time, so the number of moves, and hence the partner, depends on the speed of the node.
Rerunning a seed (`FAMILY=ws sbatch --array=12 scripts/spec_div/run_spec_div.sh`, then
`run_gain_all.sh` and `spec_div_csv_gen.py` as in "Usage") revisits the same
references in the same order. It yields partners with the same constraints (λ₂, λ_N
and m matched within `--rtol`) and statistically equivalent divergence, but not the
published graphs. For WS, whose yield is below 100%, the set of references that
produce a pair can differ too. For the published pairs, the shipped CSVs are the record.

## Cost

At one pair per 120 s reference, 700 pairs ≈ 23 h per task, 40 tasks in total.
Gains take about 1 min per seed, each CSV pass about 1 min, and the notebooks
10–15 min per (family, spectrum) run with about 48 GB of memory.
