# `data/` - chunked archives of `nets/`

The figure-level data (about 1.5 GB uncompressed) ships as eight `tar.gz` archives,
each split into `<=10MB` chunks to stay within per-file limits of the release
target. Graph (`.gt`) files and tables (CSV) are in separate archives. Archive
members are paths relative to `nets/`.

## Archives

| Archive | Extracts to | Contents | tar.gz | Parts |
|---|---|---|---|---|
| `LFC_240_ws_gt` | `nets/LFC/240/ws/` | `16_seed1/`, `16_seed2/`: 200 WS graphs at k = 16 carrying the bounded-confidence results | 189.2 MB | 19 |
| `LFC_240_mhk_gt` | `nets/LFC/240/mhk/` | `16_seed1/`: 100 MHK graphs at k = 16 carrying the bounded-confidence results | 111.5 MB | 11 |
| `spec_div_top_pairs_gt` | `nets/spec_div/` | the Fig. 1 pairs: `ws_seed_5/pair_263_G{1,2}.gt`, `mhk_seed_8/pair_376_G{1,2}.gt` | 0.8 MB | 1 |
| `LFC_240_ws_csv` | `nets/LFC/240/ws/` | 232 per-seed `_props`, `_gains`, `_bc_gains` CSVs (every k and seed) | 42.0 MB | 5 |
| `LFC_240_mhk_csv` | `nets/LFC/240/mhk/` | 231 per-seed CSVs | 41.9 MB | 4 |
| `spec_div_ws_csv` | `nets/spec_div/` | WS eigenvalue (unnormalised, normalised) and H² CSVs for 28,000 graphs, and 20 `pairs_index.csv` | 151.7 MB | 15 |
| `spec_div_mhk_csv` | `nets/spec_div/` | the same for MHK | 152.2 MB | 15 |
| `spec_div_pair_properties` | `nets/spec_div/pair_properties/` | per-pair properties and within-pair differences, with the scripts that built them | 8.6 MB | 1 |
| **Total** | | | **697.9 MB** | **71** |

## Not shipped (regenerable)

About 25 GB of graph (`.gt`) files are not part of this release, because they
exceed what a Git repository can reasonably hold. No figure opens them: every
quantity the figures use is already in the shipped CSVs.

| Set | Graphs | Raw size | Rebuilt exactly? | How |
|---|---|---|---|---|
| LFC, WS: k = 16 seeds 3–100, and seed 1 at the 15 other k | 11,300 | 6.7 GB | yes, edge for edge | [scripts/LFC/README.md](../scripts/LFC/README.md#regenerating-the-unshipped-graphs) |
| LFC, MHK: k = 16 seeds 2–100, and seed 1 at the 15 other k | 11,400 | 6.8 GB | yes, edge for edge | same |
| Spectral divergence, reference graphs `pair_*_G1.gt` | 27,998 | 5.7 GB | yes, edge for edge | [scripts/spec_div/README.md](../scripts/spec_div/README.md#regenerating-the-unshipped-pair-graphs) |
| Spectral divergence, annealed partners `pair_*_G2.gt` | 27,998 | 5.7 GB | **no**, only statistically | same |

The release has 28,000 pairs (14,000 per family, 56,000 graphs). Only the two Fig. 1
pairs ship as graphs (`spec_div_top_pairs_gt`), hence 27,998 of each kind above.

The LFC graphs are a deterministic function of `(model_type, k, p, seed)`, and each
reference graph G1 of `(base_type, k, p, ref_seed)` as recorded in `pairs_index.csv`.
Both were checked by rebuilding graphs from these parameters and comparing their edge
sets with the originals. A partner G2 comes from simulated annealing on a wall-clock
budget, so a rerun produces an equivalent but not identical partner. For the
published pairs, the shipped CSVs (both spectra and H² of every G1 and G2) are the record.

## Reassembling

From the repo root, `bash reassemble.sh <archive>` or `bash reassemble.sh all`.
Manually:

```bash
cat data/<name>.tar.gz.part* > data/<name>.tar.gz
sha256sum -c <(grep "data/<name>.tar.gz$" data/checksums_full.sha256)
mkdir -p nets && tar xzf data/<name>.tar.gz -C nets
```

- `checksums_full.sha256`: sha256 of each whole (reassembled) archive.
- `checksums_parts.sha256`: sha256 of each chunk, for checking a transfer of the parts themselves.

`package_data.sh` in the repo root rebuilds these archives from `nets/`.
