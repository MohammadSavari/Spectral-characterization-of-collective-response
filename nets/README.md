# `nets/` - data

Empty in the repository. Populate it with `bash reassemble.sh all` from the repo
root (unpacks `data/`), or regenerate it with `scripts/`. Layout:

```
nets/
├── spec_div/
│   ├── all_graph_eigenvalues_{ws,mhk}.csv        # unnormalised spectrum, one row per graph
│   ├── all_graph_eigenvalues_norm_{ws,mhk}.csv   # normalised spectrum
│   ├── all_graphs_h2_data_{ws,mhk}.csv           # H²(ω), 100 frequencies per graph
│   ├── {ws,mhk}_seed_{1..20}/pairs_index.csv     # matching record, 700 pairs per seed
│   ├── ws_seed_5/pair_263_G{1,2}.gt              # Fig. 1 pair (ws); mhk_seed_8/pair_376 likewise
│   └── pair_properties/                          # per-pair properties and differences
└── LFC/240/{ws,mhk}/
    ├── <k>_seed<N>_props.csv                     # structure + spectral moments per graph
    ├── <k>_seed<N>_gains.csv                     # H2 (LFC), Lazy, metro gains per (graph, ω)
    ├── 16_seed1_bc_gains.csv                     # bounded-confidence RMS (ws also 16_seed2)
    └── 16_seed1/*.gt                             # k = 16 graphs carrying the BC results (ws also 16_seed2)
```

Graph IDs in the spectral-divergence CSVs are `G{pair}_{1|2}_{seed}` and are
unique only within a family, so tables are keyed on `(family, seed, pair)`.

Regenerated graphs land in the same tree: LFC graphs in
`LFC/240/<model_type>/<k>_seed<N>/`, beside their shipped CSVs, and spectral-divergence
pairs in `spec_div/<family>_seed_<N>/`. See `data/README.md` ("Not shipped") for
what is regenerable and how.
