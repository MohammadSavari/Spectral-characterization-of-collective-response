"""
Collect H^2 and Laplacian eigenvalues from the spec_divergence_new .gt files into
the two flat CSVs the correlation analysis reads, one pair of CSVs per family.

    python spec_div_csv_gen.py --family ws            # every ws_seed_* directory
    python spec_div_csv_gen.py --family mhk --seeds 1 2 3

Writes, next to this script:

    all_graphs_h2_data_{family}.csv     ID, freq, H2        (long, 100 rows/graph)
    all_graph_eigenvalues_{family}.csv  ID, eigenvalue_1..n (wide, one row/graph)

The family tag matters: G{pair}_{1,2}_{seed} is not unique across ws and mhk (nor
across ../spec_div), so the analyser keys pairs on (family, seed, pair).

Same schema and the same ID convention -- G{pair}_{1,2}_{seed} -- as
cospectral/all_graphs_h2_data.csv and cospectral/all_graph_eigenvalues.csv, which
cospectral/analyzer.ipynb built from the converter's per-seed
`converted_{seed}_corr_gains.csv` files.  This is the equivalent for the spec_div
pipeline, which stores both quantities on the .gt directly:

  * H2 is the mean over nodes of the collective response at each frequency, i.e.
    np.mean(gain.get_2d_array(...), 1) -- exactly what convert_csv_degroot.py
    writes into the 'H2' column.  The per-node gain comes from the
    'gain_laplacian_all' vertex property, so run gain_laplacian_all_gen.py first;
    graphs without a complete gain are counted and skipped, not silently dropped.
  * the eigenvalues are the unnormalised Laplacian spectrum stored as
    'eig_laplacian', sorted ascending and padded with NaN to the longest
    spectrum found (all graphs here are n=240, but the reference file pads and
    the analysis tolerates NaN, so this does too).

Only the .gt files are read -- the CSVs the generator can also write are not
needed, and --formats gt runs do not have them.
"""

import argparse
import glob
import os
import re
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

GAIN_PROP = 'gain_laplacian_all'
EIG_PROP = 'eig_laplacian'
NAME_RE = re.compile(r'pair_(\d+)_G([12])\.gt$')
# one directory per (family, seed); the family is a prefix, not a suffix,
# so it never collides with the seed number
SEED_RE_TEMPLATE = r'{family}_seed_(\d+)$'


def read_graph(job):
    """(ID, frequencies, mean H^2, sorted eigenvalues, metadata) for one .gt file."""
    path, gain_prop, eig_prop, seed_re = job
    import graph_tool as gt

    match = NAME_RE.search(os.path.basename(path))
    seed = re.search(seed_re, os.path.dirname(path).rstrip('/'))
    if not match or not seed:
        return path, 'name', None
    ident = f'G{match.group(1)}_{match.group(2)}_{seed.group(1)}'

    try:
        g = gt.load_graph(path)
    except Exception as exc:
        return path, f'{type(exc).__name__}: {exc}', None

    if gain_prop not in g.vp or eig_prop not in g.vp:
        return path, 'missing property', None
    freqs = np.asarray(g.gp['frequencies']) if 'frequencies' in g.gp else None
    if freqs is None:
        return path, "no 'frequencies' graph property", None

    vp = g.vp[gain_prop]
    if any(len(vp[v]) != len(freqs) for v in g.vertices()):
        return path, 'incomplete gain', None

    # (frequency, vertex) -> mean over vertices, the column convert_csv_degroot.py
    # calls H2
    h2 = np.mean(vp.get_2d_array(range(len(freqs))), 1)
    eigs = np.sort(g.vp[eig_prop].get_array())

    # base_type is the reference family, 'ws' or 'mhk'; probability is the p it was
    # drawn at, and (k, ref_seed) complete the reference's identity -- together they
    # rebuild it exactly.  The spec_div analysis showed the reference explains ~72%
    # of the variance in these quantities, so everything needed to stratify by it
    # travels with the data.
    meta = dict(base_type=str(g.gp['base_type']) if 'base_type' in g.gp else '',
                probability=float(g.gp['probability']) if 'probability' in g.gp else float('nan'),
                k=int(g.gp['k']) if 'k' in g.gp else -1,
                ref_seed=int(g.gp['ref_seed']) if 'ref_seed' in g.gp else -1,
                seed=int(g.gp['seed']) if 'seed' in g.gp else -1,
                pair=int(g.gp['pair']) if 'pair' in g.gp else -1,
                which=str(g.gp['which']) if 'which' in g.gp else '',
                n=int(g.num_vertices()), m=int(g.num_edges()))
    return path, 'ok', (ident, freqs, h2, eigs, meta)


def parse_arguments(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description='Collect H^2 and Laplacian eigenvalues from the spec_div .gt '
                    'files into all_graphs_h2_data.csv and all_graph_eigenvalues.csv')
    p.add_argument('--family', default='ws', choices=['ws', 'mhk'],
                   help='which {family}_seed_* tree to read')
    p.add_argument('--root', default=here,
                   help='directory holding the {family}_seed_* folders')
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='seeds to include (default: every seed directory found)')
    p.add_argument('--out-dir', default=here, help='where to write the two CSVs')
    # family-tagged by default: the two families share an output directory, and
    # G{pair}_{which}_{seed} is not unique across them
    p.add_argument('--h2-name', default=None,
                   help='default: all_graphs_h2_data_{family}.csv')
    p.add_argument('--eig-name', default=None,
                   help='default: all_graph_eigenvalues_{family}.csv')
    p.add_argument('--gain-prop', default=GAIN_PROP)
    p.add_argument('--eig-prop', default=EIG_PROP,
                   help="'eig_laplacian' (unnormalised) or 'eig_laplacian_norm' "
                        '(the spectrum of I - D^-1/2 A D^-1/2)')
    p.add_argument('--no-h2', action='store_true',
                   help='write only the eigenvalue CSV. The gain does not depend on '
                        '--eig-prop, so a second spectrum needs no second H2 file')
    p.add_argument('--workers', type=int, default=0,
                   help='reader processes (0 = SLURM_CPUS_PER_TASK)')
    p.add_argument('--limit', type=int, default=0,
                   help='first N .gt files per seed, for testing')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    try:
        import graph_tool  # noqa: F401
    except ImportError:
        print('graph_tool not importable -- load the module stack '
              '(StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10)', flush=True)
        return 2

    seeds = args.seeds
    seed_re = SEED_RE_TEMPLATE.format(family=args.family)
    h2_name = args.h2_name or f'all_graphs_h2_data_{args.family}.csv'
    eig_name = args.eig_name or f'all_graph_eigenvalues_{args.family}.csv'
    dirs = sorted(
        (d for d in glob.glob(os.path.join(args.root, f'{args.family}_seed_*'))
         if os.path.isdir(d)
         and (seeds is None or int(re.search(seed_re, d).group(1)) in seeds)),
        key=lambda d: int(re.search(seed_re, d).group(1)))
    if not dirs:
        print(f'no {args.family}_seed_* directories under {args.root}', flush=True)
        return 2

    files = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, 'pair_*_G[12].gt')))
        files.extend(found[:args.limit] if args.limit else found)
    print(f'{len(dirs)} seed director(ies), {len(files)} .gt file(s)', flush=True)

    workers = args.workers or int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    jobs = [(f, args.gain_prop, args.eig_prop, seed_re) for f in files]
    start = time.time()

    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            results = pool.map(read_graph, jobs, chunksize=32)
    else:
        results = [read_graph(j) for j in jobs]

    ids, h2_rows, eig_rows, metas, skipped = [], [], [], [], {}
    for path, status, payload in results:
        if status != 'ok':
            skipped.setdefault(status, []).append(path)
            continue
        ident, freqs, h2, eigs, meta = payload
        ids.append(ident)
        h2_rows.append((ident, freqs, h2))
        eig_rows.append(eigs)
        metas.append(meta)

    print(f'read {len(ids)} graph(s) in {time.time() - start:.0f}s', flush=True)
    for reason, paths in skipped.items():
        print(f'  skipped {len(paths)}: {reason}  (e.g. {os.path.relpath(paths[0], args.root)})',
              flush=True)
    if not ids:
        return 1

    # --- long H2 table: one row per (graph, frequency)
    if args.no_h2:
        print('skipping the H2 table (--no-h2)', flush=True)
    else:
        n_freq = len(h2_rows[0][1])
        h2_df = pd.DataFrame({
            'ID': np.repeat([r[0] for r in h2_rows], n_freq),
            'freq': np.concatenate([r[1] for r in h2_rows]),
            'H2': np.concatenate([r[2] for r in h2_rows]),
        })
        h2_path = os.path.join(args.out_dir, h2_name)
        h2_df.to_csv(h2_path, index=False)
        print(f'{h2_path}: {len(h2_df)} rows '
              f'({os.path.getsize(h2_path) / 1e6:.0f} MB)', flush=True)

    # --- wide eigenvalue table, NaN-padded to the longest spectrum
    width = max(len(e) for e in eig_rows)
    eigs = np.full((len(eig_rows), width), np.nan)
    for i, row in enumerate(eig_rows):
        eigs[i, :len(row)] = row
    eig_df = pd.DataFrame(eigs,
                          columns=[f'eigenvalue_{j + 1}' for j in range(width)])
    # metadata first, spectrum after: readers select the spectrum by the
    # 'eigenvalue_' prefix, so extra leading columns are transparent to them
    for key in ('m', 'n', 'ref_seed', 'k', 'probability', 'which', 'pair',
                'seed', 'base_type'):
        eig_df.insert(0, key, [meta[key] for meta in metas])
    eig_df.insert(0, 'ID', ids)
    eig_path = os.path.join(args.out_dir, eig_name)
    eig_df.to_csv(eig_path, index=False)
    print(f'{eig_path}: {len(eig_df)} rows x {width} eigenvalues '
          f'({os.path.getsize(eig_path) / 1e6:.0f} MB)', flush=True)
    print('  base types: ' + ', '.join(
        f'{k}={v}' for k, v in eig_df['base_type'].value_counts().items()), flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
