'''
Helpers for Figure_generator.ipynb-style notebooks to read LFC data that is
now spread across one directory per seed (nets/{model}/{n}/{n_type}/{k}_seed{seed}/),
instead of the old single directory per (model, n, n_type, k).

Each function below pools data across every seed found for a given
(model, nodes, network, k) and returns a DataFrame with the exact same
shape/index as the single-file `pd.read_csv(...)` calls the notebook used
to make, so downstream plotting code (which already does
`.groupby(level=...).mean()` over the pooled rows) does not need to change.

Requires the per-seed CSVs produced by LFC_props_csv_gen.py to already
exist (run it once after generating .gt files with LFC_net_gen_args.py).
'''

import glob
import os

import graph_tool as gt
import graph_tool.spectral
import numpy as np
import pandas as pd


def _seed_dirs(model, nodes, network, k, root='nets'):
    # Seed prefixes '<...>/{k}_seed{N}'. `{k}_seed*` also matches the sibling
    # CSVs (e.g. '16_seed1_props.csv'), so those are reduced to their prefix
    # rather than used as-is. A seed counts if its .gt directory exists or, as
    # in the data release (which ships every table but only some graphs), its
    # '_props.csv' does; on a complete tree both give the same seeds.
    base = f'{root}/{model}/{nodes}/{network}'
    found = {p for p in glob.glob(f'{base}/{k}_seed*') if os.path.isdir(p)}
    found |= {p[:-len('_props.csv')] for p in glob.glob(f'{base}/{k}_seed*_props.csv')}
    return sorted(found)


def _perc_label(perc):
    return int(perc) if float(perc).is_integer() else perc


def pooled_props(model, nodes, network, k, root='nets'):
    '''
    Pools '<seed_dir>_props.csv' across every seed found for this
    (model, nodes, network, k). Same shape as the old single
    f'networks/{model}/{nodes}/{network}/{k}_props.csv' read
    (index_col=[0,1], i.e. indexed by (ID, p)).
    '''
    frames = []
    for seed_dir in _seed_dirs(model, nodes, network, k, root):
        frames.append(pd.read_csv(f'{seed_dir}_props.csv', sep='\t', index_col=[0, 1]))
    if not frames:
        raise FileNotFoundError(f'No props CSVs found under {root}/{model}/{nodes}/{network}/{k}_seed*_props.csv')
    return pd.concat(frames)


def pooled_gains(model, nodes, network, k, t_b, perc, centrality='degree', root='nets'):
    '''
    Pools '<seed_dir>_{t_b}_{perc}_corr_gains_{centrality}.csv' across every
    seed found for this (model, nodes, network, k). Same shape as the old
    single f'networks/{model}/{nodes}/{network}/{k}_{t_b}_{perc}_corr_gains_{centrality}.csv'
    read (index_col=[1], i.e. indexed by freq, with ID/p/H2 as columns).

    LFC_net_gen_args.py only computes gains for the top 5% and bottom 5%
    of nodes by degree, so only (t_b, perc) in {('top', 5), ('bot', 5)}
    will find matching CSVs from the current pipeline.
    '''
    perc_label = _perc_label(perc)
    frames = []
    for seed_dir in _seed_dirs(model, nodes, network, k, root):
        path = f'{seed_dir}_{t_b}_{perc_label}_corr_gains_{centrality}.csv'
        frames.append(pd.read_csv(path, sep='\t', index_col=[1]))
    if not frames:
        raise FileNotFoundError(
            f'No gains CSVs found under {root}/{model}/{nodes}/{network}/{k}_seed*_{t_b}_{perc_label}_corr_gains_{centrality}.csv'
        )
    return pd.concat(frames)


def pooled_gains_all(model, nodes, network, k, root='nets'):
    '''
    Legacy-only: pools '<seed_dir>_gains.csv' (H2 averaged across ALL
    nodes, no top/bot selector) across every seed found for this
    (model, nodes, network, k). LFC_net_gen_args.py no longer computes
    gains for every node (only the top 5% / bottom 5% by degree, via
    gains_top5 / gains_bot5), so LFC_props_csv_gen.py no longer writes
    '_gains.csv' - this will raise FileNotFoundError against any data
    produced by the current pipeline. Kept only so old '_gains.csv' files
    from a previous run (or the notebook cell that still calls this) don't
    hit an ImportError.
    '''
    frames = []
    for seed_dir in _seed_dirs(model, nodes, network, k, root):
        frames.append(pd.read_csv(f'{seed_dir}_gains.csv', sep='\t', index_col=[1]))
    if not frames:
        raise FileNotFoundError(f'No gains CSVs found under {root}/{model}/{nodes}/{network}/{k}_seed*_gains.csv')
    return pd.concat(frames)


def find_graph_for_p(model, nodes, network, k, p, root='nets', tol=1e-9):
    '''
    Loads one representative .gt file (from any seed) whose stored
    'probability' graph property is closest to p. Replaces the old
    positional lookup via get_sorted_filenames(directory_path)[specialp[th]],
    which assumed exactly one file per p - no longer true with multiple
    seeds contributing files to the same k.
    '''
    best_path, best_diff, best_graph = None, None, None
    for seed_dir in _seed_dirs(model, nodes, network, k, root):
        for path in glob.glob(f'{seed_dir}/*.gt'):
            G = gt.load_graph(path)
            diff = abs(G.gp.probability - p)
            if best_diff is None or diff < best_diff:
                best_path, best_diff, best_graph = path, diff, G
            if diff <= tol:
                return G
    if best_path is None:
        raise FileNotFoundError(f'No .gt files found under {root}/{model}/{nodes}/{network}/{k}_seed*/')
    return best_graph


def _eig_cache_path(model, nodes, network, k, root='nets'):
    return f'{root}/{model}/{nodes}/{network}/{k}_eig_cache.npz'


def pooled_eig_laplacians(model, nodes, network, k, root='nets', decimals=6, use_cache=True):
    '''
    One pass over every .gt file for this (model, nodes, network, k),
    grouping normalized-Laplacian eigenvalues by their stored 'probability'
    graph property (rounded to `decimals` - run.sh derives p identically for
    every seed via np.linspace(0.001, 1, 100)[P_IDX], so realizations that
    target the same p produce bit-identical-up-to-float-roundtrip values).

    Lets the eigenspectrum panel pool eigenvalues across every seed
    realization that shares a given p, instead of reading a single
    representative graph. Returns {p: concatenated eigenvalue array}.

    This opens every individual .gt file for (model, nodes, network, k)
    (~5000 for the current 50-seed sweep) - measured >3.5 min wall time for
    only ~33s of actual CPU time when tried interactively on a login node,
    i.e. almost entirely Lustre I/O wait (opening the files, not
    the eigenvalue computation itself, dominates - so reading a stored
    eig_laplacian_norm vertex property below instead of recomputing it
    doesn't meaningfully speed this up either). Do not call this
    interactively; run precompute_eig_cache.py via submit_eig_cache.sh on a
    compute node instead, which populates the on-disk cache this function
    checks first (`use_cache=True`, the default) so later calls - e.g. from
    the figures notebook - are instant.
    '''
    cache_path = _eig_cache_path(model, nodes, network, k, root)
    if use_cache and os.path.exists(cache_path):
        with np.load(cache_path) as data:
            return {float(key[2:]): data[key] for key in data.files}

    groups = {}
    for seed_dir in _seed_dirs(model, nodes, network, k, root):
        for path in glob.glob(f'{seed_dir}/*.gt'):
            G = gt.load_graph(path)
            p = round(float(G.gp.probability), decimals)
            if G.vertex_properties.get('eig_laplacian_norm', False):
                eig = G.vp.eig_laplacian_norm.get_array()
            else:
                eig = np.linalg.eigvalsh(graph_tool.spectral.laplacian(G, norm=True).todense())
            groups.setdefault(p, []).append(eig)
    if not groups:
        raise FileNotFoundError(f'No .gt files found under {root}/{model}/{nodes}/{network}/{k}_seed*/')
    pooled = {p: np.concatenate(v) for p, v in groups.items()}

    if use_cache:
        np.savez(cache_path, **{f'p_{p:.6f}': arr for p, arr in pooled.items()})

    return pooled
