'''
Per-seed CSV generator + mode-aware pooling loaders for figure.ipynb, built
from the .gt files LFC_net_gen_diff_weights.py already produced under
nets/{model}/{n}/{network}/{k}_seed{seed}/.

Mirrors the established nets/-realizations convention (LFC_props_csv_gen.py,
lfc_data_loader.py in the sibling Scale_Free_Social_Contagion project): one
CSV per seed directory, sitting right beside it
('<seed_dir>_props.csv' / '<seed_dir>_gains.csv'), generated once and for
all - mode selection happens later, at LOAD time, by picking how many of
those per-seed CSVs to concatenate (see pooled_props/pooled_gains below).
Generation itself doesn't know about modes.

Per seed_dir ('.../{k}_seed{N}'), writes:
  - '<seed_dir>_props.csv' (tab-delimited, index=(ID, p)): network, CC, T,
    SP, Rg (same base columns LFC_props_csv_gen.py writes, minus l2/lmax_l2
    - superseded by the four lambda_* columns below), plus
    l_mean/l_variance/l_skewness/l_kurtosis (KDE-smoothed moments of the
    normalized-Laplacian spectrum - see moments() below, identical formula
    to network_functions.py's moments()/get_moments()) and
    lambda_2_norm/lambda_max_norm/lambda_2_unnorm/lambda_max_unnorm
    (first-nonzero and largest eigenvalues of the normalized and
    unnormalized Laplacian respectively - read straight off the
    already-stored eig_laplacian/eig_laplacian_norm vertex properties, no
    eigendecomposition recomputed). Does NOT dump the full l_0..l_{n-1}
    spectrum the older spec_lfc.py-style pipeline wrote.
  - '<seed_dir>_gains.csv' (tab-delimited, index=(ID, freq, p)): H2
    (laplacian weight), Lazy (lazy random walk weight), metro (Metropolis-
    Hastings weight) - one column per weight type in the SAME file, since
    all three share the same (ID, freq, p) grid for a given graph (unlike
    the old top-5%/bottom-5% selector split, which needed separate files
    because each selector picked a different node subset).
  - '<seed_dir>_bc_gains.csv' (tab-delimited, index=(ID, freq, p)): one
    BC_{eps} column per epsilon any .gt in this seed dir has been swept at
    (e.g. 'BC_0.2', 'BC_0.05' - auto-discovered per graph from its
    bc_frequencies_eps* graph properties, not a single hardcoded epsilon),
    bounded-confidence collective RMS from bc_rms_sweep.py's
    bc_rms_eps{eps} vertex property: each leader's stored value is already
    a mean over its followers (bc_rms_sweep.py's own per-leader
    normalization), so it is first converted back to that leader's sum
    over followers (mean * (n-1)), then averaged over all nodes-as-leader -
    i.e. per-leader SUM, then MEAN over leaders, matching H2/Lazy/metro's
    per-leader mean convention at the leader-axis step. Kept in its OWN
    file rather than folded into '_gains.csv' because each epsilon's
    frequency grid is independent of H2/Lazy/metro's shared 100-point grid
    (and of each other's) - the 'freq' column here holds whichever grid
    that row's epsilon used, so a given (ID, freq, p) row is populated in
    only the BC_{eps} column(s) whose grid contains that freq; other
    epsilon columns are NaN there. Only written for seed dirs where at
    least one .gt already has BC data (bc_rms_sweep.py run against it);
    silently skipped otherwise, since BC coverage lags behind props/gains
    (as of writing: ws and mhk k=16 seed1, eps 0.2 and 0.05).

Also symlinks each k's seed1 .gt files into a flat, mode-independent
'../LFC/{n}/{network}/{k}/' directory for figure.ipynb's row5 eigenvalue-
histogram panel (which needs raw .gt files, not CSVs, and was never
mode-dependent to begin with).

Example:
    python LFC_diffw_props_csv_gen.py --network ws
    python LFC_diffw_props_csv_gen.py --network mhk --force
'''

import argparse
from pathlib import Path

import graph_tool as gt
import numpy as np
import pandas as pd

WEIGHT_COLUMNS = {'laplacian': 'H2', 'lazy': 'Lazy', 'metropolis': 'metro'}
MODES = ('seed1', 'pool_gains', 'pool_both')


# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────

def seed_dirs(nets_root, network, k, model='LFC', n=240):
    base = Path(f'{nets_root}/{model}/{n}/{network}')
    return sorted(p for p in base.glob(f'{k}_seed*') if p.is_dir())


def all_seed_dirs(nets_root, network, model='LFC', n=240):
    base = Path(f'{nets_root}/{model}/{n}/{network}')
    return sorted(p for p in base.glob('*_seed*') if p.is_dir())


def _k_of(seed_dir):
    return int(seed_dir.name.split('_seed')[0])


def _props_csv_path(seed_dir):
    return seed_dir.parent / f'{seed_dir.name}_props.csv'


def _gains_csv_path(seed_dir):
    return seed_dir.parent / f'{seed_dir.name}_gains.csv'


def _bc_gains_csv_path(seed_dir):
    return seed_dir.parent / f'{seed_dir.name}_bc_gains.csv'


# ─────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────

def gaussian_kernel(x, lambdas, sigma):
    kernel_sum = np.zeros_like(x)
    for lam in lambdas:
        kernel_sum += (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - lam) ** 2) / (2 * sigma**2))
    return kernel_sum


def moments(eigenvalues):
    '''KDE-smoothed mean/variance/skewness/kurtosis of a normalized-Laplacian
    spectrum, x in [0, 2), sigma=0.015 - identical to network_functions.py's
    moments()/get_moments().'''
    x = np.arange(0, 2, 0.001)
    gamma = gaussian_kernel(x, eigenvalues, 0.015)
    gamma_normalized = gamma / np.sum(gamma)
    mean_val = np.sum(gamma_normalized * x)
    variance_val = np.sum(gamma_normalized * (x - mean_val) ** 2)
    skewness_val = np.sum(gamma_normalized * (x - mean_val) ** 3) / (variance_val**1.5)
    kurtosis_val = np.sum(gamma_normalized * (x - mean_val) ** 4) / (variance_val**2)
    return mean_val, variance_val, skewness_val, kurtosis_val


def extract_props(G, network_type):
    n = G.num_vertices()
    eig_unnorm = np.sort(G.vp.eig_laplacian.get_array())
    eig_norm = np.sort(G.vp.eig_laplacian_norm.get_array())
    l_mean, l_variance, l_skewness, l_kurtosis = moments(eig_norm)
    return {
        'network': network_type,
        'CC': sum(G.vp.local_clustering.get_array()) / n,
        'T': G.gp.transitivity,
        'SP': G.gp.get('shortest_path'),
        'Rg': n * np.sum(1 / eig_unnorm[1:]),
        'l_mean': l_mean,
        'l_variance': l_variance,
        'l_skewness': l_skewness,
        'l_kurtosis': l_kurtosis,
        'lambda_2_norm': eig_norm[1],        # first nonzero normalized-Laplacian eigenvalue
        'lambda_max_norm': eig_norm[-1],     # largest normalized-Laplacian eigenvalue
        'lambda_2_unnorm': eig_unnorm[1],    # first nonzero unnormalized-Laplacian eigenvalue
        'lambda_max_unnorm': eig_unnorm[-1], # largest unnormalized-Laplacian eigenvalue
    }


def extract_gains(G):
    '''H2/Lazy/metro, each averaged over ALL nodes (t_b="all", perc=100
    convention) - no top/bottom-degree subsetting. All three weight types
    share the same frequency grid for a given graph, so they come back
    keyed by the same freq.'''
    freqs = list(G.gp.frequencies)
    out = {}
    for weight_type, col in WEIGHT_COLUMNS.items():
        gains_2d = G.vp[f'gains_{weight_type}'].get_2d_array(range(len(freqs)))
        mean_gains = np.mean(gains_2d, axis=1)
        out[col] = dict(zip(freqs, mean_gains))
    return out


def discover_bc_epsilons(G):
    '''Every epsilon this graph has a bc_rms_sweep.py sweep stored for,
    read off its bc_frequencies_eps* graph properties - lets bc_gains.csv
    pick up whichever epsilons happen to be on disk (0.2, 0.05, ...)
    instead of a single hardcoded one.'''
    prefix = 'bc_frequencies_eps'
    return sorted(k[len(prefix):] for k in G.gp.keys() if k.startswith(prefix))


def extract_bc_gains(G):
    '''Bounded-confidence collective AC-RMS: per-leader SUM over followers,
    then MEAN over all nodes-as-leader. bc_rms_sweep.py's bc_rms_eps{eps}
    vertex property already stores each leader's value as a MEAN over its
    239 followers (norm_ac[others].mean() in bc_rms_sweep.py), so before
    averaging over the leader axis we first undo that per-leader mean by
    multiplying back by the follower count (n-1), recovering each leader's
    true sum over followers - matching the intended two-stage definition
    (sum over followers per leader, mean over leaders), rather than the
    leader-axis sum this used to compute. nanmean rather than mean:
    bc_rms_sweep.py's sweep is resumable/leader-chunked, so a graph can have
    some leaders not yet run (NaN) even once it has the property at all -
    nanmean treats those as absent (averaging over however many leaders are
    actually done) rather than poisoning the whole result with NaN. The
    follower count (n-1) is a fixed structural property of the graph, not
    tied to how many leaders have been swept so far, so it's always
    bc_2d.shape[1] - 1 regardless of partial leader coverage.

    Returns {eps_str: {freq: value}} for every epsilon discover_bc_epsilons
    finds on this graph (empty dict if it has no BC data at all yet) -
    callers must treat a missing epsilon as absent rather than zero.

    freq is rounded to 5 decimals - same fix figure.ipynb's load_row
    already applies to H2/Lazy/metro's freq for the same reason: the grid
    is recomputed by np.logspace() independently per bc_rms_sweep.py CLI
    invocation (one per graph/leader over a long-running SLURM array,
    unlike H2/Lazy/metro's single shared grid computed once per generation
    run), so one graph's stored grid can differ from the rest by ~1e-13
    floating noise - confirmed on ws k=16 seed1, where this collapses 71
    spurious distinct values back down to the intended 60.'''
    out = {}
    for eps in discover_bc_epsilons(G):
        tag = f'eps{eps}'
        key = f'bc_rms_{tag}'
        if key not in G.vp:
            continue
        freqs = [round(f, 5) for f in G.gp[f'bc_frequencies_{tag}']]
        bc_2d = G.vp[key].get_2d_array(range(len(freqs)))
        n_followers = bc_2d.shape[1] - 1   # bc_rms_sweep.py stored a mean over this many followers
        mean_over_leaders = np.nanmean(bc_2d, axis=1) * n_followers
        out[eps] = dict(zip(freqs, mean_over_leaders))
    return out


def scan_seed_dir(seed_dir, network_type):
    '''One graph_tool load per .gt file in seed_dir - props, all three
    gains columns, and (where present) BC gains are extracted in the same
    pass.'''
    records = []
    for path in sorted(seed_dir.glob('*.gt')):
        G = gt.load_graph(str(path))
        records.append({
            # path.stem, not G.gp.ID: G.gp.ID is a millisecond timestamp and
            # collides across graphs generated in the same millisecond (only
            # 73/100 unique in ws/16_seed1) - the .gt filename stem
            # (timestamp + SLURM job/task ID, or + pid for manual runs, from
            # LFC_net_gen_diff_weights.py) is the genuinely unique identifier.
            'ID': path.stem,
            'p': G.gp.probability,
            'props': extract_props(G, network_type),
            'gains': extract_gains(G),
            'bc_gains': extract_bc_gains(G),
        })
    return records


def props_dataframe(records):
    rows = {(r['ID'], r['p']): r['props'] for r in records}
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.names = ['ID', 'p']
    return df


def gains_dataframe(records):
    rows = {}
    for r in records:
        freqs = next(iter(r['gains'].values())).keys()
        for freq in freqs:
            key = (r['ID'], freq, r['p'])
            rows[key] = {col: r['gains'][col][freq] for col in WEIGHT_COLUMNS.values()}
    df = pd.DataFrame.from_dict(rows, orient='index', columns=list(WEIGHT_COLUMNS.values()))
    df.index.names = ['ID', 'freq', 'p']
    return df


def bc_gains_dataframe(records):
    '''None (rather than an empty/all-NaN frame) when no .gt in this seed
    dir has BC data yet - generate() skips writing the CSV in that case.
    Wide-form: one BC_{eps} column per epsilon found across this seed
    dir's graphs. Unlike gains_dataframe, freq here is each epsilon's own
    grid (see extract_bc_gains), not H2/Lazy/metro's - a given (ID, freq, p)
    row is populated only in the column(s) whose grid contains that freq,
    NaN in the others.'''
    rows = {}
    for r in records:
        for eps, freq_map in r['bc_gains'].items():
            col = f'BC_{eps}'
            for freq, val in freq_map.items():
                rows.setdefault((r['ID'], freq, r['p']), {})[col] = val
    if not rows:
        return None
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.names = ['ID', 'freq', 'p']
    return df


def populate_flat_gt_dir(seed_dir, target_dir):
    '''Symlinks seed_dir's .gt files into a flat target_dir - reproduces the
    old one-directory-per-k raw .gt layout that figure.ipynb's row5
    eigenvalue-histogram panel and Degroot_gain_gen.py's network_path both
    expect (../LFC/{n}/{network}/{k}/*.gt). Always sourced from seed1 only -
    this directory is not mode-dependent.'''
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for gt_path in seed_dir.glob('*.gt'):
        link_name = target_dir / gt_path.name
        if not link_name.exists():
            link_name.symlink_to(gt_path.resolve())


# ─────────────────────────────────────────────────────────────────────────
# Generation (mode-agnostic: one CSV pair per seed_dir, regardless of k)
# ─────────────────────────────────────────────────────────────────────────

def generate(network_type, nets_root, lfc_flat_root, model='LFC', n=240, force=False):
    for seed_dir in all_seed_dirs(nets_root, network_type, model=model, n=n):
        props_path = _props_csv_path(seed_dir)
        gains_path = _gains_csv_path(seed_dir)
        bc_gains_path = _bc_gains_csv_path(seed_dir)
        need_props_gains = force or not (props_path.exists() and gains_path.exists())
        need_bc = force or not bc_gains_path.exists()

        if not (need_props_gains or need_bc):
            print(f'{network_type} {seed_dir.name}: already generated - skipping. '
                  f'Pass force=True / --force to regenerate.')
        else:
            # BC coverage lags props/gains (only some seed dirs have had
            # bc_rms_sweep.py run against them yet), so need_bc can be True
            # even when need_props_gains is False - scan once regardless
            # and let each _dataframe() below decide what it has to write.
            records = scan_seed_dir(seed_dir, network_type)
            print(f'{network_type} {seed_dir.name}: scanned {len(records)} .gt files')

            if need_props_gains:
                props_dataframe(records).to_csv(props_path, sep='\t', mode='w', header=True)
                print(props_path)
                gains_dataframe(records).to_csv(gains_path, sep='\t', mode='w', header=True)
                print(gains_path)

            if need_bc:
                bc_df = bc_gains_dataframe(records)
                if bc_df is not None:
                    bc_df.to_csv(bc_gains_path, sep='\t', mode='w', header=True)
                    print(bc_gains_path)
                else:
                    print(f'{network_type} {seed_dir.name}: no BC data on any .gt yet - skipping bc_gains.csv')

        if seed_dir.name.endswith('_seed1'):
            k = _k_of(seed_dir)
            populate_flat_gt_dir(seed_dir, Path(f'{lfc_flat_root}/{n}/{network_type}/{k}'))

    print('Done all!')


# ─────────────────────────────────────────────────────────────────────────
# Loading / pooling (imported by figure.ipynb - picks the mode at read time)
# ─────────────────────────────────────────────────────────────────────────

def _dirs_for_mode(nets_root, network, k, mode, pool_this):
    '''pool_this: whether THIS quantity (props or gains) should be pooled
    across every seed found for (network, k), or restricted to seed1.
    Pooling is driven purely by what's on disk (seed_dirs naturally
    returns just [seed1] for any k that only has seed1), so no
    special-casing of k==16 is needed here.'''
    assert mode in MODES, f"Unknown mode '{mode}', expected one of {MODES}"
    dirs = seed_dirs(nets_root, network, k)
    if not dirs:
        raise FileNotFoundError(f'No seed dirs found under {nets_root}/LFC/240/{network}/{k}_seed*')
    if pool_this:
        return dirs
    return [d for d in dirs if d.name == f'{k}_seed1']


def pooled_props(network, k, mode, nets_root='nets'):
    dirs = _dirs_for_mode(nets_root, network, k, mode, pool_this=(mode == 'pool_both'))
    frames = [pd.read_csv(_props_csv_path(d), sep='\t', index_col=[0, 1]) for d in dirs]
    return pd.concat(frames)


def pooled_gains(network, k, mode, nets_root='nets'):
    dirs = _dirs_for_mode(nets_root, network, k, mode, pool_this=(mode in ('pool_gains', 'pool_both')))
    frames = [pd.read_csv(_gains_csv_path(d), sep='\t', index_col=[1]) for d in dirs]
    return pd.concat(frames)


def pooled_bc_gains(network, k, mode, nets_root='nets'):
    '''Same pooling rule as pooled_gains, but BC coverage on disk is
    partial (not every seed dir has had bc_rms_sweep.py run against it
    yet), so dirs missing a bc_gains.csv are silently dropped rather than
    raising - the caller sees however many seeds' worth of BC data
    actually exist today.'''
    dirs = _dirs_for_mode(nets_root, network, k, mode, pool_this=(mode in ('pool_gains', 'pool_both')))
    paths = [p for p in (_bc_gains_csv_path(d) for d in dirs) if p.exists()]
    if not paths:
        raise FileNotFoundError(
            f'No bc_gains.csv found under {nets_root}/LFC/240/{network}/{k}_seed* '
            '- run bc_rms_sweep.py + LFC_diffw_props_csv_gen.py for this network/k first.')
    frames = [pd.read_csv(p, sep='\t', index_col=[1]) for p in paths]
    return pd.concat(frames)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=str, choices=['ws', 'mhk'], required=True)
    parser.add_argument('--nets-root', type=str, default='nets')
    parser.add_argument('--lfc-flat-root', type=str, default='output/LFC_flat',
                         help='Where to symlink the flat per-k raw .gt directories used by the histogram panel')
    parser.add_argument('--force', action='store_true',
                         help='Regenerate CSVs even if the output files already exist (default: skip already-done seed dirs)')
    return parser.parse_args()


def main():
    args = parse_args()
    generate(args.network, args.nets_root, args.lfc_flat_root, force=args.force)


if __name__ == '__main__':
    main()
