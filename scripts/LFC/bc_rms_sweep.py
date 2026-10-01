'''
Leader-driven bounded-confidence RMS frequency sweep over the realization
graphs in ../LFC/{n}/{network}/{k}/*.gt — the fast, converged replacement
for Non_linear/bc_model/bc_adaptive.py's sweep loop.

Model (identical rule to bc_adaptive.py, verified equivalent): all opinions
start at 0; the leader node's opinion is forced to 0.5*(1 + sin(2*pi*f*t));
every step each other node jumps to the mean of {itself + neighbors within
epsilon} (strict <, leader written in place before the averaging step).
Differences from bc_adaptive.py, all validated interactively first:

  * vectorized update (dense masked averaging, ~0.14 ms/step at n=240 vs
    ~4 ms for the per-node Python loop). Dense n x n is the right trade for
    n up to a few thousand.
  * converge-then-measure instead of a fixed period count: whole periods run
    until max|x(nT) - x((n-1)T)| < tol (the driven orbit is period-1, so
    this detects lock-in), then exactly one more period is measured. A
    max_periods cap marks the result unconverged (lock stored as -1) rather
    than silently keeping it. (The old num_periods=3 measured the transient
    for every f above ~0.02 Hz at n=240, k=16.)
  * O(n) memory: per-period sum/sum-of-squares accumulators, no
    opinion_history.
  * AC (mean-subtracted) RMS as the response measure: raw RMS is dominated
    by the DC offset the gating leaves behind (a node frozen at 0.5 would
    report a normalized "gain" of ~0.82). The raw quantity is kept alongside
    for continuity with the old bc_model CSVs.
  * loops over leaders (--leaders): every node, a degree-percentile selector
    (top:P / bot:P / rand:P, as in the polarization-speed pipelines), an
    explicit list, or an index range.

dt is FIXED across the sweep at 1/(10**fmax * samples_per_period), as in
bc_adaptive.py. Deliberate and load-bearing: the update jumps all the way to
the neighborhood mean each step, so a step (not a second) is the model's
time unit and the dynamics depend only on steps-per-period; a per-frequency
dt would give every frequency the same steps-per-period and a flat response.

Results are stored INTO each .gt, mirroring the gains_{wt} convention of
LFC_net_gen_diff_weights.py (vector<double> vertex property over a
frequency-grid graph property, cached in the graph, saved atomically):

  graph properties (per epsilon):
    bc_frequencies_eps{eps}   vector<double>, the sweep grid
    bc_dt_eps{eps}, bc_tol_eps{eps}   run parameters
  vertex properties (per epsilon), entry j = value at frequency j WHEN THIS
  VERTEX IS THE LEADER (NaN where that vertex never served as leader):
    bc_rms_eps{eps}       collective AC-RMS: mean over non-leader nodes of
                          per-node AC-RMS normalized by the leader's AC-RMS
    bc_rms_raw_eps{eps}   same with raw (DC-included) RMS
    bc_lock_eps{eps}      period at which the orbit locked (-1 = hit
                          max_periods without locking at all - see below)
    bc_dfinal_eps{eps}    the period-to-period distance d at lock time -
                          near-zero (< tol) means a tight period-1 lock;
                          a small nonzero plateau value means a quasi-
                          periodic/subharmonic lock (see run_one()'s
                          docstring) - both are legitimate steady states,
                          this is a quality flag, not a failure signal

The .gt write is SAFE FOR CONCURRENT SLURM-ARRAY ACCESS: the (long) per-leader
compute happens with no file open at all, and only the (short) merge+save is
wrapped in an exclusive flock on a sibling `<gt_file>.lock`, during which the
graph is re-read from disk before the local leader's results are written in
— so two array tasks writing different leaders into the same .gt file cannot
clobber each other, regardless of which finishes first.

An optional per-graph CSV (one row per (leader, frequency), appended as
computed; also lock-protected) can be written to --output_dir for monitoring
and notebook use via --write_csv (default OFF — the .gt properties are the
source of truth; the CSV is just a convenience mirror and doubles the I/O
across a large array). Finished (leader, frequency) pairs found in the .gt
properties are skipped on re-run unless --force, so interrupted sweeps
resume, and SLURM-array chunks over --leaders fill the same properties
incrementally.

Cost at n=240, k=16, the default logspace(-3, 0, 60) grid: ~13 min per
(graph, leader) on a compute node. --leaders all is therefore ~52 h per
graph — chunk it over a SLURM array instead, e.g. one task per leader:
    python bc_rms_sweep.py --gt_file <seed.gt> --leaders $SLURM_ARRAY_TASK_ID

Examples:
    # one seed graph, single default leader (node 0), ~13 min
    python bc_rms_sweep.py --gt_file ../LFC/240/ws/16/<seed>.gt
    # all ws k=16 seeds, hub leaders only
    python bc_rms_sweep.py --network ws --k 16 --leaders top:5
    # quick smoke test
    python bc_rms_sweep.py --gt_file <seed.gt> --leaders 0 --nfreq 6 --fmin -1
'''

import argparse
import contextlib
import fcntl
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import graph_tool.all as gt

SCRIPT_DIR = Path(__file__).resolve().parent


@contextlib.contextmanager
def gt_lock(gt_file):
    '''Exclusive lock on a sibling .lock file, held across a reload-merge-save
    critical section so concurrent processes never clobber each other's
    writes to the same .gt file.'''
    lock_path = str(gt_file) + '.lock'
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ─────────────────────────────────────────────────────────────────────────
# Leader selection
# ─────────────────────────────────────────────────────────────────────────

def parse_leaders(spec, degrees, rng):
    '''all | top:P | bot:P | rand:P (percent by degree) | 3,7,9 | 10-40 | 0'''
    n = len(degrees)
    if spec == 'all':
        return list(range(n))
    for tag, pick in (('top:', lambda m: np.argsort(degrees)[::-1][:m]),
                      ('bot:', lambda m: np.argsort(degrees)[:m]),
                      ('rand:', lambda m: rng.choice(n, size=m, replace=False))):
        if spec.startswith(tag):
            m = max(1, int(round(n * float(spec[len(tag):]) / 100)))
            return sorted(int(v) for v in pick(m))
    if '-' in spec and ',' not in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(',')]


# ─────────────────────────────────────────────────────────────────────────
# Simulation core (vectorized, converge-then-measure)
# ─────────────────────────────────────────────────────────────────────────

def run_one(A, leader, freq, dt, eps, tol, max_periods,
           plateau_rel_tol=0.01, plateau_window=3):
    '''Simulate one (leader, frequency) pair from rest until one period past
    lock-in. Returns (rms_ac, rms_raw, lock_period, converged, d_final); the
    RMS arrays are per node, over the measurement period.

    "Locked" covers two distinct cases, both legitimate steady states for
    RMS purposes - d_final records which one this was:
      - tight lock (d_final < tol): the orbit is exactly period-1, as at
        most frequencies.
      - plateau lock (the last `plateau_window` d-values agree within
        plateau_rel_tol of each other, but not near zero): some (graph,
        frequency) combinations - confirmed interactively, reproducible
        across very different rewiring probabilities at the same
        frequency - settle onto a quasi-periodic/subharmonic attractor
        instead of a period-1 orbit, because the epsilon-gated network's
        own relaxation rate isn't commensurate with the drive. d(nT vs
        (n-1)T) then stabilizes at a small but nonzero level and would
        never satisfy d < tol at ANY tol tight enough to be meaningful -
        checking for a stable plateau (not just d -> 0) is what makes
        this case terminate at all instead of burning max_periods.
    Only hitting max_periods with neither achieved counts as unconverged
    (returned converged=False, lock_period=-1).'''
    n = A.shape[0]
    S = int(round(1.0 / (freq * dt)))
    x = np.zeros(n)
    t = 0.0
    prev = x.copy()
    locked = False
    lock_period = -1
    d_history = []
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    for period in range(1, max_periods + 1):
        s1[:] = 0.0
        s2[:] = 0.0
        for _ in range(S):
            t += dt
            x[leader] = 0.5 * (1.0 + np.sin(2.0 * np.pi * freq * t))
            M = (A & (np.abs(x[:, None] - x[None, :]) < eps)).astype(float)
            xn = (x + M @ x) / (1.0 + M.sum(axis=1))
            xn[leader] = x[leader]
            x = xn
            s1 += x
            s2 += x * x
        if locked:                       # this was the measurement period
            break
        d = np.abs(x - prev).max()
        prev = x.copy()
        d_history.append(d)
        tight = d < tol
        plateaued = False
        if len(d_history) >= plateau_window:
            recent = d_history[-plateau_window:]
            plateaued = (max(recent) - min(recent)
                        < max(tol, plateau_rel_tol * max(recent)))
        if tight or plateaued:
            locked = True
            lock_period = period
    m = s1 / S
    rms_ac = np.sqrt(np.maximum(s2 / S - m * m, 0.0))
    rms_raw = np.sqrt(s2 / S)
    d_final = d_history[-1] if d_history else float('nan')
    return rms_ac, rms_raw, lock_period, locked, d_final


# ─────────────────────────────────────────────────────────────────────────
# .gt property storage (gains_{wt}-style caching)
# ─────────────────────────────────────────────────────────────────────────

def get_or_init_props(g, eps, freqs, dt, tol):
    '''Fetch the per-epsilon bc_* properties, creating NaN-filled ones on
    first use. Errors out if the stored frequency grid differs.'''
    tag = f'eps{eps}'
    fkey = f'bc_frequencies_{tag}'
    if fkey in g.gp:
        stored = np.asarray(g.gp[fkey])
        if len(stored) != len(freqs) or not np.allclose(stored, freqs):
            sys.exit(f'{fkey} already stored with a different grid '
                     f'({len(stored)} pts) — rerun with the same grid or '
                     'a different epsilon, or --force to wipe it.')
    else:
        g.gp[fkey] = g.new_graph_property('vector<double>', val=freqs)
        g.gp[f'bc_dt_{tag}'] = g.new_graph_property('double', val=dt)
        g.gp[f'bc_tol_{tag}'] = g.new_graph_property('double', val=tol)
    props = {}
    for name in (f'bc_rms_{tag}', f'bc_rms_raw_{tag}', f'bc_lock_{tag}',
                f'bc_dfinal_{tag}'):
        if name not in g.vp:
            p = g.new_vertex_property('vector<double>')
            for v in g.vertices():
                p[v] = np.full(len(freqs), np.nan)
            g.vp[name] = p
        props[name] = g.vp[name]
    return tag, props


def wipe_props(g, eps):
    tag = f'eps{eps}'
    for name in (f'bc_rms_{tag}', f'bc_rms_raw_{tag}', f'bc_lock_{tag}',
                f'bc_dfinal_{tag}'):
        if name in g.vp:
            del g.vp[name]
    for name in (f'bc_frequencies_{tag}', f'bc_dt_{tag}', f'bc_tol_{tag}'):
        if name in g.gp:
            del g.gp[name]


def save_atomic(g, path):
    tmp = str(path) + '.tmp'
    g.save(tmp, fmt='gt')
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────
# Per-graph sweep, resumable, concurrency-safe .gt merges, optional CSV
# ─────────────────────────────────────────────────────────────────────────

def pending_freqs(gt_file, eps, leader, freqs, dt, tol):
    '''Under lock: reload, ensure props exist, return which frequency
    indices are still missing for this leader (fresh — reflects whatever
    other concurrent tasks have already finished).'''
    with gt_lock(gt_file):
        g = gt.load_graph(str(gt_file))
        tag, props = get_or_init_props(g, eps, freqs, dt, tol)
        save_atomic(g, gt_file)          # persist newly-created NaN props
        done = ~np.isnan(np.asarray(props[f'bc_rms_{tag}'][g.vertex(leader)]))
    return [j for j in range(len(freqs)) if not done[j]]


def merge_leader_results(gt_file, eps, leader, freqs, dt, tol, results):
    '''Under lock: reload the current .gt (picking up any concurrent
    writes), write this leader's newly-computed (collective_ac,
    collective_raw, lock, d_final) values for the given frequency indices,
    save.'''
    with gt_lock(gt_file):
        g = gt.load_graph(str(gt_file))
        tag, props = get_or_init_props(g, eps, freqs, dt, tol)
        v = g.vertex(leader)
        rms_vec = np.asarray(props[f'bc_rms_{tag}'][v])
        raw_vec = np.asarray(props[f'bc_rms_raw_{tag}'][v])
        lock_vec = np.asarray(props[f'bc_lock_{tag}'][v])
        dfinal_vec = np.asarray(props[f'bc_dfinal_{tag}'][v])
        for j, (coll_ac, coll_raw, lock, d_final) in results.items():
            rms_vec[j] = coll_ac
            raw_vec[j] = coll_raw
            lock_vec[j] = lock
            dfinal_vec[j] = d_final
        props[f'bc_rms_{tag}'][v] = rms_vec
        props[f'bc_rms_raw_{tag}'][v] = raw_vec
        props[f'bc_lock_{tag}'][v] = lock_vec
        props[f'bc_dfinal_{tag}'][v] = dfinal_vec
        save_atomic(g, gt_file)


def append_csv_rows(gt_file, out_csv, rows):
    with gt_lock(gt_file):        # reuse the same lock: cheap, avoids a 2nd file
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, mode='a',
                                  header=not out_csv.exists(), index=False)


def sweep_graph(gt_file, leaders_spec, freqs, dt, eps, tol, max_periods,
                out_csv, per_node, write_csv, force, rng):
    if force:
        with gt_lock(gt_file):
            g = gt.load_graph(str(gt_file))
            wipe_props(g, eps)
            save_atomic(g, gt_file)
        if write_csv:
            out_csv.unlink(missing_ok=True)

    with gt_lock(gt_file):
        g = gt.load_graph(str(gt_file))
    A = gt.adjacency(g).toarray() > 0
    A = A | A.T
    np.fill_diagonal(A, False)
    n = A.shape[0]
    degrees = A.sum(axis=1)
    leaders = parse_leaders(leaders_spec, degrees, rng)

    print(f'{gt_file.name}: n={n}, leaders={len(leaders)} ({leaders_spec}), '
          f'{len(freqs)} freqs, dt={dt:g}'
          + (f' -> {out_csv}' if write_csv else ' (CSV off)'), flush=True)

    others = np.ones(n, dtype=bool)
    n_done_total = 0
    for L in leaders:
        pending = pending_freqs(gt_file, eps, L, freqs, dt, tol)
        if not pending:
            continue
        results = {}
        for j in pending:
            f = freqs[j]
            rms_ac, rms_raw, lock, conv, d_final = run_one(
                A, L, f, dt, eps, tol, max_periods)
            others[:] = True
            others[L] = False
            norm_ac = (rms_ac / rms_ac[L]) if rms_ac[L] > 0 else rms_ac * np.nan
            norm_raw = (rms_raw / rms_raw[L]) if rms_raw[L] > 0 else rms_raw * np.nan
            coll_ac = norm_ac[others].mean()
            coll_raw = norm_raw[others].mean()
            results[j] = (coll_ac, coll_raw, lock, d_final)
            n_done_total += 1
            quality = 'tight' if d_final < tol else ('plateau' if conv else 'FAILED')
            print(f'  leader={L} f={f:.5g} Hz: collective_rms={coll_ac:.4f} '
                  f'lock={lock} d_final={d_final:.2e} ({quality})', flush=True)
            if write_csv:
                row = {'graph': gt_file.stem, 'leader': L,
                       'leader_degree': int(degrees[L]), 'frequency': f,
                       'collective_rms': coll_ac, 'collective_rms_raw': coll_raw,
                       'lock_period': lock, 'd_final': d_final,
                       'converged': bool(conv)}
                if per_node:
                    row.update({f'node_{v}': norm_ac[v] for v in range(n)})
                append_csv_rows(gt_file, out_csv, [row])
        merge_leader_results(gt_file, eps, L, freqs, dt, tol, results)
    if n_done_total == 0:
        print('  nothing to do (already complete; use --force to redo)')


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Bounded-confidence collective-RMS frequency sweep; '
                    'results cached into the .gt as bc_* properties.',
        formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--gt_file', type=Path, help='single .gt graph')
    src.add_argument('--network', choices=('ws', 'mhk'),
                     help='sweep every .gt in nets/LFC/{n}/{network}/{k}_seed1/')
    ap.add_argument('--k', type=int, help='degree dir (with --network)')
    ap.add_argument('--n', type=int, default=240,
                    help='size dir for --network mode (default 240)')
    ap.add_argument('--leaders', default='all',
                    help='all | top:P | bot:P | rand:P | 3,7,9 | 10-40 '
                         '(default all; ~13 min per leader at n=240)')
    ap.add_argument('--epsilon', type=float, default=0.2)
    ap.add_argument('--fmin', type=float, default=-3,
                    help='log10 of lowest frequency (default -3)')
    ap.add_argument('--fmax', type=float, default=0,
                    help='log10 of highest frequency (default 0)')
    ap.add_argument('--nfreq', type=int, default=60)
    ap.add_argument('--samples_per_period', type=int, default=20,
                    help='sets fixed dt = 1/(10**fmax * spp)')
    ap.add_argument('--tol', type=float, default=1e-9,
                    help='period-to-period lock-in tolerance')
    ap.add_argument('--max_periods', type=int, default=100,
                    help='safety cap; plateau detection (see run_one()) '
                         'normally locks within ~15 periods, so this '
                         'should rarely bind (default 100)')
    ap.add_argument('--write_csv', action='store_true',
                    help='also append a per-(leader,frequency) CSV row '
                         '(default off — .gt properties are authoritative)')
    ap.add_argument('--per_node', action='store_true',
                    help='with --write_csv, also add per-node normalized '
                         'AC-RMS columns')
    ap.add_argument('--output_dir', type=Path, default=None,
                    help='CSV dir (default: bc_rms/[{network}/{k}] beside '
                         'this script)')
    ap.add_argument('--seed', type=int, default=0, help='rand: selector seed')
    ap.add_argument('--force', action='store_true',
                    help='wipe stored bc_* properties/CSV and redo')
    args = ap.parse_args()

    if args.network and args.k is None:
        ap.error('--network requires --k')

    freqs = np.logspace(args.fmin, args.fmax, args.nfreq)
    dt = 1.0 / (10.0 ** args.fmax * args.samples_per_period)
    rng = np.random.default_rng(args.seed)

    # .resolve() everywhere: ../LFC/{n}/{network}/{k}/ is a flat symlinked
    # view onto ../nets/LFC/{n}/{network}/{k}_seed{N}/ (see
    # LFC_diffw_props_csv_gen.py). os.replace() on a symlink destination
    # replaces the symlink itself, not its target - writing through the
    # symlinked path would silently detach it from the real file and leave
    # the canonical nets/ copy without the bc_* properties. Resolving first
    # makes every save land on the real file, keeping the symlink intact.
    if args.gt_file:
        gt_files = [args.gt_file.resolve()]
        out_dir = args.output_dir or SCRIPT_DIR / 'bc_rms'
    else:
        src_dir = (SCRIPT_DIR.parents[1] / 'nets' / 'LFC' / str(args.n) / args.network
                   / f'{args.k}_seed1')
        gt_files = sorted(p.resolve() for p in src_dir.glob('*.gt'))
        if not gt_files:
            sys.exit(f'no .gt files in {src_dir}')
        out_dir = args.output_dir or SCRIPT_DIR / 'bc_rms' / args.network / str(args.k)

    for gt_file in gt_files:
        out_csv = out_dir / f'bc_rms_{gt_file.stem}_eps{args.epsilon}.csv'
        sweep_graph(gt_file, args.leaders, freqs, dt, args.epsilon, args.tol,
                    args.max_periods, out_csv, args.per_node, args.write_csv,
                    args.force, rng)


if __name__ == '__main__':
    main()
