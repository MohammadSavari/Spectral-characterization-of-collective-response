"""
Store the all-node Laplacian collective response on every .gt file of one seed
directory, under the vertex property 'gain_laplacian_all'.

spectral_divergence_gen.py writes the pairs without gains (its --gains default is
'none') because they are n x 100 doubles and dwarf everything else on disk.  This
script adds them back after the fact, in place, for a whole seed at a time:

    python gain_laplacian_all_gen.py --seed 1

Files that already carry a complete gain -- the property exists and every vertex
holds a full 100-frequency vector -- are skipped, so the script is resumable and
safe to re-run while spectral_divergence_gen.py is still appending pairs to the
same directory.  A partially written property (some vertices empty, which is what
a killed run of an earlier gain writer would leave) counts as missing and is
recomputed.  Each file is written to a temporary name and moved into place with
os.replace, so an interrupted run never leaves a truncated .gt behind.

H^2 definition
--------------
Identical to spectral_divergence_gen.collective_response(kind='laplacian'), i.e.
to convert_csv_degroot.get_gain(weight_type='laplacian'): the random-walk
Laplacian D^-1 L grounded at each node in turn, on w = logspace(-4, 1, 100).
--method reference calls that function directly; --method fast (the default) is
the algebraic shortcut below.  Measured on pairs from spec_div_seed_1, the two
agree to a worst relative deviation of 8.3e-12 (--verify re-measures this on any
files you point it at).

Why the fast method
-------------------
The reference implementation grounds each node in turn and solves an (n-1)x(n-1)
complex system per frequency: n x 100 = 24 000 dense solves per graph, 18 s at
n=240 on one HPC core.  Over the ~1000-8000 files a seed accumulates, times 20
seeds, that is hundreds of CPU-hours.

All n grounded solves at one frequency are columns of a single inverse.  With
L_rw = D^-1 L, N = L_rw + i w I and b_j = L_rw[j, g], the grounded system
M x = -b (M = N with row/column g deleted) is solved by y = (y_g = 1, y_k = x_k),
because (N y)_j = (M x)_j + b_j = 0 for every j != g.  So N y = r e_g, and y is
the g-th column of N^-1 rescaled to y_g = 1:

    H^2(g, w) = ||x||^2 = sum_{j != g} |N^-1[j, g]|^2 / |N^-1[g, g]|^2

N is not symmetric, but L_rw = D^-1/2 S D^1/2 with S = D^-1/2 L D^-1/2 the
symmetric normalised Laplacian, so with S = Q diag(lam) Q^T computed once per
graph and G_w = Q diag(1/(lam + i w)) Q^T,

    N^-1[j, g] = (d_g / d_j)^1/2 G_w[j, g]
    H^2(g, w)  = d_g * sum_{j != g} |G_w[j, g]|^2 / d_j  /  |G_w[g, g]|^2

Per frequency that is two real 240x240 matmuls instead of 240 complex solves:
one eigendecomposition plus 100 x 4n^3 flops, 0.12 s per graph against 18 s, a
~150x saving.  The j = g term is dropped before summing rather than by
subtracting 1 from the total (||x||^2 = ||y||^2 - 1), which keeps the
high-frequency values -- where that term dominates the column -- free of
cancellation.

Threads
-------
OMP_NUM_THREADS and friends are pinned to 1 before numpy imports, for the reason
spectral_divergence_gen.py documents: OpenBLAS sizes its pool from the node's
physical cores (192 on the cluster used), not the SLURM allocation, and at n=240 that is
~1180x slower than a single thread.  Parallelism here is over files instead --
--workers processes, each single-threaded.
"""

import argparse
import os
import signal
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_divergence_gen import FREQUENCIES, collective_response  # noqa: E402

PROP = 'gain_laplacian_all'

_stop = False


def _handle_stop(signum, frame):
    """SIGTERM (scancel/timeout) or SIGINT: stop dispatching new files.

    Every file is saved atomically, so whatever is complete on disk stays
    complete and re-running the script picks up where this one left off.
    """
    global _stop
    if not _stop:
        print(f"\nReceived signal {signum}; stopping after the files in flight.",
              flush=True)
    _stop = True


# --------------------------------------------------------------------------
# the gain itself
# --------------------------------------------------------------------------
def gain_all_nodes(A, w=FREQUENCIES):
    """H^2(node, frequency) for every grounded node, via one eigendecomposition.

    Exactly the quantity collective_response(A, kind='laplacian') returns; see the
    module docstring for the derivation.
    """
    n = len(A)
    d = A.sum(1)
    if np.any(d <= 0):
        raise ValueError('isolated vertex: the random-walk Laplacian D^-1 L is '
                         'undefined (graphs from this pipeline are connected)')

    inv_sqrt = 1.0 / np.sqrt(d)
    S = np.eye(n) - inv_sqrt[:, None] * A * inv_sqrt[None, :]
    lam, Q = np.linalg.eigh(S)

    inv_d = 1.0 / d
    QT = np.ascontiguousarray(Q.T)
    h2 = np.empty((n, len(w)))
    for i, f in enumerate(w):
        scale = 1.0 / (lam + 1j * f)      # lam >= 0 and f > 0, so never singular
        # |G_w|^2 elementwise, from the real and imaginary parts separately:
        # two real matmuls are cheaper than one complex one.
        P = ((Q * scale.real) @ QT) ** 2 + ((Q * scale.imag) @ QT) ** 2
        diag = np.diagonal(P).copy()      # |G_w[g, g]|^2
        np.fill_diagonal(P, 0.0)          # drop j = g before summing, not after
        h2[:, i] = d * (inv_d @ P) / diag
    return h2


def adjacency_array(g):
    """Dense adjacency of a graph-tool graph, in vertex-index order."""
    from graph_tool.spectral import adjacency
    return np.asarray(adjacency(g).todense(), dtype=float)


def has_full_gain(g, prop, nfreq):
    """True only if every vertex carries a full, finite frequency vector.

    graph_tool leaves unset vertices of a vector<double> property as empty
    vectors, so a run that died partway through writing one shows up here as
    missing rather than as present-but-wrong.
    """
    if prop not in g.vp:
        return False
    vp = g.vp[prop]
    if any(len(vp[v]) != nfreq for v in g.vertices()):
        return False
    return bool(np.all(np.isfinite(vp.get_2d_array(range(nfreq)))))


def store_gain(g, h2, prop, w):
    vp = g.new_vertex_property('vector<double>')
    vp.set_2d_array(h2.T)                 # set_2d_array takes (frequency, vertex)
    g.vertex_properties[prop] = vp
    if 'frequencies' not in g.gp:
        # the name spectral_divergence_gen.py and the realizations pipeline use
        g.graph_properties['frequencies'] = g.new_graph_property(
            'vector<double>', val=w)


def process_file(job):
    """Add the gain to one .gt file.  Runs in a worker process."""
    path, prop, method, force, w = job
    import graph_tool as gt

    t0 = time.time()
    try:
        g = gt.load_graph(path)
    except Exception as exc:
        # spectral_divergence_gen.py may be mid-write on this very file; a later
        # run picks it up.
        return path, 'error', 0.0, f'{type(exc).__name__}: {exc}'

    if not force and has_full_gain(g, prop, len(w)):
        return path, 'skip', time.time() - t0, ''

    try:
        A = adjacency_array(g)
        h2 = (gain_all_nodes(A, w) if method == 'fast'
              else collective_response(A, kind='laplacian', w=w, workers=1))
        store_gain(g, h2, prop, w)
        tmp = f'{path}.tmp{os.getpid()}'
        g.save(tmp, fmt='gt')
        os.replace(tmp, path)
    except Exception as exc:
        return path, 'error', time.time() - t0, f'{type(exc).__name__}: {exc}'
    return path, 'done', time.time() - t0, ''


# --------------------------------------------------------------------------
def seed_files(directory, limit=0):
    files = sorted(f for f in os.listdir(directory) if f.endswith('.gt'))
    if limit:
        files = files[:limit]
    return [os.path.join(directory, f) for f in files]


def verify(files, w, count):
    """Check the fast method against the reference one on real graphs."""
    import graph_tool as gt

    worst = 0.0
    for path in files[:count]:
        g = gt.load_graph(path)
        A = adjacency_array(g)
        t0 = time.time()
        fast = gain_all_nodes(A, w)
        t_fast = time.time() - t0
        t0 = time.time()
        ref = collective_response(A, kind='laplacian', w=w, workers=1)
        t_ref = time.time() - t0
        dev = float(np.max(np.abs(fast - ref) / np.abs(ref)))
        worst = max(worst, dev)
        print(f"{os.path.basename(path):<22} n={g.num_vertices():<4} "
              f"max rel dev {dev:.3e}   fast {t_fast:6.2f}s  "
              f"reference {t_ref:7.2f}s  ({t_ref / max(t_fast, 1e-9):5.1f}x)",
              flush=True)
    print(f"\nworst relative deviation over {min(count, len(files))} graph(s): "
          f"{worst:.3e}", flush=True)
    return 0 if worst < 1e-8 else 1


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(
        description="Store the all-node Laplacian collective response on every .gt "
                    "of a spec_div seed directory, under 'gain_laplacian_all'")
    p.add_argument('--seed', type=int, required=True,
                   help='seed whose directory is processed')
    p.add_argument('--root', default=os.path.dirname(os.path.abspath(__file__)),
                   help='directory holding the spec_div_seed_* folders '
                        '(default: this script\'s directory)')
    p.add_argument('--dir', default=None,
                   help='process this directory instead of root/spec_div_seed_{seed}')
    p.add_argument('--prop', default=PROP,
                   help=f'vertex property to write (default: {PROP})')
    p.add_argument('--method', default='fast', choices=['fast', 'reference'],
                   help="'fast' is the eigendecomposition route, ~150x quicker and "
                        "equal to ~1e-11; 'reference' calls "
                        'spectral_divergence_gen.collective_response directly')
    p.add_argument('--workers', type=int, default=0,
                   help='processes, one file each (0 = SLURM_CPUS_PER_TASK)')
    p.add_argument('--force', action='store_true',
                   help='recompute even where a complete gain is already stored')
    p.add_argument('--limit', type=int, default=0,
                   help='only the first N .gt files, for testing')
    p.add_argument('--verify', type=int, default=0, metavar='N',
                   help='compare both methods on N graphs, write nothing, exit')
    p.add_argument('--dry-run', action='store_true',
                   help='report how many files need the gain, write nothing')
    p.add_argument('--progress-every', type=int, default=50,
                   help='files between progress lines')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        import graph_tool  # noqa: F401
    except ImportError:
        print('graph_tool not importable -- load the module stack '
              '(StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10)', flush=True)
        return 2

    directory = args.dir or os.path.join(args.root, f'spec_div_seed_{args.seed}')
    if not os.path.isdir(directory):
        print(f'no such directory: {directory}', flush=True)
        return 2

    files = seed_files(directory, args.limit)
    if not files:
        print(f'no .gt files in {directory}', flush=True)
        return 1

    w = np.asarray(FREQUENCIES)
    print(f"{directory}: {len(files)} .gt file(s), property '{args.prop}', "
          f"{len(w)} frequencies", flush=True)

    if args.verify:
        return verify(files, w, args.verify)

    if args.dry_run:
        import graph_tool as gt
        todo = sum(not has_full_gain(gt.load_graph(f), args.prop, len(w))
                   for f in files)
        print(f'{todo} file(s) need the gain, {len(files) - todo} already have it',
              flush=True)
        return 0

    workers = args.workers or int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f'method={args.method}  workers={workers}'
          f'{"  (forcing recompute)" if args.force else ""}', flush=True)

    jobs = [(f, args.prop, args.method, args.force, w) for f in files]
    counts = {'done': 0, 'skip': 0, 'error': 0}
    start = time.time()

    def report(result):
        path, status, _, message = result
        counts[status] += 1
        if status == 'error':
            print(f'  {os.path.basename(path)}: {message}', flush=True)
        n_seen = sum(counts.values())
        if n_seen % args.progress_every == 0 or n_seen == len(jobs):
            elapsed = time.time() - start
            left = len(jobs) - n_seen
            rate = counts['done'] / elapsed if counts['done'] else 0.0
            eta = f'{left / rate / 60:.0f} min' if rate else '-'
            print(f"  {n_seen}/{len(jobs)}  computed {counts['done']}  "
                  f"skipped {counts['skip']}  errors {counts['error']}  "
                  f"{elapsed / 60:.1f} min elapsed  ETA {eta}", flush=True)

    if workers > 1:
        import multiprocessing as mp

        # Children must keep the default SIGTERM handler, or pool.terminate()
        # cannot reap them; only the parent decides when to wind up.
        def _reset_signals():
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)

        pool = mp.Pool(workers, initializer=_reset_signals)
        try:
            for result in pool.imap_unordered(process_file, jobs, chunksize=1):
                report(result)
                if _stop:
                    break
        finally:
            pool.terminate()
            pool.join()
    else:
        for job in jobs:
            report(process_file(job))
            if _stop:
                break

    elapsed = time.time() - start
    print(f"\n{counts['done']} computed, {counts['skip']} already present, "
          f"{counts['error']} error(s) in {elapsed / 60:.1f} min", flush=True)
    if _stop or counts['error']:
        print('re-run the same command to finish the remaining files', flush=True)
    # Non-zero on any error, including the benign one of catching a file while
    # spectral_divergence_gen.py is writing it: either way the directory is not
    # fully covered yet and the run should be repeated.
    return 1 if counts['error'] else 0


if __name__ == '__main__':
    sys.exit(main())
