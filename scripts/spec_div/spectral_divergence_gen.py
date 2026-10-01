"""
Generate pairs of graphs that are matched on lambda_2 and lambda_max but differ
maximally in the bulk of the Laplacian spectrum, using WS and MHK references.

This is realizations/spec_div/spectral_divergence_gen.py with one thing changed:
the reference graph G1.  spec_div anneals away from six synthetic topologies
(path, cycle, wheel, star, ladder, barbell); this anneals away from the network
families the rest of the project studies, so the spectrum-to-H^2 relationship can
be tested on them.  See Purpose.txt.

Everything else -- the matching criterion, the annealing search, the objective,
the .gt schema, the output layout -- is deliberately identical, so results from
the two directories are directly comparable.  base_type carries 'ws'/'mhk' where
spec_div carries 'path'/'barbell'/..., which is what lets the shared downstream
scripts stratify by family with no change at all.

Two differences follow from the reference change:

  * references come from the project's own generators, copied from
    realizations/scripts/LFC_net_gen_diff_weights.py (see mhk_reference for why
    that copy and not network_functions.py's);
  * randomness is derived per reference from (run seed, reference index) rather
    than from one shared stream, so reference i is reproducible regardless of how
    many annealing steps fitted into the budgets before it.  spec_div's single
    stream is consumed by a wall-clock-bounded anneal, which leaves only its first
    reference reproducible from --seed.

This supersedes cospectral_file_gen.py, whose matching criterion did not hold the
pair fixed in the way the study needs.  Measured on the 200 pairs it produced in
cospectral/graph_pairs_seed_{1,13,25,33}:

    rel |dlambda_2|   mean 0.117  max 1.602      71% of pairs off by >5%
    rel |dlambda_max| mean 0.006  max 0.015
    rel |dm|          mean 0.114                 38% of pairs off by >10%
    corr( rel|dm| , spectrum_diff_score ) = 0.766

i.e. the spectral gap -- the quantity that governs the dynamics -- was pinned ~19x
more loosely than lambda_max, and 59% of the variance in the "spectral diversity"
being maximised was explained by the two graphs simply having different edge counts.

What changed here:

  * relative tolerance on BOTH lambda_2 and lambda_max, so the two constraints are
    equally tight regardless of scale (--rtol, default 1%);
  * m(G1) == m(G2) enforced structurally, so trace(L) = 2m is identical and a bulk
    difference cannot be manufactured out of a density difference;
  * one dimensionless objective (normalised 1-Wasserstein between the spectral
    bulks) instead of three unnormalised norms of the same difference vector;
  * simulated annealing over m-preserving edge relocations instead of rejection
    sampling.  Benchmarked at n=240, 45 s per reference graph: rejection sampling
    found 0 pairs meeting a 1% relative criterion, annealing found ~16 000.

The degree sequence is deliberately left free.  Freezing it too (double-edge swaps)
also satisfies the criterion but reaches ~15x less bulk divergence, because there is
almost no room left to move the spectrum.

Output filenames follow cospectral_file_gen.py's contract so that
cospectral/convert_csv_degroot.py consumes this directory unchanged.
"""

import argparse
import csv
import math
import os
import random
import signal
import sys
import time

# OpenBLAS sizes its thread pool from the node's physical core count (192 on
# the cluster used), not from the SLURM allocation, and oversubscription is catastrophic for
# matrices this small.  Measured here for a 240x240 eigvalsh:
#     1 thread    1.70 ms      4 threads   2.32 ms
#     8 threads  18.34 ms      default   2005.35 ms   (1180x slower)
# This must be set before numpy imports its BLAS.  setdefault so the caller can
# still override from the environment.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import matplotlib
matplotlib.use("Agg")            # generator runs headless under SLURM
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx            # noqa: E402
import numpy as np               # noqa: E402

# Annealing schedule.  T is in units of the objective (a normalised W1 distance,
# typically 0-0.5), so these are absolute, not relative to any problem scale.
T_START = 0.05
T_END = 0.002
PENALTY = 40.0   # weight on constraint violation in the annealing objective

_stop = False


def _handle_stop(signum, frame):
    """SLURM sends SIGTERM on scancel/timeout; Ctrl-C sends SIGINT.  Both mean
    'wind up cleanly and leave the manifest complete'.

    This is the only time limit the generator observes -- there is no internal
    wall-clock budget.  SLURM's --time governs, and the grace period before
    SIGKILL is ample: the current pair finishes annealing its remaining seconds
    and is written out in well under a second.
    """
    global _stop
    if not _stop:
        print(f"\nReceived signal {signum}; finishing current pair and stopping.",
              flush=True)
    _stop = True


# --------------------------------------------------------------------------
# spectra and the matching criterion
# --------------------------------------------------------------------------
def laplacian_eigs(A):
    """Sorted Laplacian eigenvalues of a dense adjacency array.

    eigvalsh (not eigvals) because the Laplacian is symmetric: the result is real
    and ascending by construction, so no np.real / np.sort is needed.
    """
    L = np.diag(A.sum(1)) - A
    return np.linalg.eigvalsh(L)


def rel(a, b):
    """Relative difference against the mean of the two values.  Symmetric."""
    denom = 0.5 * (abs(a) + abs(b))
    if denom == 0.0:
        return 0.0
    return abs(a - b) / denom


def bulk_w1(eigs1, eigs2, m):
    """Normalised 1-Wasserstein distance between the two spectral bulks.

    Both spectra are sorted, so the sorted-pairing mean absolute difference IS the
    W1 distance between the empirical spectral measures.  The bulk excludes
    lambda_1 = 0 and the two matched eigenvalues lambda_2 and lambda_max.  Dividing
    by the mean degree 2m/n makes the result dimensionless and comparable across
    densities; because m is equal for the two graphs, sum(lambda) is equal too, so
    this cannot be inflated by a shift in total spectral mass.
    """
    n = len(eigs1)
    return float(np.abs(eigs1[2:-1] - eigs2[2:-1]).mean() / (2.0 * m / n))


def matched(eigs1, eigs2, m1, m2, rtol):
    """The pair criterion.  m1 == m2 is structural, not a tolerance."""
    return (m1 == m2
            and rel(eigs1[1], eigs2[1]) <= rtol
            and rel(eigs1[-1], eigs2[-1]) <= rtol)


# --------------------------------------------------------------------------
# reference pool: the project's own WS / MHK families
# --------------------------------------------------------------------------
FAMILIES = ('ws', 'mhk')

# The rewiring grid run_diff_weights.sh sweeps for the LFC realizations, so a
# reference here sits at a p this project has already characterised elsewhere.
P_GRID = np.linspace(0.001, 1, 100)

# Only used to decorrelate the two families' construction seeds; the p draw is
# deliberately family-independent so ws_seed_N and mhk_seed_N are paired on p.
FAMILY_ID = {'ws': 0, 'mhk': 1}


def ws_reference(n, k, p, seed):
    """Watts-Strogatz, as networkx.

    network_functions.ws_network / LFC_net_gen_diff_weights.ws_network wrap exactly
    this call and convert the result to graph_tool; the search here needs the
    networkx object, so take it before the conversion rather than converting back.
    """
    return nx.connected_watts_strogatz_graph(n, k, p, tries=1000000, seed=seed)


def mhk_reference(n, k, p, seed):
    """Modified Holme-Kim, as networkx.

    Copied from diff_net_realizations/net_functions.py's mhk_network (branch
    100_realization_fixed_mhk) in the Scale_Free_Social_Contagion project, which is
    now the single source of truth for this construction.  logs/check_mhk_equiv.py
    asserts the two are edge-for-edge identical.

    This replaces the earlier copy of LFC_net_gen_diff_weights.mhk_network, which
    was wrong in three ways.  It added k in-loop edges PLUS a trailing anchor edge
    per grown node, so <k> came out at ~2k -- the references here ran at m ~ 3250-3450
    (<k> ~ 27-29) against ws's m = 1920, and duplicate-edge collapse made the realized
    <k> drift with p and seed on top of that.  And it chose the anchor uniformly over
    nodes, so there was no preferential attachment at all: the family was never
    actually scale-free.

    The version below fixes both.  A global edge budget E_target = round(k*n/2) is
    split into a shuffled per-node allowance and duplicates are rejected and retried,
    so 2m == round(n*k) exactly regardless of p, seed or collisions -- at k=16 the
    references are edge-count-matched to the ws ones.  The anchor is drawn from a
    degree-weighted pool, restoring the preferential attachment that makes the degree
    distribution heavy-tailed.

    network_functions.py's mhk_network remains the one version not to copy: it
    computes a seed and then never uses it, drawing from the global numpy RNG, so it
    cannot be reproduced.
    """
    if seed is None:
        seed = np.random.randint(2**63)
    rng = np.random.default_rng(seed)

    # --- bootstrap ring lattice ---
    k0 = max(3, int(np.ceil(k)) + 1)
    G = nx.connected_watts_strogatz_graph(k0, 2, 0, seed=seed)

    # --- fix the TOTAL edge budget so average degree == k exactly ---
    E_target = int(round(k * n / 2))
    E_seed = G.number_of_edges()
    n_new_nodes = n - k0
    if n_new_nodes <= 0:
        raise ValueError("n must be larger than the bootstrap ring size")

    E_remaining = E_target - E_seed
    if E_remaining < n_new_nodes:
        raise ValueError(
            "target average degree too small for this n "
            "(each new node needs at least 1 edge to attach)"
        )

    base, extra = divmod(E_remaining, n_new_nodes)
    edge_budget = [base + 1 if i < extra else base for i in range(n_new_nodes)]
    rng.shuffle(edge_budget)  # avoid systematically favoring early/late nodes

    # --- degree-weighted sampling pool for O(1) preferential attachment ---
    # (classic trick: a node with degree d appears d times in this list, so
    # rng.choice(repeated_nodes) samples proportional to degree)
    repeated_nodes = []
    for node, deg in G.degree():
        repeated_nodes.extend([node] * deg)

    for idx, node in enumerate(range(k0, n)):
        budget = max(1, edge_budget[idx])

        # --- preferential attachment: pick anchor proportional to degree ---
        anchor = rng.choice(repeated_nodes)
        anchor_neigh = list(G.neighbors(anchor))

        G.add_edge(anchor, node)
        repeated_nodes.extend([anchor, node])  # both endpoints gained +1 degree
        added = 1

        attempts, max_attempts = 0, 200 * budget
        while added < budget and attempts < max_attempts:
            attempts += 1
            cand = None

            if rng.random() < p and anchor_neigh:
                # triad formation: close a triangle with a neighbor of anchor
                pool = [z for z in anchor_neigh if z != node and not G.has_edge(z, node)]
                if pool:
                    cand = rng.choice(pool)

            if cand is None:
                # preferential attachment among eligible remaining nodes
                for _ in range(50):
                    candidate = rng.choice(repeated_nodes)
                    if candidate != node and candidate != anchor and not G.has_edge(candidate, node):
                        cand = candidate
                        break

            if cand is not None:
                G.add_edge(cand, node)
                repeated_nodes.extend([cand, node])
                added += 1

    return G


def reference_parameters(run_seed, ref_index, family):
    """(p, ref_seed) for one reference, derived from its INDEX, not from stream
    position.

    spec_div drew everything -- topology deck, reference, and every annealing move
    -- from one random.Random(seed) stream.  anneal() runs on a wall-clock budget,
    so the number of draws it consumes depends on how fast the node is, and only
    the first reference of a run is reproducible from --seed.  Deriving each
    reference from (run_seed, ref_index) instead makes reference i the same graph
    however many annealing steps preceded it.

    p is family-independent, so ws_seed_N's reference i and mhk_seed_N's reference i
    sit at the same rewiring probability -- a paired comparison across families.
    The construction seed does carry the family, so the two are not the same draw.
    """
    p_rng = np.random.default_rng(np.random.SeedSequence([run_seed, ref_index]))
    p = float(P_GRID[p_rng.integers(len(P_GRID))])

    seed_rng = np.random.default_rng(
        np.random.SeedSequence([run_seed, ref_index, FAMILY_ID[family]]))
    return p, int(seed_rng.integers(2 ** 31))


def reference_graph(n, k, family, p, ref_seed):
    """One WS or MHK reference on n nodes.

    Both constructions are connected by design -- connected_watts_strogatz_graph
    retries until it is, MHK attaches every new node to an anchor -- but the search
    needs connectivity to hold, so it is checked rather than assumed.
    """
    build = ws_reference if family == 'ws' else mhk_reference
    G = build(n, k, p, ref_seed)
    G = nx.convert_node_labels_to_integers(G)
    G.add_nodes_from(range(n))
    if not nx.is_connected(G):
        raise ValueError(f'{family} reference at p={p}, seed={ref_seed} is disconnected')
    return G


# --------------------------------------------------------------------------
# annealing state: dense adjacency + edge list with reversible m-preserving moves
# --------------------------------------------------------------------------
class GraphState:
    """Mutable graph carrying its own adjacency matrix and edge list.

    Moves are proposed, applied, evaluated, and either kept or undone in place, so
    a step costs one eigendecomposition and no graph rebuilding.
    """

    def __init__(self, G, n):
        self.n = n
        self.A = nx.to_numpy_array(G, nodelist=range(n))
        self.edges = [tuple(sorted(e)) for e in G.edges()]
        self.edge_set = set(self.edges)

    @property
    def m(self):
        return len(self.edges)

    def propose(self, rng, max_tries=200):
        """Remove a uniformly random edge, add a uniformly random non-edge.

        Preserves m exactly (hence trace(L)); leaves the degree sequence free.
        """
        idx = rng.randrange(len(self.edges))
        old = self.edges[idx]
        for _ in range(max_tries):
            u, v = rng.randrange(self.n), rng.randrange(self.n)
            if u == v:
                continue
            new = (u, v) if u < v else (v, u)
            if new not in self.edge_set:
                return (idx, old, new)
        return None

    def apply(self, move):
        idx, old, new = move
        self.A[old[0], old[1]] = self.A[old[1], old[0]] = 0.0
        self.A[new[0], new[1]] = self.A[new[1], new[0]] = 1.0
        self.edge_set.discard(old)
        self.edge_set.add(new)
        self.edges[idx] = new

    def undo(self, move):
        idx, old, new = move
        self.A[new[0], new[1]] = self.A[new[1], new[0]] = 0.0
        self.A[old[0], old[1]] = self.A[old[1], old[0]] = 1.0
        self.edge_set.discard(new)
        self.edge_set.add(old)
        self.edges[idx] = old

    def to_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(self.edges)
        return G


def anneal(eigs_ref, G_ref, rtol, budget, rng, min_sep, min_w1, max_keep):
    """Search for partners of G_ref, maximising bulk W1 subject to the criterion.

    Starts from G_ref itself (bulk W1 = 0, constraints trivially satisfied) and
    walks away from it under a penalised objective, so the constraints are never
    violated for long.

    The full budget is always spent: bulk W1 grows along the trajectory, so
    stopping as soon as max_keep partners exist would return only the weakly
    divergent ones found in the first seconds.  Candidates are pooled as the walk
    proceeds (a new one is pooled only if it is more than min_sep from every
    pooled W1, which both deduplicates the long runs of near-identical accepted
    states and bounds the pool), and the most divergent max_keep are returned.
    """
    n = G_ref.number_of_nodes()
    m = G_ref.number_of_edges()
    state = GraphState(G_ref, n)
    l2_ref, lmax_ref = eigs_ref[1], eigs_ref[-1]

    def evaluate():
        eigs = laplacian_eigs(state.A)
        if eigs[1] < 1e-8:          # disconnected; connectivity comes free here
            return None
        r2 = rel(l2_ref, eigs[1])
        rmax = rel(lmax_ref, eigs[-1])
        w1 = bulk_w1(eigs_ref, eigs, m)
        obj = w1 - PENALTY * (max(0.0, r2 - rtol) + max(0.0, rmax - rtol))
        return eigs, r2, rmax, w1, obj

    current = evaluate()
    if current is None:
        return [], 0

    pool = []
    steps = 0
    start = time.time()
    while not _stop:
        elapsed = time.time() - start
        if elapsed >= budget:
            break
        temperature = T_START * (T_END / T_START) ** (elapsed / budget)

        move = state.propose(rng)
        if move is None:
            continue
        state.apply(move)
        steps += 1
        candidate = evaluate()

        if candidate is None:
            state.undo(move)
            continue

        delta = candidate[4] - current[4]
        if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
            current = candidate
            eigs, r2, rmax, w1, _ = candidate
            # A pair that satisfies the criterion but has W1 ~ 0 is just two
            # copies of the same spectrum and is useless for the study.  Hub
            # graphs (star, wheel) sit here permanently: lambda_max = n is pinned
            # by the hub degree, so any relocation that moves the bulk also
            # breaks the lambda_max constraint.
            if (r2 <= rtol and rmax <= rtol and w1 >= min_w1
                    and all(abs(w1 - p['bulk_w1']) > min_sep for p in pool)):
                pool.append(dict(graph=state.to_graph(), eigs=eigs.copy(),
                                 rel_lambda2=r2, rel_lambdamax=rmax,
                                 bulk_w1=w1, anneal_steps=steps))
        else:
            state.undo(move)

    pool.sort(key=lambda p: p['bulk_w1'], reverse=True)
    return pool[:max_keep], steps


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
# 'ref' identifies the reference graph a pair was annealed away from.  Every pair
# sharing a ref shares a byte-identical G1; G1 changes only when ref does.  Without
# this column that block structure is not recoverable from the manifest.
#
# 'k', 'p' and 'ref_seed' identify the reference itself: reference_graph(n, k,
# base_type, p, ref_seed) rebuilds it byte-for-byte from these columns alone.
INDEX_FIELDS = ['pair', 'ref', 'n', 'm', 'base_type', 'k', 'p', 'ref_seed',
                'lambda2_1', 'lambda2_2', 'rel_lambda2',
                'lambdamax_1', 'lambdamax_2', 'rel_lambdamax',
                'bulk_w1', 'anneal_steps']


def resume_state(output_dir):
    """(pairs_saved, next_ref) recovered from an existing pairs_index.csv.

    Resuming is only sound because the reference draw is index-derived rather than
    stream-position-derived (see reference_parameters): reference i is the same graph
    whether it is reached in one run or in the second half of a resumed one, and the
    walk seeds itself the same way from (seed, ref).  So a topped-up seed is the same
    sample an uninterrupted run would have produced.

    The 'ref' column is 1-based (save_pair is handed ref + 1), so its maximum is
    already the next 0-based index to search.  References that yielded no partner
    leave no row, so a run that ended on a run of failures re-searches that tail --
    correct, since those references contributed nothing, but not free.
    """
    index = os.path.join(output_dir, 'pairs_index.csv')
    if not os.path.exists(index):
        raise SystemExit(
            f"--resume: {index} does not exist.  Drop --resume to start a fresh run; "
            f"passing it against a missing index would silently generate from scratch "
            f"under resume semantics.")
    pairs, refs = [], []
    with open(index, newline='') as f:
        for row in csv.DictReader(f):
            pairs.append(int(row['pair']))
            refs.append(int(row['ref']))
    if not pairs:
        raise SystemExit(
            f"--resume: {index} holds a header but no pairs.  Drop --resume.")
    return max(pairs), max(refs)


def init_output(output_dir, args, resume=False):
    os.makedirs(output_dir, exist_ok=True)

    # Resuming continues the run already on disk, so neither the pair_* wipe nor the
    # truncating header rewrite below may happen -- both would destroy it.  save_pair
    # opens the index and the summary in append mode, so nothing else is needed.
    if resume:
        with open(os.path.join(output_dir, 'summary.txt'), 'a') as f:
            f.write(f"\n--- resumed {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
        return

    # Pair numbering restarts at 1, so files from a previous run of this seed
    # would otherwise survive as orphans: absent from pairs_index.csv but still
    # matched by convert_csv_degroot.py's '*_adjacency.csv' glob.
    stale = [f for f in os.listdir(output_dir) if f.startswith('pair_')]
    for f in stale:
        os.remove(os.path.join(output_dir, f))
    if stale:
        print(f"removed {len(stale)} file(s) from a previous run of this seed",
              flush=True)
    with open(os.path.join(output_dir, 'pairs_index.csv'), 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=INDEX_FIELDS).writeheader()
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Spectral-divergence graph pairs, n={args.n}\n")
        f.write(f"References: {args.family}, k={args.k}, "
                f"p drawn from linspace(0.001, 1, 100)\n")
        f.write(f"Criterion: rel|dlambda_2| <= {args.rtol} and "
                f"rel|dlambda_max| <= {args.rtol} and m(G1) == m(G2)\n")
        f.write("Objective: normalised bulk 1-Wasserstein distance (maximised)\n")
        f.write(f"Run seed: {args.seed} (per-reference seeds are derived from it; "
                f"see the ref_seed column)\n\n")


# --------------------------------------------------------------------------
# graph properties, matching the names the rest of the project stores on .gt
# --------------------------------------------------------------------------
FREQUENCIES = np.logspace(-4, 1, 100)   # the grid convert_csv_degroot.py uses

# 'laplacian' is the random-walk weighting; the other two are the Degroot variants.
GAIN_KINDS = ('laplacian', 'lazy', 'metropolis')


def transition_laplacian(A, kind):
    """I - W for the requested weighting, matching convert_csv_degroot.get_gain."""
    n = len(A)
    d = A.sum(1)
    if kind == 'laplacian':
        L = np.diag(d) - A
        return (L / L.diagonal()).T                 # D^-1 L, random walk
    if kind == 'lazy':
        W = A / (2.0 * np.maximum(d, 1e-12))[:, None]
        np.fill_diagonal(W, 0.5)
        return np.eye(n) - W
    if kind == 'metropolis':
        W = A / (1.0 + np.maximum(d[:, None], d[None, :]))
        W = W * (A != 0)
        np.fill_diagonal(W, 0.0)
        np.fill_diagonal(W, 1.0 - W.sum(1))
        return np.eye(n) - W
    raise ValueError(f'unknown weighting {kind!r}')


def _gain_chunk(args):
    """H^2 over all frequencies for a block of grounded nodes."""
    L, nodes, w = args
    n = len(L)
    out = np.empty((len(nodes), len(w)))
    for r, g in enumerate(nodes):
        keep = np.arange(n) != g
        M = L[np.ix_(keep, keep)].astype(complex)
        b = L[np.ix_(keep, ~keep)]
        diag0 = M.diagonal().copy()
        for i, f in enumerate(w):
            np.fill_diagonal(M, diag0 + 1j * f)
            out[r, i] = np.linalg.norm(np.linalg.solve(M, -b)) ** 2
    return nodes, out


def collective_response(A, kind='laplacian', w=FREQUENCIES, workers=1):
    """Per-node H^2(w): n grounded solves per frequency.

    This is the expensive property by a wide margin -- everything else here is
    milliseconds.  Parallelised over grounded nodes, which are independent.
    """
    L = transition_laplacian(A, kind)
    n = len(A)
    h2 = np.empty((n, len(w)))
    chunks = [np.arange(i, n, max(workers, 1)) for i in range(max(workers, 1))]
    chunks = [c for c in chunks if len(c)]
    if workers > 1:
        import multiprocessing as mp
        # Workers inherit the parent's SIGTERM/SIGINT handlers, so tearing the
        # pool down made every child print the "finishing current pair" notice.
        # Reset them in the children; only the parent should own shutdown.
        def _reset_signals():
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)

        with mp.Pool(workers, initializer=_reset_signals) as pool:
            for nodes, block in pool.map(_gain_chunk, [(L, c, w) for c in chunks]):
                h2[nodes] = block
    else:
        for c in chunks:
            nodes, block = _gain_chunk((L, c, w))
            h2[nodes] = block
    return h2


def normalised_eigs(A):
    d = A.sum(1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return np.linalg.eigvalsh(np.eye(len(A)) - np.diag(inv_sqrt) @ A @ np.diag(inv_sqrt))


def spectral_moments(eig_norm):
    """Moments of the Gaussian-smoothed normalised spectral density.

    Same construction as network_functions.moments(): sigma = 0.015 on a 0.001
    grid over [0, 2), renormalised to sum 1.
    """
    x = np.arange(0, 2, 0.001)
    sigma = 0.015
    gamma = np.zeros_like(x)
    for lam in eig_norm:
        gamma += (1 / np.sqrt(2 * np.pi * sigma ** 2)) * \
                 np.exp(-((x - lam) ** 2) / (2 * sigma ** 2))
    g = gamma / np.sum(gamma)
    mean = np.sum(g * x)
    var = np.sum(g * (x - mean) ** 2)
    return (float(mean), float(var),
            float(np.sum(g * (x - mean) ** 3) / var ** 1.5),
            float(np.sum(g * (x - mean) ** 4) / var ** 2))


def save_gt(path, G, eigs, meta, gains='none', workers=1, probability=None):
    """Write a graph-tool .gt carrying the full property set.

    Property names follow the ones already on the project's .gt files (see
    realizations/nets/.../*.gt and cospectral/converted_*/*.gt), so code that
    reads those -- small_cospectral/gen_figs.ipynb's eigs_from_gt/props_from_gt,
    realizations/scripts/lfc_data_loader.py -- works on these unchanged.

    Collective-response gains are NOT stored by default.  They are n x 100
    doubles per weighting and dwarf everything else on disk (785 kB with all
    three variants against 17 kB without), and they are cheap to recompute from
    the topology when needed -- pair_analysis_fig.py does exactly that.  Pass
    gains='laplacian'/'all' to store them anyway; 'gains' is then written
    alongside 'gains_laplacian' because the cospectral pipeline reads the former
    and the realizations pipeline the latter.

    graph_tool is imported lazily: it is needed only for this output format.
    """
    import graph_tool as gt

    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n))

    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(sorted(G.edges()))

    # --- vertex properties
    eig_norm = normalised_eigs(A)
    for name, values in (('eig_laplacian', eigs),
                         ('eig_laplacian_norm', eig_norm),
                         ('local_clustering',
                          np.array([nx.clustering(G)[v] for v in range(n)]))):
        vp = g.new_vertex_property('double')
        vp.a = values
        g.vertex_properties[name] = vp

    kinds = () if gains == 'none' else (
        GAIN_KINDS if gains == 'all' else ('laplacian',))
    for kind in kinds:
        h2 = collective_response(A, kind=kind, workers=workers)
        vp = g.new_vertex_property('vector<double>')
        for v in range(n):
            vp[v] = h2[v]
        g.vertex_properties[f'gains_{kind}'] = vp
        if kind == 'laplacian':
            g.vertex_properties['gains'] = vp   # name the cospectral side reads

    # --- graph properties
    l_mean, l_var, l_skew, l_kurt = spectral_moments(eig_norm)
    props = dict(meta)
    props.update(transitivity=float(nx.transitivity(G)),
                 shortest_path=float(nx.average_shortest_path_length(G)),
                 l_mean=l_mean, l_variance=l_var,
                 l_skewness=l_skew, l_kurtosis=l_kurt,
                 # the reference's rewiring probability, not a placeholder: this is
                 # the p the WS/MHK reference was drawn at
                 probability=float(probability) if probability is not None else 0.0,
                 original_name=os.path.basename(path))
    if kinds:
        fp = g.new_graph_property('vector<double>')
        fp = g.new_graph_property('vector<double>', val=FREQUENCIES)
        g.graph_properties['frequencies'] = fp

    for key, value in props.items():
        kind = ('int' if isinstance(value, (int, np.integer)) and
                not isinstance(value, bool)
                else 'double' if isinstance(value, (float, np.floating))
                else 'string')
        g.graph_properties[key] = g.new_graph_property(kind, val=value)
    g.save(path)


def save_pair(output_dir, pair_number, ref_number, G1, eigs1, partner, base_type,
              formats=('csv', 'gt'), seed=None, gains='none', workers=1,
              k=None, p=None, ref_seed=None):
    """CSV filenames match cospectral_file_gen.py so convert_csv_degroot.py works."""
    G2, eigs2 = partner['graph'], partner['eigs']

    if 'gt' in formats:
        for i, (tag, G, eigs) in enumerate((('G1', G1, eigs1), ('G2', G2, eigs2)), 1):
            save_gt(os.path.join(output_dir, f"pair_{pair_number}_{tag}.gt"), G, eigs,
                    # ID matches what convert_csv_degroot.py would have assigned,
                    # so anything keyed on it works either way
                    dict(ID=f'G{pair_number}_{i}',
                         pair=int(pair_number), ref=int(ref_number),
                         seed=int(seed if seed is not None else -1),
                         which=tag, ntype=str(base_type), base_type=str(base_type),
                         # the reference is rebuildable from these three
                         k=int(k if k is not None else -1),
                         ref_seed=int(ref_seed if ref_seed is not None else -1),
                         lambda2=float(eigs[1]), lambda_max=float(eigs[-1]),
                         bulk_w1=float(partner['bulk_w1'])),
                    gains=gains, workers=workers, probability=p)

    if 'csv' in formats:
        for tag, G in (('G1', G1), ('G2', G2)):
            # fmt='%d', not savetxt's default '%.18e'.  The matrix is binary, so
            # the default spends 25 characters ("0.000000000000000000e+00") on
            # each of n^2 entries -- 1.44 MB to store ~300 edges at n=240, versus
            # 115 kB here.  Still a plain CSV that pd.read_csv parses identically,
            # so convert_csv_degroot.py is unaffected.
            np.savetxt(
                os.path.join(output_dir, f"pair_{pair_number}_{tag}_adjacency.csv"),
                nx.to_numpy_array(G, nodelist=range(G.number_of_nodes())),
                delimiter=',', fmt='%d')
        for tag, eigs in (('G1', eigs1), ('G2', eigs2)):
            np.savetxt(
                os.path.join(output_dir, f"pair_{pair_number}_{tag}_eigenvalues.csv"),
                eigs, delimiter=',')

    row = dict(pair=pair_number, ref=ref_number,
               n=G1.number_of_nodes(), m=G1.number_of_edges(),
               base_type=base_type, k=k, p=p, ref_seed=ref_seed,
               lambda2_1=eigs1[1], lambda2_2=eigs2[1],
               rel_lambda2=partner['rel_lambda2'],
               lambdamax_1=eigs1[-1], lambdamax_2=eigs2[-1],
               rel_lambdamax=partner['rel_lambdamax'],
               bulk_w1=partner['bulk_w1'], anneal_steps=partner['anneal_steps'])
    with open(os.path.join(output_dir, 'pairs_index.csv'), 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=INDEX_FIELDS).writerow(row)

    with open(os.path.join(output_dir, 'summary.txt'), 'a') as f:
        f.write(f"Pair {pair_number} (ref {ref_number}, base {base_type}, "
                f"m={row['m']}):\n")
        f.write(f"  lambda_2   {eigs1[1]:.6f} vs {eigs2[1]:.6f}   "
                f"rel {partner['rel_lambda2']:.6f}\n")
        f.write(f"  lambda_max {eigs1[-1]:.6f} vs {eigs2[-1]:.6f}   "
                f"rel {partner['rel_lambdamax']:.6f}\n")
        f.write(f"  bulk W1    {partner['bulk_w1']:.6f}\n\n")


def plot_pair(output_dir, pair_number, eigs1, eigs2):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    idx = np.arange(len(eigs1))
    axes[0].plot(idx, eigs1, label='G1', lw=1.4)
    axes[0].plot(idx, eigs2, label='G2', lw=1.4, ls='--')
    axes[0].set_xlabel('index i')
    axes[0].set_ylabel(r'$\lambda_i$')
    axes[0].set_title('Laplacian spectra (matched at $\\lambda_2$, $\\lambda_{max}$)')
    axes[0].legend()
    axes[0].grid(True, ls='--', alpha=0.5)

    axes[1].plot(idx[2:-1], np.abs(eigs1[2:-1] - eigs2[2:-1]), color='purple', lw=1.2)
    axes[1].set_xlabel('index i')
    axes[1].set_ylabel(r'$|\lambda_i^{(1)} - \lambda_i^{(2)}|$')
    axes[1].set_title('Bulk difference')
    axes[1].grid(True, ls='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"pair_{pair_number}_spectra.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
def parse_arguments(argv=None):
    p = argparse.ArgumentParser(
        description='Generate graph pairs matched on lambda_2, lambda_max and edge '
                    'count, with maximally divergent spectral bulk')
    p.add_argument('--family', choices=FAMILIES, required=True,
                   help='reference network family; also names the output directory')
    p.add_argument('--seed', type=int, default=42,
                   help='run seed.  Indexes the run and names the output directory; '
                        'it is NOT the seed the LFC pipeline passes to '
                        'ws_network/mhk_network -- the per-reference construction '
                        'seed is derived from it and recorded as ref_seed')
    p.add_argument('--n', type=int, default=240, help='nodes per graph')
    p.add_argument('--k', type=int, default=16,
                   help="mean degree of the reference; 16 is the value the LFC "
                        "realizations use for every seed past the first")
    p.add_argument('--rtol', type=float, default=0.01,
                   help='relative tolerance on lambda_2 AND lambda_max')
    p.add_argument('--refs', type=int, default=20,
                   help='number of reference graphs G1 to search around')
    p.add_argument('--budget', type=float, default=120.0,
                   help='annealing seconds per reference graph')
    p.add_argument('--max-per-ref', type=int, default=1,
                   help='partners kept per reference graph.  1 (the default) means '
                        'every pair has its own G1, so pairs are independent; raise '
                        'it to get more pairs per unit compute, at the cost of '
                        'blocks that share a reference graph')
    p.add_argument('--formats', nargs='+', default=['csv', 'gt'],
                   choices=['csv', 'gt'],
                   help="output formats.  'csv' is what convert_csv_degroot.py "
                        "reads; 'gt' needs graph_tool")
    p.add_argument('--gains', default='none', choices=['none', 'laplacian', 'all'],
                   help='collective-response variants stored on the .gt.  Off by '
                        'default: the gain arrays are n x 100 doubles per variant '
                        'and dominate the file size (785 kB with all three, 17 kB '
                        'without).  Compute them downstream instead')
    p.add_argument('--workers', type=int, default=0,
                   help='processes for the gain computation (0 = SLURM_CPUS_PER_TASK)')
    p.add_argument('--min-sep', type=float, default=0.01,
                   help='minimum bulk-W1 separation between kept partners')
    p.add_argument('--min-w1', type=float, default=0.01,
                   help='discard partners whose bulk divergence is below this; '
                        'hub-dominated references produce little above it, so mhk '
                        'may need this lowered (see the README)')
    p.add_argument('--output-dir', default=None,
                   help='default: {family}_seed_{seed}')
    p.add_argument('--resume', action='store_true',
                   help='continue an existing output directory instead of wiping it, '
                        'appending to its pairs_index.csv and picking up at the next '
                        'reference.  Use to top a seed up to the target pair count '
                        'after it hit the wall clock short: without it, a resubmit '
                        'deletes every pair_* file and restarts from pair 1')
    p.add_argument('--plot', action='store_true',
                   help='also write a spectrum figure per pair')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    # No run-wide RNG: references and walks each get their own index-derived
    # stream, so nothing depends on how much of the previous budget was consumed.
    output_dir = args.output_dir or f"{args.family}_seed_{args.seed}"
    if args.resume:
        pairs_saved, start_ref = resume_state(output_dir)
        print(f"resuming {output_dir}: {pairs_saved} pair(s) already present, "
              f"continuing from reference {start_ref + 1}", flush=True)
    else:
        pairs_saved, start_ref = 0, 0
    start_pairs = pairs_saved
    init_output(output_dir, args, resume=args.resume)

    if 'gt' in args.formats:
        try:
            import graph_tool  # noqa: F401
        except ImportError:
            print("graph_tool not importable -- load the module stack "
                  "(StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10) or pass "
                  "--formats csv", flush=True)
            return 2
    if 'csv' not in args.formats:
        print("note: without 'csv', convert_csv_degroot.py cannot read this output",
              flush=True)

    workers = args.workers or int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    if 'gt' in args.formats and args.gains != 'none':
        print(f"storing gains={args.gains} on {workers} worker(s); this is what "
              f"makes the .gt files large", flush=True)

    print(f"family={args.family}  n={args.n}  k={args.k}  rtol={args.rtol}  "
          f"refs={args.refs}  budget={args.budget}s/ref  -> {output_dir}", flush=True)

    start = time.time()
    for ref in range(start_ref, args.refs):
        if _stop:
            break
        # Reference randomness comes from the reference's index, not from the
        # position of a shared stream, so it does not depend on how many annealing
        # steps the previous references happened to fit into their budgets.
        p, ref_seed = reference_parameters(args.seed, ref, args.family)
        G1 = reference_graph(args.n, args.k, args.family, p, ref_seed)
        eigs1 = laplacian_eigs(nx.to_numpy_array(G1, nodelist=range(args.n)))
        m = G1.number_of_edges()
        print(f"ref {ref + 1}/{args.refs}  {args.family} p={p:<7.4f} "
              f"ref_seed={ref_seed:<11} m={m:<6} "
              f"<k>={2 * m / args.n:6.2f}  lambda_2={eigs1[1]:.5f} "
              f"lambda_max={eigs1[-1]:.5f}", flush=True)

        # The walk gets its own per-reference stream for the same reason.
        anneal_rng = random.Random((args.seed, ref))
        partners, steps = anneal(eigs1, G1, args.rtol, args.budget, anneal_rng,
                                 args.min_sep, args.min_w1, args.max_per_ref)

        for partner in partners:
            # Belt and braces: re-check the criterion on what is actually written.
            G2 = partner['graph']
            assert matched(eigs1, partner['eigs'], m, G2.number_of_edges(), args.rtol)
            pairs_saved += 1
            save_pair(output_dir, pairs_saved, ref + 1, G1, eigs1, partner,
                      args.family, formats=args.formats, seed=args.seed,
                      gains=args.gains, workers=workers,
                      k=args.k, p=p, ref_seed=ref_seed)
            if args.plot:
                plot_pair(output_dir, pairs_saved, eigs1, partner['eigs'])

        if partners:
            best = max(p['bulk_w1'] for p in partners)
            print(f"    {len(partners)} pair(s) kept from {steps} steps, "
                  f"best bulk W1 = {best:.4f}", flush=True)
        else:
            print(f"    no partner met the criterion in {steps} steps", flush=True)

    # pairs_saved counts everything in the directory, which on a resumed run includes
    # what was already there; report both so the log says what THIS run contributed.
    added = pairs_saved - start_pairs
    print(f"\nSaved {added} pairs to {output_dir} "
          f"in {time.time() - start:.0f}s ({pairs_saved} in total)", flush=True)
    return 0 if added else 1


if __name__ == '__main__':
    sys.exit(main())
