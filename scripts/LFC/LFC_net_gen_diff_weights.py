'''
Generates the gt files (no CSVs).

Combines LFC_net_gen_args.py's network generation (ws/mhk, seed/model_type/p/k as
CLI args, one output .gt per (model_type, k, p, seed)) with consensus H2 gain
computation across all three weight types (laplacian, metropolis, lazy), with every
node treated as leader, plus both normalized and unnormalized Laplacian eigenvalues.

mhk generator provenance
------------------------
mhk_network below is a verbatim copy of net_functions.py::mhk_network in the
companion Scale-Free-Social-Contagion repository (scripts/net_functions.py),
which is the single source of truth for this construction.  The copy exists
because this project deliberately carries no cross-project imports; the cost of
copying is drift, and scripts/spec_div/checks/check_mhk_equiv.py is the gate that
pins the two together.

It guarantees two invariants the pre-2026-08 version did not:

  * 2m == round(N*k) EXACTLY, for every seed and every p.  The old version
    added k in-loop edges PLUS a trailing anchor edge per grown node, so <k>
    came out at ~2k (label k=16 measured at <k> 26.85, label k=8 at 15.65),
    and duplicate-edge collapse made the realized <k> drift with p and seed on
    top of that (Spearman(<k>, p) = -0.87, +-3.9 across seeds).
  * Degree-weighted (preferential) attachment.  The old version drew the anchor
    uniformly over nodes, so the family was never actually scale-free.

Note that for mhk, `p` is the TRIAD-FORMATION probability -- the chance a new
node's additional edges close a triangle with a neighbour of its anchor.  It is
NOT the rewiring probability `p` means for ws, and the two are not comparable
on a shared axis.

ws_network is unchanged: it always realized <k> == k and is the regression
baseline for the mhk regeneration.

Example:
    python LFC_net_gen_diff_weights.py --seed 1 --model_type ws --p 0.1 --k 16
'''

import argparse
import numpy as np
import networkx as nx
import scipy as sp
from scipy import linalg
import graph_tool.clustering
import graph_tool.spectral
import graph_tool.topology
import graph_tool as gt
import os
import time

WEIGHT_TYPES = ('laplacian', 'metropolis', 'lazy')


def ws_network(N, k, p, seed=None):
    """
    Function for creating a Watts-Strogatz network
    Takes inputs:
       N: int, Number of nodes
       k: integer, The mean degree of nodes
       p: Probability of rewireing of edges on the graph
       seed: Integer for the random seed
    Returns:
       G: a graphtool graph

    """
    if seed is None:
        seed = np.random.randint(2**63)

    G = gt.Graph(directed=False)
    G.add_edge_list(np.transpose(sp.sparse.tril(nx.adjacency_matrix(nx.connected_watts_strogatz_graph(N, k, p, tries=1000000, seed=seed))).nonzero()))

    return G


def nx_to_gt(G, expected_n=None):
    '''
    Convert a networkx simple graph to an undirected graph_tool Graph.

    Same edge extraction as the inline idiom in ws_network above, plus two
    guards that the exact-degree mhk generator makes load-bearing:

      * add_vertex(n) pre-allocates the vertices, so a graph whose
        highest-labelled node ends up isolated still yields exactly n
        vertices. The bare add_edge_list-only form silently drops trailing
        isolated nodes, which would desync get_gain's np.arange(N) indexing
        and its np.zeros((N, N)) weight matrices.
      * nodelist=sorted(...) makes vertex i in the gt graph the same as node
        i in the nx graph regardless of nx insertion order.
    '''
    n = G.number_of_nodes()
    assert nx.number_of_selfloops(G) == 0, 'self-loops are not supported downstream'
    if expected_n is not None:
        assert n == expected_n, f'generator returned {n} nodes, expected {expected_n}'
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
    T = gt.Graph(directed=False)
    T.add_vertex(n)
    T.add_edge_list(np.transpose(sp.sparse.tril(A).nonzero()))
    return T


def mhk_network(N, k, p, seed=None):
    """
    Modified Holme-Kim network generator.

    Restores the preferential-attachment mechanism (degree-weighted node
    selection) that makes the network scale-free, while guaranteeing the
    realized average degree matches `k` exactly -- regardless of seed, p,
    or duplicate-edge collisions.

    Parameters
    ----------
    N : int
        Number of nodes in the final graph.
    k : float
        Target average degree of the graph (k >= 4 recommended).
    p : float
        Triad-formation probability (0 <= p <= 1): probability that a new
        node's additional edges close a triangle with a neighbor of its
        anchor, versus attaching to another node via preferential attachment.
    seed : int, optional
        Random seed. If None, a random seed is drawn.

    Returns
    -------
    G : networkx.Graph
        The generated undirected graph.  Call nx_to_gt() on it before handing
        it to any of the get_* helpers, which all operate on graph_tool graphs.
    """
    if seed is None:
        seed = np.random.randint(2**63)
    rng = np.random.default_rng(seed)

    # --- bootstrap ring lattice ---
    k0 = max(3, int(np.ceil(k)) + 1)
    G = nx.connected_watts_strogatz_graph(k0, 2, 0, seed=seed)

    # --- fix the TOTAL edge budget so average degree == k exactly ---
    E_target = int(round(k * N / 2))
    E_seed = G.number_of_edges()
    n_new_nodes = N - k0
    if n_new_nodes <= 0:
        raise ValueError("N must be larger than the bootstrap ring size")

    E_remaining = E_target - E_seed
    if E_remaining < n_new_nodes:
        raise ValueError(
            "target average degree too small for this N "
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

    for idx, node in enumerate(range(k0, N)):
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


def get_laplacian_eigenvalues(G):
    if not G.vertex_properties.get('eig_laplacian', False):
        eig_lap = np.linalg.eigvalsh(gt.spectral.laplacian(G, norm=False).todense())
        G.vp['eig_laplacian'] = G.new_vertex_property('double', vals=eig_lap)
    return G


def get_laplacian_eigenvalues_norm(G):
    if not G.vertex_properties.get('eig_laplacian_norm', False):
        eig_lap_norm = np.linalg.eigvalsh(gt.spectral.laplacian(G, norm=True).todense())
        G.vp['eig_laplacian_norm'] = G.new_vertex_property('double', vals=eig_lap_norm)
    return G


def get_local_clutsering(G):
    if not G.vertex_properties.get('local_clustering', False):
        G.vertex_properties['local_clustering'] = graph_tool.clustering.local_clustering(G)
    return G


def get_transitivity(G):
    if not G.gp.get('transitivity', False):
        trans = G.new_graph_property('double', val=graph_tool.clustering.global_clustering(G)[0])
        G.graph_properties['transitivity'] = trans
    return G


def get_ave_shortest_path(G):
    if not G.gp.get('shortest_path', False):
        G.gp['shortest_path'] = G.new_graph_property('double', val=np.sum(graph_tool.topology.shortest_distance(G).get_2d_array(range(G.num_vertices()))) / (G.num_vertices() * (G.num_vertices() - 1)))
    return G


def get_gain(graph, w, N, weight_type):
    '''
    graph: graph-tool Graph object
    w: array of frequencies
    N: number of nodes
    weight_type: 'laplacian', 'metropolis', or 'lazy'

    Computes the H2 gain with every node as leader (no top/bottom-% subsetting).
    '''
    L = gt.spectral.laplacian(graph, norm=False)

    if weight_type == 'laplacian':
        # Random walk normalization: D^-1 L
        L = (L / L.diagonal()).T
    elif weight_type == 'metropolis':
        # Metropolis-Hastings weights
        W = np.zeros((N, N))
        for i in range(N):
            neighbors = [int(v) for v in graph.vertex(i).out_neighbors()]
            if len(neighbors) > 0:
                degree_i = len(neighbors)
                for j in neighbors:
                    degree_j = graph.vertex(j).out_degree()
                    W[i, j] = 1.0 / (max(degree_i, degree_j))
                W[i, i] = 1.0 - np.sum(W[i, :])
        L = np.eye(N) - W
    elif weight_type == 'lazy':
        # Lazy random walk
        W = np.zeros((N, N))
        for i in range(N):
            neighbors = [int(v) for v in graph.vertex(i).out_neighbors()]
            degree_i = len(neighbors)
            if degree_i > 0:
                for j in neighbors:
                    W[i, j] = 1.0 / (2.0 * degree_i)
                W[i, i] = 0.5
        L = np.eye(N) - W

    L = L.toarray() if hasattr(L, 'toarray') else L
    h2 = graph.new_vertex_property('vector<double>')

    for g in range(N):
        ida = np.arange(N) != g
        idb = np.arange(N) == g
        A = L[np.ix_(ida, ida)].astype(complex)
        B = L[np.ix_(ida, idb)]

        A_diag_original = A.diagonal().copy()

        H2 = []
        for f in w:
            np.fill_diagonal(A, A_diag_original + 1j * f)
            h = linalg.solve(A, -B)
            H2.append(linalg.norm(h) ** 2)

        h2[g] = H2

    return h2


def parse_args():
    parser = argparse.ArgumentParser(description='Generate LFC networks for a given seed, model type, and probability, with '
                                                   'consensus gains (laplacian/metropolis/lazy weights, all nodes as leaders) '
                                                   'and normalized+unnormalized Laplacian eigenvalues.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for graph generation')
    parser.add_argument('--model_type', type=str, choices=['mhk', 'ws'], required=True, help='Network model type: mhk or ws')
    parser.add_argument('--p', type=float, required=True, help='Rewiring/edge probability value')
    parser.add_argument('--k', type=int, required=True, help='Mean degree of nodes')
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    n_type = args.model_type
    prob = args.p
    k = args.k

    model = 'LFC'
    n = 240
    w = np.logspace(-4, 1, 100)

    network_path = f"nets/{model}/{n}/{n_type}/{k}_seed{seed}"
    os.makedirs(network_path, exist_ok=True)

    if n_type == "mhk":
        # mhk_network returns networkx; nx_to_gt's guards keep the vertex count
        # and vertex-index ordering intact for get_gain's np.arange(N) indexing.
        G = nx_to_gt(mhk_network(n, k, prob, seed=seed), expected_n=n)
    else:
        G = ws_network(n, k, prob, seed=seed)

    G.graph_properties['ID'] = G.new_graph_property('int64_t', val=int(time.time() * 1000))
    G.graph_properties['ntype'] = G.new_graph_property('string', val=n_type)
    G.graph_properties['probability'] = G.new_graph_property('double', prob)
    G.graph_properties['seed'] = G.new_graph_property('int64_t', val=seed)
    G.graph_properties['frequencies'] = G.new_graph_property('vector<double>', val=w)

    G = get_local_clutsering(G)
    G = get_transitivity(G)
    G = get_ave_shortest_path(G)
    G = get_laplacian_eigenvalues(G)
    G = get_laplacian_eigenvalues_norm(G)

    for wt in WEIGHT_TYPES:
        G.vertex_properties[f'gains_{wt}'] = get_gain(G, w, n, wt)

    # G.gp.ID alone (millisecond timestamp) can collide between concurrent
    # SLURM array tasks writing into the same network_path, silently
    # overwriting each other's file. SLURM_ARRAY_JOB_ID+SLURM_ARRAY_TASK_ID
    # is unique per task cluster-wide; fall back to PID for manual runs.
    job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    unique_suffix = f'{job_id}_{task_id}' if job_id is not None else str(os.getpid())
    out_path = f'{network_path}/{G.gp.ID}_{unique_suffix}.gt'
    G.save(out_path)
    print(out_path)


if __name__ == '__main__':
    main()
