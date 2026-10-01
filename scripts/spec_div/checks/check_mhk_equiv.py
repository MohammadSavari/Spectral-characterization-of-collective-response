"""Assert spectral_divergence_gen.mhk_reference == net_functions.mhk_network.

mhk_reference is a COPY of the fixed generator in the Scale_Free_Social_Contagion
project rather than an import: spectral_divergence_gen.py deliberately carries no
cross-project imports, and it defers `import graph_tool` into save_gt so the CSV path
runs without the module stack (net_functions.py imports graph_tool at module top).
The cost of copying is drift, so this check pins the two together -- it is the gate on
launching the 20-task generator array.

The sys.path insert below is confined to this file on purpose; it must not leak into
spectral_divergence_gen.py.

net_functions.py is the one in the companion Scale-Free-Social-Contagion repository
(its scripts/ folder); point NET_FUNCTIONS_DIR at it. Run on a compute node (via
checks/check_equiv.sh), never on the login node:

    NET_FUNCTIONS_DIR=<path-to-Scale-Free-Social-Contagion>/scripts \
        sbatch scripts/spec_div/checks/check_equiv.sh
"""

import os
import sys

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.dirname(HERE)
NET_FUNCTIONS_DIR = os.environ.get('NET_FUNCTIONS_DIR',
                                   '<path-to-Scale-Free-Social-Contagion>/scripts')

sys.path.insert(0, GEN_DIR)
sys.path.insert(0, NET_FUNCTIONS_DIR)

from spectral_divergence_gen import mhk_reference          # noqa: E402
from net_functions import mhk_network                      # noqa: E402

N = 240
K = 16
P_VALUES = (0.001, 0.3, 0.7, 1.0)
SEEDS = (1, 7, 12345, 2096660437)   # the last is a real ref_seed from the old run


def edges(G):
    return {tuple(sorted(e)) for e in G.edges()}


def main():
    print(f"mhk_reference   <- {GEN_DIR}/spectral_divergence_gen.py")
    print(f"mhk_network     <- {NET_FUNCTIONS_DIR}/net_functions.py")
    print(f"n={N} k={K}, p in {P_VALUES}, {len(SEEDS)} seeds\n")

    failures = []
    for p in P_VALUES:
        for seed in SEEDS:
            G_ours = mhk_reference(N, K, p, seed)
            G_theirs = mhk_network(N, K, p, seed)

            e_ours, e_theirs = edges(G_ours), edges(G_theirs)
            m = len(e_ours)
            degs = np.array([d for _, d in G_ours.degree()])

            checks = {
                'edge-for-edge identical': e_ours == e_theirs,
                'n == 240': G_ours.number_of_nodes() == N,
                '2m == n*k': 2 * m == round(N * K),
                'connected': nx.is_connected(G_ours),
                'no self-loops': nx.number_of_selfloops(G_ours) == 0,
            }
            bad = [name for name, ok in checks.items() if not ok]
            status = 'ok' if not bad else 'FAIL: ' + ', '.join(bad)
            if bad:
                failures.append((p, seed, bad))

            print(f"p={p:<6} seed={seed:<11} m={m:<5} <k>={2 * m / N:5.2f}  "
                  f"deg min/med/max={degs.min():3d}/{int(np.median(degs)):3d}/"
                  f"{degs.max():3d}  {status}")

    # The old generator was near-regular because its anchor was uniform; the fixed one
    # is degree-weighted, so the tail should be well clear of the mean.
    G = mhk_reference(N, K, 0.3, 1)
    degs = np.array([d for _, d in G.degree()])
    print(f"\nheavy-tail spot check (p=0.3, seed=1): max degree {degs.max()} "
          f"vs mean {degs.mean():.1f}  -> ratio {degs.max() / degs.mean():.2f}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S) -- do not launch the array")
        for p, seed, bad in failures:
            print(f"  p={p} seed={seed}: {', '.join(bad)}")
        return 1

    print(f"\nAll {len(P_VALUES) * len(SEEDS)} cases pass. Safe to launch.")
    return 0


_rc = main()
print('EQUIV_CHECK_RESULT=' + ('PASS' if _rc == 0 else 'FAIL'))

# Exit code under sbatch/CLI, printed verdict under the persistent kernel -- where
# sys.exit would surface as a SystemExit traceback rather than a status.
try:
    get_ipython()          # noqa: F821  -- only defined inside an IPython kernel
except NameError:
    sys.exit(_rc)
