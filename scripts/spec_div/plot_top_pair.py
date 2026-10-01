'''
Render the five-panel matched-pair figure for the highest-scoring pair of a family.

Reproduces the panel layout of the paper's Fig. 1 (pair1561_analysis.pdf), which was
made under the superseded construction on an n=10 pair:

    (a) Laplacian eigenvalues by index, both graphs
    (b) normalised-Laplacian eigenvalue density
    (c) collective response H^2(omega)
    (d),(e) the two graphs, annotated with <k>, C, l_mean, R_g

"Highest-scoring" means the largest bulk_w1 in the family's pairs_index.csv, i.e. the
pair for which the annealer found the greatest normalised 1-Wasserstein separation of the
spectral bulks while holding lambda_2, lambda_max and the edge count.

One visible difference from the old figure is load-bearing rather than cosmetic: the
revised construction fixes m(G_1) = m(G_2), so <k> is now identical in panels (d) and (e).
In the old figure the two graphs had <k> = 2.6 and 5.4, i.e. the pair differed in density
as well as in spectral shape, which is exactly what the edge-count constraint removes.

H^2 is read from all_graphs_h2_data_{family}.csv rather than recomputed, so the curve is
the same one the correlation analysis consumed.

Data is read from nets/spec_div/ (override with SPEC_DIV_DATA); output goes to
output/spec_div/top_pair_{family}.pdf unless --out is given.

Run through sbatch or an interactive allocation, never the login node.
'''

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import graph_tool as gt

plt.rcParams.update({'axes.labelsize': 15, 'axes.titlesize': 16,
                      'xtick.labelsize': 13, 'ytick.labelsize': 13})

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.environ.get('SPEC_DIV_DATA', os.path.join(REPO, 'nets', 'spec_div'))
BLUE, RED = '#1f77b4', '#d62728'


def top_pair(family):
    '''(seed_dir, pair_number, row) for the largest bulk_w1 in the family.'''
    best = None
    for seed in range(1, 21):
        idx = os.path.join(HERE, f'{family}_seed_{seed}', 'pairs_index.csv')
        if not os.path.exists(idx):
            continue
        for r in csv.DictReader(open(idx)):
            w = float(r['bulk_w1'])
            if best is None or w > best[0]:
                best = (w, f'{family}_seed_{seed}', int(r['pair']), r, seed)
    if best is None:
        raise SystemExit(f'no pairs_index.csv found for {family}')
    return best


def load_nx(path):
    G = gt.load_graph(path)
    H = nx.Graph()
    H.add_nodes_from(range(G.num_vertices()))
    H.add_edges_from((int(e.source()), int(e.target())) for e in G.edges())
    return H


def metrics(H):
    n = H.number_of_nodes()
    eig = np.sort(np.linalg.eigvalsh(nx.laplacian_matrix(H).todense().astype(float)))
    return {
        'k': 2 * H.number_of_edges() / n,
        'C': nx.average_clustering(H),
        'l': nx.average_shortest_path_length(H),
        # Kirchhoff index, same definition as the paper: R_g = n * sum_{i>=2} 1/lambda_i
        'Rg': n * np.sum(1.0 / eig[1:]),
        'eig': eig,
        'eig_norm': np.sort(np.linalg.eigvalsh(
            nx.normalized_laplacian_matrix(H).todense().astype(float))),
    }


def h2_curves(family, pair, seed):
    '''{which: (freqs, H2)} from the family's H2 CSV.'''
    want = {f'G{pair}_{w}_{seed}': w for w in ('1', '2')}
    acc = defaultdict(list)
    with open(os.path.join(HERE, f'all_graphs_h2_data_{family}.csv')) as fh:
        for r in csv.DictReader(fh):
            w = want.get(r['ID'])
            if w:
                acc[w].append((float(r['freq']), float(r['H2'])))
    out = {}
    for w, rows in acc.items():
        rows.sort()
        out[w] = (np.array([x for x, _ in rows]), np.array([y for _, y in rows]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', default='ws', choices=('ws', 'mhk'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--layout-seed', type=int, default=7)
    args = ap.parse_args()

    w1, seed_dir, pair, row, seed = top_pair(args.family)
    print(f'top {args.family} pair: {seed_dir} pair {pair}, bulk_w1 = {w1:.6f}')
    print(f"  n={row['n']} m={row['m']} k={row['k']} p={float(row['p']):.6f} "
          f"ref_seed={row['ref_seed']}")
    print(f"  lambda2  {float(row['lambda2_1']):.6f} vs {float(row['lambda2_2']):.6f} "
          f"(rel {float(row['rel_lambda2']):.5f})")
    print(f"  lambdamax {float(row['lambdamax_1']):.4f} vs {float(row['lambdamax_2']):.4f} "
          f"(rel {float(row['rel_lambdamax']):.5f})")

    G1 = load_nx(os.path.join(HERE, seed_dir, f'pair_{pair}_G1.gt'))
    G2 = load_nx(os.path.join(HERE, seed_dir, f'pair_{pair}_G2.gt'))
    m1, m2 = metrics(G1), metrics(G2)
    assert G1.number_of_edges() == G2.number_of_edges(), 'edge counts must match'
    curves = h2_curves(args.family, pair, seed)
    if set(curves) != {'1', '2'}:
        raise SystemExit(f'H2 rows missing for G{pair}_*_{seed}: found {sorted(curves)}')

    fig = plt.figure(figsize=(15, 8.4), dpi=600)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92], hspace=0.3, wspace=0.3)

    # (a) eigenvalues by index -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    idx = np.arange(1, len(m1['eig']) + 1)
    ax.plot(idx, m1['eig'], color=BLUE, lw=1.4, label='Graph 1')
    ax.plot(idx, m2['eig'], color=RED, lw=1.4, ls='--', label='Graph 2')
    ax.set_xlabel(r'Eigenvalue index $i$')
    ax.set_ylabel(r'$\lambda_i$')
    ax.set_title('(a)', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=11)
    ax.grid(alpha=0.3, ls=':')

    # (b) normalised spectral density ------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 2, 60)
    ax.hist(m1['eig_norm'], bins=bins, density=True, alpha=0.55, color=BLUE, label='Graph 1')
    ax.hist(m2['eig_norm'], bins=bins, density=True, alpha=0.55, color=RED, label='Graph 2')
    ax.set_xlabel('Normalized Eigenvalues')
    ax.set_ylabel('Density')
    ax.set_title('(b)', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=11)
    ax.grid(alpha=0.3, ls=':')

    # (c) collective response ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    f1, y1 = curves['1']
    f2, y2 = curves['2']
    ax.loglog(f1, y1, color=BLUE, lw=1.6, label='Graph 1')
    ax.loglog(f2, y2, color=RED, lw=1.6, ls='--', label='Graph 2')
    ax.set_xlabel(r'Frequency $(\omega)$')
    ax.set_ylabel(r'Collective response ($H^2$)')
    ax.set_title('(c)', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, which='major', alpha=0.5, ls=':')
    ax.grid(True, which='minor', alpha=0.15, ls=':')

    # (d), (e) the two graphs ---------------------------------------------------
    # Both drawn on G_1's layout: the pair differs by a small number of relocated
    # edges, so a shared layout makes that difference visible instead of hiding it
    # behind two unrelated spring embeddings.
    pos = nx.spring_layout(G1, seed=args.layout_seed, iterations=120)
    for col, (G, mm, colour, tag) in enumerate(
            [(G1, m1, BLUE, '(d)'), (G2, m2, RED, '(e)')]):
        ax = fig.add_subplot(gs[1, col])
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.18, alpha=0.28, edge_color='0.35')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=11,
                               node_color=colour, linewidths=0)
        # Pad the view beyond the node layout's tight autoscaled extent, so
        # the stats box below has empty margin to sit in instead of
        # overlapping the node cloud in the top-left corner.
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xpad, ypad = 0.2 * (xlim[1] - xlim[0]), 0.2 * (ylim[1] - ylim[0])
        ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
        ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)
        ax.set_title(tag, loc='left', fontweight='bold')
        # Inside the axes, not to the right of it: at x=1.02 this ran into the
        # neighbouring subplot's text.
        ax.text(0.01, 0.99, '\n'.join([
            r'$\langle k \rangle = %.1f$' % mm['k'],
            r'$C = %.3f$' % mm['C'],
            r'$\ell_{\mathrm{mean}} = %.3f$' % mm['l'],
            r'$R_g \approx %.0f$' % mm['Rg'],
        ]), transform=ax.transAxes, va='top', ha='left', fontsize=13,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.8', alpha=0.85))
        ax.set_axis_off()

    # summary panel instead of an empty third cell
    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()
    ax.text(0.06, 0.5, '\n'.join([
        r'$N = %d$,   $m = %d$' % (G1.number_of_nodes(),
                                   G1.number_of_edges()),
        r'$\lambda_2$:  $%.4f$  vs  $%.4f$' % (
            m1['eig'][1], m2['eig'][1]),
        r'$\lambda_N$:  $%.3f$  vs  $%.3f$' % (
            m1['eig'][-1], m2['eig'][-1]),
        r'bulk divergence  $\mathcal{D} = %.4f$' % w1,
    ]), transform=ax.transAxes, va='center', ha='left', fontsize=17)

    out = args.out or os.path.join(REPO, 'output', 'spec_div',
                                   f'top_pair_{args.family}.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    print('wrote', out)

    print('\n--- panel (d)/(e) annotations ---')
    for tag, mm in (('G1', m1), ('G2', m2)):
        print(f"  {tag}: <k>={mm['k']:.1f} C={mm['C']:.4f} "
              f"l_mean={mm['l']:.4f} Rg={mm['Rg']:.1f}")


if __name__ == '__main__':
    main()
