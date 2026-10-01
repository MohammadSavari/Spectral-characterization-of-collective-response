"""Figs. S4 (WS) and S5 (MHK): one 2-row figure per network family, row 1 at
eps = 0.05 and row 2 at eps = 0.3, each row being Fig. 3/S3's bounded-confidence
(row-4) panels for an epsilon not shown there. Run on a compute node only.
"""
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# repo root: REPO_ROOT if set, else the nearest ancestor of the cwd holding scripts/LFC
REPO = os.path.abspath(os.environ.get('REPO_ROOT', os.getcwd()))
while not os.path.isdir(os.path.join(REPO, 'scripts', 'LFC')):
    if os.path.dirname(REPO) == REPO:
        raise RuntimeError('run from inside the repository or set REPO_ROOT')
    REPO = os.path.dirname(REPO)
os.chdir(REPO)
warnings.filterwarnings("ignore")
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathptmx}')

REALIZATIONS_ROOT = 'nets'
FIG_DIR = 'output/figures'

my_new_colors = ['darkslateblue', 'darkcyan', 'coral', 'blue']
main_props = ['CC', 'Rg', 'l_skewness', 'l_kurtosis']
prop_label = {
    'CC': r'$C$',
    'Rg': r'$R_g$',
    'l_skewness': r'$S$',
    'l_kurtosis': r'$K$',
}
cf = -3
sf = 0
SIG_ALPHA = 0.05

NET_CONFIG = {
    'ws': dict(specialp=[51, 30, 12, 7], label='WS'),
    'mhk': dict(specialp=[0, 50, 83, 99], label='MHK'),
}

ARROW_X = (0.0, 0.4425, 0.535, 1.0)
SIMPLE_COMPLEX_Y_ARROW = 0.045
SIMPLE_COMPLEX_Y_TEXT = 0.10


def _draw_simple_complex(ax):
    x0, x1, x2, x3 = ARROW_X
    arrowprops = dict(facecolor='black', shrink=0.01, width=0.005, headwidth=3)
    ax.annotate('', xy=(x0, SIMPLE_COMPLEX_Y_ARROW), xytext=(x1, SIMPLE_COMPLEX_Y_ARROW),
                xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops)
    ax.annotate('', xy=(x3, SIMPLE_COMPLEX_Y_ARROW), xytext=(x2, SIMPLE_COMPLEX_Y_ARROW),
                xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops)
    ax.text((x0 + x1) / 2, SIMPLE_COMPLEX_Y_TEXT, 'Simple', transform=ax.transAxes,
            fontsize=11, ha='center', verticalalignment='center')
    ax.text((x2 + x3) / 2, SIMPLE_COMPLEX_Y_TEXT, 'Complex', transform=ax.transAxes,
            fontsize=11, ha='center', verticalalignment='center')


def _corr_and_pvals(props_df, gains_series, cols):
    aligned_props, aligned_gains = props_df.align(gains_series, join='inner', axis=0)
    rhos, pvals = [], []
    for col in cols:
        if len(aligned_gains) < 2:
            rhos.append(np.nan); pvals.append(np.nan)
            continue
        rho, p = spearmanr(aligned_props[col].values, aligned_gains.values)
        rhos.append(rho); pvals.append(p)
    return rhos, pvals


def load_row4(net, specialp, in_k=16, model='LFC', root=REALIZATIONS_ROOT, eps=0.2):
    ix = pd.IndexSlice
    intended_k = in_k
    nodes = 240
    seed_dir = f'{root}/{model}/{nodes}/{net}/{intended_k}_seed1'

    bc_props = pd.read_csv(f'{seed_dir}_props.csv', sep='\t', index_col=[0, 1])
    network_props = pd.DataFrame(index=bc_props.index)
    network_props['p'] = bc_props.index.get_level_values(1)
    network_props['k'] = intended_k
    network_props['rawCC'] = bc_props['CC']
    network_props['rawSP'] = bc_props['SP']
    network_props['rawRg'] = bc_props['Rg']
    network_props['rawT'] = bc_props['T']
    network_props['l_mean'] = bc_props['l_mean']
    network_props['l_variance'] = bc_props['l_variance']
    network_props['l_skewness'] = bc_props['l_skewness']
    network_props['l_kurtosis'] = bc_props['l_kurtosis']
    network_props['CC'] = network_props['rawCC'] / network_props['rawCC'].max()
    network_props['T'] = network_props['rawT'] / network_props['rawT'].max()
    network_props['Rg'] = network_props['rawRg'] / network_props['rawRg'].min()
    network_props['SP'] = network_props['rawSP'] / network_props['rawSP'].min()

    network_gains = pd.read_csv(f'{seed_dir}_bc_gains.csv', sep='\t', index_col=[1])
    network_gains = network_gains.rename(columns={f'BC_{eps}': 'H2'})
    network_gains = network_gains.dropna(subset=['H2'])

    for f in network_gains.index.unique():
        network_gains.loc[f, 'normH2'] = (
            network_gains.loc[f].H2 / network_gains.loc[f, 'H2'].max()
        ) * (240 / 12)
    if 'k' not in network_gains.columns:
        network_gains['k'] = intended_k

    network_gains = network_gains.reset_index()
    network_gains['freq'] = network_gains['freq'].apply(lambda x: round(x, 5))
    network_gains.set_index(['ID', 'freq'], inplace=True)

    freq16 = network_gains.index.get_level_values(1).unique()
    corr_index_16 = pd.MultiIndex.from_tuples(list(zip(
        main_props * len(freq16),
        ['H2'] * len(freq16) * len(main_props),
        sorted(list(freq16) * len(main_props)))))
    network_corr_16 = pd.DataFrame(index=corr_index_16)
    corr_values_16, pval_values_16 = [], []

    prop_special_grouped = network_props.groupby(level=1)[main_props].mean()
    network_gains_indexed = network_gains.reset_index().set_index(['ID', 'freq', 'p'])

    for f in freq16.sort_values():
        try:
            rhos, pvals = _corr_and_pvals(prop_special_grouped, network_gains_indexed.loc[ix[:, f], 'H2'], main_props)
        except Exception:
            rhos, pvals = [np.nan] * len(main_props), [np.nan] * len(main_props)
        corr_values_16 += rhos
        pval_values_16 += pvals

    if len(corr_values_16) > 0:
        network_corr_16.loc[corr_index_16, 'spearman'] = np.reshape(corr_values_16, (len(corr_values_16), 1))
        network_corr_16.loc[corr_index_16, 'pvalue'] = np.reshape(pval_values_16, (len(pval_values_16), 1))

    prop_special = network_props.groupby(level=1).mean()
    gain_grouped = network_gains_indexed.groupby(level=[2, 1])
    gain_special = gain_grouped.mean()
    gain_special_std = gain_grouped.std()

    return dict(network_corr_16=network_corr_16, specialp=specialp, specialk=intended_k,
                model=model, net=net, prop_special=prop_special, gain_special=gain_special,
                gain_special_std=gain_special_std,
                ylabel_a=r'Collective response $(RMS)$', ylabel_c=r'Correlation $r_s(RMS,\chi)$')


def draw_subplot_a(ax_a, data, row_num, show_legend=False, legend_loc='upper right'):
    prop_special = data['prop_special']
    gain_special = data['gain_special']
    gain_special_std = data.get('gain_special_std')
    specialp = data['specialp']

    ax_a.set_xscale('log')
    ax_a.set_yscale('log')
    ax_a.set_xlabel(r'Frequency $(\omega)$', labelpad=2, fontsize=13)
    ax_a.set_ylabel(data['ylabel_a'], labelpad=2.5, fontsize=13)

    for p in gain_special.index.get_level_values(0).unique()[specialp]:
        C_label = str(prop_special.loc[p].rawCC.round(2))
        if len(C_label) < 4:
            C_label += '0'
        r_label = str((prop_special.loc[p].rawRg.mean() / 1000).round(1))
        r_label += r'\! \times \! 10^{3}'
        label_string = (
            fr'${C_label}  |  {r_label}  |'
            fr'    {prop_special.loc[p].l_mean.round(2)}'
            fr'     \,|     {prop_special.loc[p].l_variance.round(2)}'
            fr'    \,|     {prop_special.loc[p].l_skewness.round(2)}'
            fr'    \,|      {prop_special.loc[p].l_kurtosis.round(2)}$'
        )
        y = gain_special.loc[p].H2.iloc[sf:cf]
        line, = ax_a.plot(y, label=label_string)

        if gain_special_std is not None:
            y_std = gain_special_std.loc[p].H2.iloc[sf:cf]
            lower = (y - y_std).clip(lower=1e-4)
            upper = y + y_std
            ax_a.fill_between(y.index, lower, upper, color=line.get_color(),
                              alpha=0.2, linewidth=0, zorder=line.get_zorder() - 1)

    if show_legend:
        leg1 = ax_a.legend(
            title=(r'$ \;\,\;\;\; C \;\,\;\,\;\; | \;\,\,\;\,\, R_{g}'
                   r'  \;\:\,\;\,\,\;\,  | \,\;\,\;\ M   \,\;\,\ |'
                   r' \;\,\,\;\,\, V \;\,\,\;\,| \,\;\,\;\ S   \,\;\,\ |'
                   r' \,\;\,\;\ K  $'),
            loc=legend_loc, ncol=1, borderpad=0.2, markerscale=0.8,
            handlelength=0.9, handletextpad=0.4, labelspacing=0.3, fontsize=10)
        leg1.get_title().set_position((3.55, 0))
        leg1.get_title().set_fontsize('10')
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_alpha(1.0)
        leg1.get_frame().set_edgecolor('white')

    ax_a.set_xlim([0.0001, 2])
    ax_a.set_ylim([0.017, 520])
    ax_a.grid(True, which="major", ls=":")
    ax_a.tick_params(axis='x', labelsize=13)
    ax_a.tick_params(axis='y', labelsize=13)
    ax_a.text(-0.02, 1.03, rf'$\bf{{({row_num}a)}}$', transform=ax_a.transAxes, fontsize=14)
    _draw_simple_complex(ax_a)


def draw_subplot_b(ax_b, data, row_num, show_legend=False, legend_loc='center left'):
    network_corr_16 = data['network_corr_16']

    for idx, metric in enumerate(main_props):
        sub = network_corr_16.loc[(metric, 'H2')].iloc[sf:cf]
        color = my_new_colors[idx % 4]
        ax_b.plot(
            sub.index, sub['spearman'],
            label=prop_label[metric],
            zorder=[3, 2, 1, 4][idx % 4],
            c=color,
            linewidth=2,
            markersize=5,
            ls=[':', '--', '-', 'dashdot'][idx % 4],
            markeredgewidth=1,
            markerfacecolor='none')

        sig = sub[sub['pvalue'] < SIG_ALPHA]
        ax_b.plot(
            sig.index, sig['spearman'],
            linestyle='None', marker='o', markersize=3.5,
            markerfacecolor=color, markeredgecolor=color, markeredgewidth=0,
            zorder=10)

    if show_legend:
        leg2 = ax_b.legend(
            title=r'$\chi$', loc=legend_loc, borderpad=0.2, markerscale=1,
            handlelength=2, handletextpad=0.4, fontsize=12)
        leg2.get_title().set_position((0, 0))
        leg2.get_title().set_fontsize('13')
        leg2.get_frame().set_facecolor('white')
        leg2.get_frame().set_alpha(1.0)
        leg2.get_frame().set_edgecolor('white')

    ax_b.set_xscale('log')
    ax_b.set_ylabel(data['ylabel_c'], labelpad=2.5, fontsize=13)
    ax_b.set_xlabel(r'Frequency $( \omega )$', labelpad=2, fontsize=13)
    ax_b.set_ylim([-1.5, 1.05])
    ax_b.set_xlim([0.0001, 2])
    ax_b.grid(True, which="major", ls=":")
    ax_b.tick_params(axis='x', labelsize=13)
    ax_b.tick_params(axis='y', labelsize=13)
    ax_b.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([-1, 0, 1]))
    ax_b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_b.fill_between(x=[0.015, 2.9], y1=[-4, -4], y2=[-1.72, -1.72], facecolor='w', zorder=-10)
    ax_b.text(-0.02, 1.04, rf'$\bf{{({row_num}b)}}$', transform=ax_b.transAxes, fontsize=14)
    _draw_simple_complex(ax_b)


def build_bc_supplement(net, eps_list=(0.05, 0.3)):
    """2-row figure for one network family: row i = BC row-4 panels
    (a: collective response, b: Spearman correlation) at eps_list[i-1].
    Mirrors Fig. 3/S3's row 4 exactly, just pulled out on its own for the
    two epsilons that aren't in the main text/S3."""
    specialp = NET_CONFIG[net]['specialp']
    data_list = [load_row4(net, specialp, eps=eps) for eps in eps_list]

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 6, hspace=0.45, wspace=0.85)

    for i, data in enumerate(data_list, start=1):
        ax_a = fig.add_subplot(gs[2 * (i - 1):2 * i, 0:3])
        ax_b = fig.add_subplot(gs[2 * (i - 1):2 * i, 3:6])
        draw_subplot_a(ax_a, data, i, show_legend=(i == 1))
        draw_subplot_b(ax_b, data, i, show_legend=(i == 1))
        ax_a.set_xlim([1e-3, 1e-1]); ax_b.set_xlim([1e-3, 1e-1])

    os.makedirs(FIG_DIR, exist_ok=True)
    label = NET_CONFIG[net]['label']
    out = f'{FIG_DIR}/{label}_BC_supplement.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


if __name__ == '__main__':
    for net in ('ws', 'mhk'):
        path = build_bc_supplement(net)
        print('wrote', path, os.path.getsize(path), 'bytes')
