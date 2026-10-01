'''
Rebuild every {family}_seed_{N}/pairs_index.csv from pair_properties_per_pair.csv.

spectral_divergence_gen.py writes one pairs_index.csv per seed directory (the
matching/annealing record of each pair). The original files are not part of this
release; pair_properties/pair_props_gen.py joined all of them, row for row, into
pair_properties_per_pair.csv, renaming base_type to `family` and adding a derived
`mean_degree`. This script inverts that join: it restores base_type, keeps exactly
the generator's INDEX_FIELDS in the generator's order, and writes one file per seed.
Values are copied unchanged (the per-pair table stores them at full precision).

plot_top_pair.py (Fig. 1) is the consumer: it picks the pair with the largest
bulk_w1 across a family's pairs_index.csv files.

    python scripts/spec_div/rebuild_pairs_index.py            # from the repo root
'''

import argparse
import os

import pandas as pd

INDEX_FIELDS = ['pair', 'ref', 'n', 'm', 'base_type', 'k', 'p', 'ref_seed',
                'lambda2_1', 'lambda2_2', 'rel_lambda2',
                'lambdamax_1', 'lambdamax_2', 'rel_lambdamax',
                'bulk_w1', 'anneal_steps']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--per-pair',
                    default='nets/spec_div/pair_properties/pair_properties_per_pair.csv')
    ap.add_argument('--out-root', default='nets/spec_div',
                    help='directory that receives the {family}_seed_{N}/ folders')
    args = ap.parse_args()

    pp = pd.read_csv(args.per_pair).rename(columns={'family': 'base_type'})
    for (family, seed), d in pp.groupby(['base_type', 'seed'], sort=True):
        out_dir = os.path.join(args.out_root, f'{family}_seed_{seed}')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'pairs_index.csv')
        d.sort_values('pair')[INDEX_FIELDS].to_csv(path, index=False)
        print(f'{path}: {len(d)} pairs')


if __name__ == '__main__':
    main()
