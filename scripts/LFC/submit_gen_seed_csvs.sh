#!/bin/bash
set -euo pipefail

# Submits a 2-task array (ws, mhk) that writes one props.csv + gains.csv
# per seed directory under nets/LFC/240/{network}/{k}_seed{N}/. Generation
# is mode-agnostic - run this once; figure.ipynb's MODE toggle picks how
# many seeds to pool at load time via pooled_props/pooled_gains.

mkdir -p logs

sbatch --job-name="gen_seed_csvs" --array=0-1 scripts/LFC/run_gen_seed_csvs.sh
echo "Submitted array 0-1 (ws, mhk)"
