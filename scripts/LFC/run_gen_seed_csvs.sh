#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --mem=4G
#SBATCH --time=0-01:00:00 # k=16 alone is up to 100 seeds x 100 .gt files; other k values are seed1-only (fast)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-1

set -euo pipefail

module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

# Slurm copies this script into a spool directory before running it, so
# dirname "${BASH_SOURCE[0]}" resolves to that spool copy's path, not
# scripts/ - use $SLURM_SUBMIT_DIR (the directory sbatch was invoked from)
# instead. submit_gen_seed_csvs.sh must be run from the repo root.
cd "$SLURM_SUBMIT_DIR"

NETWORKS=(ws mhk)
NETWORK=${NETWORKS[$SLURM_ARRAY_TASK_ID]}

# LFC_diffw_props_csv_gen.py skips any seed dir whose CSVs already exist, so a
# re-run after the .gt files changed is a silent no-op that still exits 0. The
# case that needs FORCE=1 is after a bc_rms_sweep.py sweep: that writes bc_rms_*
# properties into existing .gt files, and _bc_gains.csv only appears once the
# extractor re-reads them.
FORCE_ARGS=()
if [ "${FORCE:-0}" != "0" ]; then
    FORCE_ARGS=(--force)
fi

echo "network=$NETWORK (array task $SLURM_ARRAY_TASK_ID) ${FORCE_ARGS[*]:-}"
python scripts/LFC/LFC_diffw_props_csv_gen.py --network "$NETWORK" ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}
