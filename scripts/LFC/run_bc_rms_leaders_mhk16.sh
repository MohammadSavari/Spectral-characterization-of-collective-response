#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --mem=2G
#SBATCH --time=0-00:20:00
#SBATCH --array=0-23999%50
#SBATCH --output=logs/%x_%A_%a.out

# One array task = one (graph, leader) pair out of the 100 mhk/16_seed1
# graphs (../LFC/240/mhk/16/*.gt) x 240 leaders = 24000 tasks. Writes the
# collective-RMS frequency sweep straight into the graph's bc_* properties
# (see bc_rms_sweep.py's docstring) - concurrency-safe via flock, so many
# array tasks can hit the same .gt file at once. --array is always
# overridden per chunk by submit_bc_rms_leaders_mhk16.sh; the header above
# is just the full-range default for a direct sbatch call.
#
# Time budget: measured ~13 min/leader at n=240, k=16 on the vectorized
# converge-then-measure sweep; 20 min gives ~50% margin.
#
# Slurm copies this script into a spool directory before running it, so
# dirname "${BASH_SOURCE[0]}" resolves to that spool copy's path, not
# scripts/ - use $SLURM_SUBMIT_DIR instead (matches run_gen_seed_csvs.sh's
# convention). Must be submitted from the repo root.
#
# TASK_OFFSET: the cluster's MaxArraySize=10000 caps --array task IDs to
# 0-9999, well under this sweep's 24000 tasks - submit_bc_rms_leaders_mhk16.sh
# resets --array to 0-9999-relative per megachunk and passes the true
# offset via --export so TASK_ID = SLURM_ARRAY_TASK_ID + TASK_OFFSET still
# spans the full 0-23999 range.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

N_LEADERS=240
# Points straight at the real nets/ seed1 directory rather than the flat symlinked
# view at ../LFC/240/mhk/16 (which is populated as a side effect of
# LFC_diffw_props_csv_gen.py and was found empty at the 2026-08 mhk regeneration,
# which would have made this sweep fail on set -u at GT_FILES[$GRAPH_IDX]).
# Matches run_bc_rms_leaders_mhk16_eps005.sh / _eps03.sh, which already did this.
GT_DIR="nets/LFC/240/mhk/16_seed1"
mapfile -t GT_FILES < <(ls "$GT_DIR"/*.gt | sort)

TASK_ID=$(( SLURM_ARRAY_TASK_ID + ${TASK_OFFSET:-0} ))
LEADER=$(( TASK_ID % N_LEADERS ))
GRAPH_IDX=$(( TASK_ID / N_LEADERS ))
GT_FILE=${GT_FILES[$GRAPH_IDX]}

echo "graph=$GT_FILE leader=$LEADER (array task $TASK_ID, graph_idx=$GRAPH_IDX)"

python scripts/LFC/bc_rms_sweep.py --gt_file "$GT_FILE" --leaders "$LEADER"
