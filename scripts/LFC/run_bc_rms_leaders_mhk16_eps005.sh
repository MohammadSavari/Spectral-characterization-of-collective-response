#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --mem=2G
#SBATCH --time=0-00:20:00
#SBATCH --array=0-23999%50
#SBATCH --output=logs/%x_%A_%a.out

# eps=0.05 counterpart of run_bc_rms_leaders_mhk16.sh (which did eps=0.2).
# One array task = one (graph, leader) pair out of the 100 mhk/16_seed1
# graphs (nets/LFC/240/mhk/16_seed1/*.gt) x 240 leaders = 24000 tasks.
# Points GT_DIR straight at the real nets/ seed1 directory instead of the
# ../LFC/240/mhk/16 flat symlink view, which is currently empty.
#
# No eps0.05 properties exist yet anywhere in this directory (fresh sweep),
# so there is no pre-existing frequency grid to match here - but the grid
# is kept identical to the ws16 eps0.05 run (fmin=-3, fmax=-0.699 - the
# literal rounded value actually used there, not log10(0.2); see that
# script's comment - nfreq=40, samples_per_period=20 default, tol=1e-9
# default) for consistency across the pooled mhk/ws eps0.05 dataset.
#
# Time budget: same 20 min/leader budget as the eps0.2 sweep, even though
# nfreq=40 here (vs 60) - convergence time per frequency dominates, not
# strictly proportional to grid size, so kept conservative.
#
# Slurm copies this script into a spool directory before running it, so
# dirname "${BASH_SOURCE[0]}" resolves to that spool copy's path, not
# scripts/ - use $SLURM_SUBMIT_DIR instead. Must be submitted from
# the repo root.
#
# TASK_OFFSET: the cluster's MaxArraySize=10000 caps --array task IDs to
# 0-9999, well under this sweep's 24000 tasks - submit_bc_rms_leaders_mhk16_eps005.sh
# resets --array to 0-9999-relative per megachunk and passes the true
# offset via --export so TASK_ID = SLURM_ARRAY_TASK_ID + TASK_OFFSET still
# spans the full 0-23999 range.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

N_LEADERS=240
GT_DIR="nets/LFC/240/mhk/16_seed1"
mapfile -t GT_FILES < <(ls "$GT_DIR"/*.gt | sort)

TASK_ID=$(( SLURM_ARRAY_TASK_ID + ${TASK_OFFSET:-0} ))
LEADER=$(( TASK_ID % N_LEADERS ))
GRAPH_IDX=$(( TASK_ID / N_LEADERS ))
GT_FILE=${GT_FILES[$GRAPH_IDX]}

echo "graph=$GT_FILE leader=$LEADER (array task $TASK_ID, graph_idx=$GRAPH_IDX)"

python scripts/LFC/bc_rms_sweep.py --gt_file "$GT_FILE" --leaders "$LEADER" \
    --epsilon 0.05 --fmin -3 --fmax -0.699 --nfreq 40
