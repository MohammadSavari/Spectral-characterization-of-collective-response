#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --mem=2G
#SBATCH --time=0-00:20:00
#SBATCH --array=0-23999%50
#SBATCH --output=logs/%x_%A_%a.out

# eps=0.05 counterpart of run_bc_rms_leaders_ws16.sh (which did eps=0.2).
# One array task = one (graph, leader) pair out of the 100 ws/16_seed1
# graphs (nets/LFC/240/ws/16_seed1/*.gt) x 240 leaders = 24000 tasks.
# Unlike the eps0.2 run, this points GT_DIR straight at the real nets/
# seed1 directory instead of the ../LFC/240/ws/16 flat symlink view, which
# is currently empty (nothing repopulates it) - going direct avoids
# depending on that.
#
# Frequency grid MUST match what's already stored in the 84/100 files that
# already have partial bc_*_eps0.05 properties (get_or_init_props() in
# bc_rms_sweep.py errors out on a grid mismatch): fmin=-3, fmax=-0.699
# (the literal rounded value actually used originally - NOT log10(0.2) -
# confirmed by reverse-engineering from bc_dt_eps0.05=0.25001726748848924,
# which only reproduces from fmax=-0.699 exactly, not the more precise
# log10(0.2)=-0.698970...; a first attempt using the precise value failed
# every task on the grid-mismatch guard), nfreq=40, samples_per_period=20
# (default), tol=1e-9 (default). Per-leader work is idempotent/resumable
# (bc_rms_sweep.py skips already-completed (leader, freq) pairs via
# pending_freqs()), so re-running all 240 leaders against files that are
# already fully done for this epsilon is safe and fast (they just get
# skipped).
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
# 0-9999, well under this sweep's 24000 tasks - submit_bc_rms_leaders_ws16_eps005.sh
# resets --array to 0-9999-relative per megachunk and passes the true
# offset via --export so TASK_ID = SLURM_ARRAY_TASK_ID + TASK_OFFSET still
# spans the full 0-23999 range.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

N_LEADERS=240
GT_DIR="nets/LFC/240/ws/16_seed1"
mapfile -t GT_FILES < <(ls "$GT_DIR"/*.gt | sort)

TASK_ID=$(( SLURM_ARRAY_TASK_ID + ${TASK_OFFSET:-0} ))
LEADER=$(( TASK_ID % N_LEADERS ))
GRAPH_IDX=$(( TASK_ID / N_LEADERS ))
GT_FILE=${GT_FILES[$GRAPH_IDX]}

echo "graph=$GT_FILE leader=$LEADER (array task $TASK_ID, graph_idx=$GRAPH_IDX)"

python scripts/LFC/bc_rms_sweep.py --gt_file "$GT_FILE" --leaders "$LEADER" \
    --epsilon 0.05 --fmin -3 --fmax -0.699 --nfreq 40
