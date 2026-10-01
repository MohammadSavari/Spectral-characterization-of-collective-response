#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --mem=2G
#SBATCH --time=0-00:20:00
#SBATCH --array=0-23999%50
#SBATCH --output=logs/%x_%A_%a.out

# seed2 counterpart of run_bc_rms_leaders_ws16.sh (which did seed1). One
# array task = one (graph, leader) pair out of the 100 ws/16_seed2 graphs
# (nets/LFC/240/ws/16_seed2/*.gt) x 240 leaders = 24000 tasks. Writes the
# collective-RMS frequency sweep straight into the graph's bc_* properties
# (see bc_rms_sweep.py's docstring) - concurrency-safe via flock, so many
# array tasks can hit the same .gt file at once. --array is always
# overridden per chunk by submit_bc_rms_leaders_ws16_seed2.sh; the header
# above is just the full-range default for a direct sbatch call.
#
# Points GT_DIR straight at the real nets/ seed2 directory rather than a
# flat symlink view (there is no seed2 flat view - populate_flat_gt_dir()
# in LFC_diffw_props_csv_gen.py only ever populates *_seed1), matching the
# direct-path convention already used by run_bc_rms_leaders_ws16_eps005.sh.
#
# PREREQUISITE: before first use, run_bc_rms_sweep.py --force --leaders 1-0
# must be run against the 4 seed2 graphs previously touched by
# pilot_bc_seed2_noise_check.py, which stored a mismatched (8-point)
# frequency grid on them - get_or_init_props() in bc_rms_sweep.py will
# sys.exit on those graphs otherwise. See realizations/README.md.
#
# Time budget: same 20 min/leader budget as the seed1 sweep (measured ~13
# min/leader at n=240, k=16 on the vectorized converge-then-measure sweep).
#
# Slurm copies this script into a spool directory before running it, so
# dirname "${BASH_SOURCE[0]}" resolves to that spool copy's path, not
# scripts/ - use $SLURM_SUBMIT_DIR instead (matches run_gen_seed_csvs.sh's
# convention). Must be submitted from the repo root.
#
# TASK_OFFSET: the cluster's MaxArraySize=10000 caps --array task IDs to
# 0-9999, well under this sweep's 24000 tasks -
# submit_bc_rms_leaders_ws16_seed2.sh resets --array to 0-9999-relative per
# megachunk and passes the true offset via --export so TASK_ID =
# SLURM_ARRAY_TASK_ID + TASK_OFFSET still spans the full 0-23999 range.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

N_LEADERS=240
GT_DIR="nets/LFC/240/ws/16_seed2"
mapfile -t GT_FILES < <(ls "$GT_DIR"/*.gt | sort)

TASK_ID=$(( SLURM_ARRAY_TASK_ID + ${TASK_OFFSET:-0} ))
LEADER=$(( TASK_ID % N_LEADERS ))
GRAPH_IDX=$(( TASK_ID / N_LEADERS ))
GT_FILE=${GT_FILES[$GRAPH_IDX]}

echo "graph=$GT_FILE leader=$LEADER (array task $TASK_ID, graph_idx=$GRAPH_IDX)"

python scripts/LFC/bc_rms_sweep.py --gt_file "$GT_FILE" --leaders "$LEADER"
