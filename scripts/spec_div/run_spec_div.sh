#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --job-name=spec_div_new
#SBATCH --mem=8G
#SBATCH --time=6-12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/spec_div_new_%A_%a.out
#SBATCH --mail-user=<your-email>
#SBATCH --mail-type=FAIL,END

# One array task per seed, one array per family. Submit from the repo root:
#     mkdir -p logs
#     FAMILY=ws  sbatch --array=1-20 scripts/spec_div/run_spec_div.sh
#     FAMILY=mhk sbatch --array=1-20 scripts/spec_div/run_spec_div.sh
# Check headroom first: squeue -u $USER -r | tail -n +2 | wc -l
#
# To top up seeds that ended short of the target (see RESUME below):
#     FAMILY=mhk RESUME=1 sbatch --array=3,7,11 scripts/spec_div/run_spec_div.sh
#
# Output goes to nets/spec_div/{family}_seed_{seed}/, so the two families never
# share a directory and either can be cancelled on its own.

source <path-to-venv>/bin/activate
module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPTS="$REPO/scripts/spec_div"
mkdir -p "$REPO/nets/spec_div" && cd "$REPO/nets/spec_div"

# OpenBLAS otherwise sizes its pool from the node's physical cores (192 on the cluster used)
# rather than the SLURM allocation; for 240x240 eigendecompositions that is ~1180x
# slower than a single thread.  The script also sets these, belt and braces.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEED=${SLURM_ARRAY_TASK_ID:-1}
FAMILY=${FAMILY:-ws}

# No internal time limit: SLURM's --time above governs.  On timeout or scancel
# SLURM sends SIGTERM, the generator finishes the pair it is on, writes it, and
# leaves a complete pairs_index.csv.
#
# --refs is an upper bound, not a target: stop_at_target.sh cancels each task once its
# seed reaches TARGET pairs (700 by default).  With --max-per-ref 1 the pair count can
# never exceed the reference count, so --refs must not be allowed to bind before the
# target does.  It was 4000, which at 120 s/ref is 133 h -- below the 6d12h wall, so at
# any yield under ~17.5% a seed would have stopped short of 700 with allocation still
# unused.  20000 puts it out of reach (the wall searches ~4680 references at this
# budget), leaving the wall clock and stop_at_target.sh as the only limits.
#
# Raise --refs to use more of the allocation, not --budget -- a longer anneal on one
# reference has diminishing returns compared with more independent references.  Do not
# change --budget/--rtol/--min-w1 either: the ws seeds were produced at these values
# and the ws<->mhk comparison is paired.
#
# RESUME=1 appends to an existing seed directory instead of wiping it, to top a seed
# up to 700 after it hit the wall short.  Without it a resubmit deletes every pair_*
# file and restarts from pair 1.
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

python "$SCRIPTS/spectral_divergence_gen.py" \
    --family "$FAMILY" \
    --seed "$SEED" \
    --n 240 \
    --k 16 \
    --rtol 0.01 \
    --refs 20000 \
    --budget 120 \
    --max-per-ref 1 \
    --formats gt \
    $RESUME_FLAG

# --formats gt (no CSV) keeps the scratch inode count manageable: 2 files per pair
# instead of 6.  The .gt files already carry every property convert_csv_degroot.py
# would have computed, apart from the gains -- add those with run_gain_all.sh.
