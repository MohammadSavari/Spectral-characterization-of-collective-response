#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --job-name=gain_all_new
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gain_all_%A_%a.out
#SBATCH --mail-user=<your-email>
#SBATCH --mail-type=FAIL,END

# One array task per seed, one array per family. Submit from the repo root:
#     mkdir -p logs
#     FAMILY=ws  sbatch --array=1-20 scripts/spec_div/run_gain_all.sh
#     FAMILY=mhk sbatch --array=1-20 scripts/spec_div/run_gain_all.sh
# Check headroom first: squeue -u $USER -r | tail -n +2 | wc -l

source <path-to-venv>/bin/activate
module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPTS="$REPO/scripts/spec_div"
cd "$REPO/nets/spec_div"

# OpenBLAS otherwise sizes its pool from the node's physical cores (192 on the cluster used)
# rather than the SLURM allocation; at n=240 that is ~1180x slower than a single
# thread.  Parallelism here is over files, one per worker process.  The script
# also sets these, belt and braces.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEED=${SLURM_ARRAY_TASK_ID:-1}
FAMILY=${FAMILY:-ws}

# Measured 0.12 s per graph on one core, and the run is I/O-bound past ~4 workers:
# 100 files took 12 s on 1 worker and 7 s on 4.  A 1000-file seed is ~1 min, the
# ~8000 files a full spec_div run leaves behind ~8 min.  The hour is slack.
#
# Files that already carry a complete gain are skipped, so this is resumable:
# on timeout or scancel, resubmit the same command.  It is also safe to run while
# spectral_divergence_gen.py is still appending pairs to the same directory --
# each .gt is rewritten atomically, and a file caught mid-write is reported and
# left for the next run.  That case also makes the task exit non-zero (and so
# mail on FAIL): re-run, do not treat it as a bug.
python "$SCRIPTS/gain_laplacian_all_gen.py" --seed "$SEED" --dir "${FAMILY}_seed_${SEED}"
