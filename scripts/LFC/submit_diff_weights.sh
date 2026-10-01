#!/bin/bash
set -euo pipefail

# Total tasks: seed 1 gets the full k(16 values) x p(100 values) grid = 1600 tasks;
# seeds 2-100 get k=16 only x p(100 values) = 100 tasks each.
# 1600 + 99*100 = 11,500 tasks/.gt files total, for whichever MODEL_TYPE is set below.
# Override with e.g. `MODEL_TYPE=mhk bash submit_diff_weights.sh`.
#
# Resumable: after each chunk submits, the next seed is written to
# .submit_diff_weights_progress_${MODEL_TYPE}. A plain re-run with no args picks up
# from there automatically. This matters because the loop below polls squeue for
# hours (11,500 tasks at ~6 min each), and a re-run from scratch would NOT overwrite
# the already-generated .gt files -- their names embed SLURM_ARRAY_JOB_ID, so a
# second pass produces a second copy per (k, p), silently duplicating (ID, p) rows in
# every downstream CSV. Same failure mode submit_bc_rms_leaders_mhk16.sh guards
# against, same mechanism.
#
# Pass a START seed explicitly to override the checkpoint (e.g. to redo one seed):
#     MODEL_TYPE=mhk bash submit_diff_weights.sh 57
# Note the checkpoint is seed-granular: an interrupted seed's partial chunk is NOT
# tracked, so on resume verify that seed's dir with the coverage check before
# trusting it (the .gt filenames encode SLURM_ARRAY_TASK_ID, which maps
# deterministically to the p index -- see run_diff_weights.sh).

MODEL_TYPE=${MODEL_TYPE:-ws}
CHUNK_SIZE=400    # tasks per sbatch submission (well under the 1000 cap)
MAX_IN_QUEUE=900  # safety margin below the account's 1000-job cap
POLL_INTERVAL=30  # seconds between queue-depth checks while waiting
CONCURRENCY=100   # max simultaneously running tasks per array (the %N throttle)

LAST_SEED=100
PROGRESS_FILE=".submit_diff_weights_progress_${MODEL_TYPE}"

mkdir -p logs

current_queue_depth() {
    squeue -u "$USER" -r -h -o "%i" | wc -l
}

submit_chunked () {  # $1 = seed, $2 = total_tasks
    local SEED=$1 TOTAL=$2 START=0
    while [ "$START" -lt "$TOTAL" ]; do
        END=$(( START + CHUNK_SIZE - 1 ))
        if [ "$END" -ge "$TOTAL" ]; then END=$(( TOTAL - 1 )); fi
        CHUNK_LEN=$(( END - START + 1 ))
        while [ "$(( $(current_queue_depth) + CHUNK_LEN ))" -gt "$MAX_IN_QUEUE" ]; do
            echo "Queue near cap, waiting ${POLL_INTERVAL}s before submitting seed=${SEED} array=${START}-${END} ..."
            sleep "$POLL_INTERVAL"
        done
        sbatch --job-name="LFC_diffw_${MODEL_TYPE}_seed${SEED}" --array="${START}-${END}%${CONCURRENCY}" --export="SEED=${SEED},MODEL_TYPE=${MODEL_TYPE}" scripts/LFC/run_diff_weights.sh
        echo "Submitted seed=${SEED} array=${START}-${END} (${CHUNK_LEN} tasks)"
        START=$(( END + 1 ))
    done
}

# Resume point: explicit arg > checkpoint file > seed 1.
if [ "${1:-}" != "" ]; then
    FIRST_SEED="$1"
elif [ -f "$PROGRESS_FILE" ]; then
    FIRST_SEED=$(cat "$PROGRESS_FILE")
else
    FIRST_SEED=1
fi

if [ "$FIRST_SEED" -gt "$LAST_SEED" ]; then
    echo "Checkpoint ${PROGRESS_FILE} says seed ${FIRST_SEED} > ${LAST_SEED}: nothing left to submit."
    echo "Delete the checkpoint or pass an explicit START seed to re-run."
    exit 0
fi

echo "MODEL_TYPE=${MODEL_TYPE}, starting from seed ${FIRST_SEED}, concurrency %${CONCURRENCY}"

for SEED in $(seq "$FIRST_SEED" "$LAST_SEED"); do
    if [ "$SEED" -eq 1 ]; then
        submit_chunked 1 1600      # full k x p grid
    else
        submit_chunked "$SEED" 100 # k=16, p-only
    fi
    echo "$(( SEED + 1 ))" > "$PROGRESS_FILE"
done

echo "All submissions complete."
