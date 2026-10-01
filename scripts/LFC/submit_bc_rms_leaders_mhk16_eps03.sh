#!/bin/bash
# Submits run_bc_rms_leaders_mhk16_eps03.sh's 24000-task sweep (100
# mhk/16_seed1 graphs x 240 leaders each, epsilon=0.3) - eps0.3 counterpart
# of submit_bc_rms_leaders_mhk16.sh (which did eps0.2). Chunks submission
# against the account's ~1000-task queue cap, same pattern.
#
# Resumable: after each chunk submits, the next START is written to
# .submit_bc_rms_progress_mhk16_eps03. A plain re-run with no args picks up
# from there automatically. Pass a START value explicitly to override.
#
# MaxArraySize=10000 on this cluster caps --array task IDs to 0-9999, well
# under this sweep's 24000 tasks, so each sbatch call below uses array
# indices LOCAL = GLOBAL - OFFSET (OFFSET = the 10000-wide megachunk GLOBAL
# falls in) and passes OFFSET via --export so run_bc_rms_leaders_mhk16_eps03.sh
# can recover the true global task ID. CHUNK_SIZE chunks never cross a
# megachunk boundary (clamped below) so OFFSET is constant within a chunk.
#
# Usage (from the repo root): bash scripts/LFC/submit_bc_rms_leaders_mhk16_eps03.sh [START]

set -euo pipefail
cd "$(dirname "$0")/../.."

CHUNK_SIZE=400    # tasks per sbatch submission (well under the 1000 cap)
MAX_IN_QUEUE=450  # half the account's 1000-job cap - this driver runs
                  # concurrently with submit_bc_rms_leaders_ws16_eps03.sh
                  # sharing the same account-wide cap; 900 each let both
                  # combined blow past 1000 and hit AssocMaxSubmitJobLimit
POLL_INTERVAL=30  # seconds between queue-depth checks while submitting
N_GRAPHS=100      # nets/LFC/240/mhk/16_seed1/*.gt count - must match run_bc_rms_leaders_mhk16_eps03.sh's GT_DIR
N_LEADERS=240
TOTAL_TASKS=$(( N_GRAPHS * N_LEADERS ))
PROGRESS_FILE=".submit_bc_rms_progress_mhk16_eps03"

mkdir -p logs

current_queue_depth() {
    squeue -u "$USER" -r -h -o "%i" | wc -l
}

if [ "${1:-}" != "" ]; then
    START="$1"
elif [ -f "$PROGRESS_FILE" ]; then
    START=$(cat "$PROGRESS_FILE")
else
    START=0
fi

MAX_ARRAY_SIZE=10000

while [ "$START" -lt "$TOTAL_TASKS" ]; do
    OFFSET=$(( (START / MAX_ARRAY_SIZE) * MAX_ARRAY_SIZE ))
    MEGACHUNK_END=$(( OFFSET + MAX_ARRAY_SIZE - 1 ))

    END=$(( START + CHUNK_SIZE - 1 ))
    if [ "$END" -ge "$TOTAL_TASKS" ]; then
        END=$(( TOTAL_TASKS - 1 ))
    fi
    if [ "$END" -gt "$MEGACHUNK_END" ]; then
        END=$MEGACHUNK_END
    fi
    CHUNK_LEN=$(( END - START + 1 ))
    LOCAL_START=$(( START - OFFSET ))
    LOCAL_END=$(( END - OFFSET ))

    while [ "$(( $(current_queue_depth) + CHUNK_LEN ))" -gt "$MAX_IN_QUEUE" ]; do
        echo "Queue near cap, waiting ${POLL_INTERVAL}s before submitting bc_rms_leaders_eps03 array=${START}-${END} (offset=${OFFSET}) ..."
        sleep "$POLL_INTERVAL"
    done

    sbatch --job-name="bc_rms_leaders_mhk16_eps03" --array="${LOCAL_START}-${LOCAL_END}%50" \
        --export="ALL,TASK_OFFSET=${OFFSET}" scripts/LFC/run_bc_rms_leaders_mhk16_eps03.sh
    echo "Submitted bc_rms_leaders_eps03 global=${START}-${END} (local=${LOCAL_START}-${LOCAL_END}, offset=${OFFSET}, ${CHUNK_LEN} tasks)"

    START=$(( END + 1 ))
    echo "$START" > "$PROGRESS_FILE"
done

echo "All bc_rms_leaders_mhk16_eps03 submissions complete (${TOTAL_TASKS} tasks)."
