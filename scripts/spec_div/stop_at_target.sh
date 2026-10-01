#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --job-name=spec_div_new_stop
#SBATCH --mem=256M
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/stop_at_target_%j.out

# Watches a spec_divergence_new generator array and cancels each seed's task once that seed
# has reached TARGET pairs, so every seed stops at the same sample size instead of
# running to the 6d12h wall clock.
#
#     FAMILY=ws ARRAY=<jobid> sbatch scripts/spec_div/stop_at_target.sh        # from the repo root
#     FAMILY=mhk ARRAY=<jobid> TARGET=700 sbatch scripts/spec_div/stop_at_target.sh
#
# One watcher per family, since each family is its own array job.
#
# scancel sends SIGTERM, which spectral_divergence_gen.py handles: it finishes the
# pair it is annealing, writes it, and leaves pairs_index.csv complete.  So a seed
# can land one pair past the target -- by design, not by accident.
#
# Cancel the watcher itself with scancel <its own jobid>; the generators then simply
# run to their wall clock.

TARGET=${TARGET:-700}
ARRAY=${ARRAY:?set ARRAY to the generator array job id}
FAMILY=${FAMILY:-ws}
INTERVAL=${INTERVAL:-120}
DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/nets/spec_div"

echo "watching array $ARRAY (family $FAMILY), stopping each seed at $TARGET pairs, polling ${INTERVAL}s"

while true; do
    live=$(squeue -j "$ARRAY" -h -r -o "%i" 2>/dev/null)
    if [ -z "$live" ]; then
        echo "$(date '+%F %T')  no tasks of array $ARRAY left; watcher done"
        break
    fi

    for task in $live; do
        seed=${task##*_}
        index="$DIR/${FAMILY}_seed_${seed}/pairs_index.csv"
        [ -r "$index" ] || continue

        rows=$(wc -l < "$index" 2>/dev/null)
        # a header plus one row per pair; ignore anything that is not a clean number
        case "$rows" in
            ''|*[!0-9]*) continue ;;
        esac
        pairs=$((rows - 1))
        [ "$pairs" -ge 0 ] || continue

        if [ "$pairs" -ge "$TARGET" ]; then
            echo "$(date '+%F %T')  seed $seed reached $pairs pairs -> scancel $task"
            scancel "$task"
        fi
    done

    # one progress line per poll round, compact
    echo "$(date '+%F %T')  $(echo "$live" | wc -w) task(s) live: $(
        for task in $live; do
            seed=${task##*_}
            index="$DIR/${FAMILY}_seed_${seed}/pairs_index.csv"
            [ -r "$index" ] && echo -n "$seed:$(( $(wc -l < "$index") - 1 )) "
        done)"

    sleep "$INTERVAL"
done
