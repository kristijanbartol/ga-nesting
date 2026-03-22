#!/bin/bash
# Auto-resuming wrapper for run_experiments.py.
#
# Usage:
#   ./run_experiments_loop.sh [args for run_experiments.py]
#
# Example:
#   ./run_experiments_loop.sh --garment lower --num_bodies 5 10 25 50 100 --runs 20 --pop 50 --gens 10 --w1 1.0 --w2 10.0
#
# run_experiments.py handles its own resume logic by scanning the results
# directory — just re-run the same command on failure.  This wrapper retries
# automatically until all body counts and seeds are complete (exit 0).

MAX_RETRIES=200
SLEEP_ON_FAIL=5

echo "[loop] args: $@"

for attempt in $(seq 1 $MAX_RETRIES); do
    echo ""
    echo "=========================================="
    echo "[loop] Attempt $attempt / $MAX_RETRIES"
    echo "=========================================="

    python run_experiments.py "$@"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "[loop] All experiments complete (exit 0)."
        exit 0
    fi

    echo "[loop] Exited with code $EXIT_CODE. Retrying in ${SLEEP_ON_FAIL}s..."
    sleep $SLEEP_ON_FAIL
done

echo "[loop] Reached max retries ($MAX_RETRIES). Giving up."
exit 1
