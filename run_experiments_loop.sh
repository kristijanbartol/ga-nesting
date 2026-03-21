#!/bin/bash
# Auto-resuming wrapper for run_experiments.py.
#
# Usage:
#   ./run_experiments_loop.sh [args for run_experiments.py]
#
# Example:
#   ./run_experiments_loop.sh --garment lower --num_bodies 10 --runs 20 --pop 50 --gens 10 --w1 1.0 --w2 10.0
#
# On failure the script finds the latest results directory and resumes
# automatically.  Exits 0 only when run_experiments.py reports all runs done.

MAX_RETRIES=200
SLEEP_ON_FAIL=5

# Extract --garment value from args to locate the results directory.
GARMENT="upper"
args=("$@")
for i in "${!args[@]}"; do
    if [ "${args[$i]}" = "--garment" ]; then
        GARMENT="${args[$((i+1))]}"
    fi
done

RESULTS_BASE="results/experiments/$GARMENT"

echo "[loop] garment=$GARMENT  results_base=$RESULTS_BASE"

for attempt in $(seq 1 $MAX_RETRIES); do
    echo ""
    echo "=========================================="
    echo "[loop] Attempt $attempt / $MAX_RETRIES"
    echo "=========================================="

    # Find the most recently modified results directory (if any).
    LATEST=$(ls -td "$RESULTS_BASE"/[0-9]* 2>/dev/null | head -1)

    if [ -z "$LATEST" ]; then
        echo "[loop] No existing results — fresh start."
        python run_experiments.py "$@"
    else
        echo "[loop] Resuming from $LATEST"
        python run_experiments.py --resume "$LATEST"
    fi

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
