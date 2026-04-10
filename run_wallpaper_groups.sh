#!/usr/bin/env bash
# Run GA for each wallpaper group (num_bodies=1, quick settings) and
# save the simulation PLY + best_individual.json to a per-group directory.
#
# Usage:
#   bash run_wallpaper_groups.sh              # upper garment (default)
#   bash run_wallpaper_groups.sh lower

set -e

GARMENT=${1:-upper}
WALLPAPER_GROUPS=(stripes diagonal_stripes grid p4 p4m pg pmg pgg)
OUT_ROOT="results/wallpaper_groups/${GARMENT}"
PYTHONPATH_EXTRA="potpourri3d/src"

echo "[wallpaper] garment=${GARMENT}  groups=${WALLPAPER_GROUPS[*]}"
echo "[wallpaper] output root: ${OUT_ROOT}"

for GROUP in "${WALLPAPER_GROUPS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Running GA: wallpaper=${GROUP}  garment=${GARMENT}  $(date '+%H:%M:%S')"
    echo "============================================================"

    PYTHONPATH="${PYTHONPATH_EXTRA}" conda run -n ga-nesting-env python -u run_ga.py \
        --garment  "${GARMENT}" \
        --wallpaper "${GROUP}" \
        --num_bodies 1 \
        --pop 20 \
        --gens 5 \
        --seed 0 \
        --w1 1.0 \
        --w2 1.0 \
        --w4 1.0 \
        --no_vis
    echo "  [done] wallpaper=${GROUP}  $(date '+%H:%M:%S')"

    # Save results to a stable per-group location before the next run overwrites them.
    OUT_DIR="${OUT_ROOT}/${GROUP}"
    mkdir -p "${OUT_DIR}"

    SIM_PLY="results/simulation/${GARMENT}/cloth_00000.ply"
    BEST_JSON="results/pattern/best/best_individual.json"

    if [ -f "${SIM_PLY}" ]; then
        cp "${SIM_PLY}" "${OUT_DIR}/cloth_00000.ply"
        echo "[wallpaper] saved PLY  -> ${OUT_DIR}/cloth_00000.ply"
    else
        echo "[wallpaper] WARNING: simulation PLY not found at ${SIM_PLY}"
    fi

    if [ -f "${BEST_JSON}" ]; then
        cp "${BEST_JSON}" "${OUT_DIR}/best_individual.json"
        echo "[wallpaper] saved JSON -> ${OUT_DIR}/best_individual.json"
    else
        echo "[wallpaper] WARNING: best_individual.json not found"
    fi
done

echo ""
echo "[wallpaper] All groups done. Run plot_wallpaper_figure.py to generate the figure."
