"""Generate a publication-quality B0 vs GA comparison figure.

Usage:
    python gen_comparison_figure.py [--garment upper] [--out paper/figures/comparison.pdf]

Requires: run_ga.py must have been run first (produces best_individual.json
and patches in results/pattern/best/).
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from copy import deepcopy

from ga.geometry_block import build_instance, run_geometry_blackbox_timeout
from ga.real_evaluator import load_patch_vertices_full_from_latest
from nesting.loader import PatchLoader
from nesting.engine import NestingEngine
from nesting.phase_utils import TextureLattice, Rigid2D, seam_phase_mismatch
from nesting.stage2_global_align import (
    load_seam_constraints_from_dir,
    solve_global_alignment_all_components,
)
from nesting.vis_utils import (
    PATCH_COLORS,
    _polygon_to_path,
    _draw_texture_in_patch,
)
from spec import SeamPathType

LATEST_ROOT = "results/pattern/latest"
PERIOD_U = 50.0
PERIOD_V = 50.0
FABRIC_W = 1500.0
K = 8


def _seam_importance_map(instance):
    return {
        s.name: s.importance
        for s in instance.active_seam_definitions
        if s.path_type == SeamPathType.GEODESIC
    }


def _weights_by_filename(seam_dir, importance):
    import re
    result = {}
    if not os.path.isdir(seam_dir):
        return result
    for fn in os.listdir(seam_dir):
        m = re.match(r"seam-(.+)_\d+-\d+\.txt$", fn)
        if m:
            result[fn] = importance.get(m.group(1), 0.0)
    return result


def draw_layout(ax, fabric_state, texture_spec, title):
    """Draw nesting layout on an axes (no plt.show)."""
    max_h = max(fabric_state.total_height + 50, 200)
    width = fabric_state.width

    rect = mpatches.Rectangle(
        (0, 0), width, fabric_state.total_height,
        linewidth=1.5, edgecolor="black", facecolor="#FAFAFA", zorder=1,
    )
    ax.add_patch(rect)

    for idx, (item, cx, cy, poly) in enumerate(fabric_state.placed_items):
        color = PATCH_COLORS[idx % len(PATCH_COLORS)]
        x, y = poly.exterior.xy
        ax.fill(x, y, color=color, alpha=0.25, zorder=2)
        ax.plot(x, y, color=color, lw=1.2, zorder=3)
        _draw_texture_in_patch(ax, poly, texture_spec, color, lw=1.0, alpha=0.8)
        ax.text(cx + 5, cy + 5, item.name, fontsize=5, color="black", zorder=16)

    ax.set_xlim(-10, width + 10)
    ax.set_ylim(-10, max_h)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")


def run_b0(garment_type, instance, mesh, num_bodies=1):
    """Run B0 baseline reusing already-computed geometry."""
    seam_dir = f"data/seamlines/{garment_type}"
    loader = PatchLoader(LATEST_ROOT, garment_type)
    base_items = loader.load_items()

    all_items = []
    for b in range(num_bodies):
        for it in base_items:
            clone = deepcopy(it)
            clone.name = f"body_{b}/{it.name}" if num_bodies > 1 else it.name
            all_items.append(clone)

    engine = NestingEngine(fabric_width=FABRIC_W, texture_spec=instance.texture)
    fabric_state = engine.nest(all_items)

    importance = _seam_importance_map(instance)
    constraints = load_seam_constraints_from_dir(
        seam_dir,
        weights_by_filename=_weights_by_filename(seam_dir, importance),
        default_weight=0.0,
    )
    lattice = TextureLattice(
        u_dir=np.array([1.0, 0.0]),
        v_dir=np.array([0.0, 1.0]),
        period_u=PERIOD_U,
        period_v=PERIOD_V,
    )
    V = load_patch_vertices_full_from_latest(
        LATEST_ROOT, garment_part=garment_type, scale_mm=1000.0, center_by_boundary=True,
    )
    kappas = {pid: 0 for pid in V}
    patch_ids = sorted(V.keys())
    T0 = {pid: Rigid2D(0, 0, 0) for pid in patch_ids}
    Tsol = solve_global_alignment_all_components(
        patch_ids=patch_ids,
        constraints=constraints,
        patch_vertices_by_id=V,
        lattice=lattice,
        kappas_by_id=kappas,
        K=K,
        initial_transforms=T0,
        max_iters=15,
        verbose=False,
    )

    f2 = 0.0
    for c in constraints:
        if c.patch_i not in V or c.patch_j not in V:
            continue
        Ti = Tsol.get(c.patch_i, Rigid2D(0, 0, 0))
        Tj = Tsol.get(c.patch_j, Rigid2D(0, 0, 0))
        f2 += seam_phase_mismatch(
            seam_pairs=c.pairs,
            patch_i_vertices_xy=Ti.apply(V[c.patch_i]),
            patch_j_vertices_xy=Tj.apply(V[c.patch_j]),
            lattice=lattice,
            kappa_i=0, kappa_j=0, K=K, weight=c.weight,
        )

    return {
        "f1_mm": fabric_state.total_height,
        "f1_norm": fabric_state.total_height / (FABRIC_W * num_bodies),
        "f2": f2,
        "fabric_state": fabric_state,
    }


def run_ga_from_json(garment_type, instance, best_json_path, num_bodies=1):
    """Re-evaluate GA best individual from saved JSON for nesting visualization."""
    with open(best_json_path) as f:
        data = json.load(f)

    kappas_by_id_raw = data.get("kappas_by_id", {})
    kappas_by_id_json = {int(k): v for k, v in kappas_by_id_raw.items()}

    loader = PatchLoader(LATEST_ROOT, garment_type)
    base_items = loader.load_items()

    # Apply kappa from GA
    for item in base_items:
        pid = item.patch_idx
        item.kappa = kappas_by_id_json.get(pid, 0)

    all_items = []
    for b in range(num_bodies):
        for it in base_items:
            clone = deepcopy(it)
            clone.name = f"body_{b}/{it.name}" if num_bodies > 1 else it.name
            all_items.append(clone)

    engine = NestingEngine(fabric_width=FABRIC_W, texture_spec=instance.texture)
    fabric_state = engine.nest(all_items)

    seam_dir = f"data/seamlines/{garment_type}"
    importance = _seam_importance_map(instance)
    constraints = load_seam_constraints_from_dir(
        seam_dir,
        weights_by_filename=_weights_by_filename(seam_dir, importance),
        default_weight=0.0,
    )
    lattice = TextureLattice(
        u_dir=np.array([1.0, 0.0]),
        v_dir=np.array([0.0, 1.0]),
        period_u=PERIOD_U,
        period_v=PERIOD_V,
    )
    V = load_patch_vertices_full_from_latest(
        LATEST_ROOT, garment_part=garment_type, scale_mm=1000.0, center_by_boundary=True,
    )
    kappas_by_id = {pid: kappas_by_id_json.get(pid, 0) for pid in V}
    patch_ids = sorted(V.keys())
    T0 = {pid: Rigid2D(0, 0, 0) for pid in patch_ids}
    Tsol = solve_global_alignment_all_components(
        patch_ids=patch_ids,
        constraints=constraints,
        patch_vertices_by_id=V,
        lattice=lattice,
        kappas_by_id=kappas_by_id,
        K=K,
        initial_transforms=T0,
        max_iters=15,
        verbose=False,
    )

    f2 = 0.0
    for c in constraints:
        if c.patch_i not in V or c.patch_j not in V:
            continue
        Ti = Tsol.get(c.patch_i, Rigid2D(0, 0, 0))
        Tj = Tsol.get(c.patch_j, Rigid2D(0, 0, 0))
        f2 += seam_phase_mismatch(
            seam_pairs=c.pairs,
            patch_i_vertices_xy=Ti.apply(V[c.patch_i]),
            patch_j_vertices_xy=Tj.apply(V[c.patch_j]),
            lattice=lattice,
            kappa_i=kappas_by_id.get(c.patch_i, 0),
            kappa_j=kappas_by_id.get(c.patch_j, 0),
            K=K, weight=c.weight,
        )

    return {
        "f1_mm": fabric_state.total_height,
        "f1_norm": fabric_state.total_height / (FABRIC_W * num_bodies),
        "f2": f2,
        "fabric_state": fabric_state,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--garment", default="upper")
    parser.add_argument("--out", default="paper/figures/comparison.pdf")
    parser.add_argument("--json", default="results/pattern/best/best_individual.json")
    args = parser.parse_args()

    gt = args.garment
    instance, mesh = build_instance(
        mesh_path="data/SMPL_FEMALE.ply",
        fabric_width=FABRIC_W / 1000.0,
        garment_type=gt,
    )

    # Load GA JSON for metadata
    with open(args.json) as f:
        data = json.load(f)

    # Use the geometry already on disk (from the GA run)
    print("Running B0 baseline...")
    b0 = run_b0(gt, instance, mesh)
    print(f"  B0: f1={b0['f1_mm']:.0f}mm, f2={b0['f2']:.4f}")

    print("Loading GA best individual...")
    ga = run_ga_from_json(gt, instance, args.json)
    print(f"  GA: f1={ga['f1_mm']:.0f}mm, f2={ga['f2']:.4f}")

    # Side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    draw_layout(
        ax1, b0["fabric_state"], instance.texture,
        f"B0: no texture alignment\n"
        f"$f_{{\\mathrm{{align}}}}$ = {b0['f2']:.3f}",
    )
    draw_layout(
        ax2, ga["fabric_state"], instance.texture,
        f"GA: optimised seams + phases\n"
        f"$f_{{\\mathrm{{align}}}}$ = {ga['f2']:.3f}",
    )

    M = len(data.get("kappa", []))
    fig.suptitle(
        f"$M={M}$ patches, $N=1$ body, stripes",
        fontsize=12, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.out}")

    # Also save PNG
    png_path = args.out.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
