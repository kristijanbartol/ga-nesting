"""Baseline B0: pure greedy nesting, no texture alignment.

Decisions:
  delta  = 0.5 for all landmarks  (seam positions at quad midpoints)
  kappa  = 0 for all patches      (no phase offset)
  rho    = 0 for all patches      (no rotation)
  pi     = area-sorted descending (engine default, permutation=None)
  h      = 0                      (bottom-left heuristic)
  Stage2 = skipped

Reports f1 (fabric height) and f2 (raw seam phase mismatch) so results
are directly comparable to GA runs.
"""
import re
import os
import numpy as np

from ga.geometry_block import build_instance, run_geometry_blackbox_timeout
from ga.real_evaluator import load_patch_vertices_full_from_latest
from nesting.loader import PatchLoader
from nesting.engine import NestingEngine
from nesting.phase_utils import TextureLattice, Rigid2D, seam_phase_mismatch
from nesting.stage2_global_align import load_seam_constraints_from_dir
from nesting.vis_utils import visualize_layout, plot_seam_mismatch
from spec import SeamPathType


GARMENT_TYPE  = "upper"
LATEST_ROOT   = "results/pattern/latest"
SEAM_DIR      = f"data/seamlines/{GARMENT_TYPE}"
PERIOD_U_MM   = 50.0
PERIOD_V_MM   = 50.0
FABRIC_WIDTH_MM = 150.0 * 10.0
K             = 8


def _seam_importance_map(instance):
    return {
        seam.name: seam.importance
        for seam in instance.active_seam_definitions
        if seam.path_type == SeamPathType.GEODESIC
    }


def _weights_by_filename(seam_dir, importance_by_name):
    result = {}
    if not os.path.isdir(seam_dir):
        return result
    for fn in os.listdir(seam_dir):
        if not (fn.startswith("seam-") and fn.endswith(".txt")):
            continue
        m = re.match(r"seam-(.+)_\d+-\d+\.txt$", fn)
        if m:
            result[fn] = importance_by_name.get(m.group(1), 0.0)
    return result


def main():
    # ── 1. Geometry (baseline delta: all landmarks at quad midpoint) ──────────
    instance, mesh = build_instance(
        mesh_path="data/SMPL_FEMALE.ply",
        fabric_width=FABRIC_WIDTH_MM / 1000.0,
        garment_type=GARMENT_TYPE,
    )
    delta_baseline = np.array([0.5, 0.5] * instance.num_sampled_landmarks, dtype=float)
    print("[B0] Running geometry with baseline delta (all 0.5)...")
    run_geometry_blackbox_timeout(instance, mesh, delta_baseline, garment_part=GARMENT_TYPE)

    # ── 2. Load patches and nest (area-sorted, kappa=0, rho=0) ───────────────
    loader = PatchLoader(LATEST_ROOT, GARMENT_TYPE)
    items = loader.load_items()
    # phase_offset defaults to (0, 0) in NestingItem, so kappa=0 is implicit.

    engine = NestingEngine(fabric_width=FABRIC_WIDTH_MM, texture_spec=instance.texture)
    print("[B0] Nesting (area-sorted, no kappa, no rho)...")
    fabric_state = engine.nest(items)  # permutation=None → area-sorted descending

    f1 = fabric_state.total_height
    f1_norm = f1 / FABRIC_WIDTH_MM
    print(f"[B0] f1 = {f1:.1f} mm  (normalised: {f1_norm:.4f})")

    # ── 3. Compute f2 (no Stage 2, all kappa=0, identity transforms) ─────────
    importance_by_name = _seam_importance_map(instance)
    constraints = load_seam_constraints_from_dir(
        SEAM_DIR,
        weights_by_filename=_weights_by_filename(SEAM_DIR, importance_by_name),
        default_weight=0.0,
    )

    lattice = TextureLattice(
        u_dir=np.array([1.0, 0.0]),
        v_dir=np.array([0.0, 1.0]),
        period_u=PERIOD_U_MM,
        period_v=PERIOD_V_MM,
    )

    V_centered_by_id = load_patch_vertices_full_from_latest(
        LATEST_ROOT, garment_part=GARMENT_TYPE, scale_mm=1000.0, center_by_boundary=True
    )
    kappas_by_id = {pid: 0 for pid in V_centered_by_id}
    transforms   = {pid: Rigid2D(0.0, 0.0, 0.0) for pid in V_centered_by_id}

    f2 = 0.0
    for c in constraints:
        if c.patch_i not in V_centered_by_id or c.patch_j not in V_centered_by_id:
            continue
        f2 += seam_phase_mismatch(
            seam_pairs=c.pairs,
            patch_i_vertices_xy=V_centered_by_id[c.patch_i],
            patch_j_vertices_xy=V_centered_by_id[c.patch_j],
            lattice=lattice,
            kappa_i=0,
            kappa_j=0,
            K=K,
            weight=c.weight,
        )

    print(f"[B0] f2 = {f2:.4f}")
    print(f"[B0] fitness sum = {f1_norm + f2:.4f}")

    # ── 4. Visualise ──────────────────────────────────────────────────────────
    visualize_layout(fabric_state, instance.texture, title="B0 — Pure Greedy (no texture alignment)")
    plot_seam_mismatch(constraints, V_centered_by_id, lattice, kappas_by_id, K, transforms,
                       title="B0 — Seam Phase Mismatch")


if __name__ == "__main__":
    main()
