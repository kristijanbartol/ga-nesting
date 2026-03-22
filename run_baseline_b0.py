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
from nesting.stage2_global_align import load_seam_constraints_from_dir, solve_global_alignment_all_components
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


def run(garment_type: str = GARMENT_TYPE, num_bodies: int = 1) -> dict:
    """Run B0 headlessly and return {f1_mm, f1_norm, f2, f_sum, fabric_state, ...}."""
    from copy import deepcopy
    seam_dir = f"data/seamlines/{garment_type}"

    instance, mesh = build_instance(
        mesh_path="data/SMPL_FEMALE.ply",
        fabric_width=FABRIC_WIDTH_MM / 1000.0,
        garment_type=garment_type,
    )
    delta_baseline = np.array([0.5, 0.5] * instance.num_sampled_landmarks, dtype=float)
    print("[B0] Running geometry with baseline delta (all 0.5)...")
    run_geometry_blackbox_timeout(instance, mesh, delta_baseline, garment_part=garment_type)

    loader = PatchLoader(LATEST_ROOT, garment_type)
    base_items = loader.load_items()

    all_items = []
    for b in range(num_bodies):
        for it in base_items:
            clone = deepcopy(it)
            clone.name = f"body_{b}/{it.name}"
            all_items.append(clone)

    engine = NestingEngine(fabric_width=FABRIC_WIDTH_MM, texture_spec=instance.texture)
    print(f"[B0] Nesting ({num_bodies} bodies, area-sorted, no kappa, no rho)...")
    fabric_state = engine.nest(all_items)

    f1 = fabric_state.total_height
    f1_norm = f1 / (FABRIC_WIDTH_MM * num_bodies)

    importance_by_name = _seam_importance_map(instance)
    constraints = load_seam_constraints_from_dir(
        seam_dir,
        weights_by_filename=_weights_by_filename(seam_dir, importance_by_name),
        default_weight=0.0,
    )
    lattice = TextureLattice(
        u_dir=np.array([1.0, 0.0]),
        v_dir=np.array([0.0, 1.0]),
        period_u=PERIOD_U_MM,
        period_v=PERIOD_V_MM,
    )
    V_centered_by_id = load_patch_vertices_full_from_latest(
        LATEST_ROOT, garment_part=garment_type, scale_mm=1000.0, center_by_boundary=True
    )
    kappas_by_id = {pid: 0 for pid in V_centered_by_id}
    patch_ids    = sorted(V_centered_by_id.keys())

    print("[B0] Running Stage2 (LM alignment)...")
    T0   = {pid: Rigid2D(0.0, 0.0, 0.0) for pid in patch_ids}
    Tsol = solve_global_alignment_all_components(
        patch_ids=patch_ids,
        constraints=constraints,
        patch_vertices_by_id=V_centered_by_id,
        lattice=lattice,
        kappas_by_id=kappas_by_id,
        K=K,
        initial_transforms=T0,
        max_iters=15,
        verbose=False,
    )

    f2 = 0.0
    for c in constraints:
        if c.patch_i not in V_centered_by_id or c.patch_j not in V_centered_by_id:
            continue
        Ti = Tsol.get(c.patch_i, Rigid2D(0, 0, 0))
        Tj = Tsol.get(c.patch_j, Rigid2D(0, 0, 0))
        f2 += seam_phase_mismatch(
            seam_pairs=c.pairs,
            patch_i_vertices_xy=Ti.apply(V_centered_by_id[c.patch_i]),
            patch_j_vertices_xy=Tj.apply(V_centered_by_id[c.patch_j]),
            lattice=lattice,
            kappa_i=0, kappa_j=0, K=K, weight=c.weight,
        )

    print(f"[B0] f1={f1:.1f}mm  f2={f2:.4f}  f_sum={f1_norm + f2:.4f}")
    return {
        "f1_mm": f1, "f1_norm": f1_norm, "f2": f2, "f_sum": f1_norm + f2,
        "fabric_state": fabric_state, "constraints": constraints,
        "V_centered_by_id": V_centered_by_id, "lattice": lattice,
        "kappas_by_id": kappas_by_id, "transforms": Tsol,
        "instance": instance,
    }


def main():
    result = run(GARMENT_TYPE, num_bodies=1)
    visualize_layout(result["fabric_state"], result["instance"].texture,
                     title="B0 — Pure Greedy (no texture alignment)")
    plot_seam_mismatch(result["constraints"], result["V_centered_by_id"],
                       result["lattice"], result["kappas_by_id"], K,
                       result["transforms"], title="B0 — Seam Phase Mismatch")


if __name__ == "__main__":
    main()
