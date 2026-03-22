"""Baseline B2: exhaustive optimal kappa search.

Decisions:
  delta  = 0.5 for all landmarks  (seam positions at quad midpoints)
  kappa  = globally optimal: enumerate all K^M combinations, pick min f2
  rho    = 0 for all patches      (no rotation)
  pi     = area-sorted descending (engine default, permutation=None)
  h      = 0                      (bottom-left heuristic)
  Stage2 = skipped

This gives a lower bound on f2 achievable with fixed geometry and discrete
kappa offsets.  Tractable for small M and K:
  K=8, M=4 patches  →  8^4  =    4 096 combinations
  K=8, M=6 patches  →  8^6  =  262 144 combinations  (~seconds)
  K=8, M=8 patches  →  8^8  = 16 777 216 combinations (may be slow)

For num_bodies > 1, all bodies share the same geometry so the optimal
kappa assignment is body-invariant — K^M search is sufficient regardless
of the number of bodies.

Reports f1/f2 directly comparable to B0, B1, and the GA.
"""
import itertools
import re
import os
from copy import deepcopy

import numpy as np

from ga.geometry_block import build_instance, run_geometry_blackbox_timeout
from ga.real_evaluator import load_patch_vertices_full_from_latest
from nesting.loader import PatchLoader
from nesting.engine import NestingEngine
from nesting.phase_utils import TextureLattice, Rigid2D, seam_phase_mismatch
from nesting.stage2_global_align import load_seam_constraints_from_dir
from nesting.vis_utils import visualize_layout, plot_seam_mismatch
from spec import SeamPathType


GARMENT_TYPE    = "upper"
LATEST_ROOT     = "results/pattern/latest"
PERIOD_U_MM     = 50.0
PERIOD_V_MM     = 50.0
FABRIC_WIDTH_MM = 150.0 * 10.0
K               = 8


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


def _compute_f2(kappas_by_id, constraints, V_centered_by_id, lattice):
    f2 = 0.0
    for c in constraints:
        if c.patch_i not in V_centered_by_id or c.patch_j not in V_centered_by_id:
            continue
        f2 += seam_phase_mismatch(
            seam_pairs=c.pairs,
            patch_i_vertices_xy=V_centered_by_id[c.patch_i],
            patch_j_vertices_xy=V_centered_by_id[c.patch_j],
            lattice=lattice,
            kappa_i=kappas_by_id.get(c.patch_i, 0),
            kappa_j=kappas_by_id.get(c.patch_j, 0),
            K=K,
            weight=c.weight,
        )
    return f2


def run(garment_type: str = GARMENT_TYPE, num_bodies: int = 1) -> dict:
    """Run B2 headlessly and return {f1_mm, f1_norm, f2, f_sum, ...}."""
    seam_dir = f"data/seamlines/{garment_type}"

    instance, mesh = build_instance(
        mesh_path="data/SMPL_FEMALE.ply",
        fabric_width=FABRIC_WIDTH_MM / 1000.0,
        garment_type=garment_type,
    )
    delta_baseline = np.array([0.5, 0.5] * instance.num_sampled_landmarks, dtype=float)
    print("[B2] Running geometry with baseline delta (all 0.5)...")
    run_geometry_blackbox_timeout(instance, mesh, delta_baseline, garment_part=garment_type)

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

    # ── Exhaustive kappa search ───────────────────────────────────────────────
    patch_ids = sorted(V_centered_by_id.keys())
    M = len(patch_ids)
    total = K ** M
    print(f"[B2] Exhaustive kappa search: K={K}, M={M} patches, {total} combinations...")

    best_kappas = {pid: 0 for pid in patch_ids}
    best_f2 = float("inf")

    for i, combo in enumerate(itertools.product(range(K), repeat=M)):
        kappas = dict(zip(patch_ids, combo))
        f2 = _compute_f2(kappas, constraints, V_centered_by_id, lattice)
        if f2 < best_f2:
            best_f2 = f2
            best_kappas = kappas
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"[B2]   {i + 1}/{total}  best_f2={best_f2:.4f}", end="\r")

    print(f"\n[B2] Optimal kappas: {best_kappas}  f2={best_f2:.4f}")

    # ── Nest with optimal kappas (all bodies share the same kappas) ───────────
    loader = PatchLoader(LATEST_ROOT, garment_type)
    base_items = loader.load_items()
    tx = instance.texture.period_x
    ty = instance.texture.period_y
    pid_re = re.compile(r"patch_(\d+)")

    all_items = []
    for b in range(num_bodies):
        for it in base_items:
            m = pid_re.search(it.name)
            pid = int(m.group(1)) if m else -1
            clone = deepcopy(it)
            clone.name = f"body_{b}/{it.name}"
            k = best_kappas.get(pid, 0)
            clone.phase_offset = ((k / float(K)) * tx, (k / float(K)) * ty)
            all_items.append(clone)

    engine = NestingEngine(fabric_width=FABRIC_WIDTH_MM, texture_spec=instance.texture)
    print(f"[B2] Nesting ({num_bodies} bodies, area-sorted, optimal kappa)...")
    fabric_state = engine.nest(all_items)

    f1 = fabric_state.total_height
    f1_norm = f1 / (FABRIC_WIDTH_MM * num_bodies)
    transforms = {pid: Rigid2D(0.0, 0.0, 0.0) for pid in V_centered_by_id}

    print(f"[B2] f1={f1:.1f}mm  f2={best_f2:.4f}  f_sum={f1_norm + best_f2:.4f}")
    return {
        "f1_mm": f1, "f1_norm": f1_norm, "f2": best_f2, "f_sum": f1_norm + best_f2,
        "fabric_state": fabric_state, "constraints": constraints,
        "V_centered_by_id": V_centered_by_id, "lattice": lattice,
        "kappas_by_id": best_kappas, "transforms": transforms,
        "instance": instance,
    }


def main():
    result = run(GARMENT_TYPE, num_bodies=1)
    visualize_layout(result["fabric_state"], result["instance"].texture,
                     title="B2 — Exhaustive optimal kappa")
    plot_seam_mismatch(result["constraints"], result["V_centered_by_id"],
                       result["lattice"], result["kappas_by_id"], K,
                       result["transforms"], title="B2 — Seam Phase Mismatch")


if __name__ == "__main__":
    main()
