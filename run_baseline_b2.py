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
import argparse
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
from nesting.stage2_global_align import load_seam_constraints_from_dir, solve_global_alignment_all_components
from nesting.vis_utils import visualize_layout, plot_seam_mismatch
from spec import SeamPathType
from wallpaper import get_policy


GARMENT_TYPE    = "upper"
WALLPAPER_GROUP = "stripes"
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


def _compute_f2(kappas_by_id, constraints, V_centered_by_id, lattice, phase_axes=None):
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
            phase_axes=phase_axes,
        )
    return f2


def run(garment_type: str = GARMENT_TYPE, num_bodies: int = 1,
        wallpaper_group: str = WALLPAPER_GROUP) -> dict:
    """Run B2 headlessly and return {f1_mm, f1_norm, f2, f_sum, ...}."""
    seam_dir = f"data/seamlines/{garment_type}"
    policy = get_policy(wallpaper_group)
    u_dir, v_dir = policy.lattice_directions()
    pa = policy.phase_axes()

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
        u_dir=u_dir,
        v_dir=v_dir,
        period_u=PERIOD_U_MM,
        period_v=PERIOD_V_MM,
    )
    V_centered_by_id = load_patch_vertices_full_from_latest(
        LATEST_ROOT, garment_part=garment_type, scale_mm=1000.0, center_by_boundary=True
    )

    # ── Kappa search ────────────────────────────────────────────────────────
    patch_ids = sorted(V_centered_by_id.keys())
    M = len(patch_ids)
    total = K ** M
    MAX_EXHAUSTIVE = 500_000  # ~seconds on modern hardware

    best_kappas = {pid: 0 for pid in patch_ids}
    best_f2 = float("inf")

    if total <= MAX_EXHAUSTIVE:
        print(f"[B2] Exhaustive kappa search: K={K}, M={M} patches, {total} combinations...")
        for i, combo in enumerate(itertools.product(range(K), repeat=M)):
            kappas = dict(zip(patch_ids, combo))
            f2 = _compute_f2(kappas, constraints, V_centered_by_id, lattice, phase_axes=pa)
            if f2 < best_f2:
                best_f2 = f2
                best_kappas = kappas
            if (i + 1) % 500 == 0 or (i + 1) == total:
                print(f"[B2]   {i + 1}/{total}  best_f2={best_f2:.4f}", end="\r")
        print()
    else:
        # Random sampling fallback for large M (e.g. onesie with 11 patches).
        N_SAMPLES = MAX_EXHAUSTIVE
        rng = np.random.RandomState(42)
        print(f"[B2] K^M={total:.2e} too large for exhaustive search."
              f"  Random sampling {N_SAMPLES} kappa configurations...")
        for i in range(N_SAMPLES):
            combo = tuple(rng.randint(0, K, size=M))
            kappas = dict(zip(patch_ids, combo))
            f2 = _compute_f2(kappas, constraints, V_centered_by_id, lattice, phase_axes=pa)
            if f2 < best_f2:
                best_f2 = f2
                best_kappas = kappas
            if (i + 1) % 500 == 0 or (i + 1) == N_SAMPLES:
                print(f"[B2]   {i + 1}/{N_SAMPLES}  best_f2={best_f2:.4f}", end="\r")
        print()

    print(f"[B2] Best kappas: {best_kappas}  f2={best_f2:.4f}")

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

    print("[B2] Running Stage2 (LM alignment)...")
    patch_ids = sorted(V_centered_by_id.keys())
    T0   = {pid: Rigid2D(0.0, 0.0, 0.0) for pid in patch_ids}
    Tsol = solve_global_alignment_all_components(
        patch_ids=patch_ids,
        constraints=constraints,
        patch_vertices_by_id=V_centered_by_id,
        lattice=lattice,
        kappas_by_id=best_kappas,
        K=K,
        initial_transforms=T0,
        max_iters=15,
        verbose=False,
        phase_axes=pa,
    )

    # Recompute f2 with Stage2 transforms applied.
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
            kappa_i=best_kappas.get(c.patch_i, 0),
            kappa_j=best_kappas.get(c.patch_j, 0),
            K=K, weight=c.weight,
            phase_axes=pa,
        )

    print(f"[B2] f1={f1:.1f}mm  f2={f2:.4f}  f_sum={f1_norm + f2:.4f}")
    return {
        "f1_mm": f1, "f1_norm": f1_norm, "f2": f2, "f_sum": f1_norm + f2,
        "fabric_state": fabric_state, "constraints": constraints,
        "V_centered_by_id": V_centered_by_id, "lattice": lattice,
        "kappas_by_id": best_kappas, "transforms": Tsol,
        "instance": instance,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline B2: exhaustive optimal kappa")
    parser.add_argument("--simulate", action="store_true",
                        help="Run cloth simulation after nesting")
    parser.add_argument("--garment-type", default=GARMENT_TYPE,
                        help=f"Garment type (default: {GARMENT_TYPE})")
    parser.add_argument("--num-bodies", type=int, default=1,
                        help="Number of bodies to nest (default: 1)")
    parser.add_argument("--wallpaper", default=WALLPAPER_GROUP,
                        choices=["stripes", "diagonal_stripes", "grid", "p4", "p4m", "pg", "pmg", "pgg"],
                        help=f"Texture wallpaper group (default: {WALLPAPER_GROUP})")
    args = parser.parse_args()

    garment_type = args.garment_type
    result = run(garment_type, num_bodies=args.num_bodies, wallpaper_group=args.wallpaper)
    visualize_layout(result["fabric_state"], result["instance"].texture,
                     title="B2 — Exhaustive optimal kappa")
    policy = get_policy(args.wallpaper)
    plot_seam_mismatch(result["constraints"], result["V_centered_by_id"],
                       result["lattice"], result["kappas_by_id"], K,
                       result["transforms"], title="B2 — Seam Phase Mismatch",
                       phase_axes=policy.phase_axes(),
                       save_path=f"results/graphs/{garment_type}/b2_seam_mismatch.png")

    if args.simulate:
        from geometry.simulation import run_headless_simulation
        out_dir = f"results/simulation/{garment_type}/b2"
        print(f"\n[B2] Running cloth simulation → {out_dir}/")
        run_headless_simulation(
            avatar='data/SMPL_FEMALE.ply',
            garment_type=garment_type,
            tsol=result["transforms"],
            kappas_by_id=result["kappas_by_id"],
            K=K,
            period_u_mm=PERIOD_U_MM,
            period_v_mm=PERIOD_V_MM,
            pattern_root=LATEST_ROOT,
            patches_dir=f"data/patches/{garment_type}",
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
