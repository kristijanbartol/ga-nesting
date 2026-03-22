"""Baseline B1: seam-graph-ordered greedy nesting with greedy kappa selection.

Decisions:
  delta  = 0.5 for all landmarks  (seam positions at quad midpoints)
  kappa  = greedy per patch: try all K values, pick the one minimising
           phase mismatch with already-placed seam neighbours
  rho    = 0 for all patches      (no rotation)
  pi     = BFS order on seam graph, starting from the largest patch;
           ensures that when patch j is placed, most of its seam
           neighbours are already placed (making greedy kappa effective)
  h      = 0                      (bottom-left heuristic)
  Stage2 = skipped

Reports the same f1/f2 metrics as B0 and the GA for direct comparison.
"""
import re
import os
from collections import deque
from copy import deepcopy

import numpy as np

from ga.geometry_block import build_instance, run_geometry_blackbox_timeout
from ga.real_evaluator import load_patch_vertices_full_from_latest
from nesting.loader import PatchLoader
from nesting.engine import NestingEngine
from nesting.phase_utils import TextureLattice, Rigid2D, seam_phase_mismatch
from nesting.stage2_global_align import load_seam_constraints_from_dir, SeamConstraint
from nesting.vis_utils import visualize_layout, plot_seam_mismatch
from spec import SeamPathType


GARMENT_TYPE    = "upper"
LATEST_ROOT     = "results/pattern/latest"
SEAM_DIR        = f"data/seamlines/{GARMENT_TYPE}"
PERIOD_U_MM     = 50.0
PERIOD_V_MM     = 50.0
FABRIC_WIDTH_MM = 150.0 * 10.0
K               = 8


# ── helpers ──────────────────────────────────────────────────────────────────

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


def _patch_id(item) -> int:
    m = re.search(r"patch_(\d+)", item.name)
    return int(m.group(1)) if m else 10 ** 9


def _bfs_order(patch_ids: list[int], constraints: list[SeamConstraint],
               area_by_pid: dict[int, float]) -> list[int]:
    """BFS traversal of the seam adjacency graph.

    Starts from the largest patch in each connected component so that
    the heaviest geometry anchors the traversal order.  Isolated patches
    (not connected to any active seam) are appended last, also by area.
    """
    # Build adjacency from active (weight > 0) constraints only.
    adj: dict[int, set[int]] = {pid: set() for pid in patch_ids}
    for c in constraints:
        if c.weight <= 0.0:
            continue
        if c.patch_i in adj and c.patch_j in adj:
            adj[c.patch_i].add(c.patch_j)
            adj[c.patch_j].add(c.patch_i)

    visited: set[int] = set()
    order: list[int] = []

    # Process connected components, always starting from the largest patch.
    remaining = sorted(patch_ids, key=lambda p: area_by_pid.get(p, 0.0), reverse=True)
    for start in remaining:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        while queue:
            pid = queue.popleft()
            order.append(pid)
            # Visit neighbours in area-descending order for determinism.
            for nb in sorted(adj[pid], key=lambda p: area_by_pid.get(p, 0.0), reverse=True):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

    return order


def _greedy_kappa(pid: int,
                  placed_kappas: dict[int, int],
                  constraints: list[SeamConstraint],
                  V_centered: dict[int, np.ndarray],
                  lattice: TextureLattice,
                  K: int) -> int:
    """Choose kappa for `pid` that minimises total mismatch with already-placed neighbours."""
    # Collect constraints that connect pid to an already-placed patch.
    active: list[tuple[SeamConstraint, int]] = []   # (constraint, neighbour_pid)
    for c in constraints:
        if c.weight <= 0.0:
            continue
        if c.patch_i == pid and c.patch_j in placed_kappas:
            active.append((c, c.patch_j))
        elif c.patch_j == pid and c.patch_i in placed_kappas:
            active.append((c, c.patch_i))

    if not active:
        return 0  # no placed neighbours → phase doesn't matter yet

    best_k, best_cost = 0, float("inf")
    for k in range(K):
        cost = 0.0
        for c, nb_pid in active:
            k_nb = placed_kappas[nb_pid]
            if c.patch_i == pid:
                Vi, Vj = V_centered[pid], V_centered[nb_pid]
                ki, kj = k, k_nb
            else:
                Vi, Vj = V_centered[nb_pid], V_centered[pid]
                ki, kj = k_nb, k
            cost += seam_phase_mismatch(
                seam_pairs=c.pairs,
                patch_i_vertices_xy=Vi,
                patch_j_vertices_xy=Vj,
                lattice=lattice,
                kappa_i=ki,
                kappa_j=kj,
                K=K,
                weight=c.weight,
            )
        if cost < best_cost:
            best_cost = cost
            best_k = k

    return best_k


# ── main ─────────────────────────────────────────────────────────────────────

def run(garment_type: str = GARMENT_TYPE, num_bodies: int = 1) -> dict:
    """Run B1 headlessly and return {f1_mm, f1_norm, f2, f_sum, fabric_state, ...}."""
    from copy import deepcopy
    seam_dir = f"data/seamlines/{garment_type}"

    instance, mesh = build_instance(
        mesh_path="data/SMPL_FEMALE.ply",
        fabric_width=FABRIC_WIDTH_MM / 1000.0,
        garment_type=garment_type,
    )
    delta_baseline = np.array([0.5, 0.5] * instance.num_sampled_landmarks, dtype=float)
    print("[B1] Running geometry with baseline delta (all 0.5)...")
    run_geometry_blackbox_timeout(instance, mesh, delta_baseline, garment_part=garment_type)

    importance_by_name = _seam_importance_map(instance)
    constraints = load_seam_constraints_from_dir(
        seam_dir,
        weights_by_filename=_weights_by_filename(seam_dir, importance_by_name),
        default_weight=0.0,
    )
    for c in constraints:
        print(f"  seam '{c.name}'  weight={c.weight:.3f}")

    lattice = TextureLattice(
        u_dir=np.array([1.0, 0.0]),
        v_dir=np.array([0.0, 1.0]),
        period_u=PERIOD_U_MM,
        period_v=PERIOD_V_MM,
    )
    V_centered_by_id = load_patch_vertices_full_from_latest(
        LATEST_ROOT, garment_part=garment_type, scale_mm=1000.0, center_by_boundary=True
    )

    loader = PatchLoader(LATEST_ROOT, garment_type)
    base_items = loader.load_items()
    items_by_pid = {_patch_id(it): it for it in base_items}
    patch_ids = sorted(items_by_pid.keys())
    area_by_pid = {pid: items_by_pid[pid].area for pid in patch_ids}

    # Greedy kappa computed once on a single body; same kappas applied to all bodies.
    bfs_order = _bfs_order(patch_ids, constraints, area_by_pid)
    print(f"[B1] BFS patch order: {bfs_order}")

    placed_kappas: dict[int, int] = {}
    kappas_by_id: dict[int, int] = {}
    tx = instance.texture.period_x
    ty = instance.texture.period_y

    for pid in bfs_order:
        k = _greedy_kappa(pid, placed_kappas, constraints, V_centered_by_id, lattice, K)
        placed_kappas[pid] = k
        kappas_by_id[pid] = k

    print(f"[B1] Greedy kappas: {kappas_by_id}")

    # Duplicate items for each body, applying the same greedy kappas.
    all_items = []
    for b in range(num_bodies):
        for pid in bfs_order:
            if pid not in items_by_pid:
                continue
            it = deepcopy(items_by_pid[pid])
            it.name = f"body_{b}/{it.name}"
            k = kappas_by_id[pid]
            it.phase_offset = ((k / float(K)) * tx, (k / float(K)) * ty)
            all_items.append(it)

    # Permutation: BFS order repeated for each body, in body-major order.
    permutation = list(range(len(all_items)))

    engine = NestingEngine(fabric_width=FABRIC_WIDTH_MM, texture_spec=instance.texture)
    print(f"[B1] Nesting ({num_bodies} bodies, BFS order, greedy kappa)...")
    fabric_state = engine.nest(all_items, permutation=permutation)

    f1 = fabric_state.total_height
    f1_norm = f1 / (FABRIC_WIDTH_MM * num_bodies)

    transforms = {pid: Rigid2D(0.0, 0.0, 0.0) for pid in V_centered_by_id}
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
            K=K, weight=c.weight,
        )

    print(f"[B1] f1={f1:.1f}mm  f2={f2:.4f}  f_sum={f1_norm + f2:.4f}")
    return {
        "f1_mm": f1, "f1_norm": f1_norm, "f2": f2, "f_sum": f1_norm + f2,
        "fabric_state": fabric_state, "constraints": constraints,
        "V_centered_by_id": V_centered_by_id, "lattice": lattice,
        "kappas_by_id": kappas_by_id, "transforms": transforms,
        "instance": instance,
    }


def main():
    result = run(GARMENT_TYPE, num_bodies=1)
    visualize_layout(result["fabric_state"], result["instance"].texture,
                     title="B1 — Greedy seam-order + greedy kappa")
    plot_seam_mismatch(result["constraints"], result["V_centered_by_id"],
                       result["lattice"], result["kappas_by_id"], K,
                       result["transforms"], title="B1 — Seam Phase Mismatch")


if __name__ == "__main__":
    main()
