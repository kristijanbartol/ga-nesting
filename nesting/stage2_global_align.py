# stage2_global_align.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any
import os
import numpy as np

from .phase_utils import (
    Rigid2D,
    TextureLattice,
    load_exported_seamfile,
    seam_phase_residuals_uv,
)


@dataclass(frozen=True)
class SeamConstraint:
    patch_i: int
    patch_j: int
    pairs: List[Tuple[int, int]]
    weight: float
    name: str = ""


def load_seam_constraints_from_dir(
    seam_dir: str,
    weights_by_filename: Optional[Dict[str, float]] = None,
    default_weight: float = 1.0,
) -> List[SeamConstraint]:
    weights_by_filename = weights_by_filename or {}
    out: List[SeamConstraint] = []
    for fn in sorted(os.listdir(seam_dir)):
        if not (fn.startswith("seam-") and fn.endswith(".txt")):
            continue
        p = os.path.join(seam_dir, fn)
        i, j, pairs = load_exported_seamfile(p)
        w = float(weights_by_filename.get(fn, default_weight))
        out.append(SeamConstraint(i, j, pairs, w, name=fn))
    return out


def connected_components(patch_ids: Iterable[int], constraints: List[SeamConstraint]) -> List[List[int]]:
    patch_ids = list(patch_ids)
    adj: Dict[int, Set[int]] = {pid: set() for pid in patch_ids}
    for c in constraints:
        if c.weight <= 0.0:
            continue
        if c.patch_i in adj and c.patch_j in adj:
            adj[c.patch_i].add(c.patch_j)
            adj[c.patch_j].add(c.patch_i)

    seen: Set[int] = set()
    comps: List[List[int]] = []
    for pid in patch_ids:
        if pid in seen:
            continue
        stack = [pid]
        seen.add(pid)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def _pack_params(comp: List[int], root: int, transforms: Dict[int, Rigid2D]) -> Tuple[np.ndarray, List[int]]:
    order = [pid for pid in comp if pid != root]
    x = np.zeros((3 * len(order),), dtype=float)
    for k, pid in enumerate(order):
        T = transforms.get(pid, Rigid2D(0.0, 0.0, 0.0))
        x[3*k + 0] = T.theta
        x[3*k + 1] = T.tx
        x[3*k + 2] = T.ty
    return x, order


def _unpack_params(x: np.ndarray, order: List[int], root: int) -> Dict[int, Rigid2D]:
    Ts: Dict[int, Rigid2D] = {root: Rigid2D(0.0, 0.0, 0.0)}
    for k, pid in enumerate(order):
        Ts[pid] = Rigid2D(float(x[3*k+0]), float(x[3*k+1]), float(x[3*k+2]))
    return Ts


def _residuals_for_component(
    comp: List[int],
    constraints: List[SeamConstraint],
    patch_vertices_by_id: Dict[int, np.ndarray],
    lattice: TextureLattice,
    kappas_by_id: Dict[int, int],
    K: int,
    transforms: Dict[int, Rigid2D],
    phase_axes: Optional[Tuple[bool, bool]] = None,
) -> np.ndarray:
    comp_set = set(comp)
    chunks: List[np.ndarray] = []
    for c in constraints:
        if c.weight <= 0.0:
            continue
        if c.patch_i not in comp_set or c.patch_j not in comp_set:
            continue

        Vi = patch_vertices_by_id[c.patch_i]
        Vj = patch_vertices_by_id[c.patch_j]
        Ti = transforms.get(c.patch_i, Rigid2D(0.0, 0.0, 0.0))
        Tj = transforms.get(c.patch_j, Rigid2D(0.0, 0.0, 0.0))
        ki = kappas_by_id.get(c.patch_i, 0)
        kj = kappas_by_id.get(c.patch_j, 0)

        r = seam_phase_residuals_uv(
            seam_pairs=c.pairs,
            patch_i_vertices_xy=Vi,
            patch_j_vertices_xy=Vj,
            lattice=lattice,
            kappa_i=ki,
            kappa_j=kj,
            K=K,
            weight=c.weight,
            transform_i=Ti,
            transform_j=Tj,
            phase_axes=phase_axes,
        )
        if r.size:
            chunks.append(r)

    if not chunks:
        return np.zeros((0,), dtype=float)
    return np.concatenate(chunks, axis=0)


def _finite_difference_jacobian(fun, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    f0 = fun(x)
    m = f0.size
    n = x.size
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        x1 = x.copy()
        x1[j] += eps
        f1 = fun(x1)
        J[:, j] = (f1 - f0) / eps
    return J


def solve_component_global_alignment(
    comp: List[int],
    constraints: List[SeamConstraint],
    patch_vertices_by_id: Dict[int, np.ndarray],
    lattice: TextureLattice,
    kappas_by_id: Dict[int, int],
    K: int,
    initial_transforms: Optional[Dict[int, Rigid2D]] = None,
    root: Optional[int] = None,
    max_iters: int = 25,
    lm_lambda: float = 1e-2,
    fd_eps: float = 1e-6,
    verbose: bool = False,
    phase_axes: Optional[Tuple[bool, bool]] = None,
) -> Dict[int, Rigid2D]:
    if root is None:
        root = comp[0]

    transforms0 = dict(initial_transforms or {})
    transforms0[root] = Rigid2D(0.0, 0.0, 0.0)

    x, order = _pack_params(comp, root, transforms0)

    def fun(xv: np.ndarray) -> np.ndarray:
        Ts = _unpack_params(xv, order, root)
        return _residuals_for_component(
            comp=comp,
            constraints=constraints,
            patch_vertices_by_id=patch_vertices_by_id,
            lattice=lattice,
            kappas_by_id=kappas_by_id,
            K=K,
            transforms=Ts,
            phase_axes=phase_axes,
        )

    for it in range(max_iters):
        r = fun(x)
        cost = float(r @ r)

        if verbose:
            print(f"[Stage2 LM] iter {it:02d}: cost={cost:.6e}, m={r.size}, n={x.size}")

        if r.size == 0 or x.size == 0:
            break

        J = _finite_difference_jacobian(fun, x, eps=fd_eps)
        A = J.T @ J
        b = -(J.T @ r)
        A_reg = A + lm_lambda * np.eye(A.shape[0], dtype=float)

        try:
            dx = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(A_reg, b, rcond=None)[0]

        x_new = x + dx
        r_new = fun(x_new)
        cost_new = float(r_new @ r_new)

        if cost_new < cost:
            x = x_new
            lm_lambda *= 0.7
        else:
            lm_lambda *= 2.0

        if np.linalg.norm(dx) < 1e-8:
            break

    return _unpack_params(x, order, root)


def solve_global_alignment_all_components(
    patch_ids: Iterable[int],
    constraints: List[SeamConstraint],
    patch_vertices_by_id: Dict[int, np.ndarray],
    lattice: TextureLattice,
    kappas_by_id: Dict[int, int],
    K: int,
    initial_transforms: Optional[Dict[int, Rigid2D]] = None,
    max_iters: int = 25,
    verbose: bool = False,
    phase_axes: Optional[Tuple[bool, bool]] = None,
) -> Dict[int, Rigid2D]:
    patch_ids = list(patch_ids)
    comps = connected_components(patch_ids, constraints)

    out: Dict[int, Rigid2D] = dict(initial_transforms or {})
    for comp in comps:
        Ts = solve_component_global_alignment(
            comp=comp,
            constraints=constraints,
            patch_vertices_by_id=patch_vertices_by_id,
            lattice=lattice,
            kappas_by_id=kappas_by_id,
            K=K,
            initial_transforms=out,
            root=comp[0],
            max_iters=max_iters,
            verbose=verbose,
            phase_axes=phase_axes,
        )
        out.update(Ts)
    return out
