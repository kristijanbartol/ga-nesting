# phase_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np


# ----------------------------
# Texture lattice definition
# ----------------------------

@dataclass(frozen=True)
class TextureLattice:
    """
    Periodic texture lattice in R^2.

    Lattice vectors:
        U = period_u * u_dir
        V = period_v * v_dir

    u_dir and v_dir need not be axis-aligned.
    """
    u_dir: np.ndarray        # shape (2,)
    v_dir: np.ndarray        # shape (2,)
    period_u: float
    period_v: float

    def matrix(self) -> np.ndarray:
        """2x2 matrix [U V] with lattice vectors as columns."""
        U = self.u_dir / (np.linalg.norm(self.u_dir) + 1e-12) * self.period_u
        V = self.v_dir / (np.linalg.norm(self.v_dir) + 1e-12) * self.period_v
        return np.column_stack([U, V])  # shape (2,2)


def frac(x: np.ndarray) -> np.ndarray:
    """Fractional part in [0,1)."""
    return x - np.floor(x)


def phase_uv(points_xy: np.ndarray, lattice: TextureLattice) -> np.ndarray:
    """
    Compute lattice phase for points in fabric coordinates.

    Returns:
        phases: (N,2) in [0,1) x [0,1)
        where phases[:,0] is along U, phases[:,1] along V.

    This is the natural interpretation of "Phase(x) computed using basis B reduced modulo one period".
    """
    A = lattice.matrix()  # [U V]
    A_inv = np.linalg.inv(A)
    # coordinates in lattice basis: c = A_inv * x
    coords = (points_xy @ A_inv.T)  # (N,2)
    return frac(coords)


def wrapped_delta_scalar(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Wrapped difference for scalar phases in [0,1).
    Vectorized over arrays.
    """
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)


def wrapped_delta_uv(phi_a_uv: np.ndarray, phi_b_uv: np.ndarray, reduce: str = "mean") -> np.ndarray:
    """
    Wrapped delta for 2D phase vectors. Your LaTeX defines Delta for scalars;
    for a 2D lattice we need a scalar mismatch per seam sample.
    This implements a simple reduction:
        - mean of component-wise wrapped deltas (default), or
        - max of components.
    """
    du = wrapped_delta_scalar(phi_a_uv[:, 0], phi_b_uv[:, 0])
    dv = wrapped_delta_scalar(phi_a_uv[:, 1], phi_b_uv[:, 1])
    if reduce == "max":
        return np.maximum(du, dv)
    # default "mean"
    return 0.5 * (du + dv)


# ----------------------------
# Rigid transforms (optional)
# ----------------------------

@dataclass(frozen=True)
class Rigid2D:
    """
    Simple 2D rigid transform for later stages:
        x' = R(theta) x + t
    """
    theta: float
    tx: float
    ty: float

    def R(self) -> np.ndarray:
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        return np.array([[c, -s],
                         [s,  c]], dtype=float)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.R().T + np.array([self.tx, self.ty], dtype=float)


# ----------------------------
# Seam IO (your export format)
# ----------------------------

def load_exported_seamfile(path: str) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    Reads seam correspondence file written by export_seamlines():

        line 1: symmetric flag (ignored here)
        line 2: patch_i
        line 3: patch_j
        remaining lines: "vidx_i vidx_j"

    Returns: (patch_i, patch_j, pairs)
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 4:
        raise ValueError(f"Invalid seam file: {path}")

    patch_i = int(lines[1])
    patch_j = int(lines[2])

    pairs: List[Tuple[int, int]] = []
    for ln in lines[3:]:
        a, b = ln.split()
        pairs.append((int(a), int(b)))

    return patch_i, patch_j, pairs


# ----------------------------
# Seam mismatch (your LaTeX)
# ----------------------------

def seam_phase_mismatch(
    seam_pairs: List[Tuple[int, int]],
    patch_i_vertices_xy: np.ndarray,
    patch_j_vertices_xy: np.ndarray,
    lattice: TextureLattice,
    kappa_i: int,
    kappa_j: int,
    K: int,
    weight: float,
    transform_i: Optional[Rigid2D] = None,
    transform_j: Optional[Rigid2D] = None,
    reduce_uv: str = "mean",
) -> float:
    """
    Implements (discrete approximation of) your LaTeX:

        Mismatch(s) = (1/L) ∫ Delta( Phase_i(x_i(l)), Phase_j(x_j(l)) ) dl

    Here we approximate by averaging over seam correspondence samples.

    - Seam weights w_s in [0,1] multiply the mismatch.
    - w=0 disables the seam influence (returns 0).
    - κ/K offset is added and wrapped mod 1.

    Notes:
    - Phase is computed in 2D (u,v). Since Delta is defined for scalar phases in your LaTeX,
      we reduce (du,dv) to a scalar per sample by mean (default) or max.
    - Transforms are optional: for now you can pass None (identity),
      later the global solver will pass Rigid2D per patch.
    """
    if weight <= 0.0:
        return 0.0
    if K <= 0:
        raise ValueError("K must be positive")

    if len(seam_pairs) == 0:
        return 0.0

    # Gather seam sample points
    idx_i = np.array([a for (a, _) in seam_pairs], dtype=int)
    idx_j = np.array([b for (_, b) in seam_pairs], dtype=int)

    pts_i = patch_i_vertices_xy[idx_i]  # (N,2)
    pts_j = patch_j_vertices_xy[idx_j]  # (N,2)

    # Apply optional transforms (for later stages)
    if transform_i is not None:
        pts_i = transform_i.apply(pts_i)
    if transform_j is not None:
        pts_j = transform_j.apply(pts_j)

    # Compute base lattice phase in [0,1)^2
    phi_i = phase_uv(pts_i, lattice)
    phi_j = phase_uv(pts_j, lattice)

    # Apply discrete κ/K offset (mod 1)
    off_i = (kappa_i / float(K)) % 1.0
    off_j = (kappa_j / float(K)) % 1.0

    phi_i = frac(phi_i + off_i)
    phi_j = frac(phi_j + off_j)

    # Wrapped delta per sample (scalar)
    d = wrapped_delta_uv(phi_i, phi_j, reduce=reduce_uv)  # (N,)

    # Average mismatch and apply seam weight
    return float(weight * np.mean(d))


def seam_phase_mismatch_from_file(
    seamfile_path: str,
    patch_vertices_by_id: Dict[int, np.ndarray],
    lattice: TextureLattice,
    kappas_by_id: Dict[int, int],
    K: int,
    weight: float,
    transforms_by_id: Optional[Dict[int, Rigid2D]] = None,
    reduce_uv: str = "mean",
) -> float:
    """
    Convenience wrapper: reads seamfile, looks up vertices + κ for the two patches,
    and returns the weighted seam mismatch.
    """
    patch_i, patch_j, pairs = load_exported_seamfile(seamfile_path)

    Vi = patch_vertices_by_id[patch_i]
    Vj = patch_vertices_by_id[patch_j]

    ki = kappas_by_id.get(patch_i, 0)
    kj = kappas_by_id.get(patch_j, 0)

    Ti = None
    Tj = None
    if transforms_by_id is not None:
        Ti = transforms_by_id.get(patch_i, None)
        Tj = transforms_by_id.get(patch_j, None)

    return seam_phase_mismatch(
        seam_pairs=pairs,
        patch_i_vertices_xy=Vi,
        patch_j_vertices_xy=Vj,
        lattice=lattice,
        kappa_i=ki,
        kappa_j=kj,
        K=K,
        weight=weight,
        transform_i=Ti,
        transform_j=Tj,
        reduce_uv=reduce_uv,
    )
