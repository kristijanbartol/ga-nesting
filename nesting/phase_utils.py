# phase_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class TextureLattice:
    """
    Periodic texture lattice in R^2.

    Lattice vectors:
        U = period_u * u_dir_normalized
        V = period_v * v_dir_normalized
    """
    u_dir: np.ndarray        # (2,)
    v_dir: np.ndarray        # (2,)
    period_u: float
    period_v: float

    def matrix(self) -> np.ndarray:
        U = self.u_dir / (np.linalg.norm(self.u_dir) + 1e-12) * self.period_u
        V = self.v_dir / (np.linalg.norm(self.v_dir) + 1e-12) * self.period_v
        return np.column_stack([U, V])  # 2x2


def frac(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)


def phase_uv(points_xy: np.ndarray, lattice: TextureLattice) -> np.ndarray:
    """
    Returns phases in [0,1)^2 for each point.
    """
    A = lattice.matrix()          # [U V]
    A_inv = np.linalg.inv(A)
    coords = (points_xy @ A_inv.T)  # lattice coordinates
    return frac(coords)


def wrap_signed_phase_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Signed wrapped phase diff in [-0.5, 0.5).
    """
    return ((a - b + 0.5) % 1.0) - 0.5


@dataclass(frozen=True)
class Rigid2D:
    """
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


def load_exported_seamfile(path: str) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    export_seamlines format:

        line1: symmetric flag (ignored)
        line2: patch_i
        line3: patch_j
        rest : "vidx_i vidx_j"
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


def seam_phase_residuals_uv(
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
) -> np.ndarray:
    """
    Residual vector for least squares:
      r = sqrt(weight) * [du0, dv0, du1, dv1, ...]
    where du,dv are signed wrapped phase diffs in [-0.5,0.5).

    weight=0 -> returns empty.
    """
    if weight <= 0.0 or len(seam_pairs) == 0:
        return np.zeros((0,), dtype=float)
    if K <= 0:
        raise ValueError("K must be positive")

    idx_i = np.array([a for (a, _) in seam_pairs], dtype=int)
    idx_j = np.array([b for (_, b) in seam_pairs], dtype=int)

    pts_i = patch_i_vertices_xy[idx_i]
    pts_j = patch_j_vertices_xy[idx_j]

    if transform_i is not None:
        pts_i = transform_i.apply(pts_i)
    if transform_j is not None:
        pts_j = transform_j.apply(pts_j)

    phi_i = phase_uv(pts_i, lattice)
    phi_j = phase_uv(pts_j, lattice)

    off_i = (kappa_i / float(K)) % 1.0
    off_j = (kappa_j / float(K)) % 1.0

    phi_i = frac(phi_i + off_i)
    phi_j = frac(phi_j + off_j)

    du = wrap_signed_phase_diff(phi_i[:, 0], phi_j[:, 0])
    dv = wrap_signed_phase_diff(phi_i[:, 1], phi_j[:, 1])

    r = np.empty((2 * len(du),), dtype=float)
    r[0::2] = du
    r[1::2] = dv

    r *= np.sqrt(weight)
    return r


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
) -> float:
    """
    Spec-compliant seam phase mismatch (f2 contribution for one seam).

    For each corresponding pair of seam points, computes the wrapped absolute
    phase difference per lattice axis:

        Delta(phi_a, phi_b) = min(|phi_a - phi_b|, 1 - |phi_a - phi_b|)

    in [0, 0.5].  The per-point mismatch is the mean of the U and V deltas.
    The seam mismatch is the mean over all pairs (approximating the 1/L integral),
    scaled by weight:

        Mismatch(s) = weight * mean_over_pairs( mean(Delta_u, Delta_v) )

    weight=0 returns 0.0 immediately.
    """
    if weight <= 0.0 or len(seam_pairs) == 0:
        return 0.0
    if K <= 0:
        raise ValueError("K must be positive")

    idx_i = np.array([a for (a, _) in seam_pairs], dtype=int)
    idx_j = np.array([b for (_, b) in seam_pairs], dtype=int)

    pts_i = patch_i_vertices_xy[idx_i]
    pts_j = patch_j_vertices_xy[idx_j]

    if transform_i is not None:
        pts_i = transform_i.apply(pts_i)
    if transform_j is not None:
        pts_j = transform_j.apply(pts_j)

    phi_i = phase_uv(pts_i, lattice)   # (N, 2) in [0,1)
    phi_j = phase_uv(pts_j, lattice)   # (N, 2) in [0,1)

    # Apply kappa offsets then re-wrap to [0,1)
    phi_i = frac(phi_i + (kappa_i / float(K)))
    phi_j = frac(phi_j + (kappa_j / float(K)))

    # Wrapped absolute difference per axis: Delta in [0, 0.5]
    diff = np.abs(phi_i - phi_j)           # (N, 2)
    delta = np.minimum(diff, 1.0 - diff)   # (N, 2)

    # Per-point mismatch: mean over both axes
    mismatch_per_point = delta.mean(axis=1)   # (N,)

    # Arc-length weighting (trapezoidal rule, approximates the 1/L_s integral).
    # Segment lengths are computed on the i-side; pairs are ordered along the seam.
    # For N=1 fall back to uniform (no neighbours to compute a length from).
    if pts_i.shape[0] < 2:
        return float(weight * mismatch_per_point.mean())

    seg_lengths = np.linalg.norm(np.diff(pts_i, axis=0), axis=1)  # (N-1,)
    # Trapezoid weights: each interior point owns half of its left and right segment.
    arc_weights = np.empty(pts_i.shape[0], dtype=float)
    arc_weights[0]    = seg_lengths[0] / 2.0
    arc_weights[1:-1] = (seg_lengths[:-1] + seg_lengths[1:]) / 2.0
    arc_weights[-1]   = seg_lengths[-1] / 2.0

    total_length = arc_weights.sum()
    if total_length < 1e-12:
        # Degenerate seam (all points coincide): fall back to uniform mean
        return float(weight * mismatch_per_point.mean())

    # Weighted mean ~ (1/L) * integral Delta dl
    return float(weight * np.dot(arc_weights, mismatch_per_point) / total_length)


# Alias kept for backward compatibility with existing callers
seam_phase_mismatch_scalar = seam_phase_mismatch