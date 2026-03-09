"""
Visualize a simulated garment PLY with stripe texture in Polyscope.

UV coordinates stored in the PLY (s, t) are in fabric-space mm, produced by
the simulation pipeline with Stage2 + kappa transforms applied.  Dividing by
the stripe period gives normalized UV where one full tile = one stripe period.
Stripe color is computed analytically per vertex — identical to image texture
mapping for a 1-D periodic pattern, but requires no image file and guarantees
the period matches the phase definition used by the GA exactly.
"""

import numpy as np
from plyfile import PlyData
import polyscope as ps


# ── Config ──────────────────────────────────────────────────────────────────
PERIOD_U_MM = 50.0
PERIOD_V_MM = 50.0
PLY_PATH = "results/simulation/upper/cloth_00000.ply"

# Stripe colors: (R, G, B) in [0, 1] for the two bands
COLOR_A = np.array([0.10, 0.15, 0.35])   # dark navy
COLOR_B = np.array([0.85, 0.88, 0.95])   # light blue-white
# ────────────────────────────────────────────────────────────────────────────


def load_ply_with_uv(ply_path: str):
    """Load mesh vertices, faces, and (s, t) UV from PLY."""
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    vertices = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
    uv = np.stack([np.asarray(v['s'], dtype=np.float64),
                   np.asarray(v['t'], dtype=np.float64)], axis=1)
    faces = np.array([f[0] for f in ply['face'].data], dtype=np.int32)
    return vertices, faces, uv


def stripe_colors(uv_mm: np.ndarray,
                  period_v_mm: float,
                  color_a: np.ndarray,
                  color_b: np.ndarray) -> np.ndarray:
    """Per-vertex stripe color from fabric-space UV (in mm).

    Horizontal stripes vary along v.  Smooth cosine blend gives
    the same visual as a sinusoidal stripe image with tile size = period.
    """
    v_norm = uv_mm[:, 1] / period_v_mm          # one full period = 1.0
    t = 0.5 + 0.5 * np.cos(2.0 * np.pi * v_norm)  # [0, 1], smooth
    return np.outer(t, color_a) + np.outer(1.0 - t, color_b)


def visualize(ply_path: str = PLY_PATH,
              period_u_mm: float = PERIOD_U_MM,
              period_v_mm: float = PERIOD_V_MM):
    vertices, faces, uv_mm = load_ply_with_uv(ply_path)

    print(f"Loaded '{ply_path}': {len(vertices)} vertices, {len(faces)} faces")
    print(f"UV range — s: [{uv_mm[:, 0].min():.1f}, {uv_mm[:, 0].max():.1f}] mm  "
          f"t: [{uv_mm[:, 1].min():.1f}, {uv_mm[:, 1].max():.1f}] mm")

    colors = stripe_colors(uv_mm, period_v_mm, COLOR_A, COLOR_B)

    ps.init()
    ps_mesh = ps.register_surface_mesh("garment", vertices, faces, smooth_shade=True)
    ps_mesh.add_color_quantity("stripe_texture", colors, defined_on='vertices', enabled=True)
    ps.show()


if __name__ == "__main__":
    visualize()
