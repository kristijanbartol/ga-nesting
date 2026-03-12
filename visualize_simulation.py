"""
Visualize a simulated garment PLY with stripe or grid texture in Polyscope,
and debug the UV pipeline by coloring the 2D sewing pattern with the
same UV values.

UV coordinates stored in the PLY (s, t) are in fabric-space mm, produced by
the simulation pipeline with Stage2 + kappa transforms applied.  Dividing by
the period gives normalized UV where one full tile = one texture period.
"""

import os
import numpy as np
from plyfile import PlyData
import polyscope as ps


# ── Config ──────────────────────────────────────────────────────────────────
PERIOD_U_MM    = 50.0
PERIOD_V_MM    = 50.0
WALLPAPER_GROUP = "stripes"   # "stripes" | "grid"
PLY_PATH      = "results/simulation/upper/cloth_00000.ply"
PARAM_DIR     = "results/pattern/best/upper"
PATCHES_3D_DIR = "data/patches/best/upper"
BACK_LABELS   = "data/labels/upper/back.txt"

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


def _texture_value(uv_mm: np.ndarray,
                   period_u_mm: float,
                   period_v_mm: float,
                   wallpaper_group: str) -> np.ndarray:
    """
    Per-vertex texture scalar in [0, 1].

    stripes: varies along v only  — 0.5 + 0.5·cos(2π·v/T_v)
    grid:    product of both axes — (0.5 + 0.5·cos(2π·u/T_u))
                                  · (0.5 + 0.5·cos(2π·v/T_v))
    """
    t_v = 0.5 + 0.5 * np.cos(2.0 * np.pi * uv_mm[:, 1] / period_v_mm)
    if wallpaper_group == 'grid':
        t_u = 0.5 + 0.5 * np.cos(2.0 * np.pi * uv_mm[:, 0] / period_u_mm)
        return t_u * t_v
    return t_v


def texture_colors(uv_mm: np.ndarray,
                   period_u_mm: float,
                   period_v_mm: float,
                   wallpaper_group: str,
                   color_a: np.ndarray = COLOR_A,
                   color_b: np.ndarray = COLOR_B) -> np.ndarray:
    """Per-vertex (N, 3) RGB texture color from fabric-space UV in mm."""
    t = _texture_value(uv_mm, period_u_mm, period_v_mm, wallpaper_group)
    return np.outer(t, color_a) + np.outer(1.0 - t, color_b)


def _load_2d_pattern(param_dir: str, back_labels: str):
    """Load subdivided 2D sewing pattern in the same order as the simulation."""
    import trimesh
    with open(back_labels) as f:
        back_idxs = list(map(int, f.read().split()))
    meshes = []
    for d in sorted(os.listdir(param_dir)):
        path = os.path.join(param_dir, d, 'optim_final-seams.ply')
        if not os.path.exists(path):
            continue
        m = trimesh.load(path)
        if int(d[-2:]) in back_idxs:
            m.vertices[:, 1] *= -1
        meshes.append(m.subdivide())
    import trimesh as _tr
    merged = _tr.util.concatenate(meshes)
    return _tr.Trimesh(vertices=merged.vertices, faces=merged.faces, process=False)


def debug_uv_2d(ply_path: str = PLY_PATH,
                param_dir: str = PARAM_DIR,
                back_labels: str = BACK_LABELS,
                period_u_mm: float = PERIOD_U_MM,
                period_v_mm: float = PERIOD_V_MM,
                wallpaper_group: str = WALLPAPER_GROUP):
    """
    Color the 2D sewing pattern by the UV values stored in the PLY and
    show the UV distribution in fabric space.

    Left panel:  2D parameterisation vertices colored by texture(s, t) from PLY.
                 Texture must be continuous across seam boundaries if the
                 UV pipeline is correct.
    Right panel: UV scatter in fabric space (mm) — sanity-checks magnitude and
                 distribution; patches should occupy distinct, non-overlapping
                 regions separated by roughly the patch size in mm.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import LinearSegmentedColormap

    _, faces, uv_mm = load_ply_with_uv(ply_path)
    pattern = _load_2d_pattern(param_dir, back_labels)
    verts_2d = pattern.vertices[:, :2]

    assert len(verts_2d) == len(uv_mm), (
        f"Vertex count mismatch: 2D pattern={len(verts_2d)}, PLY UV={len(uv_mm)}. "
        "Ensure PARAM_DIR matches the patches used during simulation."
    )

    t_val = _texture_value(uv_mm, period_u_mm, period_v_mm, wallpaper_group)
    cmap  = LinearSegmentedColormap.from_list("texture", [COLOR_B, COLOR_A])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: 2D sewing pattern colored by PLY UV
    tri = mtri.Triangulation(verts_2d[:, 0], verts_2d[:, 1], faces)
    ax1.tripcolor(tri, t_val, cmap=cmap, vmin=0, vmax=1, shading='gouraud')
    ax1.set_aspect('equal')
    ax1.set_title(f'2D sewing pattern — {wallpaper_group} texture from PLY UV\n'
                  '(texture must be continuous across seam boundaries)')
    ax1.set_xlabel('u  (m, parameterisation space)')
    ax1.set_ylabel('v  (m, parameterisation space)')

    # Right: UV scatter in fabric space
    ax2.scatter(uv_mm[:, 0], uv_mm[:, 1],
                c=t_val, cmap=cmap, s=0.3, vmin=0, vmax=1)
    ax2.set_aspect('equal')
    ax2.set_title('UV distribution in fabric space (mm)\n'
                  '(patches should occupy distinct, non-overlapping regions)')
    ax2.set_xlabel('s  (mm, fabric u)')
    ax2.set_ylabel('t  (mm, fabric v)')

    plt.suptitle(f'UV debug  —  {ply_path}', fontsize=9)
    plt.tight_layout()
    plt.show()


def show_best_nesting(json_path: str = "results/pattern/best/best_individual.json"):
    """
    Reproduce the 'BEST (GA kappa)' nesting layout using the same util functions
    as run_ga.py.  Requires results/pattern/best/best_individual.json written by
    run_ga.py after the GA finishes.
    """
    import json
    from ga_spec import Genome
    from nesting.phase_utils import TextureLattice, Rigid2D
    from spec import TextureSpec
    from run_ga import nest_and_show

    with open(json_path) as f:
        d = json.load(f)

    genome = Genome(
        delta=np.array(d['delta']),
        rho=np.array(d['rho']),
        kappa=np.array(d['kappa']),
        pi=np.array(d['pi']),
        h=d['h'],
    )
    lattice = TextureLattice(
        u_dir=np.array([1.0, 0.0]),
        v_dir=np.array([0.0, 1.0]),
        period_u=d['period_u_mm'],
        period_v=d['period_v_mm'],
    )
    texture = TextureSpec(name=d.get('wallpaper_group', 'stripes'),
                          period_x=d['period_u_mm'],
                          period_y=d['period_v_mm'],
                          wallpaper_group=d.get('wallpaper_group', 'stripes'))

    # Restore the exact Stage2 transforms and kappas used during evaluation
    # so the nesting layout matches the run_ga.py BEST visualization exactly.
    tsol = {int(pid): Rigid2D(*vals) for pid, vals in d['Tsol'].items()}
    kappas_by_id = {int(pid): int(k) for pid, k in d['kappas_by_id'].items()}

    nest_and_show(
        latest_root=d['best_root'],
        seam_dir=d['best_seam_dir'],
        lattice=lattice,
        texture=texture,
        fabric_width=d['fabric_width_mm'],
        genome=genome,
        K=d['K'],
        title="BEST (GA kappa) — reference",
        garment_part=d['garment_part'],
        num_bodies=d['num_bodies'],
        show_layout=True,
        precomputed_tsol=tsol,
        precomputed_kappas_by_id=kappas_by_id,
    )


def visualize(ply_path: str = PLY_PATH,
              period_u_mm: float = PERIOD_U_MM,
              period_v_mm: float = PERIOD_V_MM,
              wallpaper_group: str = WALLPAPER_GROUP):
    """3D Polyscope render of the simulated garment with texture."""
    vertices, faces, uv_mm = load_ply_with_uv(ply_path)

    print(f"Loaded '{ply_path}': {len(vertices)} vertices, {len(faces)} faces")
    print(f"UV range — s: [{uv_mm[:, 0].min():.1f}, {uv_mm[:, 0].max():.1f}] mm  "
          f"t: [{uv_mm[:, 1].min():.1f}, {uv_mm[:, 1].max():.1f}] mm")

    colors = texture_colors(uv_mm, period_u_mm, period_v_mm, wallpaper_group)

    ps.init()
    ps_mesh = ps.register_surface_mesh("garment", vertices, faces, smooth_shade=True)
    ps_mesh.add_color_quantity("texture", colors, defined_on='vertices', enabled=True)
    ps.show()


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Visualize simulated garment texture.")
    p.add_argument("--wallpaper", default=WALLPAPER_GROUP, choices=["stripes", "grid"],
                   help="Texture wallpaper group (default: stripes)")
    p.add_argument("--ply", default=PLY_PATH,
                   help="Path to simulated garment PLY (default: %(default)s)")
    p.add_argument("--json", default="results/pattern/best/best_individual.json",
                   help="Path to best_individual.json (default: %(default)s)")
    p.add_argument("--period", type=float, default=PERIOD_U_MM,
                   help="Texture period in mm, applied to both U and V (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    show_best_nesting(json_path=args.json)                                 # 1) reference nesting layout
    debug_uv_2d(ply_path=args.ply,
                period_u_mm=args.period, period_v_mm=args.period,
                wallpaper_group=args.wallpaper)                            # 2) UV debug: 2D pattern colored by PLY UV
    visualize(ply_path=args.ply,
              period_u_mm=args.period, period_v_mm=args.period,
              wallpaper_group=args.wallpaper)                              # 3) 3D Polyscope render
