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
WALLPAPER_GROUP = "diagonal_stripes"   # "stripes" | "diagonal_stripes" | "grid"
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

    stripes:          0.5 + 0.5·cos(2π·v/T_v)
    diagonal_stripes: 0.5 + 0.5·cos(2π·(v-u) / (T_v·√2))
    grid:             (0.5 + 0.5·cos_u) · (0.5 + 0.5·cos_v)
    p4:               0.5 + 0.5·sin(4·θ)  — pinwheel per cell, θ=arctan2(v_frac, u_frac)
    p4m:              clip(1 - r/0.32, 0, 1)  — polka dots, r = dist to nearest lattice pt
    """
    if wallpaper_group == 'diagonal_stripes':
        return 0.5 + 0.5 * np.cos(
            2.0 * np.pi * (uv_mm[:, 1] - uv_mm[:, 0]) / (period_v_mm * np.sqrt(2.0))
        )
    if wallpaper_group == 'grid':
        cos_u = np.cos(2.0 * np.pi * uv_mm[:, 0] / period_u_mm)
        cos_v = np.cos(2.0 * np.pi * uv_mm[:, 1] / period_v_mm)
        return (0.5 + 0.5 * cos_u) * (0.5 + 0.5 * cos_v)
    if wallpaper_group == 'p4':
        # Pinwheel: sin(4·θ) within each cell has 4-fold rotation but no mirror
        # symmetry — correctly represents P4.  θ is the angle from the cell centre.
        u_frac = (uv_mm[:, 0] % period_u_mm) / period_u_mm - 0.5   # [-0.5, 0.5)
        v_frac = (uv_mm[:, 1] % period_v_mm) / period_v_mm - 0.5
        angle = np.arctan2(v_frac, u_frac)
        return 0.5 + 0.5 * np.sin(4.0 * angle)
    if wallpaper_group == 'p4m':
        # Polka dots: linear falloff from each lattice point; bright circle on
        # dark background.  dot_radius=0.32 gives ~64% coverage per cell.
        u_frac = (uv_mm[:, 0] % period_u_mm) / period_u_mm
        v_frac = (uv_mm[:, 1] % period_v_mm) / period_v_mm
        du = u_frac - np.round(u_frac)
        dv = v_frac - np.round(v_frac)
        r = np.sqrt(du ** 2 + dv ** 2)
        return np.clip(1.0 - r / 0.32, 0.0, 1.0)
    if wallpaper_group == 'pg':
        # Herringbone: alternating rows of diagonal stripes reversing every half-period.
        # Even rows (v_cell < 0.5): use (v - u) diagonal; odd rows: use (v + u) diagonal.
        u_frac = (uv_mm[:, 0] % period_u_mm) / period_u_mm
        v_frac = (uv_mm[:, 1] % period_v_mm) / period_v_mm
        row_parity = np.floor(v_frac * 2.0).astype(int) % 2
        phase_ne = (v_frac - u_frac) * 2.0   # NE diagonal within cell
        phase_nw = (v_frac + u_frac) * 2.0   # NW diagonal within cell
        return 0.5 + 0.5 * np.cos(2.0 * np.pi * np.where(row_parity == 0, phase_ne, phase_nw))
    if wallpaper_group == 'pmg':
        # Chevron: V-shapes via rows that alternate slope direction, producing
        # mirror symmetry on vertical axes but glide on horizontal.
        u_frac = (uv_mm[:, 0] % period_u_mm) / period_u_mm
        v_frac = (uv_mm[:, 1] % period_v_mm) / period_v_mm
        row_parity = np.floor(v_frac * 2.0).astype(int) % 2
        phase_ne = (v_frac * 2.0 % 1.0) - u_frac
        phase_nw = (v_frac * 2.0 % 1.0) + u_frac
        return 0.5 + 0.5 * np.cos(2.0 * np.pi * np.where(row_parity == 0, phase_ne, phase_nw))
    if wallpaper_group == 'pgg':
        # Brick bond: staggered rectangles — odd rows offset by half a period.
        u_frac = (uv_mm[:, 0] % period_u_mm) / period_u_mm
        v_frac = (uv_mm[:, 1] % period_v_mm) / period_v_mm
        row = np.floor(v_frac * 2.0).astype(int) % 2
        u_shifted = np.where(row == 1, (u_frac + 0.5) % 1.0, u_frac)
        cos_u = np.cos(2.0 * np.pi * u_shifted)
        cos_v = np.cos(2.0 * np.pi * v_frac * 2.0)
        return (0.5 + 0.5 * cos_u) * (0.5 + 0.5 * cos_v)
    return 0.5 + 0.5 * np.cos(2.0 * np.pi * uv_mm[:, 1] / period_v_mm)


def texture_colors(uv_mm: np.ndarray,
                   period_u_mm: float,
                   period_v_mm: float,
                   wallpaper_group: str,
                   color_a: np.ndarray = COLOR_A,
                   color_b: np.ndarray = COLOR_B) -> np.ndarray:
    """Per-vertex (N, 3) RGB texture color from fabric-space UV in mm."""
    t = _texture_value(uv_mm, period_u_mm, period_v_mm, wallpaper_group)
    return np.outer(t, color_a) + np.outer(1.0 - t, color_b)


def render_tile_image(period_u_mm: float,
                      period_v_mm: float,
                      wallpaper_group: str,
                      resolution: int = 1024,
                      color_a: np.ndarray = COLOR_A,
                      color_b: np.ndarray = COLOR_B) -> np.ndarray:
    """Render one period of the wallpaper pattern as an (H, W, 3) float32 image.

    The image covers UV in [0, period_u) x [0, period_v) and tiles seamlessly.
    Row 0 is the top of the image (v = period_v), matching Polyscope's
    ``image_origin='upper_left'`` convention.
    """
    u = np.linspace(0, period_u_mm, resolution, endpoint=False)
    v = np.linspace(0, period_v_mm, resolution, endpoint=False)
    uu, vv = np.meshgrid(u, v)
    uv_grid = np.stack([uu.ravel(), vv.ravel()], axis=1)

    t = _texture_value(uv_grid, period_u_mm, period_v_mm, wallpaper_group)
    rgb = np.outer(t, color_a) + np.outer(1.0 - t, color_b)
    img = rgb.reshape(resolution, resolution, 3).astype(np.float32)
    return img[::-1]  # flip rows: row 0 = top (high v)


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


def _fix_wrapping_corners(uv_per_vertex: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Build per-corner UVs with period-boundary wrapping fixed.

    For each face, if any pair of corner UVs differs by more than 0.5 in
    either axis, shift the smaller values up by 1.0 so that linear
    interpolation across the triangle stays local instead of sweeping
    through the entire tile.

    Returns (F*3, 2) corner UV array in the order Polyscope expects.
    """
    # Gather per-corner UVs: (F, 3, 2)
    corner_uv = uv_per_vertex[faces]

    for axis in range(2):
        col = corner_uv[:, :, axis]             # (F, 3)
        lo = col.min(axis=1, keepdims=True)      # (F, 1)
        hi = col.max(axis=1, keepdims=True)
        wrap_mask = (hi - lo) > 0.5              # (F, 1)  faces that wrap
        # Shift corners that are in the lower half across the boundary
        shift = wrap_mask & (col < 0.5)          # (F, 3)
        corner_uv[:, :, axis] += shift.astype(float)

    return corner_uv.reshape(-1, 2)              # (F*3, 2)


def visualize(ply_path: str = PLY_PATH,
              period_u_mm: float = PERIOD_U_MM,
              period_v_mm: float = PERIOD_V_MM,
              wallpaper_group: str = WALLPAPER_GROUP,
              tex_resolution: int = 1024):
    """3D Polyscope render of the simulated garment with texture-mapped wallpaper."""
    vertices, faces, uv_mm = load_ply_with_uv(ply_path)

    print(f"Loaded '{ply_path}': {len(vertices)} vertices, {len(faces)} faces")
    print(f"UV range — s: [{uv_mm[:, 0].min():.1f}, {uv_mm[:, 0].max():.1f}] mm  "
          f"t: [{uv_mm[:, 1].min():.1f}, {uv_mm[:, 1].max():.1f}] mm")

    # Normalize UV to [0,1) within one tile
    uv_norm = np.stack([
        (uv_mm[:, 0] % period_u_mm) / period_u_mm,
        (uv_mm[:, 1] % period_v_mm) / period_v_mm,
    ], axis=1)

    # Build per-corner UVs with wrapping fixed at period boundaries
    corner_uv = _fix_wrapping_corners(uv_norm, faces)

    tile_img = render_tile_image(period_u_mm, period_v_mm, wallpaper_group,
                                 resolution=tex_resolution)

    ps.init()
    ps_mesh = ps.register_surface_mesh("garment", vertices, faces, smooth_shade=True)
    ps_mesh.add_parameterization_quantity("uv", corner_uv, defined_on='corners',
                                          enabled=False)
    ps_mesh.add_color_quantity("texture", tile_img,
                               defined_on='texture', param_name='uv',
                               enabled=True)
    ps.show()


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Visualize simulated garment texture.")
    p.add_argument("--garment", default="upper",
                   help="Garment type (default: upper)")
    p.add_argument("--baseline", default=None, choices=["b0", "b1", "b2"],
                   help="Visualize a baseline result instead of the GA (loads from results/simulation/<garment>/<baseline>/)")
    p.add_argument("--wallpaper", default=WALLPAPER_GROUP, choices=["stripes", "diagonal_stripes", "grid", "p4", "p4m", "pg", "pmg", "pgg"],
                   help="Texture wallpaper group (default: stripes)")
    p.add_argument("--ply", default=None,
                   help="Path to simulated garment PLY (default: auto from --garment/--baseline)")
    p.add_argument("--json", default="results/pattern/best/best_individual.json",
                   help="Path to best_individual.json (default: %(default)s)")
    p.add_argument("--period", type=float, default=PERIOD_U_MM,
                   help="Texture period in mm, applied to both U and V (default: %(default)s)")
    p.add_argument("--tex-resolution", type=int, default=1024,
                   help="Texture tile resolution in pixels (default: 1024)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    g = args.garment

    if args.baseline:
        ply_path    = args.ply or f"results/simulation/{g}/{args.baseline}/cloth_00000.ply"
        param_dir   = f"results/pattern/latest/{g}"
        back_labels = f"data/labels/{g}/back.txt"
    else:
        ply_path    = args.ply or f"results/simulation/{g}/cloth_00000.ply"
        param_dir   = f"results/pattern/best/{g}"
        back_labels = f"data/labels/{g}/back.txt"

    if not args.baseline:
        show_best_nesting(json_path=args.json)                             # 1) reference nesting layout (GA only)
    debug_uv_2d(ply_path=ply_path,
                param_dir=param_dir,
                back_labels=back_labels,
                period_u_mm=args.period, period_v_mm=args.period,
                wallpaper_group=args.wallpaper)                            # 2) UV debug: 2D pattern colored by PLY UV
    visualize(ply_path=ply_path,
              period_u_mm=args.period, period_v_mm=args.period,
              wallpaper_group=args.wallpaper,
              tex_resolution=args.tex_resolution)                          # 3) 3D Polyscope render
