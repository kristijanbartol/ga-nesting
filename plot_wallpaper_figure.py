"""Generate the wallpaper-groups figure for the paper.

Loads the simulation PLY saved by run_wallpaper_groups.sh for each of the
8 supported wallpaper groups and renders a front-view textured garment using
matplotlib tripcolor.  Assembles a 2×4 grid and saves to paper/figures/.

Usage:
    python plot_wallpaper_figure.py                  # upper garment (default)
    python plot_wallpaper_figure.py --garment lower
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from plyfile import PlyData


# ── Texture palette ───────────────────────────────────────────────────────────
COLOR_A = np.array([0.10, 0.15, 0.35])   # dark navy
COLOR_B = np.array([0.85, 0.88, 0.95])   # light blue-white

# ── Layout ────────────────────────────────────────────────────────────────────
GROUPS = ["stripes", "diagonal_stripes", "grid", "p4", "p4m", "pg", "pmg", "pgg"]

GROUP_LABELS = {
    "stripes":          "Stripes\n(PM)",
    "diagonal_stripes": "Diagonal stripes\n(PM)",
    "grid":             "Grid\n(PMM)",
    "p4":               "Pinwheel\n(P4)",
    "p4m":              "Polka dots\n(P4M)",
    "pg":               "Herringbone\n(PG)",
    "pmg":              "Chevron\n(PMG)",
    "pgg":              "Brick bond\n(PGG)",
}


def load_ply(ply_path: str):
    """Return (vertices, faces, uv_mm) from a simulation PLY."""
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    vertices = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
    uv = np.stack([np.asarray(v['s'], dtype=np.float64),
                   np.asarray(v['t'], dtype=np.float64)], axis=1)
    faces = np.array([f[0] for f in ply['face'].data], dtype=np.int32)
    return vertices, faces, uv


def _texture_value(uv_mm, period, group):
    """Scalar texture in [0,1] per vertex — mirrors visualize_simulation.py."""
    u, v = uv_mm[:, 0], uv_mm[:, 1]
    T = period
    if group == 'stripes':
        return 0.5 + 0.5 * np.cos(2 * np.pi * v / T)
    if group == 'diagonal_stripes':
        return 0.5 + 0.5 * np.cos(2 * np.pi * (v - u) / (T * np.sqrt(2)))
    if group == 'grid':
        return (0.5 + 0.5 * np.cos(2 * np.pi * u / T)) * \
               (0.5 + 0.5 * np.cos(2 * np.pi * v / T))
    if group == 'p4':
        uf = (u % T) / T - 0.5
        vf = (v % T) / T - 0.5
        return 0.5 + 0.5 * np.sin(4 * np.arctan2(vf, uf))
    if group == 'p4m':
        uf = (u % T) / T
        vf = (v % T) / T
        du = uf - np.round(uf)
        dv = vf - np.round(vf)
        return np.clip(1 - np.sqrt(du**2 + dv**2) / 0.32, 0, 1)
    if group == 'pg':
        uf = (u % T) / T
        vf = (v % T) / T
        parity = np.floor(vf * 2).astype(int) % 2
        phase = np.where(parity == 0, (vf - uf) * 2, (vf + uf) * 2)
        return 0.5 + 0.5 * np.cos(2 * np.pi * phase)
    if group == 'pmg':
        uf = (u % T) / T
        vf = (v % T) / T
        parity = np.floor(vf * 2).astype(int) % 2
        phase = np.where(parity == 0,
                         (vf * 2 % 1) - uf,
                         (vf * 2 % 1) + uf)
        return 0.5 + 0.5 * np.cos(2 * np.pi * phase)
    if group == 'pgg':
        uf = (u % T) / T
        vf = (v % T) / T
        row = np.floor(vf * 2).astype(int) % 2
        us = np.where(row == 1, (uf + 0.5) % 1.0, uf)
        return (0.5 + 0.5 * np.cos(2 * np.pi * us)) * \
               (0.5 + 0.5 * np.cos(2 * np.pi * vf * 2))
    return 0.5 + 0.5 * np.cos(2 * np.pi * v / T)


def _front_facing_mask(vertices, faces):
    """Boolean mask: True for faces whose normal has positive Z (faces the viewer)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals_z = np.cross(v1 - v0, v2 - v0)[:, 2]
    return normals_z > 0


def render_garment(ax, vertices, faces, uv_mm, group, period=50.0):
    """Render front-view textured garment onto ax using tripcolor."""
    mask = _front_facing_mask(vertices, faces)
    front_faces = faces[mask]

    t_val = _texture_value(uv_mm, period, group)
    cmap = LinearSegmentedColormap.from_list("tex", [COLOR_B, COLOR_A])

    # Project to X-Y (front view: camera looks along -Z)
    x, y = vertices[:, 0], vertices[:, 1]
    tri = mtri.Triangulation(x, y, front_faces)
    ax.tripcolor(tri, t_val, cmap=cmap, vmin=0, vmax=1, shading='gouraud')
    ax.set_aspect('equal')
    ax.axis('off')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--garment", default="upper", choices=["upper", "lower"])
    p.add_argument("--data_root", default="results/wallpaper_groups",
                   help="Root dir written by run_wallpaper_groups.sh")
    p.add_argument("--out_dir", default="paper/figures")
    p.add_argument("--period", type=float, default=50.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes_flat = axes.flatten()

    any_found = False
    for idx, group in enumerate(GROUPS):
        ax = axes_flat[idx]
        ply_path = os.path.join(args.data_root, args.garment, group, "cloth_00000.ply")

        if not os.path.exists(ply_path):
            print(f"[warn] missing PLY for {group}: {ply_path}")
            ax.text(0.5, 0.5, f"{group}\n(no data)", ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.axis('off')
        else:
            vertices, faces, uv_mm = load_ply(ply_path)
            render_garment(ax, vertices, faces, uv_mm, group, args.period)
            print(f"[plot] {group}  vertices={len(vertices)}  faces={len(faces)}")
            any_found = True

        ax.set_title(GROUP_LABELS[group], fontsize=8, pad=3)

    if not any_found:
        print("[error] no PLY files found — run run_wallpaper_groups.sh first")
        return

    fig.tight_layout(pad=0.5)

    for ext in ("pdf", "png"):
        out = os.path.join(args.out_dir, f"wallpaper_groups.{ext}")
        fig.savefig(out, dpi=150)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
