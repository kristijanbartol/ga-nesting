import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from nesting.phase_utils import phase_uv, frac, Rigid2D

PATCH_COLORS = ['#4C9BE8', '#F0A500', '#5DBB63', '#E85C5C', '#A06FD0', '#4ECDC4']


def _polygon_to_path(poly) -> Path:
    """Convert a Shapely polygon exterior to a matplotlib Path."""
    coords = np.array(poly.exterior.coords)
    codes = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY]
    return Path(coords, codes)


def _draw_texture_lines_in_patch(ax, poly, phase_offset_y, period_y, color, lw=1.0, alpha=0.6):
    """
    Draw horizontal stripe lines inside a single patch polygon.
    Lines are spaced period_y apart and shifted by phase_offset_y.
    Each line is clipped to the polygon boundary.
    """
    minx, miny, maxx, maxy = poly.bounds
    clip_path = PathPatch(_polygon_to_path(poly), transform=ax.transData)

    # First line position: find the nearest stripe below miny, shifted by phase
    y_start = miny - ((miny - phase_offset_y) % period_y)

    y = y_start
    while y <= maxy + period_y:
        line, = ax.plot([minx, maxx], [y, y],
                        color=color, lw=lw, alpha=alpha, zorder=4)
        line.set_clip_path(clip_path)
        y += period_y


def visualize_layout(fabric_state, texture_spec, title=None):
    fig, ax = plt.subplots(figsize=(14, 8))

    max_h = max(fabric_state.total_height + 50, 200)
    width = fabric_state.width

    # 1. Draw fabric boundary
    rect = mpatches.Rectangle((0, 0), width, fabric_state.total_height,
                               linewidth=2, edgecolor='black', facecolor='#FAFAFA', zorder=1)
    ax.add_patch(rect)

    # 2. Draw placed items + texture stripes clipped to each patch
    legend_handles = []
    for idx, (item, cx, cy, poly) in enumerate(fabric_state.placed_items):
        color = PATCH_COLORS[idx % len(PATCH_COLORS)]

        # Fill patch
        x, y = poly.exterior.xy
        ax.fill(x, y, color=color, alpha=0.25, zorder=2)
        ax.plot(x, y, color=color, lw=1.5, zorder=3)

        # Texture stripes clipped to this patch
        phase_y = getattr(item, 'phase_offset', (0.0, 0.0))[1]
        _draw_texture_lines_in_patch(
            ax, poly,
            phase_offset_y=phase_y,
            period_y=texture_spec.period_y,
            color=color, lw=1.2, alpha=0.85
        )

        # Anchor point
        ox, oy = getattr(item, 'phase_offset', (0.0, 0.0))
        ax.scatter([cx], [cy], color='red', s=25, zorder=15)
        ax.text(cx + 5, cy + 5, item.name, fontsize=8, color='black', zorder=16)

        legend_handles.append(mpatches.Patch(color=color, label=item.name))

    ax.set_xlim(-10, width + 10)
    ax.set_ylim(-10, max_h)
    ax.set_aspect('equal')
    ax.set_title(title or f"Nesting Result — Texture: {texture_spec.name}")
    ax.legend(handles=legend_handles, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()


def plot_seam_mismatch(constraints, V_full_by_id, lattice, kappas_by_id, K, transforms, title):
    """
    For each seam constraint, plot the per-point phase mismatch along the seam.
    X axis: point index along seam. Y axis: Delta(phi_i, phi_j) in [0, 0.5].
    """
    n = len(constraints)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig.suptitle(title, fontsize=12)

    for ax, c in zip(axes[0], constraints):
        if len(c.pairs) == 0:
            ax.set_title(c.name)
            continue

        idx_i = np.array([a for a, _ in c.pairs], dtype=int)
        idx_j = np.array([b for _, b in c.pairs], dtype=int)

        pts_i = V_full_by_id[c.patch_i][idx_i]
        pts_j = V_full_by_id[c.patch_j][idx_j]

        Ti = transforms.get(c.patch_i, Rigid2D(0, 0, 0))
        Tj = transforms.get(c.patch_j, Rigid2D(0, 0, 0))
        pts_i = Ti.apply(pts_i)
        pts_j = Tj.apply(pts_j)

        ki = kappas_by_id.get(c.patch_i, 0)
        kj = kappas_by_id.get(c.patch_j, 0)

        phi_i = frac(phase_uv(pts_i, lattice) + ki / float(K))
        phi_j = frac(phase_uv(pts_j, lattice) + kj / float(K))

        diff = np.abs(phi_i - phi_j)
        delta = np.minimum(diff, 1.0 - diff)          # (N, 2)
        mismatch = delta.mean(axis=1)                  # (N,) per point

        ax.plot(mismatch, lw=1.2)
        ax.axhline(mismatch.mean(), color='red', lw=1, linestyle='--', label=f'mean={mismatch.mean():.3f}')
        ax.set_ylim(0, 0.5)
        ax.set_xlabel('point index along seam')
        ax.set_ylabel('Δ phase')
        ax.set_title(c.name)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
