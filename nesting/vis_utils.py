import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from nesting.phase_utils import phase_uv, frac, Rigid2D

PATCH_COLORS = ['#4C9BE8', '#F0A500', '#5DBB63', '#E85C5C', '#A06FD0', '#4ECDC4',
                '#1A6BBF', '#C07800', '#2E8B3C', '#B03030', '#6B3A9E', '#2A9D8F']


def _polygon_to_path(poly) -> Path:
    """Convert a Shapely polygon exterior to a matplotlib Path."""
    coords = np.array(poly.exterior.coords)
    codes = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY]
    return Path(coords, codes)


def _draw_texture_hlines_in_patch(ax, poly, phase_offset_y, period_y, color, lw=1.0, alpha=0.6):
    """Draw horizontal stripe lines clipped to poly."""
    minx, miny, maxx, maxy = poly.bounds
    clip_path = PathPatch(_polygon_to_path(poly), transform=ax.transData)
    y = miny - ((miny - phase_offset_y) % period_y)
    while y <= maxy + period_y:
        line, = ax.plot([minx, maxx], [y, y], color=color, lw=lw, alpha=alpha, zorder=4)
        line.set_clip_path(clip_path)
        y += period_y


def _draw_texture_vlines_in_patch(ax, poly, phase_offset_x, period_x, color, lw=1.0, alpha=0.6):
    """Draw vertical stripe lines clipped to poly."""
    minx, miny, maxx, maxy = poly.bounds
    clip_path = PathPatch(_polygon_to_path(poly), transform=ax.transData)
    x = minx - ((minx - phase_offset_x) % period_x)
    while x <= maxx + period_x:
        line, = ax.plot([x, x], [miny, maxy], color=color, lw=lw, alpha=alpha, zorder=4)
        line.set_clip_path(clip_path)
        x += period_x


def _draw_texture_in_patch(ax, poly, texture_spec, color, lw=1.0, alpha=0.6):
    """Dispatch texture line drawing based on wallpaper group."""
    group = getattr(texture_spec, 'wallpaper_group', 'stripes')
    _draw_texture_hlines_in_patch(ax, poly, 0.0, texture_spec.period_y, color, lw, alpha)
    if group == 'grid':
        _draw_texture_vlines_in_patch(ax, poly, 0.0, texture_spec.period_x, color, lw, alpha)


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

        # Texture lines clipped to this patch (horizontal for stripes, H+V for grid).
        # phase_offset=0 so all patches share the same global texture grid.
        _draw_texture_in_patch(ax, poly, texture_spec, color, lw=1.2, alpha=0.85)

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


def visualize_population(pop, texture_spec, title=None, max_cols=6):
    """
    Draw a grid of nesting layouts for all individuals, ordered best → worst
    (lowest fitness sum first, reading left-to-right, top-to-bottom).

    Requires that each individual has ``ind.meta["fabric_state"]`` set (populated
    automatically by RealEvaluator after nesting).
    """
    sorted_pop = sorted(
        [ind for ind in pop if ind.fitness is not None],
        key=lambda ind: ind.fitness.values.sum()
    )
    n = len(sorted_pop)
    if n == 0:
        print("[visualize_population] No evaluated individuals to display.")
        return

    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    # Make each cell roughly square: width proportional to fabric width, height to total height.
    cell_w = 3.0
    cell_h = 3.5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_w, rows * cell_h))

    # Normalise axes to always be a flat list.
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for rank, ind in enumerate(sorted_pop):
        ax = axes[rank]
        fabric = ind.meta.get("fabric_state")

        f_sum = ind.fitness.values.sum()
        f1 = ind.meta.get("f1_height_mm", ind.fitness.values[0])
        f2 = ind.meta.get("f2_phase", ind.fitness.values[1])
        kappa_str = str(ind.genome.kappa.tolist())
        h_val = getattr(ind.genome, "h", "?")
        pi_str = str(ind.genome.pi.tolist())

        if fabric is None:
            ax.text(0.5, 0.5, "no fabric_state", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="red")
            ax.set_title(f"#{rank + 1}  Σ={f_sum:.3f}", fontsize=7)
            ax.axis("off")
            continue

        width = fabric.width
        height = fabric.total_height if fabric.total_height > 0 else 1.0

        # Fabric rectangle
        rect = mpatches.Rectangle(
            (0, 0), width, height,
            linewidth=1, edgecolor="black", facecolor="#FAFAFA", zorder=1
        )
        ax.add_patch(rect)

        # Patches
        for idx, (item, cx, cy, poly) in enumerate(fabric.placed_items):
            color = PATCH_COLORS[idx % len(PATCH_COLORS)]
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, alpha=0.45, zorder=2)
            ax.plot(x, y, color=color, lw=0.8, zorder=3)

        ax.set_xlim(-width * 0.02, width * 1.02)
        ax.set_ylim(-height * 0.05, height * 1.1)
        ax.set_aspect("equal")
        ax.axis("off")

        # Annotate: rank, fitness components, key genome fields
        label = (
            f"#{rank + 1}  Σ={f_sum:.3f}\n"
            f"h={f1:.0f}mm  φ={f2:.3f}\n"
            f"κ={kappa_str}  h={h_val}\n"
            f"π={pi_str}"
        )
        ax.set_title(label, fontsize=5.5, loc="left", pad=2)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title or "Population — best (top-left) to worst (bottom-right)", fontsize=10)
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

        # Arc-length weighted mean (matches the fitness function)
        if len(pts_i) >= 2:
            seg_lengths = np.linalg.norm(np.diff(pts_i, axis=0), axis=1)
            arc_weights = np.empty(len(pts_i), dtype=float)
            arc_weights[0]    = seg_lengths[0] / 2.0
            arc_weights[1:-1] = (seg_lengths[:-1] + seg_lengths[1:]) / 2.0
            arc_weights[-1]   = seg_lengths[-1] / 2.0
            total_length = arc_weights.sum()
            weighted_mean = float(np.dot(arc_weights, mismatch) / total_length) if total_length > 1e-12 else float(mismatch.mean())
        else:
            weighted_mean = float(mismatch.mean())

        ax.axhline(weighted_mean, color='red', lw=1, linestyle='--', label=f'mean={weighted_mean:.3f}')
        ax.set_ylim(0, 0.5)
        ax.set_xlabel('point index along seam')
        ax.set_ylabel('Δ phase')
        ax.set_title(c.name)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
