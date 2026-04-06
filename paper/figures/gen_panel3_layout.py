"""
Paper figure – Panel 3: annotated fabric-roll layout.

Renders a publication-quality nesting layout with:
  - Texture lines clipped to each patch (matching vis_utils style)
  - Grain-direction arrow per patch  (double-headed, encodes rho)
  - Kappa label per patch            (small subscript near centroid)
  - Optional seam-edge mismatch colouring (green → red gradient)

Usage (standalone)::

    python paper/figures/gen_panel3_layout.py

Usage (import)::

    from paper.figures.gen_panel3_layout import render_panel3
    render_panel3(fabric_state, texture_spec, rho_by_name, kappa_by_name,
                  output_path="paper/assets/panel3_layout.pdf")
"""

from __future__ import annotations

import math
import sys
import os
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Make the repo root importable when running this script directly.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nesting.vis_utils import (
    PATCH_COLORS,
    _polygon_to_path,
    _draw_texture_in_patch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grain_arrow_direction(rho: int) -> tuple[float, float]:
    """
    Return a unit vector for the warp (grain) direction given rho in {0,1,2,3}.
    rho=0 → vertical (0,1), rho=1 → horizontal (1,0), etc.
    """
    angle_rad = math.radians(rho * 90.0)
    return (math.sin(angle_rad), math.cos(angle_rad))


def _draw_grain_arrow(ax, cx: float, cy: float, rho: int, length: float,
                      color: str = "black", lw: float = 1.4, alpha: float = 0.9):
    """Draw a double-headed grain arrow centred at (cx, cy)."""
    dx, dy = _grain_arrow_direction(rho)
    half = length / 2.0
    x0, y0 = cx - dx * half, cy - dy * half
    x1, y1 = cx + dx * half, cy + dy * half

    arrowprops = dict(
        arrowstyle="<->",
        color=color,
        lw=lw,
        mutation_scale=10,
    )
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=arrowprops,
        zorder=20,
        alpha=alpha,
    )


def _mismatch_color(score: float, vmin: float = 0.0, vmax: float = 0.5) -> tuple:
    """Map a mismatch score in [vmin, vmax] to a green→red colour."""
    t = max(0.0, min(1.0, (score - vmin) / max(vmax - vmin, 1e-9)))
    r = t
    g = 1.0 - t
    b = 0.0
    return (r, g, b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_panel3(
    fabric_state,
    texture_spec,
    rho_by_name: dict[str, int],
    kappa_by_name: dict[str, int],
    seam_pairs: Optional[list] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (5, 9),
    dpi: int = 200,
    show_kappa: bool = True,
    show_grain: bool = True,
    fabric_color: str = "#F7F3EE",
    border_color: str = "#333333",
):
    """
    Render Panel 3 of the pipeline figure.

    Parameters
    ----------
    fabric_state : FabricState
        Output of NestingEngine.nest().
    texture_spec : TextureSpec
        Texture specification (period, wallpaper group, etc.).
    rho_by_name : dict
        Maps item.name → rho integer (0..3).  Used to draw grain arrows.
    kappa_by_name : dict
        Maps item.name → kappa integer.  Shown as a small label on each patch.
    seam_pairs : list of (name_i, name_j, mismatch_score), optional
        Shared seam edges to highlight.  mismatch_score in [0, 0.5].
        If None, no seam colouring is drawn.
    output_path : str, optional
        Save to this path (PDF or PNG recommended).  If None, calls plt.show().
    figsize : (width, height) in inches.
    dpi : int
    show_kappa : bool
        Whether to annotate kappa values on each patch.
    show_grain : bool
        Whether to draw grain arrows.
    fabric_color : str
        Background fill for the fabric roll.
    border_color : str
        Edge colour for fabric roll boundary and patch outlines.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    width = fabric_state.width
    total_h = fabric_state.total_height

    # Fabric roll background
    rect = mpatches.FancyBboxPatch(
        (0, 0), width, total_h,
        boxstyle="square,pad=0",
        linewidth=1.5,
        edgecolor=border_color,
        facecolor=fabric_color,
        zorder=1,
    )
    ax.add_patch(rect)

    # Build a lookup from item name to placed poly for seam drawing
    poly_by_name: dict[str, object] = {}

    for idx, (item, cx, cy, poly) in enumerate(fabric_state.placed_items):
        color = PATCH_COLORS[idx % len(PATCH_COLORS)]

        # --- Patch fill and outline ---
        x, y = poly.exterior.xy
        ax.fill(x, y, color=color, alpha=0.22, zorder=2)
        ax.plot(x, y, color=color, lw=1.4, zorder=3)

        # --- Texture lines clipped to patch ---
        _draw_texture_in_patch(ax, poly, texture_spec, color, lw=1.1, alpha=0.80)

        poly_by_name[item.name] = poly

        # --- Grain arrow ---
        if show_grain:
            rho = rho_by_name.get(item.name, 0)
            minx, miny, maxx, maxy = poly.bounds
            diag = math.hypot(maxx - minx, maxy - miny)
            arrow_len = diag * 0.35
            centroid = poly.centroid
            _draw_grain_arrow(ax, centroid.x, centroid.y, rho,
                               length=arrow_len, color=border_color)

    # --- Seam mismatch: coloured lines between centroids of seam-adjacent patches ---
    # Patches on the roll are not touching, so we draw a centroid-to-centroid line
    # coloured by mismatch score (green = well-aligned, red = misaligned).
    if seam_pairs:
        centroid_by_name = {
            item.name: poly.centroid
            for item, _cx, _cy, poly in fabric_state.placed_items
        }
        for name_i, name_j, score in seam_pairs:
            ci = centroid_by_name.get(name_i)
            cj = centroid_by_name.get(name_j)
            if ci is None or cj is None:
                continue
            seam_color = _mismatch_color(score)
            ax.plot(
                [ci.x, cj.x], [ci.y, cj.y],
                color=seam_color, lw=2.5, linestyle="--",
                solid_capstyle="round", zorder=25, alpha=0.9,
            )
            # Small dot at each endpoint so the line origin is clear
            ax.scatter([ci.x, cj.x], [ci.y, cj.y],
                       color=seam_color, s=18, zorder=26, alpha=0.9)

    # Padding
    pad = width * 0.04
    ax.set_xlim(-pad, width + pad)
    ax.set_ylim(-pad, total_h + pad)

    plt.tight_layout(pad=0.1)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        print(f"Saved Panel 3 → {output_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Standalone entry point  (python paper/figures/gen_panel3_layout.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import re
    import json
    from ga_spec import Genome, Individual
    from ga.real_evaluator import RealEvaluator, RealEvaluatorConfig

    # ------------------------------------------------------------------ config
    GARMENT      = "lower"
    WALLPAPER    = "stripes"
    BEST_JSON    = "results/pattern/best/best_individual.json"
    OUTPUT_PATH  = "paper/assets/panel3_layout.pdf"

    PERIOD_MM    = 50.0
    FABRIC_WIDTH = 150.0 * 10.0   # 1500 mm, same as run_ga.py

    # --------------------------------------------------------- build evaluator
    eval_cfg = RealEvaluatorConfig(
        garment_part=GARMENT,
        latest_root="results/pattern/latest",
        seam_dir=f"data/seamlines/{GARMENT}",
        period_u_mm=PERIOD_MM,
        period_v_mm=PERIOD_MM,
        K=8,
        fabric_width_mm=FABRIC_WIDTH,
        num_bodies=1,
        wallpaper_group=WALLPAPER,
        w1=1.0,
        w2=1.0,
    )
    evaluator = RealEvaluator(eval_cfg)

    num_patches   = len(evaluator.patch_ids)
    num_landmarks = evaluator.instance.num_sampled_landmarks
    # Deterministic patch_id -> flat genome index mapping (matches real_evaluator.py)
    pid_to_item_idx = {pid: idx for idx, pid in enumerate(evaluator.patch_ids)}

    # ------------------------------------------------------- genome: load / build
    if os.path.exists(BEST_JSON):
        print(f"[panel3] Loading genome from {BEST_JSON}")
        with open(BEST_JSON) as f:
            saved = json.load(f)
        genome = Genome(
            delta=np.array(saved["delta"]),
            rho=np.array(saved["rho"],   dtype=int),
            kappa=np.array(saved["kappa"], dtype=int),
            pi=np.array(saved["pi"],    dtype=int),
            h=int(saved["h"]),
            sigma=np.array(saved.get("sigma", [0.15] * len(saved["delta"]))),
        )
    else:
        print("[panel3] No saved genome found – using baseline (delta=0.5, kappa=0, rho=0)")
        genome = Genome(
            delta=np.full(2 * num_landmarks, 0.5),
            rho=np.zeros(num_patches, dtype=int),
            kappa=np.zeros(num_patches, dtype=int),
            pi=np.arange(num_patches, dtype=int),
            h=0,
            sigma=np.full(2 * num_landmarks, 0.15),
        )

    # ----------------------------------------------------------------- evaluate
    ind = Individual(genome=genome)
    evaluator(ind)
    fabric_state = ind.meta["fabric_state"]

    # ---------------------------------------- build annotation maps from genome
    # After multi-body cloning in real_evaluator, item names are "body_0/patch_N".
    # We parse the patch_id, look up its genome index, and read rho / kappa.
    def _pid_from_name(name: str) -> int:
        m = re.search(r"patch_(\d+)", name)
        return int(m.group(1)) if m else -1

    rho_by_name   = {}
    kappa_by_name = {}
    for item, _cx, _cy, _poly in fabric_state.placed_items:
        pid = _pid_from_name(item.name)
        idx = pid_to_item_idx.get(pid)
        if idx is not None:
            if idx < genome.rho.size:
                rho_by_name[item.name]   = int(genome.rho[idx])
            if idx < genome.kappa.size:
                kappa_by_name[item.name] = int(genome.kappa[idx])

    # --------------------------------- compute per-seam mismatch scores for render
    from nesting.phase_utils import seam_phase_mismatch

    seam_pairs_render = []
    for c in ind.meta["weighted_constraints"]:
        Ti = ind.meta["Tsol"].get(c.patch_i)
        Tj = ind.meta["Tsol"].get(c.patch_j)
        ki = ind.meta["kappas_by_id"].get(c.patch_i, 0)
        kj = ind.meta["kappas_by_id"].get(c.patch_j, 0)
        Vi = ind.meta["V_centered_by_id"].get(c.patch_i)
        Vj = ind.meta["V_centered_by_id"].get(c.patch_j)
        if Vi is None or Vj is None or not c.pairs:
            continue
        score = seam_phase_mismatch(
            seam_pairs=c.pairs,
            patch_i_vertices_xy=Vi,
            patch_j_vertices_xy=Vj,
            lattice=evaluator.lattice,
            kappa_i=ki,
            kappa_j=kj,
            K=eval_cfg.K,
            weight=1.0,          # raw mismatch, ignore seam importance for colour
            transform_i=Ti,
            transform_j=Tj,
            glide_transforms=evaluator.policy.glide_transforms(),
        )
        name_i = f"body_0/patch_{c.patch_i}"
        name_j = f"body_0/patch_{c.patch_j}"
        seam_pairs_render.append((name_i, name_j, score))
        print(f"  seam {c.patch_i}-{c.patch_j}: mismatch={score:.4f}")

    # --------------------------------------------------------------- render
    render_panel3(
        fabric_state,
        evaluator.instance.texture,
        rho_by_name=rho_by_name,
        kappa_by_name=kappa_by_name,
        seam_pairs=seam_pairs_render,
        show_kappa=False,
        show_grain=False,
        output_path=OUTPUT_PATH,
    )
