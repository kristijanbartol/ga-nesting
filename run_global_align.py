# test_stage2_real.py
import os
import glob
import re
import numpy as np
import trimesh
import matplotlib.pyplot as plt

from nesting.phase_utils import TextureLattice, Rigid2D, seam_phase_mismatch_scalar
from nesting.stage2_global_align import load_seam_constraints_from_dir, solve_global_alignment_all_components


def parse_patch_id_from_dirname(patch_dirname: str) -> int:
    """
    Supports patch_02, patch_2, patch_002 etc.
    Returns integer patch id (matches seamfile ids).
    """
    m = re.search(r"patch_(\d+)", patch_dirname)
    if not m:
        raise ValueError(f"Cannot parse patch id from {patch_dirname}")
    return int(m.group(1))


def load_patch_vertices_from_latest(root_dir: str, garment_part: str = "upper", scale_mm: float = 1000.0):
    """
    Loads 2D vertices (all mesh vertices, not boundary!) for each patch id.

    Uses:
        {root_dir}/{garment_part}/patch_*/optim_final-seams.ply
    """
    pattern = os.path.join(root_dir, garment_part, "patch_*", "optim_final-seams.ply")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")

    V_by_id = {}
    boundary_by_id = {}

    for fpath in files:
        patch_dir = os.path.basename(os.path.dirname(fpath))  # patch_02
        pid = parse_patch_id_from_dirname(patch_dir)

        mesh = trimesh.load(fpath, process=False)
        V2 = mesh.vertices[:, :2] * scale_mm
        V_by_id[pid] = V2

        # also compute one boundary loop for visualization
        unique_edge_groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
        boundary_edges = mesh.edges[unique_edge_groups]
        # build adjacency for boundary
        # easiest: extract boundary vertices by walking edges (trimesh doesn't guarantee order),
        # so for vis we can approximate by convex hull if ordering fails.
        # BUT: for nice vis, use mesh.outline() if available.
        try:
            outline = mesh.outline()
            if outline is not None and len(outline.entities) > 0:
                # take first polyline entity
                # outline.vertices are in 3D; use indices from entity points
                ent = outline.entities[0]
                idx = np.array(ent.points, dtype=int)
                boundary_by_id[pid] = (outline.vertices[idx, :2] * scale_mm)
            else:
                boundary_by_id[pid] = V2
        except Exception:
            boundary_by_id[pid] = V2

    return V_by_id, boundary_by_id


def compute_total_mismatch(V_by_id, constraints, lattice, kappas, K, transforms):
    total = 0.0
    for c in constraints:
        Ti = transforms.get(c.patch_i, Rigid2D(0, 0, 0))
        Tj = transforms.get(c.patch_j, Rigid2D(0, 0, 0))
        ki = kappas.get(c.patch_i, 0)
        kj = kappas.get(c.patch_j, 0)

        total += seam_phase_mismatch_scalar(
            seam_pairs=c.pairs,
            patch_i_vertices_xy=V_by_id[c.patch_i],
            patch_j_vertices_xy=V_by_id[c.patch_j],
            lattice=lattice,
            kappa_i=ki,
            kappa_j=kj,
            K=K,
            weight=c.weight,
            transform_i=Ti,
            transform_j=Tj,
        )
    return float(total)


def draw_grid(ax, lattice: TextureLattice, xlim, ylim, step_u=1, step_v=1, alpha=0.15):
    """
    Draw a simple parallelogram lattice grid across plot extents.
    """
    # Build lattice vectors
    A = lattice.matrix()
    U = A[:, 0]
    V = A[:, 1]

    # Choose a bounding box in UV coordinates that covers xlim/ylim roughly
    # We do a crude cover by sampling UV range.
    corners = np.array([
        [xlim[0], ylim[0]],
        [xlim[0], ylim[1]],
        [xlim[1], ylim[0]],
        [xlim[1], ylim[1]],
    ])
    A_inv = np.linalg.inv(A)
    uv = corners @ A_inv.T
    umin, vmin = np.floor(uv.min(axis=0) - 2)
    umax, vmax = np.ceil(uv.max(axis=0) + 2)

    # Draw constant-u lines and constant-v lines
    for u in range(int(umin), int(umax) + 1, step_u):
        p0 = u * U + vmin * V
        p1 = u * U + vmax * V
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=1, alpha=alpha)

    for v in range(int(vmin), int(vmax) + 1, step_v):
        p0 = umin * U + v * V
        p1 = umax * U + v * V
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=1, alpha=alpha)


def apply_T(points, T: Rigid2D):
    return T.apply(points)


def main():
    # -------------------------
    # CONFIG: paths + texture
    # -------------------------
    latest_root = "results/pattern/latest"
    garment_part = "upper"
    seam_dir = f"data/seamlines/{garment_part}"

    # Texture lattice: set to your fabric repeat (in same units as vertices!)
    # NOTE: loader multiplies vertices by 1000 -> mm, so periods should be in mm.
    lattice = TextureLattice(
        u_dir=np.array([1.0, 0.0]),
        v_dir=np.array([0.0, 1.0]),
        period_u=50.0,   # <-- set to your texture period in X (mm)
        period_v=50.0,   # <-- set to your texture period in Y (mm)
    )

    K = 8  # number of phase bins per period (κ/K)
    # For now, set kappas=0 for all patches (or set per patch for experiments)
    kappas = {}

    # Weights: minimal way -> all seams weight 1.0
    # If you want per file:
    # weights = {"seam-2_1-2.txt": 1.0, ...}
    weights = {}

    # -------------------------
    # Load real patches + seams
    # -------------------------
    V_by_id, boundary_by_id = load_patch_vertices_from_latest(latest_root, garment_part=garment_part, scale_mm=1000.0)
    patch_ids = sorted(V_by_id.keys())
    print(f"[RealData] Loaded patches: {patch_ids}")

    constraints = load_seam_constraints_from_dir(seam_dir, weights_by_filename=weights, default_weight=1.0)
    print(f"[RealData] Loaded seams: {len(constraints)}")
    for c in constraints[:10]:
        print(f"  - {c.name}: {c.patch_i}<->{c.patch_j}, pairs={len(c.pairs)}, w={c.weight}")

    # initial transforms: identity
    T0 = {pid: Rigid2D(0.0, 0.0, 0.0) for pid in patch_ids}

    # -------------------------
    # Compute mismatch before/after
    # -------------------------
    before = compute_total_mismatch(V_by_id, constraints, lattice, kappas, K, T0)
    print(f"[Stage2] Total mismatch BEFORE: {before:.6f}")

    Tsol = solve_global_alignment_all_components(
        patch_ids=patch_ids,
        constraints=constraints,
        patch_vertices_by_id=V_by_id,
        lattice=lattice,
        kappas_by_id=kappas,
        K=K,
        initial_transforms=T0,
        max_iters=25,
        verbose=True,  # set False if too chatty
    )

    after = compute_total_mismatch(V_by_id, constraints, lattice, kappas, K, Tsol)
    print(f"[Stage2] Total mismatch AFTER : {after:.6f}")

    # -------------------------
    # Visualization: before vs after
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    titles = ["BEFORE (identity transforms)", "AFTER (global Stage2 solve)"]
    Ts_list = [T0, Tsol]

    for ax, title, Ts in zip(axes, titles, Ts_list):
        # plot patch boundaries
        for pid in patch_ids:
            B = boundary_by_id[pid]
            Bt = apply_T(B, Ts.get(pid, Rigid2D(0, 0, 0)))
            ax.plot(Bt[:, 0], Bt[:, 1], linewidth=2)
            # label near first point
            ax.text(Bt[0, 0], Bt[0, 1], f"{pid}", fontsize=9)

        # plot a subset of seam correspondence segments
        # (drawing all pairs can be too heavy)
        max_pairs_to_draw = 50
        for c in constraints:
            if c.weight <= 0.0:
                continue
            Ti = Ts.get(c.patch_i, Rigid2D(0, 0, 0))
            Tj = Ts.get(c.patch_j, Rigid2D(0, 0, 0))

            Vi = V_by_id[c.patch_i]
            Vj = V_by_id[c.patch_j]
            pairs = c.pairs
            if len(pairs) > max_pairs_to_draw:
                # uniform subsample
                idx = np.linspace(0, len(pairs)-1, max_pairs_to_draw).astype(int)
                pairs = [pairs[k] for k in idx]

            for (ai, bj) in pairs:
                pi = Ti.apply(Vi[ai:ai+1])[0]
                pj = Tj.apply(Vj[bj:bj+1])[0]
                ax.plot([pi[0], pj[0]], [pi[1], pj[1]], alpha=0.10, linewidth=1)

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

    # draw lattice grid on both axes for reference
    for ax in axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        draw_grid(ax, lattice, xlim, ylim, step_u=1, step_v=1, alpha=0.15)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
