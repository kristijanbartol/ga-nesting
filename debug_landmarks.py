"""
debug_landmarks.py

Resolves every pant landmark at the baseline delta (0.5, 0.5) using the same
bilinear-interpolation + KNN-snap as the real pipeline, then writes two files:

  landmarks.ply        – point cloud of all resolved vertices (one per landmark)
  landmarks_corners.ply – point cloud of all boundary-corner vertices

Open both alongside the SMPL mesh in MeshLab to verify positions.
"""
import numpy as np
import trimesh

from geometry.geometry_utils import generate_symmetric_landmarks
from geometry.landmarks import CORE_LANDMARKS, LONG_LANDMARKS
from scipy.spatial import cKDTree


MESH_PATH = "data/SMPL_FEMALE_POSED.ply"
OUT_RESOLVED  = "landmarks.ply"
OUT_CORNERS   = "landmarks_corners.ply"


def resolve_landmark(lm, vertices, tree):
    """Bilinear interpolation at (u,v)=(0.5,0.5), snapped to nearest vertex."""
    c_ids = lm.boundary_corners
    corners = vertices[list(c_ids)]          # (4, 3)
    p00, p10, p11, p01 = corners
    p_bottom = 0.5 * p00 + 0.5 * p10
    p_top    = 0.5 * p01 + 0.5 * p11
    p_target = 0.5 * p_bottom + 0.5 * p_top
    _, idx = tree.query(p_target)
    return idx, vertices[idx]


def main():
    mesh = trimesh.load(MESH_PATH, process=False)
    V = mesh.vertices
    tree = cKDTree(V)

    source = {**CORE_LANDMARKS["Lower"], **LONG_LANDMARKS["Lower"]}
    lib = generate_symmetric_landmarks(mesh, source)

    print(f"{'Landmark':<22}  {'vtx_idx':>8}   position (x, y, z)")
    print("-" * 65)

    resolved_pts = []
    resolved_labels = []
    corner_pts = []

    for name, lm in sorted(lib.items()):
        idx, pos = resolve_landmark(lm, V, tree)
        resolved_pts.append(pos)
        resolved_labels.append(name)
        print(f"  {name:<22}  {idx:>8}   ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")

        # also collect boundary corners for second cloud
        for cidx in lm.boundary_corners:
            corner_pts.append(V[cidx])

    # --- save resolved-centroid cloud ---
    resolved_arr = np.array(resolved_pts)
    pc = trimesh.points.PointCloud(resolved_arr)
    pc.export(OUT_RESOLVED)
    print(f"\nSaved {len(resolved_pts)} resolved landmark points -> {OUT_RESOLVED}")

    # --- save corner cloud ---
    corner_arr = np.unique(np.array(corner_pts), axis=0)
    pc2 = trimesh.points.PointCloud(corner_arr)
    pc2.export(OUT_CORNERS)
    print(f"Saved {len(corner_arr)} unique corner vertices  -> {OUT_CORNERS}")


if __name__ == "__main__":
    main()
