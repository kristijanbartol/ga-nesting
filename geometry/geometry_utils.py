import sys

sys.path.append('/home/kristijan/LOOM/potpourri3d/src')

from typing import Dict
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from collections import defaultdict, Counter, deque
from itertools import combinations
import potpourri3d as pp3d
import polyscope as ps

from spec import ProblemInstance, LandmarkDefinition
import geometry.patch as patch


class LandmarkMapper:
    """
    Maps Genotype Delta (u,v) -> Vertex ID using simple 3D projection.
    """
    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        self.vertices = instance.mesh_vertices
        
        # Build ONE global tree for the whole mesh.
        # Since we assume the 4 corners are close to each other, 
        # the interpolated point won't jump across the body 
        # unless the mesh is extremely thin/concave at that spot.
        print("[Geometry] Building global mesh KDTree...")
        self.global_tree = cKDTree(self.vertices)

    def map_genotype_to_vertices(self, delta: np.ndarray) -> np.ndarray:
        """
        Args:
            delta: (2 * M) array of [u, v] coordinates.
        Returns:
            (M,) array of Global Vertex IDs.
        """
        resolved_indices = []
        
        for i, lm in enumerate(self.instance.active_landmarks):
            # 1. Extract u, v
            u = delta[2*i]
            v = delta[2*i+1]
            
            # 2. Get Corner Coordinates
            c_ids = lm.boundary_corners
            corners = self.vertices[list(c_ids)] # Shape (4, 3)
            p00, p10, p11, p01 = corners[0], corners[1], corners[2], corners[3]
            
            # 3. Bilinear Interpolation in 3D
            # This effectively creates a "virtual quad" inside the mesh volume
            p_bottom = (1 - u) * p00 + u * p10
            p_top    = (1 - u) * p01 + u * p11
            p_target = (1 - v) * p_bottom + v * p_top
            
            # 4. Snap to nearest vertex globally
            # Since p_target is weighted by corners, it is guaranteed 
            # to be spatially close to the surface patch.
            _, global_id = self.global_tree.query(p_target)
            
            resolved_indices.append(global_id)
            
        return np.array(resolved_indices, dtype=np.int32)


def generate_symmetric_landmarks(
    mesh: trimesh.Trimesh, 
    source_landmarks: Dict[str, LandmarkDefinition]
) -> Dict[str, LandmarkDefinition]:
    """
    Takes a dict of landmarks (assumed Left side) and generates
    the Right side counterparts by mirroring across the X-axis.
    """
    full_library = {}
    vertices = mesh.vertices
    
    # Build search tree for global nearest neighbor lookup
    print("[Symmetry] Building mesh KDTree...")
    tree = cKDTree(vertices)
    
    for name, lm in source_landmarks.items():
        # 1. Add Original (assume it is Left)
        l_name = f"{name}_L"
        full_library[l_name] = LandmarkDefinition(
            name=l_name,
            boundary_corners=lm.boundary_corners
        )
        
        # 2. Compute Symmetric (Right)
        # Flip X coordinate for all 4 corners
        source_coords = vertices[list(lm.boundary_corners)]
        target_coords = source_coords.copy()
        target_coords[:, 0] *= -1 # Flip X
        
        # Find indices of these flipped coordinates
        # k=1 returns (distances, indices)
        _, symmetric_indices = tree.query(target_coords, k=1)
        
        r_name = f"{name}_R"
        full_library[r_name] = LandmarkDefinition(
            name=r_name,
            boundary_corners=tuple(symmetric_indices)
        )
        print(f"   Generated {r_name} from {l_name}")

    return full_library


def _get_closest_idx(verts, query_p):
    dists = np.linalg.norm(verts - query_p, axis=1)
    closest_vertex_index = np.argmin(dists)
    return closest_vertex_index


def extract_side_idx(mesh, idx1, idx2, z_offset: float):
    mid_x = (mesh.vertices[idx1][0] + mesh.vertices[idx2][0]) / 2.
    ref_y = mesh.vertices[idx1][1]
    query_p = np.array([mid_x, ref_y, z_offset])
    return _get_closest_idx(mesh.vertices, query_p)


def find_midline_vidx(verts, faces, v_idx, front=True):
    y = verts[v_idx, 1]
    sign = 1.0 if front else -1.0
    # unique undirected edges
    edges = np.unique(
        np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
        axis=0
    )

    best = (np.inf, None, None, None, 0)  # (score, point, i0, i1)
    for edge_idx, (i0, i1) in enumerate(edges):
        v0, v1 = verts[i0], verts[i1]
        if v0[2] * sign <= 0 or v1[2] * sign <= 0:
            continue
        t = (y - v0[1]) / (v1[1] - v0[1])
        if 0 <= t <= 1:
            # minimize how far BOTH edge endpoints are from X=0
            score = max(abs(v0[0]), abs(v1[0]))
            if score < best[0]:
                best = (score, v0 + t * (v1 - v0), i0, i1, edge_idx)

    _, p, i0, i1, best_idx = best

    if abs(verts[i0][0]) < abs(verts[i1][0]):
        return i0
    else:
        return i1

