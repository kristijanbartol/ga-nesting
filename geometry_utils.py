from typing import Dict
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from spec import ProblemInstance, LandmarkDefinition


class LandmarkMapper:
    """
    Concrete implementation of the geometry mapping logic.
    """
    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        self.vertices = instance.mesh_vertices
        
        # Cache KDTrees for active landmarks
        self.region_trees = []
        self.region_indices_map = []
        
        print(f"[Geometry] Pre-building KDTrees for {len(instance.active_landmarks)} active landmarks...")
        
        for lm in instance.active_landmarks:
            # 1. Get the subset of vertices belonging to this ROI
            # In a real run, these indices come from the pre-computed flood fill.
            # For the skeleton, we assume lm.roi_vertex_indices is populated.
            roi_global_indices = list(lm.roi_vertex_indices)
            roi_coords = self.vertices[roi_global_indices]
            
            # 2. Build Tree
            tree = cKDTree(roi_coords)
            self.region_trees.append(tree)
            self.region_indices_map.append(np.array(roi_global_indices))

    def map_genotype_to_vertices(self, delta: np.ndarray) -> np.ndarray:
        """
        Args:
            delta: (2 * M) array of [u, v] coordinates.
        Returns:
            (M,) array of Global Vertex IDs.
        """
        num_landmarks = self.instance.num_landmarks
        if len(delta) != 2 * num_landmarks:
            raise ValueError(f"Delta size mismatch. Expected {2*num_landmarks}, got {len(delta)}")
            
        resolved_indices = []
        
        for i, lm in enumerate(self.instance.active_landmarks):
            # Extract u, v
            u, v = delta[2*i], delta[2*i+1]
            
            # Get Corner Coordinates
            c_indices = lm.boundary_corners
            corners = self.vertices[list(c_indices)] # (4, 3)
            p00, p10, p11, p01 = corners[0], corners[1], corners[2], corners[3]
            
            # Bilinear Interpolation
            # Bottom edge (x axis at y=0)
            p_bottom = (1 - u) * p00 + u * p10
            # Top edge (x axis at y=1)
            p_top    = (1 - u) * p01 + u * p11
            # Final point
            p_target = (1 - v) * p_bottom + v * p_top
            
            # Snap to nearest valid vertex in ROI
            _, relative_idx = self.region_trees[i].query(p_target)
            global_id = self.region_indices_map[i][relative_idx]
            
            resolved_indices.append(global_id)
            
        return np.array(resolved_indices, dtype=int)


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
            boundary_corners=lm.boundary_corners,
            reference_vertex_id=lm.boundary_corners[0], # Placeholder
            roi_vertex_indices=() # To be filled by flood fill later
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
            boundary_corners=tuple(symmetric_indices),
            reference_vertex_id=symmetric_indices[0],
            roi_vertex_indices=() 
        )
        print(f"   Generated {r_name} from {l_name}")

    return full_library
