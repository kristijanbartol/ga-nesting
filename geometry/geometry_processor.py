import numpy as np
from typing import List

from spec import ProblemInstance, SeamPathType


class BatchBuilder:
    """
    Translates Mapped Vertex IDs -> The 'landmarks_batch' structure
    required by the heavy cut_paths() function.
    """
    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        self.vertices = instance.mesh_vertices

    def _get_intermediate_idx(self, idx1: int, idx2: int, z_val: float) -> int:
        """
        Finds a vertex index close to the midpoint of idx1-idx2 
        but shifted to a specific Z-plane.
        """
        p1 = self.vertices[idx1]
        p2 = self.vertices[idx2]
        
        # Calculate geometric midpoint
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        
        # Construct query point
        # Note: We use the provided z_val as an absolute coordinate 
        # (consistent with your usage of 0.01 / -0.1)
        query_p = np.array([mid_x, mid_y, z_val])
        
        # Find closest vertex
        dists = np.linalg.norm(self.vertices - query_p, axis=1)
        return int(np.argmin(dists))

    def build_batch(self, mapped_vertex_ids: np.ndarray) -> List[List[int]]:
        """
        Args:
            mapped_vertex_ids: (M,) array of vertex indices for active landmarks.
            
        Returns:
            List of lists, e.g., [[s, e], [s, mid, e], ...]
        """
        batch = []
        
        # Iterate over the active topology
        # We need the definition to check for DUAL type and Hints
        for i, (start_ptr, end_ptr) in enumerate(self.instance.active_seam_topology):
            
            # 1. Resolve actual mesh vertex indices
            v_start = mapped_vertex_ids[start_ptr]
            v_end = mapped_vertex_ids[end_ptr]
            
            # 2. Retrieve Seam Definition (for Type and Hints)
            # We map back using the name stored in the instance
            seam_name = self.instance.seam_names[i]
            # (Assuming you have access to the library or stored definitions in instance.
            #  For now, let's assume instance has a lookup or we passed the lib).
            #  *Modification*: Let's assume active_seam_types is available in instance
            #  as defined in previous turn.
            
            seam_type = self.instance.active_seam_types[i]
            
            # 3. Build List
            if seam_type == SeamPathType.GEODESIC:
                # Simple [Start, End]
                batch.append([v_start, v_end])
                
            elif seam_type == SeamPathType.DUAL:
                # Dual [Start, Mid, End] x2
                
                # Fetch hints (or use defaults)
                # In a real impl, retrieve specific hints from SeamDefinition
                z_front = 0.05   # Default Front (slightly positive)
                z_back = -0.15   # Default Back (behind body)
                
                # Calculate intermediates
                v_front = self._get_intermediate_idx(v_start, v_end, z_front)
                v_back  = self._get_intermediate_idx(v_start, v_end, z_back)
                
                # Append two paths (Front and Back)
                batch.append([v_start, v_front, v_end])
                batch.append([v_start, v_back, v_end])
                
        return batch
