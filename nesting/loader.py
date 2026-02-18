import os
import glob
import numpy as np
import trimesh
import matplotlib.pyplot as plt

from .item import NestingItem
from .utils import boundary_loops_from_edges, polygon_area_2d


class PatchLoader:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        
    def _load_seam_anchor_indices(self) -> dict[int, int]:
        """Load a simple seam-based anchor mapping: patch_idx -> vertex_idx.

        Minimal implementation for the current experiment:
        - reads the exported correspondence file seam-2_1-2.txt
        - uses the FIRST vertex-pair as the "upper start" anchor
        - returns a dict mapping {1: v_idx_in_patch1, 2: v_idx_in_patch2}

        If the file is missing, returns an empty dict and nesting falls back
        to the default behavior.
        """
        fpath = os.path.join('data', 'seamlines', 'upper', 'seam-3_1-2.txt')
        if not os.path.exists(fpath):
            return {}

        with open(fpath, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) < 4:
            return {}

        # lines[0] = symmetric flag (ignored)
        p_a = int(lines[1])
        p_b = int(lines[2])
        v_a, v_b = map(int, lines[3].split())
        return {p_a: v_a, p_b: v_b}

    def load_items(self) -> list[NestingItem]:
        """
        Scans directory for 'optim_final-seams.ply', extracts boundaries,
        and returns a list of NestingItems.
        """
        # 1. Find all patch files
        # Pattern: results/pattern/latest/upper/patch_*/optim_final-seams.ply
        search_path = os.path.join(self.root_dir, "upper", "patch_*", "optim_final-seams.ply")
        files = glob.glob(search_path)
        
        if not files:
            raise FileNotFoundError(f"No patch files found in {search_path}")
            
        print(f"[Loader] Found {len(files)} patch files.")
        
        seam_anchor_vidx = self._load_seam_anchor_indices()
        items = []
        for i, fpath in enumerate(files):
            # Extract simple name (e.g., "patch_0")
            patch_name = os.path.basename(os.path.dirname(fpath))
            
            try:
                patch_idx = int(patch_name.split('_')[-1])
            except Exception:
                patch_idx = None
            
            # 2. Load Mesh
            mesh = trimesh.load(fpath, process=False)
            
            # 3. Flatten (Drop Z if it exists)
            vertices_2d = mesh.vertices[:, :2] * 1000.
            
            # 4. Extract Boundary Loop
            unique_edge_groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
            boundary_edges = mesh.edges[unique_edge_groups]

            loops = boundary_loops_from_edges(boundary_edges)
            if not loops:
                raise ValueError(f"No boundary loop found in {fpath}")

            # If there are multiple loops (holes etc), pick the outer one by max |area|
            areas = []
            for loop in loops:
                pts = vertices_2d[loop]
                areas.append(abs(polygon_area_2d(pts)))
            outer_loop = loops[int(np.argmax(areas))]

            boundary_verts = vertices_2d[outer_loop]
            
            # 5. Create NestingItem
            # We center it at (0,0) to normalize inputs for the packer
            centroid = np.mean(boundary_verts, axis=0)
            centered_verts = boundary_verts - centroid
            
            item = NestingItem(
                item_id=i,
                name=patch_name,
                original_vertices=centered_verts,
                patch_idx=patch_idx
            )
            
            # Optional: seam-based alignment anchor (in LOCAL coordinates)
            if patch_idx is not None and patch_idx in seam_anchor_vidx:
                v_idx = int(seam_anchor_vidx[patch_idx])
                if 0 <= v_idx < vertices_2d.shape[0]:
                    # Convert to the same local frame as the polygon: subtract the boundary centroid
                    anchor_local = vertices_2d[v_idx] - centroid
                    item.seam_anchor_local = (float(anchor_local[0]), float(anchor_local[1]))
            
            items.append(item)
            print(f"   Loaded {patch_name}: {len(boundary_verts)} vertices, Area={item.area:.2f}")
            
        return items

    def visualize_items(self, items: list[NestingItem]):
        """Debug plot to check if geometries look correct."""
        fig, axes = plt.subplots(1, len(items), figsize=(15, 5))
        if len(items) == 1: axes = [axes]
        
        for ax, item in zip(axes, items):
            # Extract x,y from the polygon exterior
            x, y = item.shape.exterior.xy
            ax.plot(x, y, color='blue', linewidth=2)
            ax.fill(x, y, color='blue', alpha=0.1)
            ax.set_title(item.name)
            ax.axis('equal')
            
        plt.tight_layout()
        plt.show()
