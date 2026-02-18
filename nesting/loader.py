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
        
        items = []
        for i, fpath in enumerate(files):
            # Extract simple name (e.g., "patch_0")
            patch_name = os.path.basename(os.path.dirname(fpath))
            
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
                original_vertices=centered_verts
            )
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
