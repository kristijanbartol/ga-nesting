from typing import Dict, List
import trimesh
import numpy as np
from spec import (
    ProblemInstance,
    TextureSpec,
    LandmarkDefinition,
    SeamDefinition,
    SeamPathType,
)


def load_experiment(
    mesh_path: str,
    landmark_lib: Dict[str, LandmarkDefinition],
    seam_lib: Dict[str, SeamDefinition],
    active_seam_names: List[str],
    texture: TextureSpec,
    fabric_width: float
) -> ProblemInstance:
    
    print(f"[Loader] Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path, process=False)
    
    # 1. Identify used landmarks
    used_landmark_names = set()
    for s_name in active_seam_names:
        if s_name not in seam_lib:
            raise ValueError(f"Seam '{s_name}' not found in library.")
        seam = seam_lib[s_name]
        used_landmark_names.add(seam.start_landmark)
        used_landmark_names.add(seam.end_landmark)
        
    # 2. Sort for determinism
    sorted_lm_names = sorted(list(used_landmark_names))
    
    # 3. Build Active Objects
    active_landmarks = tuple(landmark_lib[name] for name in sorted_lm_names)
    name_to_idx = {name: i for i, name in enumerate(sorted_lm_names)}
    
    # 4. Build Topology, Definitions, AND Types
    topology = []
    ordered_seam_names = sorted(active_seam_names)
    active_definitions = []
    active_types = []

    for s_name in ordered_seam_names:
        seam = seam_lib[s_name]
        start_i = name_to_idx[seam.start_landmark]
        end_i   = name_to_idx[seam.end_landmark]

        topology.append((start_i, end_i))
        active_definitions.append(seam)
        active_types.append(seam.path_type)

    # Collect all per-seam importances (including DUAL = 0.0).
    active_importances = [seam.importance for seam in active_definitions]

    # 5. Create Instance
    instance = ProblemInstance(
        mesh_path=mesh_path,
        fabric_width=fabric_width,
        texture=texture,
        mesh_vertices=np.array(mesh.vertices),
        mesh_faces=np.array(mesh.faces),
        active_landmarks=active_landmarks,
        active_seam_topology=tuple(topology),
        active_seam_definitions=tuple(active_definitions),
        active_seam_types=tuple(active_types),
        seam_names=tuple(ordered_seam_names),
        seam_importances=tuple(active_importances),
    )
    
    return instance
