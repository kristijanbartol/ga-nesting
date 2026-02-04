# NOTE (!!!): landmarks that create smaller patches will naturally be more "optimal"
# therefore, we should use the initial design as a target and measure deviation from this design (outer boundary / intent)

import numpy as np
import trimesh

from spec import LandmarkDefinition, TextureSpec
from experiment_loader import load_experiment
from geometry_utils import LandmarkMapper, generate_symmetric_landmarks
from topologies import build_shirt_topology

# ==============================================================================
# 0. MOCK DATA PREPARATION (User Input)
# ==============================================================================

# Pick these indices manually on your SMPL mesh.
LANDMARKS = {
    "Neck": LandmarkDefinition(
        name="Neck", 
        boundary_corners=(4301, 5279, 4199, 4762),
    ),
    "Shoulder": LandmarkDefinition(
        name="Shoulder",
        boundary_corners=(5274, 6446, 4122, 4723)
    ),
    "Armpit": LandmarkDefinition(
        name="Armpit",
        boundary_corners=(4755, 4751, 5230, 4163)
    ),
    "Waist": LandmarkDefinition(
        name="Waist",
        boundary_corners=(6524, 6557, 4984, 4921)
    )
}

# ==============================================================================
# 1. SETUP EXPERIMENT
# ==============================================================================

mesh = trimesh.load('data/SMPL_FEMALE.ply')

# 2. Generate Full Library (L + R)
full_landmark_lib = generate_symmetric_landmarks(mesh, LANDMARKS)
shirt_seams = build_shirt_topology(full_landmark_lib)

active_seams = ["Side_Seam", "Neck_Line"]
texture_spec = TextureSpec("Stripes", 10.0, 100.0)

instance = load_experiment(
    mesh_path='data/SMPL_FEMALE.ply',
    landmark_lib=full_landmark_lib,
    seam_lib=shirt_seams,
    active_seam_names=active_seams,
    texture=texture_spec,
    fabric_width=150.0
)

print("\n[Check] Instance Created Successfully")
print(f"   Num Landmarks: {instance.num_landmarks}")
print(f"   Num Seams: {instance.num_seams}")
print(f"   Active Seam Names: {instance.seam_names}")

# ==============================================================================
# 2. TEST MAPPING (Genotype -> Vertex IDs)
# ==============================================================================

print("\n[Check] Testing Geometry Mapper...")
mapper = LandmarkMapper(instance)

# Create a dummy genotype geometry vector (delta)
# 3 landmarks * 2 coords (u,v) = 6 floats
# Let's try [0.5, 0.5] for all (should act like center of the quad)
dummy_delta = np.array([0.5, 0.5] * instance.num_landmarks)

vertex_ids = mapper.map_genotype_to_vertices(dummy_delta)

print(f"   Input Delta: {dummy_delta}")
print(f"   Mapped Vertex IDs: {vertex_ids}")

# Verify output size
assert len(vertex_ids) == instance.num_landmarks
print("\n[Success] Skeleton verification complete.")
