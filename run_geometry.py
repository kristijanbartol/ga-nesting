# NOTE (!!!): landmarks that create smaller patches will naturally be more "optimal"
# therefore, we should use the initial design as a target and measure deviation from this design (outer boundary / intent)

import numpy as np
import trimesh

from spec import TextureSpec
from experiment_loader import load_experiment
from geometry.geometry_utils import (
    LandmarkMapper,
    generate_symmetric_landmarks,
    find_midline_vidx,
)
from geometry.topologies import build_shirt_topology, build_test_topology
from geometry.cut_utils import (
    perform_global_cut,
    assign_patch_labels
)
from geometry.export import export_data
from geometry.geometry_processor import BatchBuilder
from geometry.parameterization import parameterize
from geometry.landmarks import CORE_LANDMARKS, LONG_LANDMARKS, SHOULDER_KPT_IDX, HIP_KPT_IDX, ACTIVE_SEAMS

# ==============================================================================
# 1. SETUP EXPERIMENT
# ==============================================================================

mesh = trimesh.load('data/SMPL_FEMALE_POSED.ply')

# 2. Generate Full Library (L + R)
full_landmark_lib = generate_symmetric_landmarks(mesh, {**CORE_LANDMARKS["Upper"], **LONG_LANDMARKS["Upper"]})
seams = build_shirt_topology(full_landmark_lib)
#seams = build_test_topology(full_landmark_lib)

texture_spec = TextureSpec("Stripes", 50.0, 50.0)

instance = load_experiment(
    mesh_path='data/SMPL_FEMALE_POSED.ply',
    landmark_lib=full_landmark_lib,
    seam_lib=seams,
    active_seam_names=ACTIVE_SEAMS,
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
dummy_delta = np.array([0.5, 0.5] * instance.num_sampled_landmarks)

vertex_ids, _ = mapper.map_genotype_to_vertices(dummy_delta)

print(f"   Input Delta: {dummy_delta}")
print(f"   Mapped Vertex IDs: {vertex_ids}")

# Verify output size
assert len(vertex_ids) == instance.num_landmarks
print("\n[Success] Skeleton verification complete.")

builder = BatchBuilder(instance)
landmarks_batch = builder.build_batch(vertex_ids)

# step 3 - cutting
for garment_part in ['upper']:
    cut_mesh, patches, patch_faces, seamlines_dict_list, symmetric_seamline_flags, valid_patch_idxs, _seam_batch_indices = perform_global_cut(landmarks_batch, mesh.vertices, mesh.faces)
    if garment_part == 'lower':
        front_vidx = find_midline_vidx(mesh.vertices, mesh.faces, HIP_KPT_IDX, front=True)
        back_vidx  = find_midline_vidx(mesh.vertices, mesh.faces, HIP_KPT_IDX, front=False)
        ref_point = (mesh.vertices[front_vidx] + mesh.vertices[back_vidx]) / 2.0
    else:
        ref_point = mesh.vertices[SHOULDER_KPT_IDX]
    patch_labels_dict = assign_patch_labels(patches, garment_part, valid_patch_idxs, ref_point)
    
    export_data( 
        patches, 
        valid_patch_idxs,
        garment_part,
        seamlines_dict_list, 
        symmetric_seamline_flags, 
        patch_labels_dict, 
        cut_mesh
    )
    
# step 4 - parameterization
parameterize()
