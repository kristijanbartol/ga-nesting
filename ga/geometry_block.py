# ga_geometry_block.py
import numpy as np
import trimesh

from spec import LandmarkDefinition, TextureSpec
from experiment_loader import load_experiment
from geometry.geometry_utils import LandmarkMapper, generate_symmetric_landmarks
from topologies import build_shirt_topology
from geometry.cut_utils import perform_global_cut, assign_patch_labels
from geometry.export import export_data
from geometry.geometry_processor import BatchBuilder
from geometry.parameterization import parameterize


# === Copied from test_geometry.py (kept identical) ===
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
ACTIVE_SEAMS = ["Side_L", "Side_R", "Neck_Opening", "Shoulder_R", "Shoulder_L", "Armhole_R", "Armhole_L", "Waist_Hem"]
SHOULDER_KPT_IDX = 5335


def build_instance(mesh_path: str = "data/SMPL_FEMALE.ply", fabric_width: float = 150.0):
    mesh = trimesh.load(mesh_path, process=False)

    full_landmark_lib = generate_symmetric_landmarks(mesh, LANDMARKS)
    seams = build_shirt_topology(full_landmark_lib)

    # same as test_geometry.py (you can change later)
    texture_spec = TextureSpec("Stripes", 10.0, 100.0)

    instance = load_experiment(
        mesh_path=mesh_path,
        landmark_lib=full_landmark_lib,
        seam_lib=seams,
        active_seam_names=ACTIVE_SEAMS,
        texture=texture_spec,
        fabric_width=fabric_width
    )
    return instance, mesh


def run_geometry_blackbox(instance, mesh, delta_uv: np.ndarray, garment_part: str = "upper"):
    """
    Runs the exact processing steps shown in test_geometry.py:
      genotype delta -> mapped vertices -> cut -> export -> parameterize

    Writes:
      - data/seamlines/{garment_part}/...
      - results/pattern/latest/{garment_part}/patch_*/optim_final-seams.ply
    """
    mapper = LandmarkMapper(instance)
    vertex_ids = mapper.map_genotype_to_vertices(delta_uv)

    builder = BatchBuilder(instance)
    landmarks_batch = builder.build_batch(vertex_ids)

    cut_mesh, patches, patch_faces, seamlines_dict_list, symmetric_flags, valid_patch_idxs = perform_global_cut(
        landmarks_batch, mesh.vertices, mesh.faces
    )
    patch_labels_dict = assign_patch_labels(
        patches, garment_part, valid_patch_idxs, mesh.vertices[SHOULDER_KPT_IDX]
    )

    export_data(
        patches,
        valid_patch_idxs,
        garment_part,
        seamlines_dict_list,
        symmetric_flags,
        patch_labels_dict,
        cut_mesh
    )

    # writes results/pattern/latest/.../optim_final-seams.ply
    parameterize()
