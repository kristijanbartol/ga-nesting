# ga_geometry_block.py
import glob
import numpy as np
import trimesh

from spec import LandmarkDefinition, TextureSpec
from experiment_loader import load_experiment
from geometry.geometry_utils import LandmarkMapper, generate_symmetric_landmarks
from geometry.topologies import build_shirt_topology
from geometry.cut_utils import perform_global_cut, assign_patch_labels
from geometry.export import export_data
from geometry.geometry_processor import BatchBuilder
from geometry.parameterization import parameterize
from geometry.landmarks import CORE_LANDMARKS, LONG_LANDMARKS, SHOULDER_KPT_IDX, ACTIVE_SEAMS


def build_instance(mesh_path: str = "data/SMPL_FEMALE.ply", fabric_width: float = 150.0):
    mesh = trimesh.load(mesh_path, process=False)

    full_landmark_lib = generate_symmetric_landmarks(mesh, {**CORE_LANDMARKS["Upper"], **LONG_LANDMARKS["Upper"]})
    # TODO: call build_*_topology by providing the string key to select the garment type (e.g., "shirt", "pants", "dress")
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

    output_pattern = f"results/pattern/latest/{garment_part}/patch_*/optim_final-seams.ply"
    output_files = glob.glob(output_pattern)
    if not output_files:
        raise RuntimeError(
            f"Geometry blackbox produced no output files at '{output_pattern}'. "
            "Check parameterization logs above for errors."
        )
    for fpath in output_files:
        verts = trimesh.load(fpath, process=False).vertices
        if np.any(np.isnan(verts)):
            raise RuntimeError(
                f"Parameterization produced NaN coordinates in '{fpath}'. "
                "Check parameterization logs above for errors."
            )
