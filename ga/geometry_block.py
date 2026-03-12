# ga_geometry_block.py
import glob
import multiprocessing
import numpy as np
import trimesh
from smplx import SMPL
import os
import torch

from spec import LandmarkDefinition, TextureSpec
from experiment_loader import load_experiment
from geometry.geometry_utils import LandmarkMapper, generate_symmetric_landmarks
from geometry.topologies import build_pant_topology, build_sleeveless_shirt_topology
from geometry.cut_utils import perform_global_cut, assign_patch_labels
from geometry.export import export_data
from geometry.geometry_processor import BatchBuilder
from geometry.parameterization import parameterize
from geometry.landmarks import (
    CORE_LANDMARKS, LONG_LANDMARKS, SHOULDER_KPT_IDX,
    PANT_SEAMS, SHIRTLESS_SEAMS,
)


# Dispatch table keyed by garment_type (= garment_part string).
# To add a new garment, add an entry here — no other file needs changing.
_GARMENT_CONFIGS = {
    "lower": {
        "landmark_fn": lambda: {**CORE_LANDMARKS["Lower"], **LONG_LANDMARKS["Lower"]},
        "topology":    build_pant_topology,
        "active_seams": PANT_SEAMS,
    },
    "upper": {
        "landmark_fn": lambda: {**CORE_LANDMARKS["Upper"]},   # sleeveless: no sleeve landmarks
        "topology":    build_sleeveless_shirt_topology,
        "active_seams": SHIRTLESS_SEAMS,
    },
}


def build_instance(
    mesh_path: str = "data/SMPL_FEMALE_POSED.ply",
    fabric_width: float = 150.0,
    garment_type: str = "lower",
    wallpaper_group: str = "stripes",
    period_u: float = 50.0,
    period_v: float = 50.0,
):
    if garment_type not in _GARMENT_CONFIGS:
        raise ValueError(f"Unknown garment_type '{garment_type}'. Known: {list(_GARMENT_CONFIGS)}")
    cfg = _GARMENT_CONFIGS[garment_type]

    mesh = trimesh.load(mesh_path, process=False)
    full_landmark_lib = generate_symmetric_landmarks(mesh, cfg["landmark_fn"]())
    seams = cfg["topology"](full_landmark_lib)

    texture_spec = TextureSpec("Stripes", period_u, period_v, wallpaper_group=wallpaper_group)

    instance = load_experiment(
        mesh_path=mesh_path,
        landmark_lib=full_landmark_lib,
        seam_lib=seams,
        active_seam_names=cfg["active_seams"],
        texture=texture_spec,
        fabric_width=fabric_width,
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

    cut_mesh, patches, patch_faces, seamlines_dict_list, symmetric_flags, valid_patch_idxs, seam_batch_indices = perform_global_cut(
        landmarks_batch, mesh.vertices, mesh.faces
    )
    patch_labels_dict = assign_patch_labels(
        patches, garment_part, valid_patch_idxs, mesh.vertices[SHOULDER_KPT_IDX]
    )

    # Build batch position → seam name mapping so each seam file is named by its
    # seam name rather than a sequential index.  Sequential indices shift when short
    # seams are filtered out, causing importance weights to be applied to the wrong seam.
    from spec import SeamPathType
    batch_pos = 0
    batch_pos_to_seam_name = {}
    for i, seam_type in enumerate(instance.active_seam_types):
        if seam_type == SeamPathType.GEODESIC:
            batch_pos_to_seam_name[batch_pos] = instance.seam_names[i]
            batch_pos += 1
        else:  # DUAL: two paths per seam
            batch_pos += 2
    seam_names = [batch_pos_to_seam_name.get(bidx, f"unknown_{bidx}") for bidx in seam_batch_indices]

    export_data(
        patches,
        valid_patch_idxs,
        garment_part,
        seamlines_dict_list,
        symmetric_flags,
        patch_labels_dict,
        cut_mesh,
        seam_names=seam_names,
    )

    # writes results/pattern/latest/{garment_part}/patch_*/optim_final-seams.ply
    parameterize(garment_part=garment_part)

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


def _geometry_worker(instance, mesh_V, mesh_F, delta_uv, garment_part, result_queue):
    """Child-process target: reconstruct mesh, run blackbox, push result."""
    try:
        mesh = trimesh.Trimesh(vertices=mesh_V, faces=mesh_F, process=False)
        run_geometry_blackbox(instance, mesh, delta_uv, garment_part=garment_part)
        result_queue.put(None)          # None => success
    except Exception as e:
        result_queue.put(repr(e))       # string => error


def run_geometry_blackbox_timeout(instance, mesh, delta_uv: np.ndarray,
                                   garment_part: str = "upper", timeout: int = 5):
    """
    Run run_geometry_blackbox in a forked subprocess with a hard wall-clock timeout.

    This is necessary because the potpourri3d C++ geodesic solver can hang
    indefinitely on degenerate mesh configurations.  Python-level timeouts
    (signal.alarm, Thread) cannot interrupt native C++ code that holds the GIL;
    only os.kill can.
    """
    # 'fork' inherits the parent's sys.path and loaded modules, avoiding the
    # re-import overhead and path issues that come with 'spawn' on macOS.
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    p = ctx.Process(
        target=_geometry_worker,
        args=(instance, mesh.vertices, mesh.faces, delta_uv, garment_part, result_queue),
    )
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join(timeout=3)
        if p.is_alive():
            p.kill()
            p.join()
        raise RuntimeError(
            f"Geometry blackbox killed after {timeout}s — C++ geodesic solver hung"
        )

    if not result_queue.empty():
        err = result_queue.get_nowait()
        if err is not None:
            raise RuntimeError(err)
