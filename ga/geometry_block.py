# ga_geometry_block.py
import glob
import multiprocessing
import sys
import numpy as np
import trimesh
from smplx import SMPL
import os
import torch

from spec import LandmarkDefinition, TextureSpec
from experiment_loader import load_experiment
from geometry.geometry_utils import LandmarkMapper, generate_symmetric_landmarks, find_midline_vidx
from geometry.topologies import build_pant_topology, build_sleeveless_shirt_topology, build_onesie_with_sleeves_topology
from geometry.cut_utils import perform_global_cut, assign_patch_labels
from geometry.export import export_data
from geometry.geometry_processor import BatchBuilder
from geometry.parameterization import parameterize
from geometry.landmarks import (
    CORE_LANDMARKS, LONG_LANDMARKS, SHOULDER_KPT_IDX, HIP_KPT_IDX,
    PANT_SEAMS, SHIRTLESS_SEAMS,
    ONESIE_LANDMARKS, ONESIE_LONG_LANDMARKS, ONESIE_WITH_SLEEVES_SEAMS,
    LOWER_MIDLINE_LANDMARKS, LOWER_DERIVED_MIDLINE,
    UPPER_MIDLINE_LANDMARKS, UPPER_DERIVED_MIDLINE,
    ONESIE_MIDLINE_LANDMARKS, ONESIE_DERIVED_MIDLINE,
)


# Dispatch table keyed by garment_type (= garment_part string).
# To add a new garment, add an entry here — no other file needs changing.
_GARMENT_CONFIGS = {
    "lower": {
        "landmark_fn": lambda: {**CORE_LANDMARKS["Lower"], **LONG_LANDMARKS["Lower"]},
        "topology":    build_pant_topology,
        "active_seams": PANT_SEAMS,
        # Midline landmarks are added after mirroring (not mirrored themselves).
        "derived_landmarks": LOWER_MIDLINE_LANDMARKS,
        "derived_lm_specs":  LOWER_DERIVED_MIDLINE,
    },
    "upper": {
        "landmark_fn": lambda: {**CORE_LANDMARKS["Upper"]},   # sleeveless: no sleeve landmarks
        "topology":    build_sleeveless_shirt_topology,
        "active_seams": SHIRTLESS_SEAMS,
        # Hip_Front / Hip_Back are midline landmarks derived from Hip_L at runtime.
        "derived_landmarks": UPPER_MIDLINE_LANDMARKS,
        "derived_lm_specs":  UPPER_DERIVED_MIDLINE,
    },
    "onesie_sleeves": {
        # Onesie landmarks (minus midline entries) + sleeve landmarks.
        # Neck_Front, Hip_Front, Hip_Back are midline — handled as derived below.
        "landmark_fn": lambda: {
            **{k: v for k, v in ONESIE_LANDMARKS.items()
               if k not in ("Neck_Front", "Hip_Front", "Hip_Back")},
            **ONESIE_LONG_LANDMARKS,
        },
        "topology":    build_onesie_with_sleeves_topology,
        "active_seams": ONESIE_WITH_SLEEVES_SEAMS,
        "derived_landmarks": ONESIE_MIDLINE_LANDMARKS,
        "derived_lm_specs":  ONESIE_DERIVED_MIDLINE,
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
    # Add midline (derived) landmarks directly — no mirroring needed.
    full_landmark_lib.update(cfg.get("derived_landmarks", {}))
    seams = cfg["topology"](full_landmark_lib)

    texture_spec = TextureSpec("Stripes", period_u, period_v, wallpaper_group=wallpaper_group)

    instance = load_experiment(
        mesh_path=mesh_path,
        landmark_lib=full_landmark_lib,
        seam_lib=seams,
        active_seam_names=cfg["active_seams"],
        texture=texture_spec,
        fabric_width=fabric_width,
        derived_lm_specs=cfg.get("derived_lm_specs", ()),
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
    vertex_ids, name_to_vidx = mapper.map_genotype_to_vertices(delta_uv)

    builder = BatchBuilder(instance)
    landmarks_batch = builder.build_batch(vertex_ids)

    cut_mesh, patches, patch_faces, seamlines_dict_list, symmetric_flags, valid_patch_idxs, seam_batch_indices = perform_global_cut(
        landmarks_batch, mesh.vertices, mesh.faces
    )
    if garment_part == 'lower':
        front_vidx = find_midline_vidx(mesh.vertices, mesh.faces, HIP_KPT_IDX, front=True)
        back_vidx  = find_midline_vidx(mesh.vertices, mesh.faces, HIP_KPT_IDX, front=False)
        ref_point = (mesh.vertices[front_vidx] + mesh.vertices[back_vidx]) / 2.0
    else:
        ref_point = mesh.vertices[SHOULDER_KPT_IDX]
    patch_labels_dict = assign_patch_labels(
        patches, garment_part, valid_patch_idxs, ref_point
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
                                   garment_part: str = "upper",
                                   timeout: int = 15 if sys.platform == "darwin" else 10):
    """
    Run run_geometry_blackbox in a forked subprocess with a hard wall-clock timeout.

    This is necessary because the potpourri3d C++ geodesic solver can hang
    indefinitely on degenerate mesh configurations.  Python-level timeouts
    (signal.alarm, Thread) cannot interrupt native C++ code that holds the GIL;
    only os.kill can.

    macOS uses a higher default (15s) because the forked-subprocess overhead
    and slower single-thread performance make 10s too tight for complex
    garments (onesie: 26 seams, ~9s pipeline).
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
