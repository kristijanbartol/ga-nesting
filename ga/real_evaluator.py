# ga_evaluator_real.py
import os
import glob
import re
from dataclasses import dataclass
from typing import Dict

import numpy as np
import trimesh

from ga_spec import Individual, Fitness
from nesting.phase_utils import TextureLattice, Rigid2D, seam_phase_mismatch_scalar
from nesting.stage2_global_align import load_seam_constraints_from_dir, solve_global_alignment_all_components
from nesting.loader import PatchLoader
from nesting.engine import NestingEngine

from .geometry_block import build_instance, run_geometry_blackbox


def _parse_patch_id(patch_dirname: str) -> int:
    m = re.search(r"patch_(\d+)", patch_dirname)
    if not m:
        raise ValueError(f"Cannot parse patch id from {patch_dirname}")
    return int(m.group(1))


def load_patch_vertices_full_from_latest(latest_root: str, garment_part: str = "upper", scale_mm: float = 1000.0) -> Dict[int, np.ndarray]:
    """
    Load FULL mesh vertices (Nx2) for each patch id, from:
      results/pattern/latest/{garment_part}/patch_*/optim_final-seams.ply
    Needed for seam mismatch evaluation / Stage2 solver.
    """
    pattern = os.path.join(latest_root, garment_part, "patch_*", "optim_final-seams.ply")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No patch files found: {pattern}")

    V_by_id: Dict[int, np.ndarray] = {}
    for fpath in files:
        patch_dir = os.path.basename(os.path.dirname(fpath))  # patch_01
        pid = _parse_patch_id(patch_dir)
        mesh = trimesh.load(fpath, process=False)
        V2 = mesh.vertices[:, :2] * scale_mm
        V_by_id[pid] = V2
    return V_by_id


@dataclass
class RealEvaluatorConfig:
    garment_part: str = "upper"
    latest_root: str = "results/pattern/latest"
    seam_dir: str = "data/seamlines/upper"

    # texture lattice in SAME UNITS as loaded vertices (we use mm)
    period_u_mm: float = 50.0
    period_v_mm: float = 50.0

    # phase bins
    K: int = 8

    # fabric width for nesting (mm): instance uses meters? your nesting uses mm scale.
    fabric_width_mm: float = 150.0 * 1000.0  # default matches test_geometry fabric_width=150.0 *1000


class RealEvaluator:
    """
    Minimal real-data evaluator:
      - runs geometry once (baseline delta)
      - per individual: loads latest patches/seams, runs Stage2 solve, runs nesting, computes f1,f2,f3
    """
    def __init__(self, cfg: RealEvaluatorConfig):
        self.cfg = cfg

        # 1) Build instance + mesh (blackbox)
        self.instance, self.mesh = build_instance(mesh_path="data/SMPL_FEMALE.ply", fabric_width=cfg.fabric_width_mm / 1000.0)

        # 2) Baseline delta: fixed middle of each landmark quad, same as test_geometry.py
        # instance.num_landmarks landmarks => delta_uv length = 2*num_landmarks
        self.delta_baseline = np.array([0.5, 0.5] * self.instance.num_landmarks, dtype=float)

        # 3) Run geometry blackbox ONCE to generate results/pattern/latest and data/seamlines
        run_geometry_blackbox(self.instance, self.mesh, self.delta_baseline, garment_part=cfg.garment_part)

        # 4) Load seam constraints list once (order is stable: sorted filenames)
        self.constraints = load_seam_constraints_from_dir(cfg.seam_dir, weights_by_filename={}, default_weight=1.0)

        # 5) Load patch IDs once
        self.V_full_by_id = load_patch_vertices_full_from_latest(cfg.latest_root, garment_part=cfg.garment_part, scale_mm=1000.0)
        self.patch_ids = sorted(self.V_full_by_id.keys())

        # 6) Texture lattice
        self.lattice = TextureLattice(
            u_dir=np.array([1.0, 0.0]),
            v_dir=np.array([0.0, 1.0]),
            period_u=cfg.period_u_mm,
            period_v=cfg.period_v_mm
        )

        print("[RealEvaluator] Initialized.")
        print("  patches:", self.patch_ids)
        print("  seams:", len(self.constraints))

    def __call__(self, ind: Individual) -> Fitness:
        g = ind.genome
        K = self.cfg.K

        # Map per-patch kappa: assume patch IDs are 1..N corresponding to kappa[0..N-1]
        kappas_by_id = {pid: int(g.kappa[pid - 1]) for pid in self.patch_ids if (pid - 1) < g.kappa.size}

        # Map per-seam weights: constraints are sorted by filename; w[i] applies to constraints[i]
        # If genome has fewer weights than constraints, clamp.
        weights = g.w
        weighted_constraints = []
        for i, c in enumerate(self.constraints):
            w = float(weights[i]) if i < weights.size else 1.0
            weighted_constraints.append(type(c)(c.patch_i, c.patch_j, c.pairs, w, c.name))

        # Initial transforms: identity
        T0 = {pid: Rigid2D(0.0, 0.0, 0.0) for pid in self.patch_ids}

        # Stage2 solve (global per connected component)
        Tsol = solve_global_alignment_all_components(
            patch_ids=self.patch_ids,
            constraints=weighted_constraints,
            patch_vertices_by_id=self.V_full_by_id,
            lattice=self.lattice,
            kappas_by_id=kappas_by_id,
            K=K,
            initial_transforms=T0,
            max_iters=15,
            verbose=False,
        )

        # f2: total seam phase mismatch after solve (scalar)
        f2 = 0.0
        for c in weighted_constraints:
            Ti = Tsol.get(c.patch_i, Rigid2D(0, 0, 0))
            Tj = Tsol.get(c.patch_j, Rigid2D(0, 0, 0))
            ki = kappas_by_id.get(c.patch_i, 0)
            kj = kappas_by_id.get(c.patch_j, 0)
            f2 += seam_phase_mismatch_scalar(
                seam_pairs=c.pairs,
                patch_i_vertices_xy=self.V_full_by_id[c.patch_i],
                patch_j_vertices_xy=self.V_full_by_id[c.patch_j],
                lattice=self.lattice,
                kappa_i=ki,
                kappa_j=kj,
                K=K,
                weight=c.weight,
                transform_i=Ti,
                transform_j=Tj,
            )

        # f1: run nesting on boundary polygons (existing loader uses centered boundary loops)
        # We keep it minimal: use existing nesting loader + engine, ignore Stage2 transforms for nesting for now.
        loader = PatchLoader(self.cfg.latest_root)
        items = loader.load_items()
        
        import re
        # Sort items by patch id so genome.rho / genome.pi (indexed by patch_id-1)
        # deterministically map to the correct NestingItem.
        def _pid_from_item(it) -> int:
            m = re.search(r"patch_(\d+)", it.name)
            return int(m.group(1)) if m else 10**9

        items = sorted(items, key=_pid_from_item)
        
        # --- APPLY DISCRETE GRAIN ROTATIONS (rho) FOR NESTING ---
        # For nesting, only the SHAPE orientation matters; Stage2 translations are irrelevant.
        for it in items:
            pid = _pid_from_item(it)
            if 1 <= pid <= g.rho.size:
                it.set_rotation(float((int(g.rho[pid - 1]) % 4) * 90))
        
        # Apply GA kappa to nesting: per-item phase offset affects texture snapping lattice
        # item.name is like "patch_02" -> patch id = 2
        tx = self.instance.texture.period_x
        ty = self.instance.texture.period_y

        for it in items:
            m = re.search(r"patch_(\d+)", it.name)
            if not m:
                continue
            pid = int(m.group(1))

            # genome.kappa indexed by patch id-1 (assuming ids start at 1)
            if (pid - 1) < g.kappa.size:
                k = int(g.kappa[pid - 1])
            else:
                k = 0

            # Discrete phase shift within one period:
            # shift both axes for now (simple). Later we can separate u/v.
            ox = (k / float(self.cfg.K)) * tx
            oy = (k / float(self.cfg.K)) * ty
            it.phase_offset = (ox, oy)

        # rotations are fixed for now (rho not optimized yet); later we can set per-item from g.rho.
        # Optional nesting order from genome.pi (interpreted as indices into items sorted by patch id)
        pi = None
        if g.pi is not None and getattr(g.pi, "size", 0) == len(items):
            pi = [int(x) for x in g.pi.tolist()]

        nest_engine = NestingEngine(fabric_width=self.cfg.fabric_width_mm, texture_spec=self.instance.texture)
        fabric_state = nest_engine.nest(items, permutation=pi, rotations=g.rho, heuristic=int(getattr(g, "h", 0)))
        f1 = float(fabric_state.total_height)

        # f3 placeholder
        f3 = 0.0

        ind.meta["f1_height_mm"] = f1
        ind.meta["f2_phase"] = f2

        return Fitness(np.array([f1, f2, f3], dtype=float))
