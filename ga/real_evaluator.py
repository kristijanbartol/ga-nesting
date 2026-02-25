# ga_evaluator_real.py
import os
import glob
import re
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import trimesh

from ga_spec import Individual, Fitness
from nesting.phase_utils import TextureLattice, Rigid2D, seam_phase_mismatch_scalar
from nesting.stage2_global_align import load_seam_constraints_from_dir, solve_global_alignment_all_components
from nesting.loader import PatchLoader
from nesting.engine import NestingEngine

from .geometry_block import build_instance, run_geometry_blackbox_timeout


def _rho_rotate_verts(verts_xy: np.ndarray, rho: int) -> np.ndarray:
    """Rotate 2D vertices by rho * 90° CCW (matches NestingItem.set_rotation convention)."""
    r = rho % 4
    if r == 0:
        return verts_xy
    theta = r * np.pi / 2.0
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s], [s, c]], dtype=float)
    return verts_xy @ R.T


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

    period_u_mm: float = 50.0
    period_v_mm: float = 50.0
    K: int = 8
    fabric_width_mm: float = 150.0 * 1000.0

    # fitness weights: set w1=0, w2=1 to optimize only texture alignment
    w1: float = 1.0  # fabric height
    w2: float = 1.0  # seam phase mismatch
    w3: float = 0.0  # flattening distortion (stub)


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
        run_geometry_blackbox_timeout(self.instance, self.mesh, self.delta_baseline, garment_part=cfg.garment_part)

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
        print(f"  patches: {self.patch_ids}  ({len(self.patch_ids)} total)")
        print(f"  seams:   {len(self.constraints)}")
        print(f"  fabric:  {cfg.fabric_width_mm:.0f}mm wide"
              f"  period={cfg.period_u_mm:.0f}x{cfg.period_v_mm:.0f}mm  K={cfg.K}")
        print(f"  weights: w1(height)={cfg.w1}  w2(phase)={cfg.w2}  w3(distortion)={cfg.w3}")

    def __call__(self, ind: Individual) -> Fitness:
        g = ind.genome
        K = self.cfg.K
        import re

        # --- STEP 0: Geometry pipeline with this individual's delta ---
        # Regenerates patches and seams on disk for this delta.
        try:
            t0 = time.time()
            print("         [geo] running...", flush=True)
            run_geometry_blackbox_timeout(self.instance, self.mesh, g.delta, garment_part=self.cfg.garment_part)
            constraints = load_seam_constraints_from_dir(self.cfg.seam_dir, weights_by_filename={}, default_weight=1.0)
            V_full_by_id = load_patch_vertices_full_from_latest(self.cfg.latest_root, garment_part=self.cfg.garment_part, scale_mm=1000.0)
            patch_ids = sorted(V_full_by_id.keys())
            t_geo = time.time() - t0
        except Exception as e:
            print(f"         [geo] FAILED: {e} -> penalty fitness")
            penalty = 1e6 / self.cfg.fabric_width_mm  # normalised, same scale as f1_norm
            return Fitness(np.array([self.cfg.w1 * penalty, self.cfg.w2 * penalty, 0.0], dtype=float))

        kappas_by_id = {pid: int(g.kappa[pid - 1]) for pid in patch_ids if (pid - 1) < g.kappa.size}

        weighted_constraints = []
        for i, c in enumerate(constraints):
            w = float(g.w[i]) if i < g.w.size else 1.0
            weighted_constraints.append(type(c)(c.patch_i, c.patch_j, c.pairs, w, c.name))

        def _pid(it) -> int:
            m = re.search(r"patch_(\d+)", it.name)
            return int(m.group(1)) if m else 10**9

        # --- STEP 1: Stage2 from identity ---
        # Finds small phase-aligning rigid transforms in parametric space.
        T0 = {pid: Rigid2D(0.0, 0.0, 0.0) for pid in patch_ids}
        t0 = time.time()
        print("         [stage2] running...", flush=True)
        Tsol = solve_global_alignment_all_components(
            patch_ids=patch_ids,
            constraints=weighted_constraints,
            patch_vertices_by_id=V_full_by_id,
            lattice=self.lattice,
            kappas_by_id=kappas_by_id,
            K=K,
            initial_transforms=T0,
            max_iters=15,
            verbose=False,
        )
        t_stage2 = time.time() - t0

        # --- STEP 2: Apply Stage2 transforms to patch geometry ---
        loader = PatchLoader(self.cfg.latest_root)
        items = loader.load_items()
        items = sorted(items, key=_pid)

        for it in items:
            pid = _pid(it)
            T = Tsol.get(pid, Rigid2D(0.0, 0.0, 0.0))
            it.original_vertices = T.apply(it.original_vertices)
            it.shape = __import__("shapely.geometry", fromlist=["Polygon"]).Polygon(it.original_vertices)

        # Apply grain rotations (rho) and kappa phase offsets.
        for it in items:
            pid = _pid(it)
            if 1 <= pid <= g.rho.size:
                it.set_rotation(float((int(g.rho[pid - 1]) % 4) * 90))

        tx = self.instance.texture.period_x
        ty = self.instance.texture.period_y
        for it in items:
            pid = _pid(it)
            k = int(g.kappa[pid - 1]) if (pid - 1) < g.kappa.size else 0
            it.phase_offset = ((k / float(K)) * tx, (k / float(K)) * ty)

        # --- STEP 3: Nesting on phase-aligned geometry -> f1 ---
        pi = None
        if g.pi is not None and getattr(g.pi, "size", 0) == len(items):
            pi = [int(x) for x in g.pi.tolist()]

        nest_engine = NestingEngine(fabric_width=self.cfg.fabric_width_mm, texture_spec=self.instance.texture)
        t0 = time.time()
        print("         [nesting] running...", flush=True)
        fabric_state = nest_engine.nest(items, permutation=pi, rotations=g.rho, heuristic=int(getattr(g, "h", 0)))
        t_nest = time.time() - t0
        f1 = float(fabric_state.total_height)

        # --- STEP 4: f2 in fabric space (Stage2 transform + rho rotation) ---
        # Stage2 produces small corrective transforms in parameterisation space.
        # After nesting, each patch is additionally rotated by rho * 90°.  We must
        # apply that same rotation here so that f2 reflects the actual stripe
        # direction seen on the physical fabric — a 90°-rotated patch has
        # perpendicular stripes and must be penalised accordingly.
        f2 = 0.0
        for c in weighted_constraints:
            Ti = Tsol.get(c.patch_i, Rigid2D(0, 0, 0))
            Tj = Tsol.get(c.patch_j, Rigid2D(0, 0, 0))
            ki = kappas_by_id.get(c.patch_i, 0)
            kj = kappas_by_id.get(c.patch_j, 0)
            rho_i = int(g.rho[c.patch_i - 1]) % 4 if (c.patch_i - 1) < g.rho.size else 0
            rho_j = int(g.rho[c.patch_j - 1]) % 4 if (c.patch_j - 1) < g.rho.size else 0
            # Apply Stage2 transform, then rho rotation to get fabric-space coords.
            Vi = _rho_rotate_verts(Ti.apply(V_full_by_id[c.patch_i]), rho_i)
            Vj = _rho_rotate_verts(Tj.apply(V_full_by_id[c.patch_j]), rho_j)
            f2 += seam_phase_mismatch_scalar(
                seam_pairs=c.pairs,
                patch_i_vertices_xy=Vi,
                patch_j_vertices_xy=Vj,
                lattice=self.lattice,
                kappa_i=ki,
                kappa_j=kj,
                K=K,
                weight=c.weight,
                transform_i=None,   # already baked into Vi / Vj
                transform_j=None,
            )

        f3 = 0.0
        # Normalise f1 by fabric width so both objectives are dimensionless and
        # w1 / w2 become true priority knobs rather than unit-conversion factors.
        # f1_norm ≈ 0.5–3.0  (fabric height as a multiple of fabric width)
        # f2      ≈ 0.0–0.5 per seam (phase mismatch, already normalised)
        f1_norm = f1 / self.cfg.fabric_width_mm
        ind.meta["f1_height_mm"] = f1
        ind.meta["f2_phase"] = f2
        print(f"         [geo={t_geo:.1f}s  stage2={t_stage2:.2f}s  nesting={t_nest:.2f}s]"
              f"  f1={f1:.1f}mm  f2={f2:.4f}")
        return Fitness(np.array([self.cfg.w1 * f1_norm, self.cfg.w2 * f2, self.cfg.w3 * f3], dtype=float))