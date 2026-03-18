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

from spec import SeamPathType
from wallpaper import get_policy
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


def _load_patch_areas_3d(patches_3d_dir: str) -> dict[str, float]:
    """
    Load 3D surface areas (in mesh native units²) for each patch from
    data/patches/{garment_part}/patch_*/ref.ply.
    Returns {patch_dirname: area}, e.g. {"patch_00": 0.042, ...}.
    """
    areas: dict[str, float] = {}
    if not os.path.isdir(patches_3d_dir):
        return areas
    for dname in sorted(os.listdir(patches_3d_dir)):
        ref_ply = os.path.join(patches_3d_dir, dname, "ref.ply")
        if not os.path.exists(ref_ply):
            continue
        m = trimesh.load(ref_ply, process=False)
        areas[dname] = float(m.area)
    return areas


def _parse_patch_id(patch_dirname: str) -> int:
    m = re.search(r"patch_(\d+)", patch_dirname)
    if not m:
        raise ValueError(f"Cannot parse patch id from {patch_dirname}")
    return int(m.group(1))


def load_patch_vertices_full_from_latest(
    latest_root: str,
    garment_part: str = "upper",
    scale_mm: float = 1000.0,
    center_by_boundary: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Load FULL mesh vertices (Nx2) for each patch id, from:
      results/pattern/latest/{garment_part}/patch_*/optim_final-seams.ply
    Needed for seam mismatch evaluation / Stage2 solver.

    When center_by_boundary=True, vertices are shifted so the outer boundary
    loop mean is at the origin — matching the PatchLoader / NestingItem frame.
    """
    from nesting.utils import boundary_loops_from_edges, polygon_area_2d

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
        if center_by_boundary:
            ue = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
            be = mesh.edges[ue]
            loops = boundary_loops_from_edges(be)
            areas = [abs(polygon_area_2d(V2[lp])) for lp in loops]
            outer = loops[int(np.argmax(areas))]
            bm = V2[outer].mean(axis=0)
            V2 = V2 - bm
        V_by_id[pid] = V2
    return V_by_id


@dataclass
class RealEvaluatorConfig:
    garment_part: str = "upper"
    latest_root: str = "results/pattern/latest"
    seam_dir: str = f"data/seamlines/{garment_part}"

    period_u_mm: float = 50.0
    period_v_mm: float = 50.0
    K: int = 8
    fabric_width_mm: float = 150.0 * 1000.0

    # Number of bodies to nest simultaneously on one fabric roll.
    # All bodies currently share the same mesh + delta (identical cut).
    # To extend to different bodies: replace the single geometry call in __call__
    # with N separate calls (different mesh/delta), write each to a body-specific
    # directory, and load patches from those directories instead.
    num_bodies: int = 1

    wallpaper_group: str = "stripes"

    # fitness weights: set w1=0, w2=1 to optimize only texture alignment
    w1: float = 1.0  # fabric height
    w2: float = 1.0  # seam phase mismatch
    w3: float = 0.0  # flattening distortion (stub)
    w4: float = 0.0  # patch area deviation from baseline (fit preservation)


class RealEvaluator:
    """
    Minimal real-data evaluator:
      - runs geometry once (baseline delta)
      - per individual: loads latest patches/seams, runs Stage2 solve, runs nesting, computes f1,f2,f3
    """
    def __init__(self, cfg: RealEvaluatorConfig):
        self.cfg = cfg

        # 1) Build instance + mesh (topology/landmarks dispatched by garment_part)
        self.instance, self.mesh = build_instance(
            #mesh_path="data/SMPL_FEMALE_POSED.ply",
            mesh_path="data/SMPL_FEMALE.ply",
            fabric_width=cfg.fabric_width_mm / 1000.0,
            garment_type=cfg.garment_part,
            wallpaper_group=cfg.wallpaper_group,
            period_u=cfg.period_u_mm,
            period_v=cfg.period_v_mm,
        )
        self.policy = get_policy(cfg.wallpaper_group)

        # 2) Baseline delta: fixed middle of each sampled landmark quad.
        # Derived landmarks do not consume delta slots, so size = 2*num_sampled_landmarks.
        self.delta_baseline = np.array([0.5, 0.5] * self.instance.num_sampled_landmarks, dtype=float)

        # 3) Run geometry blackbox ONCE to generate results/pattern/latest and data/seamlines
        run_geometry_blackbox_timeout(self.instance, self.mesh, self.delta_baseline, garment_part=cfg.garment_part)

        # 4) Build seam name → importance mapping from topology.
        # Constraint files are now named seam-{SeamName}_{pi}-{pj}.txt so we can match
        # by name rather than by fragile sequential index (which shifts when short seams
        # such as Shoulder_L/R are filtered out by extract_seamlines, causing all
        # importances to be applied to the wrong seams and f2 to be always 0).
        self._seam_importance_by_name = {
            seam.name: seam.importance
            for seam in self.instance.active_seam_definitions
            if seam.path_type == SeamPathType.GEODESIC
        }

        # Load constraints once (for diagnostics / patch ID discovery).
        self.constraints = load_seam_constraints_from_dir(
            cfg.seam_dir,
            weights_by_filename=self._weights_by_filename(cfg.seam_dir),
            default_weight=0.0,
        )

        # 5a) Log the effective importance per constraint for diagnostics.
        for c in self.constraints:
            print(f"  seam '{c.name}' weight={c.weight:.3f}")

        # 5) Load patch IDs once
        self.V_full_by_id = load_patch_vertices_full_from_latest(cfg.latest_root, garment_part=cfg.garment_part, scale_mm=1000.0)
        self.patch_ids = sorted(self.V_full_by_id.keys())

        # 6) Texture lattice — axis directions come from the policy so rotated
        #    groups (e.g. diagonal_stripes) get the correct lattice geometry.
        u_dir, v_dir = self.policy.lattice_directions()
        self.lattice = TextureLattice(
            u_dir=u_dir,
            v_dir=v_dir,
            period_u=cfg.period_u_mm,
            period_v=cfg.period_v_mm
        )

        # 7) Baseline 3D patch areas for f4 (fit deviation).
        # Computed from the baseline geometry run above; units are m² (SMPL native).
        # The ratio in f4 is dimensionless so no scaling needed.
        self._patches_3d_dir = f"data/patches/{cfg.garment_part}"
        self.baseline_areas_3d = _load_patch_areas_3d(self._patches_3d_dir)
        self._baseline_total_area = sum(self.baseline_areas_3d.values()) or 1.0

        print("[RealEvaluator] Initialized.")
        print(f"  patches: {self.patch_ids}  ({len(self.patch_ids)} total)")
        print(f"  seams:   {len(self.constraints)}")
        print(f"  fabric:  {cfg.fabric_width_mm:.0f}mm wide"
              f"  period={cfg.period_u_mm:.0f}x{cfg.period_v_mm:.0f}mm  K={cfg.K}")
        print(f"  weights: w1(height)={cfg.w1}  w2(phase)={cfg.w2}"
              f"  w3(distortion)={cfg.w3}  w4(fit)={cfg.w4}")
        print(f"  baseline 3D patches: {len(self.baseline_areas_3d)}"
              f"  total area={self._baseline_total_area:.4f}m²")

    def _weights_by_filename(self, seam_dir: str) -> dict:
        """Build {filename: importance} by extracting seam name from each file's name."""
        import re as _re
        result = {}
        if not os.path.isdir(seam_dir):
            return result
        for fn in os.listdir(seam_dir):
            if not (fn.startswith("seam-") and fn.endswith(".txt")):
                continue
            m = _re.match(r"seam-(.+)_\d+-\d+\.txt$", fn)
            if m:
                seam_name = m.group(1)
                result[fn] = self._seam_importance_by_name.get(seam_name, 0.0)
        return result

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
            constraints = load_seam_constraints_from_dir(
                self.cfg.seam_dir,
                weights_by_filename=self._weights_by_filename(self.cfg.seam_dir),
                default_weight=0.0,
            )
            # center_by_boundary=True: vertices are shifted so the outer boundary
            # mean is at the origin, matching the PatchLoader / NestingItem frame.
            # This ensures Stage2 and f2 operate in the same coordinate system as
            # the nesting engine, so the phase computation is consistent with the
            # actual fabric placement.
            V_centered_by_id = load_patch_vertices_full_from_latest(
                self.cfg.latest_root, garment_part=self.cfg.garment_part,
                scale_mm=1000.0, center_by_boundary=True)
            patch_ids = sorted(V_centered_by_id.keys())
            t_geo = time.time() - t0

            # Snapshot raw patch files from disk into ind.meta so the best
            # individual's exact geometry can be written to best/ later without
            # re-running the non-deterministic C++ pipeline.
            patches_2d_raw = {}
            param_dir = os.path.join(self.cfg.latest_root, self.cfg.garment_part)
            for dname in sorted(os.listdir(param_dir)):
                seams_ply = os.path.join(param_dir, dname, 'optim_final-seams.ply')
                if os.path.exists(seams_ply):
                    m = trimesh.load(seams_ply, process=False)
                    patches_2d_raw[dname] = (np.array(m.vertices), np.array(m.faces))
            ind.meta["patches_2d_raw"] = patches_2d_raw

            patches_3d_raw = {}
            patches_3d_dir = f"data/patches/{self.cfg.garment_part}"
            if os.path.isdir(patches_3d_dir):
                for dname in sorted(os.listdir(patches_3d_dir)):
                    dpath = os.path.join(patches_3d_dir, dname)
                    if not os.path.isdir(dpath):
                        continue
                    ply_files = [f for f in os.listdir(dpath) if f.endswith('.ply')]
                    if ply_files:
                        m = trimesh.load(os.path.join(dpath, ply_files[0]), process=False)
                        patches_3d_raw[dname] = (np.array(m.vertices), np.array(m.faces), ply_files[0])
            ind.meta["patches_3d_raw"] = patches_3d_raw

            # Map patch_id -> item_idx (rank in sorted patch_ids).
            # IMPORTANT: do NOT use (pid - 1) as the index — patch IDs can be
            # non-sequential (e.g., [1, 2, 3, 5] when patch_04 is absent).
            # Using pid-1 would silently map patch_05 to index 4, which is either
            # out-of-bounds (kappa size=4) or into another body's genome slice in
            # the multi-body case — both produce wrong fitness signals.
            # This also raises IndexError when geometry cuts more patches than
            # expected (e.g., 5 instead of 4), which is caught below as a penalty.
            pid_to_item_idx = {pid: idx for idx, pid in enumerate(patch_ids)}
            kappas_by_id = {pid: int(g.kappa[pid_to_item_idx[pid]]) for pid in patch_ids}
        except Exception as e:
            print(f"         [geo] FAILED: {e} -> penalty fitness")
            penalty = 1e6 / self.cfg.fabric_width_mm  # normalised, same scale as f1_norm
            return Fitness(np.array([self.cfg.w1 * penalty, self.cfg.w2 * penalty, 0.0, 0.0], dtype=float))

        # Importances already baked in via _weights_by_filename during load.
        weighted_constraints = list(constraints)

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
            patch_vertices_by_id=V_centered_by_id,
            lattice=self.lattice,
            kappas_by_id=kappas_by_id,
            K=K,
            initial_transforms=T0,
            max_iters=15,
            verbose=False,
        )
        t_stage2 = time.time() - t0

        # Store Stage2 outputs so the caller can produce seam analysis plots
        # directly from this individual's data without any re-computation.
        ind.meta["Tsol"] = Tsol
        ind.meta["kappas_by_id"] = kappas_by_id
        ind.meta["V_centered_by_id"] = V_centered_by_id
        ind.meta["weighted_constraints"] = weighted_constraints

        # --- STEP 2: Build base items (Stage2 transforms applied once, shared across bodies) ---
        from copy import deepcopy
        from shapely.geometry import Polygon as _Polygon

        try:
            loader = PatchLoader(self.cfg.latest_root, self.cfg.garment_part)
            base_items = loader.load_items()
        except Exception as e:
            print(f"         [loader] FAILED: {e} -> penalty fitness")
            penalty = 1e6 / self.cfg.fabric_width_mm
            return Fitness(np.array([self.cfg.w1 * penalty, self.cfg.w2 * penalty, 0.0, 0.0], dtype=float))
        base_items = sorted(base_items, key=_pid)
        if len(base_items) != len(self.patch_ids):
            print(f"         [loader] wrong patch count: got {len(base_items)}, expected {len(self.patch_ids)} -> penalty fitness")
            penalty = 1e6 / self.cfg.fabric_width_mm
            return Fitness(np.array([self.cfg.w1 * penalty, self.cfg.w2 * penalty, 0.0, 0.0], dtype=float))
        num_base = len(base_items)

        # Apply Stage2 corrective transforms to base item geometry.
        for it in base_items:
            pid = _pid(it)
            T = Tsol.get(pid, Rigid2D(0.0, 0.0, 0.0))
            it.original_vertices = T.apply(it.original_vertices)
            it.shape = _Polygon(it.original_vertices)
            it.current_rotation = 0.0

        # Store Stage2-baked items so the caller can build alternative nestings
        # (e.g. baseline kappa=0) from the exact same geometry without any disk I/O.
        ind.meta["base_items"] = base_items

        # --- STEP 3: Clone base items N times, assign per-body kappa / rho ---
        # Genome layout: kappa[b*M : (b+1)*M] and rho[b*M : (b+1)*M] belong to body b.
        # Delta (seam positions) and w (seam weights) are shared across all bodies.
        # To extend to different bodies, replace this loop with N separate geometry
        # calls (different mesh / delta per body) and load from body-specific dirs.
        tx = self.instance.texture.period_x
        ty = self.instance.texture.period_y
        N = self.cfg.num_bodies
        M = num_base
        all_items = []

        for b in range(N):
            for item_idx, base_it in enumerate(base_items):
                it = deepcopy(base_it)
                it.name = f"body_{b}/{base_it.name}"

                pid = _pid(base_it)
                genome_idx = b * M + item_idx  # flat index into kappa / rho

                # Grain rotation (rho)
                rho_val = int(g.rho[genome_idx]) % 4 if genome_idx < g.rho.size else 0
                it.set_rotation(float(rho_val * 90))

                # Phase offset (kappa)
                k = int(g.kappa[genome_idx]) if genome_idx < g.kappa.size else 0
                it.phase_offset = ((k / float(K)) * tx, (k / float(K)) * ty)

                all_items.append(it)

        # --- STEP 4: Nest all N*M items on one fabric roll -> f1 ---
        num_total = N * M
        pi = None
        if g.pi is not None and getattr(g.pi, "size", 0) == num_total:
            pi = [int(x) for x in g.pi.tolist()]

        # rho for the engine call: flat array indexed by position in all_items
        rho_all = np.array(
            [int(g.rho[b * M + i]) % 4 if (b * M + i) < g.rho.size else 0
             for b in range(N) for i in range(M)],
            dtype=int
        )

        nest_engine = NestingEngine(fabric_width=self.cfg.fabric_width_mm, texture_spec=self.instance.texture)
        t0 = time.time()
        print("         [nesting] running...", flush=True)
        fabric_state = nest_engine.nest(all_items, permutation=pi, rotations=rho_all, heuristic=int(getattr(g, "h", 0)))
        t_nest = time.time() - t0
        f1 = float(fabric_state.total_height)
        ind.meta["fabric_state"] = fabric_state

        # --- STEP 5: f2 — phase mismatch averaged across N bodies ---
        # Each body shares the same seam structure (same weighted_constraints and Tsol)
        # but may have different kappa/rho assignments.
        # f2 is averaged over bodies so it stays in [0, 0.5*num_seams] regardless of N.

        f2_total = 0.0
        for b in range(N):
            f2_body = 0.0

            for c in weighted_constraints:
                Ti = Tsol.get(c.patch_i, Rigid2D(0, 0, 0))
                Tj = Tsol.get(c.patch_j, Rigid2D(0, 0, 0))

                # Map patch_id -> flat genome index for this body using
                # pid_to_item_idx (NOT pid-1, which breaks for non-sequential IDs).
                gi_i = b * M + pid_to_item_idx[c.patch_i]
                gi_j = b * M + pid_to_item_idx[c.patch_j]
                ki = int(g.kappa[gi_i]) if gi_i < g.kappa.size else 0
                kj = int(g.kappa[gi_j]) if gi_j < g.kappa.size else 0
                rho_i = int(g.rho[gi_i]) % 4 if gi_i < g.rho.size else 0
                rho_j = int(g.rho[gi_j]) % 4 if gi_j < g.rho.size else 0

                # Use the static per-seam importance as the f2 weight.
                # c.weight was set to seam importance in weighted_constraints above.
                seam_imp = c.weight

                # Orientation compatibility: if the two patches cannot be seam-aligned
                # given their grain rotations (policy-dependent), add a hard penalty
                # and skip the phase-mismatch computation.  The penalty is strictly
                # greater than the maximum possible phase mismatch (seam_imp * 0.5).
                if not self.policy.seam_compatible(rho_i, rho_j):
                    f2_body += seam_imp * 1.0
                    continue

                Vi = _rho_rotate_verts(Ti.apply(V_centered_by_id[c.patch_i]), rho_i)
                Vj = _rho_rotate_verts(Tj.apply(V_centered_by_id[c.patch_j]), rho_j)
                f2_body += seam_phase_mismatch_scalar(
                    seam_pairs=c.pairs,
                    patch_i_vertices_xy=Vi,
                    patch_j_vertices_xy=Vj,
                    lattice=self.lattice,
                    kappa_i=ki,
                    kappa_j=kj,
                    K=K,
                    weight=seam_imp,
                    transform_i=None,
                    transform_j=None,
                    glide_transforms=self.policy.glide_transforms(),
                )
            f2_total += f2_body

        f2 = f2_total / N  # average, scale-invariant across different N

        f3 = 0.0

        # --- f4: 3D patch area deviation from baseline (fit preservation) ---
        # Σ |area_i(δ) - area_i_baseline| / Σ area_i_baseline
        # Dimensionless ratio in [0, ∞); 0 means no change from the intended fit.
        current_areas = _load_patch_areas_3d(self._patches_3d_dir)
        if current_areas and self.baseline_areas_3d:
            f4 = sum(
                abs(current_areas.get(k, 0.0) - self.baseline_areas_3d[k])
                for k in self.baseline_areas_3d
            ) / self._baseline_total_area
        else:
            f4 = 0.0

        # Normalise f1 by fabric width so both objectives are dimensionless and
        # w1 / w2 become true priority knobs rather than unit-conversion factors.
        # f1_norm ≈ 0.5–3.0  (fabric height as a multiple of fabric width)
        # f2      ≈ 0.0–0.5 per seam (phase mismatch, already normalised)
        # f4      ≈ 0.0–0.1 typical (area deviation fraction)
        f1_norm = f1 / self.cfg.fabric_width_mm
        ind.meta["f1_height_mm"] = f1
        ind.meta["f2_phase"] = f2
        ind.meta["f4_area_dev"] = f4
        print(f"         [geo={t_geo:.1f}s  stage2={t_stage2:.2f}s  nesting={t_nest:.2f}s]"
              f"  f1={f1:.1f}mm  f2={f2:.4f}  f4={f4:.4f}")
        return Fitness(np.array(
            [self.cfg.w1 * f1_norm, self.cfg.w2 * f2, self.cfg.w3 * f3, self.cfg.w4 * f4],
            dtype=float,
        ))