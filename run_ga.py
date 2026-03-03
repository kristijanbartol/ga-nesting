# visualize_ga_effect.py
import numpy as np
from copy import deepcopy

from ga_spec import GAInstance, GAConfig, run_ga
from ga.real_evaluator import RealEvaluator, RealEvaluatorConfig
from nesting.loader import PatchLoader
from nesting.engine import NestingEngine
from nesting.vis_utils import visualize_layout, visualize_population

from nesting.phase_utils import Rigid2D
from nesting.stage2_global_align import solve_global_alignment_all_components
from nesting.stage2_global_align import load_seam_constraints_from_dir
from ga.real_evaluator import load_patch_vertices_full_from_latest



def apply_kappa_to_items(items, genome, K, texture, pid_to_item_idx):
    """Apply kappa phase offsets using pid_to_item_idx (not pid-1) to handle non-sequential patch IDs."""
    tx, ty = texture.period_x, texture.period_y
    import re
    for it in items:
        m = re.search(r"patch_(\d+)", it.name)
        if not m:
            continue
        pid = int(m.group(1))
        idx = pid_to_item_idx.get(pid)
        if idx is None or idx >= genome.kappa.size:
            continue
        k = int(genome.kappa[idx])
        it.phase_offset = ((k / float(K)) * tx, (k / float(K)) * ty)


def nest_and_show(latest_root, seam_dir, lattice, texture, fabric_width, genome, K, title, garment_part):
    loader = PatchLoader(latest_root, garment_part)
    items = loader.load_items()

    # Deterministic mapping patch_id <-> genome index
    import re
    def _pid(it):
        m = re.search(r"patch_(\d+)", it.name)
        return int(m.group(1)) if m else 10**9
    items = sorted(items, key=_pid)

    # Build pid -> item_idx mapping (do NOT use pid-1, patch IDs can be non-sequential).
    patch_ids_local = [_pid(it) for it in items]
    pid_to_item_idx = {pid: idx for idx, pid in enumerate(patch_ids_local)}

    # 1) Apply kappa -> phase_offset (snap lattice shift)
    apply_kappa_to_items(items, genome, K, texture, pid_to_item_idx)

    # 2) Stage2: compute transforms from seam constraints + kappas + weights
    constraints = load_seam_constraints_from_dir(seam_dir, weights_by_filename={}, default_weight=1.0)

    V_centered_by_id = load_patch_vertices_full_from_latest(
        latest_root, garment_part=garment_part, scale_mm=1000.0, center_by_boundary=True)
    patch_ids = sorted(V_centered_by_id.keys())

    kappas_by_id = {pid: int(genome.kappa[pid_to_item_idx[pid]]) for pid in patch_ids if pid in pid_to_item_idx}

    # use constraints as-is (default_weight=1.0 already set by load_seam_constraints_from_dir)
    weighted_constraints = constraints

    T0 = {pid: Rigid2D(0.0, 0.0, 0.0) for pid in patch_ids}

    Tsol = solve_global_alignment_all_components(
        patch_ids=patch_ids,
        constraints=weighted_constraints,
        patch_vertices_by_id=V_centered_by_id,
        lattice=lattice,
        kappas_by_id=kappas_by_id,
        K=K,
        initial_transforms=T0,
        max_iters=25,
        verbose=False
    )

    # 3) Apply Stage2 transforms to the actual nested geometry
    from shapely.geometry import Polygon as _Polygon
    for it in items:
        pid = _pid(it)
        if pid >= 10**9:
            continue
        T = Tsol.get(pid, Rigid2D(0, 0, 0))
        it.original_vertices = T.apply(it.original_vertices)
        # Rebuild shape directly from Stage2-corrected vertices so that the
        # subsequent rho rotation is applied from the correct base.
        it.shape = _Polygon(it.original_vertices)
        it.current_rotation = 0.0

    # Apply discrete grain rotations (rho) after Stage2 bake (visualization only)
    for it in items:
        pid = _pid(it)
        idx = pid_to_item_idx.get(pid)
        if idx is None or idx >= genome.rho.size:
            continue
        rho_val = int(genome.rho[idx]) % 4
        it.set_rotation(float(rho_val * 90))

    # 4) Nest + visualize
    eng = NestingEngine(fabric_width=fabric_width, texture_spec=texture)
    pi = None
    if genome.pi is not None and getattr(genome.pi, "size", 0) == len(items):
        pi = [int(x) for x in genome.pi.tolist()]
    fabric = eng.nest(items, permutation=pi, heuristic=int(getattr(genome, "h", 0)))

    print(f"{title}: height={fabric.total_height:.2f}")
    visualize_layout(fabric, texture, title=title)

    # Plot per-seam phase mismatch
    from nesting.vis_utils import plot_seam_mismatch
    import re
    kappas_by_id = {pid: int(genome.kappa[pid_to_item_idx[pid]]) for pid in patch_ids if pid in pid_to_item_idx}
    plot_seam_mismatch(weighted_constraints, V_centered_by_id, lattice, kappas_by_id, K, Tsol, title)


def main():
    GARMENT_TYPE = "upper"   # ← change this one line to switch garments: "lower" | "upper"

    eval_cfg = RealEvaluatorConfig(
        garment_part=GARMENT_TYPE,
        latest_root="results/pattern/latest",
        seam_dir=f"data/seamlines/{GARMENT_TYPE}",
        period_u_mm=50.0,
        period_v_mm=50.0,
        K=8,
        fabric_width_mm=150.0 * 10.0,
        num_bodies=1,
        w1=0,
        w2=1
    )
    evaluator = RealEvaluator(eval_cfg)

    num_patches = len(evaluator.patch_ids)

    inst = GAInstance(
        num_patches=num_patches,
        K=eval_cfg.K,
        num_landmarks=evaluator.instance.num_landmarks,
        num_bodies=eval_cfg.num_bodies,
        fixed_rho=np.zeros(num_patches, dtype=float),
        fixed_pi=None,
        fixed_h=None,
        num_heuristics=3,
    )

    cfg = GAConfig(
        seed=0,
        population_size=20,
        generations=5,
        elite_count=3,
        tournament_k=4,
        crossover_prob=0.7,
        mutation_prob=0.7,
        prob_flip_kappa=0.35,
        weight_sigma=0.20,
    )

    pop = run_ga(inst, evaluator, cfg)

    # Visualize all individuals ordered best → worst
    visualize_population(pop, evaluator.instance.texture, title="Final population — best (top-left) to worst (bottom-right)")

    best = min(pop, key=lambda ind: ind.fitness.values.sum())

    # Build a baseline genome: kappa=0 for all
    base = deepcopy(best.genome)
    base.kappa[:] = 0

    print("\nBEST fitness:", best.fitness.values)
    print("BEST kappa:", best.genome.kappa)

    # The geometry pipeline overwrites results/pattern/latest/ on every evaluation,
    # so after the GA the patches on disk belong to the LAST evaluated individual —
    # not the best.  Regenerate the best individual's patches before visualizing.
    print("\n[main] Regenerating patches for best individual...")
    from ga.geometry_block import run_geometry_blackbox_timeout
    run_geometry_blackbox_timeout(
        evaluator.instance, evaluator.mesh, best.genome.delta,
        garment_part=GARMENT_TYPE)

    # Show baseline vs best NESTING (collision-free by construction)
    nest_and_show(eval_cfg.latest_root, eval_cfg.seam_dir, evaluator.lattice, evaluator.instance.texture,
                eval_cfg.fabric_width_mm, base, eval_cfg.K, "BASELINE (kappa=0)", GARMENT_TYPE)
    nest_and_show(eval_cfg.latest_root, eval_cfg.seam_dir, evaluator.lattice, evaluator.instance.texture,
                eval_cfg.fabric_width_mm, best.genome, eval_cfg.K, "BEST (GA kappa)", GARMENT_TYPE)


if __name__ == "__main__":
    main()