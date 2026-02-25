# visualize_ga_effect.py
import numpy as np
from copy import deepcopy

from ga_spec import GAInstance, GAConfig, run_ga
from ga.real_evaluator import RealEvaluator, RealEvaluatorConfig
from nesting.loader import PatchLoader
from nesting.engine import NestingEngine
from nesting.vis_utils import visualize_layout

from nesting.phase_utils import Rigid2D
from nesting.stage2_global_align import solve_global_alignment_all_components
from nesting.stage2_global_align import load_seam_constraints_from_dir
from ga.real_evaluator import load_patch_vertices_full_from_latest



def apply_kappa_to_items(items, genome, K, texture):
    import re
    tx, ty = texture.period_x, texture.period_y
    for it in items:
        m = re.search(r"patch_(\d+)", it.name)
        if not m:
            continue
        pid = int(m.group(1))
        k = int(genome.kappa[pid - 1]) if (pid - 1) < genome.kappa.size else 0
        it.phase_offset = ((k / float(K)) * tx, (k / float(K)) * ty)


def nest_and_show(latest_root, seam_dir, lattice, texture, fabric_width, genome, K, title):
    loader = PatchLoader(latest_root)
    items = loader.load_items()

    # Deterministic mapping patch_id <-> genome index
    import re
    def _pid(it):
        m = re.search(r"patch_(\d+)", it.name)
        return int(m.group(1)) if m else 10**9
    items = sorted(items, key=_pid)

    # 1) Apply kappa -> phase_offset (snap lattice shift)
    apply_kappa_to_items(items, genome, K, texture)

    # 2) Stage2: compute transforms from seam constraints + kappas + weights
    constraints = load_seam_constraints_from_dir(seam_dir, weights_by_filename={}, default_weight=1.0)

    V_full_by_id = load_patch_vertices_full_from_latest(latest_root, garment_part="upper", scale_mm=1000.0)
    patch_ids = sorted(V_full_by_id.keys())

    kappas_by_id = {pid: int(genome.kappa[pid - 1]) for pid in patch_ids if (pid - 1) < genome.kappa.size}

    # apply genome weights by seam-file order
    weighted_constraints = []
    for i, c in enumerate(constraints):
        w = float(genome.w[i]) if i < genome.w.size else 1.0
        weighted_constraints.append(type(c)(c.patch_i, c.patch_j, c.pairs, w, c.name))

    T0 = {pid: Rigid2D(0.0, 0.0, 0.0) for pid in patch_ids}

    Tsol = solve_global_alignment_all_components(
        patch_ids=patch_ids,
        constraints=weighted_constraints,
        patch_vertices_by_id=V_full_by_id,
        lattice=lattice,
        kappas_by_id=kappas_by_id,
        K=K,
        initial_transforms=T0,
        max_iters=25,
        verbose=False
    )

    # 3) Apply Stage2 transforms to the actual nested geometry
    for it in items:
        pid = _pid(it)
        if pid >= 10**9:
            continue
        T = Tsol.get(pid, Rigid2D(0, 0, 0))

        it.original_vertices = T.apply(it.original_vertices)
        it.set_rotation(it.current_rotation)  # rebuild polygon
        
    # Apply discrete grain rotations (rho) after Stage2 bake (visualization only)
    for it in items:
        pid = _pid(it)
        if 1 <= pid <= genome.rho.size:
            it.set_rotation(float((int(genome.rho[pid - 1]) % 4) * 90))

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
    kappas_by_id = {pid: int(genome.kappa[pid-1]) for pid in patch_ids if (pid-1) < genome.kappa.size}
    plot_seam_mismatch(weighted_constraints, V_full_by_id, lattice, kappas_by_id, K, Tsol, title)


def main():
    eval_cfg = RealEvaluatorConfig(
        garment_part="upper",
        latest_root="results/pattern/latest",
        seam_dir="data/seamlines/upper",
        period_u_mm=50.0,
        period_v_mm=50.0,
        K=8,
        fabric_width_mm=150.0 * 10.0,
        w1=1,
        w2=1
    )
    evaluator = RealEvaluator(eval_cfg)

    num_patches = len(evaluator.patch_ids)
    num_seams = len(evaluator.constraints)

    inst = GAInstance(
        num_patches=num_patches,
        num_internal_seams=num_seams,
        K=eval_cfg.K,
        num_landmarks=evaluator.instance.num_landmarks,
        fixed_rho=None,
        fixed_pi=None,
        fixed_h=None,
        num_heuristics=3,
    )

    cfg = GAConfig(
        seed=0,
        population_size=20,
        generations=10,
        elite_count=2,
        tournament_k=4,
        crossover_prob=0.7,
        mutation_prob=0.7,
        prob_flip_kappa=0.35,
        weight_sigma=0.20,
    )

    pop = run_ga(inst, evaluator, cfg)
    best = min(pop, key=lambda ind: ind.fitness.values.sum())

    # Build a baseline genome: kappa=0 for all, weights=1 for all
    base = deepcopy(best.genome)
    base.kappa[:] = 0
    base.w[:] = 1.0

    print("\nBEST fitness:", best.fitness.values)
    print("BEST kappa:", best.genome.kappa)

    # Show baseline vs best NESTING (collision-free by construction)
    nest_and_show(eval_cfg.latest_root, eval_cfg.seam_dir, evaluator.lattice, evaluator.instance.texture,
                eval_cfg.fabric_width_mm, base, eval_cfg.K, "BASELINE (kappa=0)")
    nest_and_show(eval_cfg.latest_root, eval_cfg.seam_dir, evaluator.lattice, evaluator.instance.texture,
                eval_cfg.fabric_width_mm, best.genome, eval_cfg.K, "BEST (GA kappa)")


if __name__ == "__main__":
    main()