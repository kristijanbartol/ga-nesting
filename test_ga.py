# visualize_ga_effect.py
import numpy as np
from copy import deepcopy

from ga_spec import GAInstance, GAConfig, run_ga, Individual
from ga.real_evaluator import RealEvaluator, RealEvaluatorConfig
from nesting.loader import PatchLoader
from nesting.engine import NestingEngine
from nesting.vis_utils import visualize_layout


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


def nest_and_show(latest_root, texture, fabric_width, genome, K, title):
    loader = PatchLoader(latest_root)
    items = loader.load_items()

    apply_kappa_to_items(items, genome, K, texture)

    eng = NestingEngine(fabric_width=fabric_width, texture_spec=texture)
    fabric = eng.nest(items)

    print(f"{title}: height={fabric.total_height:.2f}")
    visualize_layout(fabric, texture)


def main():
    eval_cfg = RealEvaluatorConfig(
        garment_part="upper",
        latest_root="results/pattern/latest",
        seam_dir="data/seamlines/upper",
        period_u_mm=50.0,
        period_v_mm=50.0,
        K=8,
        fabric_width_mm=150.0 * 1000.0,
    )
    evaluator = RealEvaluator(eval_cfg)

    num_patches = len(evaluator.patch_ids)
    num_seams = len(evaluator.constraints)

    inst = GAInstance(
        num_patches=num_patches,
        num_internal_seams=num_seams,
        K=eval_cfg.K,
        fixed_delta=evaluator.delta_baseline,
        fixed_rho=np.zeros((num_patches,), dtype=int),
        fixed_pi=np.arange(num_patches, dtype=int),
        fixed_h=0,
    )

    cfg = GAConfig(
        seed=0,
        population_size=6,
        generations=3,
        elite_count=1,
        tournament_k=3,
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
    nest_and_show(eval_cfg.latest_root, evaluator.instance.texture, eval_cfg.fabric_width_mm, base, eval_cfg.K, "BASELINE (kappa=0)")
    nest_and_show(eval_cfg.latest_root, evaluator.instance.texture, eval_cfg.fabric_width_mm, best.genome, eval_cfg.K, "BEST (GA kappa)")


if __name__ == "__main__":
    main()
