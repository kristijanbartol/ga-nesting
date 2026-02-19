# run_ga_real_minimal.py
import numpy as np

from ga_spec import GAInstance, GAConfig, run_ga
from ga.real_evaluator import RealEvaluator, RealEvaluatorConfig


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

    # Build GA instance sizes from real exported data:
    num_patches = len(evaluator.patch_ids)
    num_seams = len(evaluator.constraints)

    # Fix everything except (kappa, w)
    inst = GAInstance(
        num_patches=num_patches,
        num_internal_seams=num_seams,
        K=eval_cfg.K,
        fixed_delta=evaluator.delta_baseline,                 # fixed geometry
        fixed_rho=np.zeros((num_patches,), dtype=int),        # fixed rotations for now
        fixed_pi=np.arange(num_patches, dtype=int),           # fixed ordering for now
        fixed_h=0,
    )

    cfg = GAConfig(
        seed=0,
        population_size=4,
        generations=2,
        elite_count=1,
        tournament_k=3,
        crossover_prob=0.7,
        mutation_prob=0.7,
        prob_flip_kappa=0.25,
        weight_sigma=0.20,
    )

    pop = run_ga(inst, evaluator, cfg)

    best = min(pop, key=lambda ind: ind.fitness.values.sum())
    print("\n=== BEST INDIVIDUAL ===")
    print("F:", best.fitness.values)
    print("kappa:", best.genome.kappa)
    print("w (first 10):", best.genome.w[:10])
    print("meta:", best.meta)


if __name__ == "__main__":
    main()
