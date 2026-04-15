"""Baseline B4: CMA-ES over delta only (kappa fixed at 0) at the GA evaluation budget.

This is the continuous-only comparison: a standard off-the-shelf CMA-ES
optimises the seam-position vector delta in [delta_lo, delta_hi]^(2m)
with all other genotype variables fixed (kappa = 0, rho = 0,
pi = area-sorted, h = bottom-left), using the same RealEvaluator as the
GA and the same total evaluation budget (pop * gens = 2000 per run).

Why this baseline: a PPSN reviewer will ask whether a learned-covariance
continuous optimiser outperforms our variable-specific mixed-integer GA
on the delta subspace alone. B4 answers that directly.

Output layout:
  results/experiments/{garment}[_tag]/b4/bodies_{N}/{timestamp}/results.json
  results/experiments/{garment}[_tag]/b4/bodies_{N}/{timestamp}/convergence.json

Auto-resume: scans each bodies_{N} directory for the latest partial run
and skips completed seeds.

Dependencies: requires `pycma` (pip install cma).
"""
import argparse
import json
import os
import re
from datetime import datetime

import numpy as np

try:
    import cma
except ImportError as exc:
    raise SystemExit(
        "pycma is required for B4. Install with: pip install cma"
    ) from exc

from ga_spec import GAInstance, Genome, Individual
from ga.real_evaluator import RealEvaluator, RealEvaluatorConfig
from nesting.loader import PatchLoader

FABRIC_WIDTH_MM = 150.0 * 10.0
PENALTY_FITNESS = 1e6  # assigned when RealEvaluator fails or returns NaN

# Threshold on fitness sum above which an evaluation is counted as a
# geometry-pipeline failure. RealEvaluator's internal penalty fitness is
# ~2e3; typical valid fitness is 1-5, so 100 cleanly separates the two.
# PENALTY_FITNESS above (1e6) is only used when RealEvaluator itself
# raises an exception here (above and beyond its own internal penalty).
FAIL_FITNESS_THRESHOLD = 100.0


def parse_args():
    p = argparse.ArgumentParser(description="Baseline B4: CMA-ES over delta.")
    p.add_argument("--garment-type", default="onesie_sleeves",
                   choices=["upper", "lower", "onesie_sleeves"])
    p.add_argument("--wallpaper", default="stripes",
                   choices=["stripes", "diagonal_stripes", "grid", "p4", "p4m",
                            "pg", "pmg", "pgg"])
    p.add_argument("--runs", type=int, default=10,
                   help="Number of seeds, 0..runs-1 (default: 10)")
    p.add_argument("--budget", type=int, default=2000,
                   help="Total evaluation budget per seed, matching GA "
                        "pop*gens (default: 2000)")
    p.add_argument("--num-bodies", type=int, nargs="+",
                   default=[5, 10, 25, 50, 100])
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=2.5)
    p.add_argument("--w4", type=float, default=10.0)
    p.add_argument("--delta_lo", type=float, default=0.2)
    p.add_argument("--delta_hi", type=float, default=0.8)
    p.add_argument("--sigma0", type=float, default=0.15,
                   help="Initial CMA-ES step size. Default 0.15 is ~1/4 of the "
                        "[0.2, 0.8] domain span, matching pycma's guidance.")
    p.add_argument("--popsize", type=int, default=None,
                   help="CMA-ES population size. Default (None) uses pycma's "
                        "automatic default 4 + floor(3 * log(n)).")
    p.add_argument("--tag", type=str, default=None,
                   help="Experiment tag (writes to "
                        "results/experiments/<garment>_<tag>/b4/...)")
    return p.parse_args()


def _metrics(ind: Individual, num_bodies: int) -> dict:
    f1_mm   = float(ind.meta.get("f1_height_mm", float("nan")))
    f2      = float(ind.meta.get("f2_phase",     float("nan")))
    f4      = float(ind.meta.get("f4_area_dev",  float("nan")))
    f1_norm = f1_mm / (FABRIC_WIDTH_MM * num_bodies)
    return {
        "f1_mm":   f1_mm,
        "f1_norm": f1_norm,
        "f2":      f2,
        "f4":      f4,
        "f_sum":   f1_norm + f2,
    }


def _find_latest_run_dir(bodies_base: str):
    try:
        entries = sorted(
            [e for e in os.scandir(bodies_base) if e.is_dir()],
            key=lambda e: e.stat().st_mtime,
            reverse=True,
        )
    except FileNotFoundError:
        return None
    for e in entries:
        if os.path.exists(os.path.join(e.path, "results.json")):
            return e.path
    return None


def _build_fixed_individual(delta: np.ndarray, inst: GAInstance) -> Individual:
    """Build an Individual with the given delta and all other variables fixed."""
    num_p   = inst.num_patches
    num_bod = inst.num_bodies

    kappa = np.zeros(num_p * num_bod, dtype=int)

    if inst.fixed_rho is not None:
        rho = inst.fixed_rho.copy()
    else:
        rho = np.zeros(num_p * num_bod, dtype=int)

    if inst.fixed_pi is not None:
        pi = inst.fixed_pi.copy()
    else:
        pi = np.arange(num_p, dtype=int)

    h = int(inst.fixed_h) if inst.fixed_h is not None else 0

    sigma = np.full(delta.shape, 0.20)  # unused; RealEvaluator does not touch sigma

    g = Genome(delta=delta.astype(float), rho=rho, kappa=kappa, pi=pi, h=h, sigma=sigma)
    return Individual(genome=g)


def run_cmaes(inst: GAInstance, evaluator: RealEvaluator, budget: int,
              seed: int, sigma0: float, popsize: int | None) -> tuple:
    """Run CMA-ES on delta for up to `budget` evaluations.

    Returns (best_individual, best_so_far_log, stats) where stats is a
    dict {"attempts": int, "failures": int} counting geometry-pipeline
    failures (penalty fitness) over the full seed.
    """
    n = 2 * inst.num_landmarks
    assert n > 0, "CMA-ES requires at least one delta dimension"

    x0 = np.full(n, 0.5, dtype=float)  # midpoint of the quad landmarks

    opts = {
        "bounds":      [[inst.delta_lo] * n, [inst.delta_hi] * n],
        "maxfevals":   budget,
        "seed":        seed + 1,  # pycma uses 1-based seeds; +1 avoids the "0 = time" sentinel
        "tolfun":      0.0,        # do not stop on small function-value change
        "tolfunhist":  0.0,
        "tolx":        0.0,        # do not stop on small x change
        "verbose":     -9,         # quiet
    }
    if popsize is not None:
        opts["popsize"] = popsize

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_ind: Individual = None
    best_sum = float("inf")
    log: list = []
    attempts = 0
    failures = 0

    evals = 0
    LOG_EVERY = 50

    while not es.stop() and evals < budget:
        xs = es.ask()
        fitnesses = []
        for x in xs:
            if evals >= budget:
                break
            ind = _build_fixed_individual(np.asarray(x, dtype=float), inst)
            try:
                ind.fitness = evaluator(ind)
                s = float(ind.fitness.values.sum())
                if not np.isfinite(s):
                    s = PENALTY_FITNESS
            except Exception as exc:
                print(f"[B4]   evaluator failed ({type(exc).__name__}: {exc}); "
                      f"assigning penalty fitness")
                s = PENALTY_FITNESS
            fitnesses.append(s)
            attempts += 1
            if s > FAIL_FITNESS_THRESHOLD:
                failures += 1

            if s < best_sum:
                best_sum = s
                best_ind = ind

            evals += 1
            if evals % LOG_EVERY == 0 or evals == budget:
                log.append({
                    "sample":     evals,
                    "best_f_sum": best_sum,
                    "best_f1_mm": float(best_ind.meta.get("f1_height_mm", float("nan"))) if best_ind else float("nan"),
                    "best_f2":    float(best_ind.meta.get("f2_phase",     float("nan"))) if best_ind else float("nan"),
                })
                fr = (failures / attempts) if attempts > 0 else 0.0
                print(f"[B4]   eval {evals}/{budget}  best_sum={best_sum:.4f}"
                      f"  valid={attempts-failures}/{attempts} ({100*(1-fr):.0f}%)")

        # Only tell CMA-ES about evaluations that were actually performed
        if len(fitnesses) == len(xs):
            es.tell(xs, fitnesses)
        else:
            # Budget exhausted mid-generation: stop the outer loop cleanly
            break

    if best_ind is None:
        raise RuntimeError(f"[B4] seed={seed}: no successful evaluation in {budget} tries")
    stats = {"attempts": attempts, "failures": failures}
    return best_ind, log, stats


def _build_instance_for(num_bodies: int, args) -> tuple:
    eval_cfg = RealEvaluatorConfig(
        garment_part=args.garment_type,
        latest_root="results/pattern/latest",
        seam_dir=f"data/seamlines/{args.garment_type}",
        period_u_mm=50.0,
        period_v_mm=50.0,
        K=8,
        fabric_width_mm=FABRIC_WIDTH_MM,
        num_bodies=num_bodies,
        wallpaper_group=args.wallpaper,
        w1=args.w1,
        w2=args.w2,
        w4=args.w4,
    )
    evaluator = RealEvaluator(eval_cfg)
    num_patches = len(evaluator.patch_ids)

    loader = PatchLoader(eval_cfg.latest_root, args.garment_type)
    items = loader.load_items()
    items_by_pid = sorted(items, key=lambda it: int(
        re.search(r"patch_(\d+)", it.name).group(1)))
    area_sorted_pi = np.array(
        sorted(range(len(items_by_pid)),
               key=lambda i: items_by_pid[i].area, reverse=True),
        dtype=int,
    )

    inst = GAInstance(
        num_patches=num_patches,
        K=eval_cfg.K,
        num_landmarks=evaluator.instance.num_sampled_landmarks,
        num_bodies=num_bodies,
        delta_lo=args.delta_lo,
        delta_hi=args.delta_hi,
        fixed_rho=np.zeros(num_patches * num_bodies, dtype=int),
        fixed_pi=area_sorted_pi,
        fixed_h=0,
        num_heuristics=3,
    )
    return evaluator, inst, eval_cfg


def _init_body_run(num_bodies: int, args) -> dict:
    exp_name = f"{args.garment_type}_{args.tag}" if args.tag else args.garment_type
    bodies_base = os.path.join("results", "experiments", exp_name, "b4",
                               f"bodies_{num_bodies}")
    latest = _find_latest_run_dir(bodies_base)

    if latest is not None:
        out_dir      = latest
        results_path = os.path.join(out_dir, "results.json")
        conv_path    = os.path.join(out_dir, "convergence.json")
        with open(results_path) as f:
            results = json.load(f)
        convergence = json.load(open(conv_path)) if os.path.exists(conv_path) else []
        completed = {r["seed"] for r in results.get("b4", [])}
        print(f"[B4] bodies={num_bodies}: resumed '{out_dir}' "
              f"({len(completed)}/{args.runs} seeds done)")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir   = os.path.join(bodies_base, timestamp)
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, "results.json")
        conv_path    = os.path.join(out_dir, "convergence.json")
        results = {
            "garment":   args.garment_type,
            "timestamp": timestamp,
            "config": {
                "method":         "b4_cmaes_delta",
                "runs":           args.runs,
                "budget":         args.budget,
                "num_bodies":     num_bodies,
                "K":              8,
                "period_u_mm":    50.0,
                "period_v_mm":    50.0,
                "fabric_width_mm": FABRIC_WIDTH_MM,
                "w1":             args.w1,
                "w2":             args.w2,
                "w4":             args.w4,
                "wallpaper":      args.wallpaper,
                "delta_lo":       args.delta_lo,
                "delta_hi":       args.delta_hi,
                "sigma0":         args.sigma0,
                "popsize":        args.popsize,
            },
            "b4": [],
        }
        convergence = []
        completed = set()
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[B4] bodies={num_bodies}: fresh start at '{out_dir}'")

    evaluator, inst, eval_cfg = _build_instance_for(num_bodies, args)

    return {
        "num_bodies":   num_bodies,
        "total_runs":   args.runs,
        "budget":       args.budget,
        "sigma0":       args.sigma0,
        "popsize":      args.popsize,
        "out_dir":      out_dir,
        "results_path": results_path,
        "conv_path":    conv_path,
        "results":      results,
        "convergence":  convergence,
        "completed":    completed,
        "evaluator":    evaluator,
        "inst":         inst,
        "eval_cfg":     eval_cfg,
    }


def _run_one_seed(state: dict, seed: int) -> None:
    if seed in state["completed"]:
        print(f"[B4] bodies={state['num_bodies']}  seed={seed} already done, skipping.")
        return

    num_bodies = state["num_bodies"]
    budget     = state["budget"]
    done       = len(state["completed"])
    total_runs = state["total_runs"]

    print("\n" + "=" * 60)
    print(f"[B4] bodies={num_bodies}  seed={seed}  budget={budget}  "
          f"sigma0={state['sigma0']}  popsize={state['popsize']}  "
          f"({done}/{total_runs} done)")
    print("=" * 60)

    best_ind, log, stats = run_cmaes(
        inst=state["inst"],
        evaluator=state["evaluator"],
        budget=budget,
        seed=seed,
        sigma0=state["sigma0"],
        popsize=state["popsize"],
    )

    run_metrics = {
        "seed":     seed,
        **_metrics(best_ind, num_bodies),
        "attempts": int(stats["attempts"]),
        "failures": int(stats["failures"]),
    }
    state["results"]["b4"].append(run_metrics)
    state["convergence"].append({"seed": seed, "log": log})
    state["completed"].add(seed)

    with open(state["results_path"], "w") as f:
        json.dump(state["results"], f, indent=2)
    with open(state["conv_path"], "w") as f:
        json.dump(state["convergence"], f, indent=2)

    done = len(state["completed"])
    f4_str = (f"{run_metrics['f4']:.4f}"
              if not np.isnan(run_metrics.get("f4", float("nan"))) else "n/a")
    att = run_metrics["attempts"]
    fail = run_metrics["failures"]
    fail_rate = (fail / att) if att > 0 else 0.0
    print(f"[B4] bodies={num_bodies}  seed={seed}"
          f"  f1={run_metrics['f1_mm']:.1f}mm"
          f"  f2={run_metrics['f2']:.4f}  f4={f4_str}  f_sum={run_metrics['f_sum']:.4f}"
          f"  valid={att-fail}/{att} ({100*(1-fail_rate):.0f}%)"
          f"  ({done}/{total_runs} done)")


def main():
    args = parse_args()
    print(f"[B4] garment={args.garment_type}  num_bodies={args.num_bodies}"
          f"  runs={args.runs}  budget={args.budget}"
          f"  sigma0={args.sigma0}  popsize={args.popsize}")

    print("\n[B4] Initialising body-count configurations...")
    states = {}
    for num_bodies in args.num_bodies:
        states[num_bodies] = _init_body_run(num_bodies, args)

    # Interleaved: outer = seed, inner = body count (matches run_experiments.py
    # and run_baseline_b3.py so partial runs accumulate one result per body
    # count before deepening).
    for seed in range(args.runs):
        print(f"\n{'#' * 60}")
        print(f"# seed={seed}")
        print(f"{'#' * 60}")
        for num_bodies in args.num_bodies:
            state = states[num_bodies]
            if len(state["completed"]) < state["total_runs"]:
                _run_one_seed(state, seed)

    print("\n[B4] All runs complete.")


if __name__ == "__main__":
    main()
