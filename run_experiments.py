"""Systematic experiment runner for GA-Nesting.

Sweeps over a list of num_bodies values with interleaved seed iteration:
  outer loop = seed (0..runs-1)
  inner loop = num_bodies

This ensures one result per body-count is available quickly before
accumulating more statistical samples.

Directory layout:
  results/experiments/{garment}/bodies_{N}/{timestamp}/results.json
  results/experiments/{garment}/bodies_{N}/{timestamp}/convergence.json

Auto-resume: on each invocation the script scans each bodies_{N} directory
for the latest partial run.  Baselines (B0/B1/B2) are run once per body
count on first encounter.  Completed seeds are skipped.  Just re-run the
same command after a crash.

results.json schema:
  {
    "garment":   str,
    "config":    {runs, pop, gens, num_bodies, K, period_u_mm, period_v_mm,
                  fabric_width_mm, w1, w2, wallpaper},
    "b0":        {f1_mm, f1_norm, f2, f_sum},
    "b1":        {f1_mm, f1_norm, f2, f_sum},
    "b2":        {f1_mm, f1_norm, f2, f_sum},
    "ga":        [{seed, f1_mm, f1_norm, f2, f_sum}, ...]
  }

convergence.json schema:
  [{seed, log: [{gen, best_f_sum, best_f1_mm, best_f2}, ...]}, ...]
"""
import argparse
import json
import os
import numpy as np
from datetime import datetime

from ga_spec import GAInstance, GAConfig, run_ga
from ga.real_evaluator import RealEvaluator, RealEvaluatorConfig
from run_baseline_b0 import run as run_b0
from run_baseline_b1 import run as run_b1
from run_baseline_b2 import run as run_b2

FABRIC_WIDTH_MM = 150.0 * 10.0


def parse_args():
    p = argparse.ArgumentParser(description="Run systematic GA-Nesting experiments.")
    p.add_argument("--garment", default="upper", choices=["upper", "lower"],
                   help="Garment type (default: upper)")
    p.add_argument("--wallpaper", default="stripes",
                   choices=["stripes", "diagonal_stripes", "grid", "p4", "p4m", "pg", "pmg", "pgg"],
                   help="Texture wallpaper group (default: stripes)")
    p.add_argument("--runs", type=int, default=20,
                   help="Number of GA runs with seeds 0..runs-1 (default: 20)")
    p.add_argument("--pop", type=int, default=50,
                   help="GA population size (default: 50)")
    p.add_argument("--gens", type=int, default=10,
                   help="GA generations per run (default: 10)")
    p.add_argument("--num_bodies", type=int, nargs="+", default=[5, 10, 25, 50, 100],
                   help="List of body counts to sweep (default: 5 10 25 50 100)")
    p.add_argument("--w1", type=float, default=1.0,
                   help="Fitness weight for fabric height f1 (default: 1.0)")
    p.add_argument("--w2", type=float, default=10.0,
                   help="Fitness weight for seam phase mismatch f2 (default: 10.0)")
    return p.parse_args()


def _metrics(ind) -> dict:
    f1_mm   = float(ind.meta.get("f1_height_mm", float("nan")))
    f2      = float(ind.meta.get("f2_phase",     float("nan")))
    f1_norm = f1_mm / FABRIC_WIDTH_MM
    return {
        "f1_mm":   f1_mm,
        "f1_norm": f1_norm,
        "f2":      f2,
        # Unweighted sum — directly comparable to baseline f_sum values.
        "f_sum":   f1_norm + f2,
    }


def _print_summary(results, out_dir):
    b0 = results["b0"]
    b1 = results["b1"]
    ga = results["ga"]
    if not ga:
        return
    sums = [r["f_sum"]  for r in ga]
    f1s  = [r["f1_mm"]  for r in ga]
    f2s  = [r["f2"]     for r in ga]
    cfg  = results["config"]
    print("\n" + "=" * 60)
    print(f"[exp] bodies={cfg['num_bodies']}  Results: {out_dir}")
    print("=" * 60)
    print(f"  {'':8s}  {'f1_mm':>10s}  {'f2':>8s}  {'f_sum':>8s}")
    print(f"  {'B0':8s}  {b0['f1_mm']:>10.1f}  {b0['f2']:>8.4f}  {b0['f_sum']:>8.4f}")
    print(f"  {'B1':8s}  {b1['f1_mm']:>10.1f}  {b1['f2']:>8.4f}  {b1['f_sum']:>8.4f}")
    b2 = results.get("b2")
    if b2:
        print(f"  {'B2':8s}  {b2['f1_mm']:>10.1f}  {b2['f2']:>8.4f}  {b2['f_sum']:>8.4f}")
    print(f"  {'GA mean':8s}  {np.mean(f1s):>10.1f}  {np.mean(f2s):>8.4f}  {np.mean(sums):>8.4f}")
    print(f"  {'GA std':8s}  {np.std(f1s):>10.1f}  {np.std(f2s):>8.4f}  {np.std(sums):>8.4f}")
    print(f"  {'GA min':8s}  {np.min(f1s):>10.1f}  {np.min(f2s):>8.4f}  {np.min(sums):>8.4f}")
    print(f"  {'GA max':8s}  {np.max(f1s):>10.1f}  {np.max(f2s):>8.4f}  {np.max(sums):>8.4f}")


def _find_latest_run_dir(bodies_base: str):
    """Return the most recently modified timestamped subdir that has results.json."""
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


def _init_body_run(num_bodies: int, args) -> dict:
    """Load or create the run state for a given num_bodies.

    Creates the output directory and runs baselines if starting fresh.
    Returns a state dict used by _run_one_seed().
    """
    bodies_base = os.path.join("results", "experiments", args.garment, f"bodies_{num_bodies}")
    latest      = _find_latest_run_dir(bodies_base)

    if latest is not None:
        out_dir      = latest
        results_path = os.path.join(out_dir, "results.json")
        conv_path    = os.path.join(out_dir, "convergence.json")

        with open(results_path) as f:
            results = json.load(f)
        convergence = json.load(open(conv_path)) if os.path.exists(conv_path) else []

        cfg        = results["config"]
        completed  = {r["seed"] for r in results["ga"]}
        total_runs = cfg["runs"]

        # Back-fill B2 if absent from older results file.
        if "b2" not in results:
            print(f"\n[exp] bodies={num_bodies}: back-filling missing B2 baseline...")
            b2 = run_b2(args.garment, num_bodies=num_bodies)
            results["b2"] = {k: b2[k] for k in ("f1_mm", "f1_norm", "f2", "f_sum")}
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        num_bodies_cfg = cfg["num_bodies"]
        pop            = cfg["pop"]
        gens           = cfg["gens"]
        w1             = cfg["w1"]
        w2             = cfg["w2"]
        wallpaper      = cfg.get("wallpaper", "stripes")

        print(f"[exp] bodies={num_bodies}: loaded '{out_dir}'"
              f"  ({len(completed)}/{total_runs} seeds done)")

    else:
        completed   = set()
        convergence = []
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir     = os.path.join(bodies_base, timestamp)
        os.makedirs(out_dir, exist_ok=True)

        results_path   = os.path.join(out_dir, "results.json")
        conv_path      = os.path.join(out_dir, "convergence.json")
        total_runs     = args.runs
        num_bodies_cfg = num_bodies
        pop            = args.pop
        gens           = args.gens
        w1             = args.w1
        w2             = args.w2
        wallpaper      = args.wallpaper

        cfg_dict = {
            "runs": total_runs, "pop": pop, "gens": gens,
            "num_bodies": num_bodies_cfg,
            "K": 8, "period_u_mm": 50.0, "period_v_mm": 50.0,
            "fabric_width_mm": FABRIC_WIDTH_MM,
            "w1": w1, "w2": w2,
            "wallpaper": wallpaper,
        }

        print(f"\n[exp] bodies={num_bodies}: running baselines...")

        print("\n" + "=" * 60)
        print(f"[exp] Baseline B0  garment={args.garment}  num_bodies={num_bodies}")
        print("=" * 60)
        b0 = run_b0(args.garment, num_bodies=num_bodies)

        print("\n" + "=" * 60)
        print(f"[exp] Baseline B1  garment={args.garment}  num_bodies={num_bodies}")
        print("=" * 60)
        b1 = run_b1(args.garment, num_bodies=num_bodies)

        print("\n" + "=" * 60)
        print(f"[exp] Baseline B2  garment={args.garment}  num_bodies={num_bodies}")
        print("=" * 60)
        b2 = run_b2(args.garment, num_bodies=num_bodies)

        results = {
            "garment":   args.garment,
            "timestamp": timestamp,
            "config":    cfg_dict,
            "b0": {k: b0[k] for k in ("f1_mm", "f1_norm", "f2", "f_sum")},
            "b1": {k: b1[k] for k in ("f1_mm", "f1_norm", "f2", "f_sum")},
            "b2": {k: b2[k] for k in ("f1_mm", "f1_norm", "f2", "f_sum")},
            "ga": [],
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[exp] bodies={num_bodies}: fresh start at '{out_dir}'")

    # ── Evaluator and GA instance (kept alive for all seeds of this config) ──
    eval_cfg = RealEvaluatorConfig(
        garment_part=args.garment,
        latest_root="results/pattern/latest",
        seam_dir=f"data/seamlines/{args.garment}",
        period_u_mm=50.0,
        period_v_mm=50.0,
        K=8,
        fabric_width_mm=FABRIC_WIDTH_MM,
        num_bodies=num_bodies_cfg,
        wallpaper_group=wallpaper,
        w1=w1,
        w2=w2,
    )
    evaluator   = RealEvaluator(eval_cfg)
    num_patches = len(evaluator.patch_ids)

    inst = GAInstance(
        num_patches=num_patches,
        K=eval_cfg.K,
        num_landmarks=evaluator.instance.num_sampled_landmarks,
        num_bodies=num_bodies_cfg,
        fixed_rho=np.zeros(num_patches * num_bodies_cfg, dtype=int),
        fixed_pi=None,
        fixed_h=None,
        num_heuristics=3,
    )

    return {
        "num_bodies":   num_bodies_cfg,
        "total_runs":   total_runs,
        "pop":          pop,
        "gens":         gens,
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
    """Run a single GA seed for the given body-count state (skips if done)."""
    if seed in state["completed"]:
        print(f"[exp] bodies={state['num_bodies']}  seed={seed} already done, skipping.")
        return

    num_bodies = state["num_bodies"]
    pop        = state["pop"]
    gens       = state["gens"]
    total_runs = state["total_runs"]
    done       = len(state["completed"])

    print("\n" + "=" * 60)
    print(f"[exp] GA  bodies={num_bodies}  seed={seed}"
          f"  pop={pop}  gens={gens}  ({done}/{total_runs} done)")
    print("=" * 60)

    ga_cfg = GAConfig(
        seed=seed,
        population_size=pop,
        generations=gens,
        elite_count=4,
        tournament_k=4,
        crossover_prob=0.7,
        mutation_prob=0.7,
        prob_flip_kappa=0.35,
        weight_sigma=0.20,
    )

    pop_result, conv_log = run_ga(state["inst"], state["evaluator"], ga_cfg)
    best = min(pop_result, key=lambda ind: ind.fitness.values.sum())

    run_metrics = {"seed": seed, **_metrics(best)}
    state["results"]["ga"].append(run_metrics)
    state["convergence"].append({"seed": seed, "log": conv_log})
    state["completed"].add(seed)

    with open(state["results_path"], "w") as f:
        json.dump(state["results"], f, indent=2)
    with open(state["conv_path"], "w") as f:
        json.dump(state["convergence"], f, indent=2)

    done = len(state["completed"])
    print(f"[exp] bodies={num_bodies}  seed={seed}"
          f"  f1={run_metrics['f1_mm']:.1f}mm"
          f"  f2={run_metrics['f2']:.4f}  f_sum={run_metrics['f_sum']:.4f}"
          f"  ({done}/{total_runs} done)")


def main():
    args = parse_args()

    print(f"[exp] garment={args.garment}  num_bodies={args.num_bodies}"
          f"  runs={args.runs}  pop={args.pop}  gens={args.gens}")

    # Initialise all body-count configurations up front (runs baselines for new ones).
    print("\n[exp] Initialising body-count configurations...")
    states = {}
    for num_bodies in args.num_bodies:
        states[num_bodies] = _init_body_run(num_bodies, args)

    # Interleaved: outer = seed, inner = body count.
    for seed in range(args.runs):
        print(f"\n{'#' * 60}")
        print(f"# seed={seed}")
        print(f"{'#' * 60}")
        for num_bodies in args.num_bodies:
            state = states[num_bodies]
            if len(state["completed"]) < state["total_runs"]:
                _run_one_seed(state, seed)

    print(f"\n[exp] All runs complete.")
    for num_bodies in args.num_bodies:
        state = states[num_bodies]
        _print_summary(state["results"], state["out_dir"])


if __name__ == "__main__":
    main()
