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
    p.add_argument("--garment", default="upper", choices=["upper", "lower", "onesie_sleeves"],
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
    p.add_argument("--w2", type=float, default=1.0,
                   help="Fitness weight for seam phase mismatch f2 (default: 1.0)")
    p.add_argument("--w4", type=float, default=10.0,
                   help="Fitness weight for 3D patch area deviation f4 (default: 10.0)")
    p.add_argument("--elite", type=int, default=4,
                   help="GA elite count (default: 4)")
    p.add_argument("--tourn", type=int, default=4,
                   help="GA tournament size (default: 4)")
    p.add_argument("--xover", type=float, default=0.7,
                   help="GA crossover probability (default: 0.7)")
    p.add_argument("--mut", type=float, default=0.7,
                   help="GA mutation probability (default: 0.7)")
    p.add_argument("--p_kappa", type=float, default=0.35,
                   help="Per-gene kappa flip probability (default: 0.35)")
    p.add_argument("--sigma_delta", type=float, default=0.20,
                   help="Gaussian sigma for delta mutation (default: 0.20)")
    p.add_argument("--self_adapt", action="store_true", default=False,
                   help="Enable MIES-style per-gene self-adaptive sigma for delta")
    p.add_argument("--delta_lo", type=float, default=0.2,
                   help="Lower bound for delta sampling and mutation (default: 0.2)")
    p.add_argument("--delta_hi", type=float, default=0.8,
                   help="Upper bound for delta sampling and mutation (default: 0.8)")
    p.add_argument("--tag", type=str, default=None,
                   help="Experiment tag — creates a separate results subdirectory "
                        "(e.g. --tag v2 writes to results/experiments/<garment>_v2/)")
    return p.parse_args()


def _metrics(ind, num_bodies: int) -> dict:
    f1_mm   = float(ind.meta.get("f1_height_mm", float("nan")))
    f2      = float(ind.meta.get("f2_phase",     float("nan")))
    f4      = float(ind.meta.get("f4_area_dev",  float("nan")))
    # Per-body normalisation matches f2 (already averaged across bodies).
    f1_norm = f1_mm / (FABRIC_WIDTH_MM * num_bodies)
    return {
        "f1_mm":   f1_mm,
        "f1_norm": f1_norm,
        "f2":      f2,
        "f4":      f4,
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
    f4s  = [r.get("f4", float("nan")) for r in ga]
    cfg  = results["config"]
    print("\n" + "=" * 60)
    print(f"[exp] bodies={cfg['num_bodies']}  Results: {out_dir}")
    print("=" * 60)
    print(f"  {'':8s}  {'f1_mm':>10s}  {'f2':>8s}  {'f4':>8s}  {'f_sum':>8s}")
    print(f"  {'B0':8s}  {b0['f1_mm']:>10.1f}  {b0['f2']:>8.4f}  {'0.0000':>8s}  {b0['f_sum']:>8.4f}")
    print(f"  {'B1':8s}  {b1['f1_mm']:>10.1f}  {b1['f2']:>8.4f}  {'0.0000':>8s}  {b1['f_sum']:>8.4f}")
    b2 = results.get("b2")
    if b2:
        print(f"  {'B2':8s}  {b2['f1_mm']:>10.1f}  {b2['f2']:>8.4f}  {'0.0000':>8s}  {b2['f_sum']:>8.4f}")
    f4_valid = [v for v in f4s if not np.isnan(v)]
    f4_mean = np.mean(f4_valid) if f4_valid else float("nan")
    f4_std  = np.std(f4_valid)  if f4_valid else float("nan")
    f4_min  = np.min(f4_valid)  if f4_valid else float("nan")
    f4_max  = np.max(f4_valid)  if f4_valid else float("nan")
    print(f"  {'GA mean':8s}  {np.mean(f1s):>10.1f}  {np.mean(f2s):>8.4f}  {f4_mean:>8.4f}  {np.mean(sums):>8.4f}")
    print(f"  {'GA std':8s}  {np.std(f1s):>10.1f}  {np.std(f2s):>8.4f}  {f4_std:>8.4f}  {np.std(sums):>8.4f}")
    print(f"  {'GA min':8s}  {np.min(f1s):>10.1f}  {np.min(f2s):>8.4f}  {f4_min:>8.4f}  {np.min(sums):>8.4f}")
    print(f"  {'GA max':8s}  {np.max(f1s):>10.1f}  {np.max(f2s):>8.4f}  {f4_max:>8.4f}  {np.max(sums):>8.4f}")


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
    exp_name = f"{args.garment}_{args.tag}" if args.tag else args.garment
    bodies_base = os.path.join("results", "experiments", exp_name, f"bodies_{num_bodies}")
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
        pop            = args.pop
        gens           = args.gens
        w1             = args.w1
        w2             = args.w2
        w4             = args.w4
        wallpaper      = args.wallpaper

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
        w4             = args.w4
        wallpaper      = args.wallpaper

        cfg_dict = {
            "runs": total_runs, "pop": pop, "gens": gens,
            "num_bodies": num_bodies_cfg,
            "K": 8, "period_u_mm": 50.0, "period_v_mm": 50.0,
            "fabric_width_mm": FABRIC_WIDTH_MM,
            "w1": w1, "w2": w2, "w4": w4,
            "wallpaper": wallpaper,
            "elite": args.elite, "tourn": args.tourn,
            "xover": args.xover, "mut": args.mut,
            "p_kappa": args.p_kappa, "sigma_delta": args.sigma_delta,
            "self_adapt": args.self_adapt,
            "delta_lo": args.delta_lo, "delta_hi": args.delta_hi,
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
        w4=w4,
    )
    evaluator   = RealEvaluator(eval_cfg)
    num_patches = len(evaluator.patch_ids)

    # Compute area-sorted-descending permutation to match the nesting
    # engine's default ordering (which baselines use).  Random pi with
    # 11! ≈ 40M orderings is unsearchable with pop=50 × 10 gens.
    from nesting.loader import PatchLoader as _PL
    import re as _re
    _loader = _PL(eval_cfg.latest_root, args.garment)
    _items = _loader.load_items()
    _items_by_pid = sorted(_items, key=lambda it: int(
        _re.search(r"patch_(\d+)", it.name).group(1)))
    _area_sorted_pi = np.array(
        sorted(range(len(_items_by_pid)),
               key=lambda i: _items_by_pid[i].area, reverse=True),
        dtype=int,
    )

    inst = GAInstance(
        num_patches=num_patches,
        K=eval_cfg.K,
        num_landmarks=evaluator.instance.num_sampled_landmarks,
        num_bodies=num_bodies_cfg,
        delta_lo=args.delta_lo,
        delta_hi=args.delta_hi,
        fixed_rho=np.zeros(num_patches * num_bodies_cfg, dtype=int),
        fixed_pi=_area_sorted_pi,
        fixed_h=0,
        num_heuristics=3,
    )

    return {
        "num_bodies":   num_bodies_cfg,
        "total_runs":   total_runs,
        "pop":          pop,
        "gens":         gens,
        "elite":        args.elite,
        "tourn":        args.tourn,
        "xover":        args.xover,
        "mut":          args.mut,
        "p_kappa":      args.p_kappa,
        "sigma_delta":  args.sigma_delta,
        "delta_lo":     args.delta_lo,
        "delta_hi":     args.delta_hi,
        "self_adapt":   args.self_adapt,
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
        elite_count=state["elite"],
        tournament_k=state["tourn"],
        crossover_prob=state["xover"],
        mutation_prob=state["mut"],
        prob_flip_kappa=state["p_kappa"],
        weight_sigma=state["sigma_delta"],
        self_adapt_sigma=state["self_adapt"],
    )

    pop_result, conv_log = run_ga(state["inst"], state["evaluator"], ga_cfg)
    best = min(pop_result, key=lambda ind: ind.fitness.values.sum())

    run_metrics = {"seed": seed, **_metrics(best, num_bodies)}
    state["results"]["ga"].append(run_metrics)
    state["convergence"].append({"seed": seed, "log": conv_log})
    state["completed"].add(seed)

    with open(state["results_path"], "w") as f:
        json.dump(state["results"], f, indent=2)
    with open(state["conv_path"], "w") as f:
        json.dump(state["convergence"], f, indent=2)

    done = len(state["completed"])
    f4_str = f"{run_metrics['f4']:.4f}" if not np.isnan(run_metrics.get("f4", float("nan"))) else "n/a"
    print(f"[exp] bodies={num_bodies}  seed={seed}"
          f"  f1={run_metrics['f1_mm']:.1f}mm"
          f"  f2={run_metrics['f2']:.4f}  f4={f4_str}  f_sum={run_metrics['f_sum']:.4f}"
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
