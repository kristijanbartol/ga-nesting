"""Systematic experiment runner for GA-Nesting.

Runs B0 and B1 baselines once (deterministic), then the GA N times with
seeds 0..N-1.  Results are written incrementally to:

  results/experiments/{garment}/{timestamp}/results.json
  results/experiments/{garment}/{timestamp}/convergence.json

Pass --resume <dir> to continue an interrupted run from the last completed
seed.  If all seeds are already done, a completion message is printed and
the script exits without touching any files.

results.json schema:
  {
    "garment":   str,
    "config":    {runs, pop, gens, num_bodies, K, period_u_mm, period_v_mm,
                  fabric_width_mm, w1, w2, wallpaper},
    "b0":        {f1_mm, f1_norm, f2, f_sum},
    "b1":        {f1_mm, f1_norm, f2, f_sum},
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
    p.add_argument("--num_bodies", type=int, default=1,
                   help="Number of bodies to nest on one fabric roll (default: 1)")
    p.add_argument("--w1", type=float, default=1.0,
                   help="Fitness weight for fabric height f1 (default: 1.0)")
    p.add_argument("--w2", type=float, default=10.0,
                   help="Fitness weight for seam phase mismatch f2 (default: 10.0)")
    p.add_argument("--resume", default=None, metavar="DIR",
                   help="Resume an interrupted run from the given results directory.")
    return p.parse_args()


def _metrics(ind, fabric_width_mm: float) -> dict:
    f1_mm  = float(ind.meta.get("f1_height_mm", float("nan")))
    f2     = float(ind.meta.get("f2_phase",     float("nan")))
    f1_norm = f1_mm / fabric_width_mm
    return {
        "f1_mm":   f1_mm,
        "f1_norm": f1_norm,
        "f2":      f2,
        # Unweighted sum — directly comparable to baseline f_sum values.
        "f_sum":   f1_norm + f2,
    }


def _print_summary(results):
    b0 = results["b0"]
    b1 = results["b1"]
    ga = results["ga"]
    if not ga:
        return
    sums = [r["f_sum"]  for r in ga]
    f1s  = [r["f1_mm"]  for r in ga]
    f2s  = [r["f2"]     for r in ga]
    print("\n" + "=" * 60)
    print(f"[exp] Results saved to {results.get('_out_dir', '?')}")
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


def main():
    args = parse_args()

    # ── Resume or fresh start ─────────────────────────────────────────────────
    if args.resume:
        out_dir      = args.resume
        results_path = os.path.join(out_dir, "results.json")
        conv_path    = os.path.join(out_dir, "convergence.json")

        with open(results_path) as f:
            results = json.load(f)
        convergence = json.load(open(conv_path)) if os.path.exists(conv_path) else []

        cfg  = results["config"]
        total_runs  = cfg["runs"]
        completed   = {r["seed"] for r in results["ga"]}

        if len(completed) >= total_runs:
            print(f"[exp] All {total_runs} experiments already complete in '{out_dir}'.")
            print("[exp] Delete the directory if you want to rerun from scratch.")
            _print_summary({**results, "_out_dir": out_dir})
            return

        # B2 may be absent from results saved before B2 was added.
        if "b2" not in results:
            print("\n[exp] Running missing B2 baseline...")
            b2 = run_b2(args.garment, num_bodies=args.num_bodies)
            results["b2"] = {k: b2[k] for k in ("f1_mm", "f1_norm", "f2", "f_sum")}
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        print(f"[exp] Resuming '{out_dir}': {len(completed)}/{total_runs} seeds done,"
              f" continuing from seed {max(completed) + 1 if completed else 0}.")

        # Restore config values for evaluator / inst setup below.
        args.garment     = results["garment"]
        args.runs        = total_runs
        args.pop         = cfg["pop"]
        args.gens        = cfg["gens"]
        args.num_bodies  = cfg["num_bodies"]
        args.w1          = cfg["w1"]
        args.w2          = cfg["w2"]
        args.wallpaper   = cfg.get("wallpaper", "stripes")

    else:
        completed   = set()
        convergence = []
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir     = os.path.join("results", "experiments", args.garment, timestamp)
        os.makedirs(out_dir, exist_ok=True)

        results_path = os.path.join(out_dir, "results.json")
        conv_path    = os.path.join(out_dir, "convergence.json")

        cfg_dict = {
            "runs": args.runs, "pop": args.pop, "gens": args.gens,
            "num_bodies": args.num_bodies,
            "K": 8, "period_u_mm": 50.0, "period_v_mm": 50.0,
            "fabric_width_mm": 150.0 * 10.0,
            "w1": args.w1, "w2": args.w2,
            "wallpaper": args.wallpaper,
        }

        # ── Baselines (deterministic — run once) ─────────────────────────────
        print("\n" + "=" * 60)
        print(f"[exp] Baseline B0  garment={args.garment}  num_bodies={args.num_bodies}")
        print("=" * 60)
        b0 = run_b0(args.garment, num_bodies=args.num_bodies)

        print("\n" + "=" * 60)
        print(f"[exp] Baseline B1  garment={args.garment}  num_bodies={args.num_bodies}")
        print("=" * 60)
        b1 = run_b1(args.garment, num_bodies=args.num_bodies)

        print("\n" + "=" * 60)
        print(f"[exp] Baseline B2  garment={args.garment}  num_bodies={args.num_bodies}")
        print("=" * 60)
        b2 = run_b2(args.garment, num_bodies=args.num_bodies)

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

    # ── GA setup (evaluator initialised once, reused for all seeds) ──────────
    eval_cfg = RealEvaluatorConfig(
        garment_part=args.garment,
        latest_root="results/pattern/latest",
        seam_dir=f"data/seamlines/{args.garment}",
        period_u_mm=50.0,
        period_v_mm=50.0,
        K=8,
        fabric_width_mm=150.0 * 10.0,
        num_bodies=args.num_bodies,
        wallpaper_group=args.wallpaper,
        w1=args.w1,
        w2=args.w2,
    )
    evaluator = RealEvaluator(eval_cfg)
    num_patches = len(evaluator.patch_ids)

    inst = GAInstance(
        num_patches=num_patches,
        K=eval_cfg.K,
        num_landmarks=evaluator.instance.num_sampled_landmarks,
        num_bodies=args.num_bodies,
        fixed_rho=np.zeros(num_patches * args.num_bodies, dtype=int),
        fixed_pi=None,
        fixed_h=None,
        num_heuristics=3,
    )

    def _flush():
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        with open(conv_path, "w") as f:
            json.dump(convergence, f, indent=2)

    # ── GA runs (results flushed after each seed) ─────────────────────────────
    for seed in range(args.runs):
        if seed in completed:
            print(f"[exp] seed={seed} already done, skipping.")
            continue

        print("\n" + "=" * 60)
        print(f"[exp] GA run {seed + 1}/{args.runs}  seed={seed}"
              f"  garment={args.garment}  pop={args.pop}  gens={args.gens}"
              f"  bodies={args.num_bodies}")
        print("=" * 60)

        ga_cfg = GAConfig(
            seed=seed,
            population_size=args.pop,
            generations=args.gens,
            elite_count=4,
            tournament_k=4,
            crossover_prob=0.7,
            mutation_prob=0.7,
            prob_flip_kappa=0.35,
            weight_sigma=0.20,
        )

        pop, conv_log = run_ga(inst, evaluator, ga_cfg)
        best = min(pop, key=lambda ind: ind.fitness.values.sum())

        run_metrics = {"seed": seed, **_metrics(best, eval_cfg.fabric_width_mm)}
        results["ga"].append(run_metrics)
        convergence.append({"seed": seed, "log": conv_log})
        completed.add(seed)
        _flush()

        print(f"[exp] seed={seed}  f1={run_metrics['f1_mm']:.1f}mm"
              f"  f2={run_metrics['f2']:.4f}  f_sum={run_metrics['f_sum']:.4f}"
              f"  ({len(completed)}/{args.runs} done)")

    print(f"\n[exp] All {args.runs} experiments complete.")
    _print_summary({**results, "_out_dir": out_dir})


if __name__ == "__main__":
    main()
