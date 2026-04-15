"""Baseline B3: random search over (delta, kappa) at the GA evaluation budget.

For each (seed, num_bodies) pair, the script maintains a persistent, on-disk
pool of random samples drawn from the same GAInstance distribution the GA
uses (delta in [delta_lo, delta_hi], kappa in {0..K-1}, rho fixed, pi fixed,
h fixed). Each sample is evaluated once with the full geometry + Stage 2 +
nesting pipeline via the same RealEvaluator, and its raw f1_mm, f1_norm, f2,
f4 values are appended as one JSON line to samples_seed{S}.jsonl alongside
results.json.

Key properties:

- Persistent pool: the full set of samples for each seed is stored on disk.
  Re-running the script with a larger --samples extends each seed's pool
  in-place without re-evaluating anything.
- Deterministic per-sample RNG: sample idx K is always the same draw,
  seeded as (seed_base * 1_000_000 + K), so crashed runs resume exactly
  where they left off, and two invocations with the same seed_base produce
  byte-identical pools up to the common prefix.
- Post-hoc reweighting: the reporting weights (--w1, --w2, --w4) do NOT
  affect which samples are drawn. They are applied only when reading the
  pool to compute best-of-pool. Re-scoring under different weights does
  not require rerunning the pipeline.
- Implicit constraint handling: geometry failures are counted as
  "failed" samples in the JSONL and excluded from best-of-pool, matching
  how the GA's selection implicitly handles the ~50% feasibility rate.

Output layout:
  results/experiments/{garment}[_tag]/b3/bodies_{N}/{timestamp}/
    results.json                  (aggregate per-seed best-of-pool metrics)
    convergence.json              (best-so-far trajectory derived from pool)
    samples_seed{S}.jsonl         (one JSON per sample, append-only)
"""
import argparse
import json
import os
import random
import re
from datetime import datetime

import numpy as np

from ga_spec import GAInstance, GAConfig, Individual, random_genome
from ga.real_evaluator import RealEvaluator, RealEvaluatorConfig
from nesting.loader import PatchLoader

FABRIC_WIDTH_MM = 150.0 * 10.0

# Per-sample RNG seed offset. seed_base * _SEED_STRIDE + idx gives each
# sample a unique, independent RNG seed, so the pool is deterministically
# resumable: sample idx K is always the same draw regardless of where we
# stop and restart. _SEED_STRIDE must be larger than any plausible budget.
_SEED_STRIDE = 1_000_000


def parse_args():
    p = argparse.ArgumentParser(description="Baseline B3: random search at GA budget.")
    p.add_argument("--garment-type", default="onesie_sleeves",
                   choices=["upper", "lower", "onesie_sleeves"])
    p.add_argument("--wallpaper", default="stripes",
                   choices=["stripes", "diagonal_stripes", "grid", "p4", "p4m",
                            "pg", "pmg", "pgg"])
    p.add_argument("--runs", type=int, default=10,
                   help="Number of seeds, 0..runs-1 (default: 10)")
    p.add_argument("--samples", type=int, default=2000,
                   help="Target total samples per seed. If a pool already "
                        "exists with fewer samples, it is extended in-place; "
                        "if the pool is already at or above this value, the "
                        "seed is re-scored with the current weights but no "
                        "new samples are drawn. Default: 2000 (matches GA's "
                        "pop * gens = 100 * 20).")
    p.add_argument("--num-bodies", type=int, nargs="+",
                   default=[5, 10, 25, 50, 100],
                   help="List of body counts (default: 5 10 25 50 100)")
    # Reporting weights only. These do NOT affect which samples are drawn;
    # they are applied when reading the pool to compute best-of-pool, so the
    # same pool can be re-scored under any weighting scheme.
    p.add_argument("--w1", type=float, default=1.0,
                   help="Reporting weight on f1_norm (fabric height). "
                        "Does not affect sampling.")
    p.add_argument("--w2", type=float, default=2.5,
                   help="Reporting weight on f2 (seam phase mismatch). "
                        "Does not affect sampling.")
    p.add_argument("--w4", type=float, default=10.0,
                   help="Reporting weight on f4 (fit preservation guard). "
                        "Does not affect sampling.")
    p.add_argument("--delta_lo", type=float, default=0.2)
    p.add_argument("--delta_hi", type=float, default=0.8)
    p.add_argument("--tag", type=str, default=None,
                   help="Experiment tag (writes to "
                        "results/experiments/<garment>_<tag>/b3/...)")
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


def _count_existing_samples(samples_path: str) -> int:
    """Return the number of complete sample lines already written to the pool file."""
    if not os.path.exists(samples_path):
        return 0
    n = 0
    with open(samples_path) as f:
        for _ in f:
            n += 1
    return n


def _weighted_score(sample: dict, w1: float, w2: float, w4: float) -> float:
    """Reporting scalar: w1*f1_norm + w2*f2 + w4*f4. NaN-safe via np.nan propagation."""
    f1n = float(sample.get("f1_norm", float("nan")))
    f2  = float(sample.get("f2",      float("nan")))
    f4  = float(sample.get("f4",      0.0))
    if not (np.isfinite(f1n) and np.isfinite(f2)):
        return float("inf")
    return w1 * f1n + w2 * f2 + w4 * f4


def _summarise_pool(samples_path: str, w1: float, w2: float, w4: float,
                    log_every: int = 50) -> tuple:
    """Read the full sample pool and return (best_sample, best_score, log, stats).

    best_sample is the raw sample dict with the lowest weighted score among
    valid (non-failed) samples. log is the best-so-far trajectory sampled
    every `log_every` draws. stats counts attempts / failures.
    """
    best: dict | None = None
    best_score = float("inf")
    log: list = []
    attempts = 0
    failures = 0

    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            attempts += 1
            if s.get("failed", False):
                failures += 1
            else:
                score = _weighted_score(s, w1, w2, w4)
                if score < best_score:
                    best_score = score
                    best = s
            if attempts % log_every == 0:
                log.append({
                    "sample":     attempts,
                    "best_f_sum": best_score if best is not None else float("inf"),
                    "best_f1_mm": float(best["f1_mm"]) if best else float("nan"),
                    "best_f2":    float(best["f2"])    if best else float("nan"),
                })
    # Always include the final point
    if attempts > 0 and (not log or log[-1]["sample"] != attempts):
        log.append({
            "sample":     attempts,
            "best_f_sum": best_score if best is not None else float("inf"),
            "best_f1_mm": float(best["f1_mm"]) if best else float("nan"),
            "best_f2":    float(best["f2"])    if best else float("nan"),
        })

    return best, best_score, log, {"attempts": attempts, "failures": failures}


def extend_sample_pool(inst: GAInstance, evaluator: RealEvaluator,
                      samples_path: str, budget: int, seed_base: int) -> None:
    """Extend the on-disk sample pool for this seed up to `budget` total samples.

    Samples are identified by a per-sample RNG (seed_base * _SEED_STRIDE + idx),
    so sample idx K is deterministic and independent of restart. Every sample
    is appended as one JSON line to `samples_path` and flushed immediately,
    so a crashed run resumes from the next unwritten idx without re-evaluating.
    Raw per-sample values (f1_mm, f1_norm, f2, f4) are stored, so the pool can
    be re-scored under any reporting weight scheme post-hoc.
    """
    start_idx = _count_existing_samples(samples_path)
    if start_idx >= budget:
        print(f"[B3]   pool already at {start_idx} samples >= budget {budget}; "
              f"no new evaluations.")
        return
    print(f"[B3]   extending pool from {start_idx} to {budget} samples...")

    cfg_stub = GAConfig(seed=0, weight_sigma=0.20)  # seed unused; per-sample RNG
    with open(samples_path, "a") as f_out:
        for idx in range(start_idx, budget):
            sample_rng = random.Random(seed_base * _SEED_STRIDE + idx)
            g = random_genome(inst, sample_rng, cfg_stub)
            ind = Individual(genome=g)
            ind.fitness = evaluator(ind)

            # Raw metrics from meta; absent on geometry failure.
            failed = "f1_height_mm" not in ind.meta
            f1_mm   = float(ind.meta.get("f1_height_mm", float("nan")))
            f1_norm = float(ind.meta.get("f1_norm",      float("nan")))
            f2      = float(ind.meta.get("f2_phase",     float("nan")))
            f4      = float(ind.meta.get("f4_area_dev",  float("nan")))

            record = {
                "idx":     idx,
                "delta":   g.delta.tolist(),
                "kappa":   g.kappa.tolist(),
                "f1_mm":   f1_mm,
                "f1_norm": f1_norm,
                "f2":      f2,
                "f4":      f4,
                "failed":  failed,
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

            done = idx + 1
            if done % 50 == 0 or done == budget:
                print(f"[B3]   sample {done}/{budget}  failed_so_far={failed}")


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
    bodies_base = os.path.join("results", "experiments", exp_name, "b3",
                               f"bodies_{num_bodies}")
    latest = _find_latest_run_dir(bodies_base)

    if latest is not None:
        out_dir      = latest
        results_path = os.path.join(out_dir, "results.json")
        conv_path    = os.path.join(out_dir, "convergence.json")
        with open(results_path) as f:
            results = json.load(f)
        convergence = json.load(open(conv_path)) if os.path.exists(conv_path) else []
        completed = {r["seed"] for r in results.get("b3", [])}
        print(f"[B3] bodies={num_bodies}: resumed '{out_dir}' "
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
                "method":         "b3_random_search",
                "runs":           args.runs,
                "samples":        args.samples,
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
            },
            "b3": [],
        }
        convergence = []
        completed = set()
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[B3] bodies={num_bodies}: fresh start at '{out_dir}'")

    evaluator, inst, eval_cfg = _build_instance_for(num_bodies, args)

    return {
        "num_bodies":   num_bodies,
        "total_runs":   args.runs,
        "samples":      args.samples,
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


def _run_one_seed(state: dict, seed: int, args) -> None:
    """Extend the sample pool for this seed up to the current budget, then
    recompute the weighted best from the full pool and write results.json.

    The seed is re-processed even if already in state["completed"]: the
    extension-and-rescoring is idempotent when the budget is unchanged, and
    picks up new samples when the budget has grown.
    """
    num_bodies = state["num_bodies"]
    budget     = state["samples"]
    total_runs = state["total_runs"]
    done       = len(state["completed"])

    samples_path = os.path.join(state["out_dir"], f"samples_seed{seed}.jsonl")
    n_existing = _count_existing_samples(samples_path)

    print("\n" + "=" * 60)
    print(f"[B3] bodies={num_bodies}  seed={seed}  budget={budget}  "
          f"pool={n_existing}  ({done}/{total_runs} seeds at-or-above-budget)")
    print("=" * 60)

    # 1. Extend the pool on disk to the current budget (no-op if already there).
    extend_sample_pool(
        inst=state["inst"],
        evaluator=state["evaluator"],
        samples_path=samples_path,
        budget=budget,
        seed_base=seed,
    )

    # 2. Read back the full pool and compute the weighted best under the
    #    reporting weights w1, w2, w4. These are reporting-only parameters;
    #    they do not affect which samples were drawn.
    best_sample, best_score, log, stats = _summarise_pool(
        samples_path=samples_path,
        w1=args.w1,
        w2=args.w2,
        w4=args.w4,
    )

    if best_sample is None:
        print(f"[B3] bodies={num_bodies}  seed={seed}: "
              f"no valid sample in pool of {stats['attempts']}")
        return

    f1_mm   = float(best_sample["f1_mm"])
    f1_norm = float(best_sample["f1_norm"])
    f2      = float(best_sample["f2"])
    f4      = float(best_sample["f4"])

    run_metrics = {
        "seed":     seed,
        "f1_mm":    f1_mm,
        "f1_norm":  f1_norm,
        "f2":       f2,
        "f4":       f4,
        "f_sum":    f1_norm + f2,  # unweighted paper-style reporting
        "f_sum_weighted": best_score,
        "best_idx": int(best_sample["idx"]),
        "attempts": int(stats["attempts"]),
        "failures": int(stats["failures"]),
    }

    # Replace existing entry for this seed (if any) rather than appending.
    state["results"]["b3"] = [r for r in state["results"]["b3"] if r["seed"] != seed]
    state["results"]["b3"].append(run_metrics)
    state["results"]["b3"].sort(key=lambda r: r["seed"])

    state["convergence"] = [c for c in state["convergence"] if c["seed"] != seed]
    state["convergence"].append({"seed": seed, "log": log})
    state["convergence"].sort(key=lambda c: c["seed"])

    # Record that this seed is at-or-above the current budget.
    if stats["attempts"] >= budget:
        state["completed"].add(seed)

    with open(state["results_path"], "w") as f:
        json.dump(state["results"], f, indent=2)
    with open(state["conv_path"], "w") as f:
        json.dump(state["convergence"], f, indent=2)

    done = len(state["completed"])
    f4_str = (f"{run_metrics['f4']:.4f}"
              if not np.isnan(run_metrics["f4"]) else "n/a")
    att = run_metrics["attempts"]
    fail = run_metrics["failures"]
    fail_rate = (fail / att) if att > 0 else 0.0
    print(f"[B3] bodies={num_bodies}  seed={seed}"
          f"  f1={f1_mm:.1f}mm  f2={f2:.4f}  f4={f4_str}  f_sum={run_metrics['f_sum']:.4f}"
          f"  valid={att-fail}/{att} ({100*(1-fail_rate):.0f}%)"
          f"  ({done}/{total_runs} done)")


def main():
    args = parse_args()
    print(f"[B3] garment={args.garment_type}  num_bodies={args.num_bodies}"
          f"  runs={args.runs}  samples={args.samples}")

    print("\n[B3] Initialising body-count configurations...")
    states = {}
    for num_bodies in args.num_bodies:
        states[num_bodies] = _init_body_run(num_bodies, args)

    # Interleaved: outer = seed, inner = body count. Same pattern as
    # run_experiments.py so partial runs accumulate one result per body
    # count before deepening.
    #
    # Note: _run_one_seed is idempotent with respect to the current budget.
    # Re-running the same command with a larger --samples extends each
    # seed's pool in-place and recomputes the weighted best from the full
    # pool. No prior results are discarded.
    for seed in range(args.runs):
        print(f"\n{'#' * 60}")
        print(f"# seed={seed}")
        print(f"{'#' * 60}")
        for num_bodies in args.num_bodies:
            state = states[num_bodies]
            _run_one_seed(state, seed, args)

    print("\n[B3] All runs complete.")


if __name__ == "__main__":
    main()
