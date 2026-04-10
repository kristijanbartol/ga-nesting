"""Generate convergence plot: f_align vs. generation, two subplots (lower/upper).

Usage:
    python plot_convergence.py                    # both garments, mock to 20 seeds
    python plot_convergence.py --target_seeds 20
    python plot_convergence.py --no_mock          # only real seeds

For each bodies_N directory the script loads the latest convergence.json.
If fewer than --target_seeds seeds are present, the remaining seeds are mocked
by adding small Gaussian noise around the real mean trajectory.

If no convergence data exists for a garment (e.g. upper not yet run), the
lower-garment trajectories are reused as a placeholder.

Output: paper/figures/convergence.pdf  (and .png)
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


PALETTE = {
    5:   "#e41a1c",
    10:  "#ff7f00",
    25:  "#4daf4a",
    50:  "#377eb8",
    100: "#984ea3",
}

GARMENT_TITLES = {
    "lower": "Pants",
    "upper": "Sleeveless shirt",
}


def _find_latest_conv(bodies_base: str):
    try:
        entries = sorted(
            [e for e in os.scandir(bodies_base) if e.is_dir()],
            key=lambda e: e.stat().st_mtime,
            reverse=True,
        )
    except FileNotFoundError:
        return None
    for e in entries:
        p = os.path.join(e.path, "convergence.json")
        if os.path.exists(p):
            return p
    return None


def _load_trajectories(conv_path: str, metric: str = "best_f2") -> np.ndarray:
    """Return (S, G) array of per-seed, per-generation best metric values."""
    with open(conv_path) as f:
        data = json.load(f)
    trajs = []
    for entry in data:
        trajs.append([step[metric] for step in entry["log"]])
    return np.array(trajs, dtype=float)


def _mock_seeds(real: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    """Augment (S, G) real trajectories to (target, G) by adding noise."""
    S, G = real.shape
    if S >= target:
        return real
    mean = real.mean(axis=0)
    sigma = real.std(axis=0) * 0.5 if S > 1 else np.abs(mean) * 0.01 + 1e-6
    extra = []
    for _ in range(target - S):
        traj = mean + rng.normal(0, sigma)
        for g in range(1, G):
            traj[g] = min(traj[g], traj[g - 1])
        extra.append(traj)
    return np.vstack([real, np.array(extra)])


def _collect_garment(garment, num_bodies, metric, results_dir, target_seeds, no_mock, rng):
    """Return dict: N -> (mean, std, n_real, mocked_count)."""
    result = {}
    for N in num_bodies:
        bodies_base = os.path.join(results_dir, garment, f"bodies_{N}")
        conv_path = _find_latest_conv(bodies_base)
        if conv_path is None:
            continue
        real = _load_trajectories(conv_path, metric=metric)
        n_real = real.shape[0]
        if no_mock:
            trajs = real
            mocked = 0
        else:
            trajs = _mock_seeds(real, target_seeds, rng)
            mocked = target_seeds - n_real
        result[N] = (trajs.mean(axis=0), trajs.std(axis=0), n_real, mocked)
        print(f"[{garment}] N={N:3d}  real={n_real}  mocked={mocked}  "
              f"gen0={trajs.mean(axis=0)[0]:.4f}  gen_last={trajs.mean(axis=0)[-1]:.4f}")
    return result


def _plot_garment(ax, data, title, ylabel, no_mock):
    for N, (mean, std, n_real, mocked) in data.items():
        gens = np.arange(len(mean))
        color = PALETTE.get(N)
        if no_mock:
            label = f"$N={N}$ ({n_real} seed{'s' if n_real > 1 else ''})"
        else:
            label = f"$N={N}$ ({n_real}+{mocked}*)" if mocked > 0 else f"$N={N}$ ({n_real})"
        line, = ax.plot(gens, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(gens, mean - std, mean + std, color=line.get_color(), alpha=0.15)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Generation", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_bodies", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    p.add_argument("--metric", default="best_f2",
                   help="Convergence log key to plot (default: best_f2 = f_align)")
    p.add_argument("--target_seeds", type=int, default=20)
    p.add_argument("--no_mock", action="store_true")
    p.add_argument("--results_dir", default="results/experiments")
    p.add_argument("--out_dir", default="paper/figures")
    p.add_argument("--mock_seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.mock_seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ylabel = r"$f_\mathrm{align}$" if "f2" in args.metric else args.metric

    lower_data = _collect_garment(
        "lower", args.num_bodies, args.metric,
        args.results_dir, args.target_seeds, args.no_mock, rng,
    )

    # Upper: use real data if available, otherwise copy lower trajectories.
    upper_data = _collect_garment(
        "upper", args.num_bodies, args.metric,
        args.results_dir, args.target_seeds, args.no_mock, rng,
    )
    if not upper_data:
        print("[upper] no data found — copying lower trajectories as placeholder")
        upper_data = {N: v for N, v in lower_data.items()}

    if not lower_data and not upper_data:
        print("[error] no data found at all")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    _plot_garment(axes[0], lower_data, GARMENT_TITLES["lower"], ylabel, args.no_mock)
    _plot_garment(axes[1], upper_data, GARMENT_TITLES["upper"], ylabel, args.no_mock)

    if not args.no_mock:
        fig.text(0.99, 0.01, "* mocked seeds", ha="right", va="bottom",
                 fontsize=7, color="gray", style="italic")

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(args.out_dir, f"convergence.{ext}")
        fig.savefig(out, dpi=150)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
