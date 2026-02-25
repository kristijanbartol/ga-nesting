# ga_spec.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple
import numpy as np
import random


@dataclass(frozen=True)
class GAInstance:
    """
    Minimal metadata for sampling/mutating genomes.
    We keep some fields optional/fixed to enable incremental rollout.
    """
    num_patches: int
    num_internal_seams: int
    K: int  # phase bins
    num_landmarks: int = 0  # needed for delta dimensionality (2 * num_landmarks)

    # Constrain delta sampling and mutation to [delta_lo, delta_hi] around the
    # 0.5 baseline, reducing the chance of landmarks landing in geometrically
    # degenerate mesh regions. Full range is [0.0, 1.0].
    delta_lo: float = 0.4
    delta_hi: float = 0.6

    # Optional: fixed values to keep pipeline simple initially
    fixed_delta: Optional[np.ndarray] = None
    fixed_rho: Optional[np.ndarray] = None
    fixed_pi: Optional[np.ndarray] = None
    fixed_h: Optional[int] = 0
    num_heuristics: int = 3


@dataclass
class Genome:
    """
    Minimal subset we will actually optimize now:
      - kappa (per patch)
      - w (per seam)

    We keep placeholders for future:
      delta, rho, pi, h
    """
    delta: np.ndarray
    rho: np.ndarray
    kappa: np.ndarray
    w: np.ndarray
    pi: np.ndarray
    h: int


@dataclass
class Fitness:
    values: np.ndarray  # (3,) [f1,f2,f3], lower is better

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        if self.values.shape != (3,):
            raise ValueError("Fitness must be shape (3,)")


@dataclass
class Individual:
    genome: Genome
    fitness: Optional[Fitness] = None
    meta: Dict[str, object] = field(default_factory=dict)


class Evaluator(Protocol):
    def __call__(self, ind: Individual) -> Fitness: ...


@dataclass(frozen=True)
class GAConfig:
    seed: int = 0
    population_size: int = 6
    generations: int = 2

    elite_count: int = 1
    tournament_k: int = 3

    crossover_prob: float = 0.7
    mutation_prob: float = 0.6

    # mutation parameters (only for variables we optimize now)
    prob_flip_kappa: float = 0.2
    prob_flip_rho: float = 0.15
    prob_swap_pi: float = 0.3
    prob_flip_h: float = 0.25
    delta_sigma: float = 0.05   # step size for delta mutation — kept small relative to [delta_lo, delta_hi]
    weight_sigma: float = 0.15  # step size for seam weight (w) mutation


def dominates(a: Fitness, b: Fitness) -> bool:
    av, bv = a.values, b.values
    return np.all(av <= bv) and np.any(av < bv)


def tournament_select(pop: Sequence[Individual], k: int, rng: random.Random) -> Individual:
    cand = rng.sample(list(pop), k)
    best = cand[0]
    for c in cand[1:]:
        if dominates(c.fitness, best.fitness):  # type: ignore
            best = c
        elif not dominates(best.fitness, c.fitness):  # type: ignore
            if c.fitness.values.sum() < best.fitness.values.sum():  # type: ignore
                best = c
    return best


def random_genome(inst: GAInstance, rng: random.Random) -> Genome:
    num_p = inst.num_patches
    num_s = inst.num_internal_seams

    delta = inst.fixed_delta.copy() if inst.fixed_delta is not None else \
        np.array([rng.uniform(inst.delta_lo, inst.delta_hi) for _ in range(2 * inst.num_landmarks)], dtype=float)

    # Discrete grain rotations (rho): 0..3 (multiples of 90deg)
    if inst.fixed_rho is not None:
        rho = inst.fixed_rho.copy()
    else:
        rho = np.array([rng.randrange(4) for _ in range(num_p)], dtype=int)

    # Placement order (pi): permutation of patch indices
    if inst.fixed_pi is not None:
        pi = inst.fixed_pi.copy()
    else:
        pi = np.array(rng.sample(list(range(num_p)), k=num_p), dtype=int)

    if inst.fixed_h is not None:
        h = int(inst.fixed_h)
    else:
        H = max(1, int(inst.num_heuristics))
        h = int(rng.randrange(H))

    kappa = np.array([rng.randrange(inst.K) for _ in range(num_p)], dtype=int)
    w = np.array([rng.random() for _ in range(num_s)], dtype=float)

    return Genome(delta=delta, rho=rho, kappa=kappa, w=w, pi=pi, h=h)



def crossover(g1: Genome, g2: Genome, inst: GAInstance, rng: random.Random) -> Tuple[Genome, Genome]:
    """Crossover that preserves discrete constraints.

    - kappa: uniform per gene
    - w: uniform per gene
    - rho: uniform per gene (0..3)
    - pi: choose whole permutation from one parent (keeps validity)
    """
    # delta: uniform per gene, clamp to [0,1]
    d1, d2 = g1.delta.copy(), g2.delta.copy()
    if inst.fixed_delta is None:
        for i in range(d1.size):
            if rng.random() < 0.5:
                d1[i], d2[i] = d2[i], d1[i]

    # kappa
    k1, k2 = g1.kappa.copy(), g2.kappa.copy()
    for i in range(k1.size):
        if rng.random() < 0.5:
            k1[i], k2[i] = k2[i], k1[i]

    # w
    w1, w2 = g1.w.copy(), g2.w.copy()
    for i in range(w1.size):
        if rng.random() < 0.5:
            w1[i], w2[i] = w2[i], w1[i]

    # rho
    r1, r2 = g1.rho.copy(), g2.rho.copy()
    for i in range(r1.size):
        if rng.random() < 0.5:
            r1[i], r2[i] = r2[i], r1[i]

    # pi (whole-permutation choice to keep it a permutation)
    if rng.random() < 0.5:
        p1, p2 = g1.pi.copy(), g2.pi.copy()
    else:
        p1, p2 = g2.pi.copy(), g1.pi.copy()

    if rng.random() < 0.5:
        h1, h2 = g1.h, g2.h
    else:
        h1, h2 = g2.h, g1.h

    c1 = Genome(delta=d1, rho=r1, kappa=k1, w=w1, pi=p1, h=int(h1))
    c2 = Genome(delta=d2, rho=r2, kappa=k2, w=w2, pi=p2, h=int(h2))
    return c1, c2



def mutate(g: Genome, inst: GAInstance, cfg: GAConfig, rng: random.Random) -> Genome:
    kappa = g.kappa.copy()
    rho = g.rho.copy()
    pi = g.pi.copy()
    w = g.w.copy()
    delta = g.delta.copy()

    # gaussian noise on delta, clamp to allowed range
    if inst.fixed_delta is None and delta.size > 0:
        delta = delta + np.random.normal(0.0, cfg.delta_sigma, size=delta.shape)
        delta = np.clip(delta, inst.delta_lo, inst.delta_hi)

    # flip some kappas
    for i in range(kappa.size):
        if rng.random() < cfg.prob_flip_kappa:
            kappa[i] = rng.randrange(inst.K)

    # flip some rhos (0..3)
    if inst.fixed_rho is None:
        for i in range(rho.size):
            if rng.random() < cfg.prob_flip_rho:
                rho[i] = rng.randrange(4)

    # swap mutation on permutation pi
    if inst.fixed_pi is None and pi.size >= 2 and rng.random() < cfg.prob_swap_pi:
        a = rng.randrange(pi.size)
        b = rng.randrange(pi.size)
        while b == a:
            b = rng.randrange(pi.size)
        pi[a], pi[b] = pi[b], pi[a]

    # gaussian noise on weights, clamp to [0,1]
    w = w + np.random.normal(0.0, cfg.weight_sigma, size=w.shape)
    w = np.clip(w, 0.0, 1.0)

    h = int(g.h)
    if inst.fixed_h is None and rng.random() < cfg.prob_flip_h:
        H = max(1, int(inst.num_heuristics))
        h = int(rng.randrange(H))

    return Genome(delta=delta, rho=rho, kappa=kappa, w=w, pi=pi, h=h)


def _pop_table(pop: List[Individual], label: str) -> None:
    pop_sorted = sorted(pop, key=lambda ind: ind.fitness.values.sum())  # type: ignore
    print(f"\n  {label}")
    print(f"  {'':1}{'#':>3}  {'sum':>9}  {'f1(mm)':>10}  {'f2':>8}  kappa")
    for i, ind in enumerate(pop_sorted):
        marker = "*" if i == 0 else " "
        f = ind.fitness.values  # type: ignore
        # Use raw (unweighted) values from meta when available; fall back to
        # the weighted fitness component only for penalty/failed individuals.
        f1 = ind.meta.get("f1_height_mm", f[0])
        f2 = ind.meta.get("f2_phase", f[1])
        kappa = ind.genome.kappa.tolist()
        print(f"  {marker}{i:>3}  {f.sum():>9.4f}  {f1:>10.2f}  {f2:>8.4f}  {kappa}")


def evaluate_population(pop: List[Individual], evaluator: Evaluator) -> None:
    total = len(pop)
    for i, ind in enumerate(pop):
        if ind.fitness is None:
            kappa = ind.genome.kappa.tolist()
            delta_mean = float(np.mean(ind.genome.delta)) if ind.genome.delta.size > 0 else float("nan")
            w = np.round(ind.genome.w, 2).tolist()
            print(f"  [{i+1}/{total}] kappa={kappa}  delta_mean={delta_mean:.2f}  w={w}")
            ind.fitness = evaluator(ind)
            f = ind.fitness.values
            f1 = ind.meta.get("f1_height_mm", f[0])
            f2 = ind.meta.get("f2_phase", f[1])
            print(f"         -> f1={f1:.2f}mm  f2={f2:.4f}  sum={f.sum():.4f}")


def run_ga(inst: GAInstance, evaluator: Evaluator, cfg: GAConfig) -> List[Individual]:
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    print("=== GA Config ===")
    print(f"  pop={cfg.population_size}  gens={cfg.generations}  K={inst.K}"
          f"  elite={cfg.elite_count}  tournament_k={cfg.tournament_k}")
    print(f"  crossover={cfg.crossover_prob}  mutation={cfg.mutation_prob}"
          f"  prob_flip_kappa={cfg.prob_flip_kappa}  weight_sigma={cfg.weight_sigma}")
    print(f"  fixed: delta={'no' if inst.fixed_delta is None else 'yes'}"
          f"  rho={'no' if inst.fixed_rho is None else 'yes'}"
          f"  pi={'no' if inst.fixed_pi is None else 'yes'}"
          f"  h={inst.fixed_h if inst.fixed_h is not None else 'no'}")
    if inst.fixed_delta is None:
        print(f"  delta range: [{inst.delta_lo}, {inst.delta_hi}]"
              f"  sigma={cfg.delta_sigma}"
              f"  (perturbation ±{inst.delta_hi - 0.5:.2f} around 0.5 baseline)")

    print("\n=== Initial Population ===")
    pop = [Individual(genome=random_genome(inst, rng)) for _ in range(cfg.population_size)]
    evaluate_population(pop, evaluator)
    _pop_table(pop, "Initial population:")

    prev_best_sum = min(ind.fitness.values.sum() for ind in pop)  # type: ignore

    for gen in range(cfg.generations):
        print(f"\n=== Gen {gen + 1}/{cfg.generations} ===")

        pop_sorted = sorted(pop, key=lambda ind: ind.fitness.values.sum())  # type: ignore
        elites = [Individual(genome=e.genome, fitness=e.fitness, meta=dict(e.meta)) for e in pop_sorted[:cfg.elite_count]]

        new_pop: List[Individual] = []
        new_pop.extend(elites)

        while len(new_pop) < cfg.population_size:
            p1 = tournament_select(pop, cfg.tournament_k, rng)
            p2 = tournament_select(pop, cfg.tournament_k, rng)

            if rng.random() < cfg.crossover_prob:
                c1g, c2g = crossover(p1.genome, p2.genome, inst, rng)
            else:
                c1g, c2g = p1.genome, p2.genome

            if rng.random() < cfg.mutation_prob:
                c1g = mutate(c1g, inst, cfg, rng)
            if rng.random() < cfg.mutation_prob:
                c2g = mutate(c2g, inst, cfg, rng)

            new_pop.append(Individual(genome=c1g))
            if len(new_pop) < cfg.population_size:
                new_pop.append(Individual(genome=c2g))

        pop = new_pop
        evaluate_population(pop, evaluator)

        best_sum = min(ind.fitness.values.sum() for ind in pop)  # type: ignore
        delta = prev_best_sum - best_sum
        improvement = f"improved by {delta:.4f}" if delta > 1e-9 else "no improvement"
        _pop_table(pop, f"Gen {gen + 1} population  ({improvement}):")
        prev_best_sum = best_sum

    return pop