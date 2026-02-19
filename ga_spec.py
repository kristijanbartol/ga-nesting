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

    # Optional: fixed values to keep pipeline simple initially
    fixed_delta: Optional[np.ndarray] = None
    fixed_rho: Optional[np.ndarray] = None
    fixed_pi: Optional[np.ndarray] = None
    fixed_h: Optional[int] = 0


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
    weight_sigma: float = 0.15


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

    delta = inst.fixed_delta.copy() if inst.fixed_delta is not None else np.zeros((0,), dtype=float)
    rho   = inst.fixed_rho.copy()   if inst.fixed_rho is not None else np.zeros((num_p,), dtype=int)
    pi    = inst.fixed_pi.copy()    if inst.fixed_pi is not None else np.arange(num_p, dtype=int)
    h     = int(inst.fixed_h) if inst.fixed_h is not None else 0

    kappa = np.array([rng.randrange(inst.K) for _ in range(num_p)], dtype=int)
    w = np.array([rng.random() for _ in range(num_s)], dtype=float)

    return Genome(delta=delta, rho=rho, kappa=kappa, w=w, pi=pi, h=h)


def crossover(g1: Genome, g2: Genome, inst: GAInstance, rng: random.Random) -> Tuple[Genome, Genome]:
    # uniform crossover on kappa and w (minimal)
    k1, k2 = g1.kappa.copy(), g2.kappa.copy()
    for i in range(k1.size):
        if rng.random() < 0.5:
            k1[i], k2[i] = k2[i], k1[i]

    w1, w2 = g1.w.copy(), g2.w.copy()
    for i in range(w1.size):
        if rng.random() < 0.5:
            w1[i], w2[i] = w2[i], w1[i]

    c1 = Genome(delta=g1.delta, rho=g1.rho, kappa=k1, w=w1, pi=g1.pi, h=g1.h)
    c2 = Genome(delta=g2.delta, rho=g2.rho, kappa=k2, w=w2, pi=g2.pi, h=g2.h)
    return c1, c2


def mutate(g: Genome, inst: GAInstance, cfg: GAConfig, rng: random.Random) -> Genome:
    kappa = g.kappa.copy()
    w = g.w.copy()

    # flip some kappas
    for i in range(kappa.size):
        if rng.random() < cfg.prob_flip_kappa:
            kappa[i] = rng.randrange(inst.K)

    # gaussian noise on weights, clamp to [0,1]
    w = w + np.random.normal(0.0, cfg.weight_sigma, size=w.shape)
    w = np.clip(w, 0.0, 1.0)

    return Genome(delta=g.delta, rho=g.rho, kappa=kappa, w=w, pi=g.pi, h=g.h)


def evaluate_population(pop: List[Individual], evaluator: Evaluator) -> None:
    for ind in pop:
        if ind.fitness is None:
            ind.fitness = evaluator(ind)


def run_ga(inst: GAInstance, evaluator: Evaluator, cfg: GAConfig) -> List[Individual]:
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    pop = [Individual(genome=random_genome(inst, rng)) for _ in range(cfg.population_size)]
    evaluate_population(pop, evaluator)

    for gen in range(cfg.generations):
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

        best = min(pop, key=lambda ind: ind.fitness.values.sum())  # type: ignore
        print(f"[GA] gen {gen:03d} best sum={best.fitness.values.sum():.6f} F={best.fitness.values}")  # type: ignore

    return pop
