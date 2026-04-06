# visualize_ga_effect.py
import json
import os
from copy import deepcopy
import numpy as np

from ga_spec import GAInstance, GAConfig, run_ga
from ga.real_evaluator import RealEvaluator, RealEvaluatorConfig
from nesting.loader import PatchLoader
from nesting.engine import NestingEngine
from nesting.vis_utils import visualize_layout, visualize_population

from nesting.phase_utils import Rigid2D
from nesting.stage2_global_align import solve_global_alignment_all_components
from nesting.stage2_global_align import load_seam_constraints_from_dir
from ga.real_evaluator import load_patch_vertices_full_from_latest


# TODO:
#   - Add fit loss component + include parameterization parameters into GA
#   - Add onesie (and dress)
#   - Update latex problem specification
#   - Write paper



def apply_kappa_to_items(items, genome, K, texture, pid_to_item_idx):
    """Apply kappa phase offsets using pid_to_item_idx (not pid-1) to handle non-sequential patch IDs."""
    tx, ty = texture.period_x, texture.period_y
    import re
    for it in items:
        m = re.search(r"patch_(\d+)", it.name)
        if not m:
            continue
        pid = int(m.group(1))
        idx = pid_to_item_idx.get(pid)
        if idx is None or idx >= genome.kappa.size:
            continue
        k = int(genome.kappa[idx])
        it.phase_offset = ((k / float(K)) * tx, (k / float(K)) * ty)


def nest_and_show(latest_root, seam_dir, lattice, texture, fabric_width, genome, K, title, garment_part,
                  seam_importance_by_name=None, num_bodies=1, show_layout=True,
                  precomputed_tsol=None, precomputed_kappas_by_id=None):
    print(f"\n[nest_and_show] '{title}'  rho={genome.rho.tolist()}  kappa={genome.kappa.tolist()}")
    loader = PatchLoader(latest_root, garment_part)
    base_items = loader.load_items()

    # Deterministic mapping patch_id <-> genome index
    import re
    def _pid(it):
        m = re.search(r"patch_(\d+)", it.name)
        return int(m.group(1)) if m else 10**9
    base_items = sorted(base_items, key=_pid)

    # Build pid -> item_idx mapping (do NOT use pid-1, patch IDs can be non-sequential).
    patch_ids_local = [_pid(it) for it in base_items]
    pid_to_item_idx = {pid: idx for idx, pid in enumerate(patch_ids_local)}
    M = len(base_items)

    # 1) Stage2: compute transforms from seam constraints + kappas + weights.
    # Stage2 operates on body-0's kappas; all bodies share the same geometry.
    import re as _re
    import os as _os

    def _weights_for_dir(d, name_to_imp):
        w = {}
        if not _os.path.isdir(d):
            return w
        for fn in _os.listdir(d):
            if not (fn.startswith("seam-") and fn.endswith(".txt")):
                continue
            m = _re.match(r"seam-(.+)_\d+-\d+\.txt$", fn)
            if m:
                w[fn] = name_to_imp.get(m.group(1), 0.0) if name_to_imp is not None else 1.0
        return w

    default_w = 0.0 if seam_importance_by_name is not None else 1.0
    constraints = load_seam_constraints_from_dir(
        seam_dir,
        weights_by_filename=_weights_for_dir(seam_dir, seam_importance_by_name),
        default_weight=default_w,
    )

    V_centered_by_id = load_patch_vertices_full_from_latest(
        latest_root, garment_part=garment_part, scale_mm=1000.0, center_by_boundary=True)
    patch_ids = sorted(V_centered_by_id.keys())

    # Use body-0 kappas for Stage2 (shared geometry across all bodies)
    kappas_by_id = (precomputed_kappas_by_id if precomputed_kappas_by_id is not None
                    else {pid: int(genome.kappa[pid_to_item_idx[pid]]) for pid in patch_ids if pid in pid_to_item_idx})

    weighted_constraints = list(constraints)

    if precomputed_tsol is not None:
        Tsol = precomputed_tsol
    else:
        T0 = {pid: Rigid2D(0.0, 0.0, 0.0) for pid in patch_ids}
        Tsol = solve_global_alignment_all_components(
            patch_ids=patch_ids,
            constraints=weighted_constraints,
            patch_vertices_by_id=V_centered_by_id,
            lattice=lattice,
            kappas_by_id=kappas_by_id,
            K=K,
            initial_transforms=T0,
            max_iters=15,
            verbose=False
        )

    # 2) Bake Stage2 transforms into base items (shared across all bodies)
    from shapely.geometry import Polygon as _Polygon
    from copy import deepcopy
    for it in base_items:
        pid = _pid(it)
        if pid >= 10**9:
            continue
        T = Tsol.get(pid, Rigid2D(0, 0, 0))
        it.original_vertices = T.apply(it.original_vertices)
        it.shape = _Polygon(it.original_vertices)
        it.current_rotation = 0.0

    # 3) Clone base items for each body, applying per-body kappa and rho
    tx, ty = texture.period_x, texture.period_y
    all_items = []
    for b in range(num_bodies):
        for item_idx, base_it in enumerate(base_items):
            it = deepcopy(base_it)
            it.name = f"body_{b}/{base_it.name}"
            genome_idx = b * M + item_idx

            rho_val = int(genome.rho[genome_idx]) % 4 if genome_idx < genome.rho.size else 0
            it.set_rotation(float(rho_val * 90))

            k = int(genome.kappa[genome_idx]) if genome_idx < genome.kappa.size else 0
            it.phase_offset = ((k / float(K)) * tx, (k / float(K)) * ty)

            all_items.append(it)

    # 4) Nest all N*M items + visualize
    eng = NestingEngine(fabric_width=fabric_width, texture_spec=texture)
    pi = None
    num_total = num_bodies * M
    if genome.pi is not None and getattr(genome.pi, "size", 0) == num_total:
        pi = [int(x) for x in genome.pi.tolist()]
    fabric = eng.nest(all_items, permutation=pi, heuristic=int(getattr(genome, "h", 0)))

    print(f"{title}: height={fabric.total_height:.2f}")
    if show_layout:
        visualize_layout(fabric, texture, title=title)

    # Plot per-seam phase mismatch (body-0 only — geometry is shared)
    from nesting.vis_utils import plot_seam_mismatch
    plot_seam_mismatch(weighted_constraints, V_centered_by_id, lattice, kappas_by_id, K, Tsol, title)

    return Tsol, kappas_by_id


def save_best_individual_data(best_ind,
                               garment_type: str,
                               latest_root: str = "results/pattern/latest",
                               seam_dir_base: str = "data/seamlines",
                               patches_3d_dir_base: str = "data/patches") -> tuple:
    """
    Write the best individual's patch geometry and seam files directly from
    ind.meta (snapshotted during evaluation) to stable best/ directories.
    This avoids re-running the non-deterministic C++ geometry pipeline, ensuring
    the saved patches are identical to what was used during evaluation.
    Returns (best_root, best_seam_dir, best_patches_3d_dir).
    """
    import shutil, os
    import trimesh as _trimesh

    best_root = os.path.join(os.path.dirname(latest_root), "best")

    # Write 2D patches (optim_final-seams.ply) from stored meta
    dst_2d = os.path.join(best_root, garment_type)
    if os.path.exists(dst_2d):
        shutil.rmtree(dst_2d)
    for dname, (verts, faces) in best_ind.meta["patches_2d_raw"].items():
        out_dir = os.path.join(dst_2d, dname)
        os.makedirs(out_dir, exist_ok=True)
        m = _trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        m.export(os.path.join(out_dir, 'optim_final-seams.ply'))

    # Write seam constraint files from the snapshotted meta (same evaluation as the patches).
    dst_seams = os.path.join(seam_dir_base, "best", garment_type)
    if os.path.exists(dst_seams):
        shutil.rmtree(dst_seams)
    os.makedirs(dst_seams)
    for fn, content in best_ind.meta.get("seam_files_raw", {}).items():
        with open(os.path.join(dst_seams, fn), 'w') as f:
            f.write(content)

    # Write 3D patches from stored meta
    dst_3d = os.path.join(patches_3d_dir_base, "best", garment_type)
    if os.path.exists(dst_3d):
        shutil.rmtree(dst_3d)
    for dname, (verts, faces, fname) in best_ind.meta["patches_3d_raw"].items():
        out_dir = os.path.join(dst_3d, dname)
        os.makedirs(out_dir, exist_ok=True)
        m = _trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        m.export(os.path.join(out_dir, fname))

    print(f"[main] Best individual data saved to '{dst_2d}', '{dst_seams}', '{dst_3d}'")
    return best_root, dst_seams, dst_3d


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Run GA-Nesting optimisation.")
    p.add_argument("--garment", default="upper", choices=["upper", "lower", "onesie_sleeves"],
                   help="Garment type (default: upper)")
    p.add_argument("--wallpaper", default="stripes", choices=["stripes", "diagonal_stripes", "grid", "p4", "p4m", "pg", "pmg", "pgg"],
                   help="Texture wallpaper group (default: stripes)")
    p.add_argument("--pop", type=int, default=50,
                   help="Population size (default: 50)")
    p.add_argument("--gens", type=int, default=1,
                   help="Number of generations (default: 1)")
    p.add_argument("--num_bodies", type=int, default=1,
                   help="Number of bodies (default: 1)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (default: 0)")
    p.add_argument("--w1", type=float, default=1.0,
                   help="Fitness weight for fabric height f1 (default: 1.0)")
    p.add_argument("--w2", type=float, default=1.0,
                   help="Fitness weight for seam phase mismatch f2 (default: 1.0)")
    p.add_argument("--w4", type=float, default=10.0,
                   help="Fitness weight for garment area reduction f4 (default: 10.0)")
    p.add_argument("--self_adapt", action="store_true", default=False,
                   help="Enable MIES-style per-gene self-adaptive sigma for delta")
    p.add_argument("--no_vis", action="store_true",
                   help="Skip all matplotlib visualization (for headless/batch runs)")
    return p.parse_args()


def main():
    args = parse_args()
    GARMENT_TYPE = args.garment

    eval_cfg = RealEvaluatorConfig(
        garment_part=GARMENT_TYPE,
        latest_root="results/pattern/latest",
        seam_dir=f"data/seamlines/{GARMENT_TYPE}",
        period_u_mm=50.0,
        period_v_mm=50.0,
        K=8,
        fabric_width_mm=150.0 * 10.0,
        num_bodies=args.num_bodies,
        wallpaper_group=args.wallpaper,
        w1=args.w1,
        w2=args.w2,
        w4=args.w4,
    )
    evaluator = RealEvaluator(eval_cfg)

    num_patches = len(evaluator.patch_ids)

    # Compute area-sorted-descending permutation (matches nesting engine default).
    import re as _re
    _loader = PatchLoader(eval_cfg.latest_root, GARMENT_TYPE)
    _items = _loader.load_items()
    _items_by_pid = sorted(_items, key=lambda it: int(
        _re.search(r"patch_(\d+)", it.name).group(1)))
    _area_sorted_pi = np.array(
        sorted(range(len(_items_by_pid)),
               key=lambda i: _items_by_pid[i].area, reverse=True),
        dtype=int,
    )

    # Fix rho=0 for all patches.  Optimizing rho is unsound for horizontal stripe
    # fabrics: rotating both patches by the same 90° preserves seam phase alignment
    # (same parity → penalty doesn't fire) but makes stripes run vertically on the
    # garment.  The GA exploits this trivially.  Grain rotation on stripe fabric
    # must always be 0; 180° flips can be re-enabled if needed.
    inst = GAInstance(
        num_patches=num_patches,
        K=eval_cfg.K,
        num_landmarks=evaluator.instance.num_sampled_landmarks,
        num_bodies=eval_cfg.num_bodies,
        fixed_rho=np.zeros(num_patches * eval_cfg.num_bodies, dtype=int),
        fixed_pi=_area_sorted_pi,
        fixed_h=0,
        num_heuristics=3,
    )

    cfg = GAConfig(
        seed=args.seed,
        population_size=args.pop,
        generations=args.gens,
        elite_count=4,
        tournament_k=4,
        crossover_prob=0.7,
        mutation_prob=0.7,
        prob_flip_kappa=0.35,
        weight_sigma=0.20,
        self_adapt_sigma=args.self_adapt,
    )

    pop, _ = run_ga(inst, evaluator, cfg)

    # Visualize all individuals ordered best → worst
    if not args.no_vis:
        visualize_population(pop, evaluator.instance.texture, title="Final population — best (top-left) to worst (bottom-right)")

    # Sort population the same way visualize_population does, then compare.
    sorted_pop = sorted(
        [ind for ind in pop if ind.fitness is not None],
        key=lambda ind: ind.fitness.values.sum()
    )
    pop_best = sorted_pop[0]

    best = min(pop, key=lambda ind: ind.fitness.values.sum())

    print("\n=== SELECTION DIAGNOSTIC ===")
    print(f"  pop_best  fitness sum = {pop_best.fitness.values.sum():.6f}  "
          f"values = {pop_best.fitness.values}  rho = {pop_best.genome.rho.tolist()}  kappa = {pop_best.genome.kappa.tolist()}")
    print(f"  best      fitness sum = {best.fitness.values.sum():.6f}  "
          f"values = {best.fitness.values}  rho = {best.genome.rho.tolist()}  kappa = {best.genome.kappa.tolist()}")
    print(f"  Same individual? {pop_best is best}")
    print("============================\n")

    # Build a baseline genome: kappa=0 for all
    base = deepcopy(best.genome)
    base.kappa[:] = 0

    # Write the best individual's patches directly from ind.meta (snapshotted during
    # evaluation) — no geometry re-run needed, so there is no risk of the
    # non-deterministic C++ pipeline producing different patches.
    print("\n[main] Saving best individual patches from evaluation snapshot...")
    best_root, best_seam_dir, best_patches_3d_dir = save_best_individual_data(
        best, GARMENT_TYPE, eval_cfg.latest_root, "data/seamlines")

    best_genome_path = os.path.join(best_root, "best_individual.json")
    _Tsol = best.meta["Tsol"]
    _kappas = best.meta["kappas_by_id"]
    with open(best_genome_path, 'w') as _f:
        json.dump({
            "kappa":          best.genome.kappa.tolist(),
            "rho":            best.genome.rho.tolist(),
            "delta":          best.genome.delta.tolist(),
            "sigma":          best.genome.sigma.tolist(),
            "pi":             best.genome.pi.tolist(),
            "h":              int(best.genome.h),
            "K":              eval_cfg.K,
            "period_u_mm":    eval_cfg.period_u_mm,
            "period_v_mm":    eval_cfg.period_v_mm,
            "fabric_width_mm": eval_cfg.fabric_width_mm,
            "garment_part":   GARMENT_TYPE,
            "wallpaper_group": eval_cfg.wallpaper_group,
            "num_bodies":     eval_cfg.num_bodies,
            "best_root":      best_root,
            "best_seam_dir":  best_seam_dir,
            "best_patches_3d_dir": best_patches_3d_dir,
            "Tsol":           {str(pid): [T.theta, T.tx, T.ty]
                               for pid, T in _Tsol.items()},
            "kappas_by_id":   {str(pid): int(k)
                               for pid, k in _kappas.items()},
        }, _f, indent=2)
    print(f"[main] Best individual genome saved to '{best_genome_path}'")

    from nesting.vis_utils import visualize_layout, plot_seam_mismatch
    from nesting.stage2_global_align import solve_global_alignment_all_components

    # ── BASELINE (kappa=0) ──────────────────────────────────────────────────
    # Use the same stored patches as the best individual; only kappas differ.
    patch_ids_best = sorted(best.meta["V_centered_by_id"].keys())
    baseline_kappas = {pid: 0 for pid in patch_ids_best}
    if not args.no_vis:
        Tsol_base = solve_global_alignment_all_components(
            patch_ids=patch_ids_best,
            constraints=best.meta["weighted_constraints"],
            patch_vertices_by_id=best.meta["V_centered_by_id"],
            lattice=evaluator.lattice,
            kappas_by_id=baseline_kappas,
            K=eval_cfg.K,
            initial_transforms={pid: Rigid2D(0.0, 0.0, 0.0) for pid in patch_ids_best},
            max_iters=15,
            verbose=False,
        )
        # Build baseline fabric_state from the stored Stage2-baked items — same
        # geometry as the best individual, kappa=0, no disk I/O.
        base_fabric_items = []
        for b in range(eval_cfg.num_bodies):
            for item_idx, base_it in enumerate(best.meta["base_items"]):
                it = deepcopy(base_it)
                it.name = f"body_{b}/{base_it.name}"
                it.phase_offset = (0.0, 0.0)
                it.set_rotation(0.0)
                base_fabric_items.append(it)
        base_engine = NestingEngine(fabric_width=eval_cfg.fabric_width_mm, texture_spec=evaluator.instance.texture)
        baseline_fabric_state = base_engine.nest(base_fabric_items)
        visualize_layout(baseline_fabric_state, evaluator.instance.texture, title="BASELINE (kappa=0)")
        plot_seam_mismatch(
            best.meta["weighted_constraints"],
            best.meta["V_centered_by_id"],
            evaluator.lattice,
            baseline_kappas,
            eval_cfg.K,
            Tsol_base,
            "BASELINE (kappa=0)",
        )

        # ── BEST (GA kappa) ─────────────────────────────────────────────────────
        # Both nesting layout and seam analysis come from stored evaluation data —
        # guaranteed to be the same individual as the population top-left thumbnail.
        visualize_layout(best.meta["fabric_state"], evaluator.instance.texture, title="BEST (GA kappa)")
        plot_seam_mismatch(
            best.meta["weighted_constraints"],
            best.meta["V_centered_by_id"],
            evaluator.lattice,
            best.meta["kappas_by_id"],
            eval_cfg.K,
            best.meta["Tsol"],
            "BEST (GA kappa)",
        )

    Tsol = best.meta["Tsol"]
    kappas_by_id = best.meta["kappas_by_id"]

    # Run cloth simulation for the best individual, passing nesting UV transforms
    print("\n[main] Running cloth simulation for best individual...")
    from geometry.simulation import run_headless_simulation
    run_headless_simulation(
        #avatar='data/SMPL_FEMALE_POSED.ply',
        avatar='data/SMPL_FEMALE.ply',
        garment_type=GARMENT_TYPE,
        tsol=Tsol,
        kappas_by_id=kappas_by_id,
        K=eval_cfg.K,
        period_u_mm=eval_cfg.period_u_mm,
        period_v_mm=eval_cfg.period_v_mm,
        pattern_root=best_root,
        patches_dir=best_patches_3d_dir,
    )


if __name__ == "__main__":
    main()