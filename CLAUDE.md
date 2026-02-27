# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

GA-Nesting optimizes garment pattern layout on fabric. It combines three stages:
1. **Geometry**: Cuts a 3D SMPL body mesh along seam lines → extracts 2D garment panels (patches)
2. **GA Optimization**: Searches for seam positions and texture phase assignments that minimize fabric waste and stripe misalignment
3. **Nesting**: Places 2D patches on a fabric roll using a bottom-left heuristic with collision detection

## Commands

```bash
# Run full GA optimization pipeline
python run_ga.py

# Run geometry processing standalone (cut mesh → export → parameterize)
python run_geometry.py

# Run nesting only (assumes patches already exist in results/pattern/latest/)
python run_nesting.py

# Run stage2 global alignment solver standalone
python run_global_align.py

# Run tests
pytest

# Run a single test file
pytest tests/test_phase_utils.py
```

## Architecture

### Data Flow: Genotype → Phenotype → Fitness

```
Genome fields:
  delta  (2*M floats, [0,1])  → LandmarkMapper → 3D seam vertices → mesh cut
  rho    (M ints, 0..3)       → grain rotation (90° steps) in nesting
  kappa  (M ints, 0..K-1)    → discrete texture phase offset per patch
  w      (S floats)           → seam constraint weights for Stage2 solver
  pi     (M permutation)      → nesting order
  h      (int, 0..2)          → nesting heuristic selector

Fitness:
  f1: total fabric height (from nesting engine)
  f2: seam phase mismatch  (stripe pattern misalignment across seams)
  f3: flattening distortion (currently stub, always 0.0)
```

### Core Files

| File | Role |
|------|------|
| `spec.py` | Core data structures: `ProblemInstance`, `TextureSpec`, `SeamDefinition`, `LandmarkDefinition` |
| `ga_spec.py` | GA types: `Genome`, `Individual`, `Fitness`, `GAConfig`, `run_ga()` |
| `topologies.py` | Seam network definitions per garment type (`build_sleeveless_shirt_topology`, `build_shirt_topology`, `build_pant_topology`) |
| `experiment_loader.py` | Loads mesh + landmarks + seam definitions into a `ProblemInstance` |
| `ga/geometry_block.py` | `LandmarkMapper`: maps `delta` → 3D vertices via bilinear interpolation + KDTree snap |
| `ga/real_evaluator.py` | `RealEvaluator`: runs geometry → Stage2 solver → nesting → returns `Fitness` |
| `geometry/cut_utils.py` | Cuts mesh along geodesic paths; extracts and labels patches |
| `geometry/geometry_utils.py` | `generate_symmetric_landmarks`, landmark utilities |
| `geometry/parameterization.py` | LSCM flattening (calls external `anisotropic-parameterization` binary) |
| `nesting/engine.py` | `NestingEngine.nest()`: places items on fabric roll with collision detection (STRtree) |
| `nesting/item.py` | `NestingItem`: patch polygon with rotation, placement, phase offset |
| `nesting/loader.py` | `PatchLoader`: reads patches from `results/pattern/latest/*.ply` |
| `nesting/phase_utils.py` | `TextureLattice`, `phase_uv()`, `seam_phase_mismatch()` |
| `nesting/stage2_global_align.py` | `Rigid2D`, `solve_global_alignment_all_components()`: Levenberg-Marquardt solver for tiny rigid transforms |
| `nesting/vis_utils.py` | `visualize_layout()`, `plot_seam_mismatch()` |

### Key Algorithms

- **Bilinear interpolation + KDTree snap** (`ga/geometry_block.py`): Maps continuous `delta` in [0,1]² to discrete mesh vertices
- **Levenberg-Marquardt** (`nesting/stage2_global_align.py`): Minimizes seam stripe-phase residuals with rigid 2D transforms per connected component; runs after nesting to compute f2
- **Tournament selection** (`ga_spec.py`): Dominance-based (Pareto) with fallback to sum of fitness values
- **Bottom-left heuristic** (`nesting/engine.py`): Places each item at lowest-leftmost valid position; snaps to texture lattice
- **Phase quantization**: `kappa` selects discrete lattice shift per patch; positions snapped to integer multiples of period

### External Dependencies

- `potpourri3d/` — geodesic path computation (vendored, built locally)
- `anisotropic-parameterization/` — LSCM flattening binary (vendored, built locally)
- NumPy, SciPy, Trimesh, Shapely

### Required Data

- `data/SMPL_FEMALE_POSED.ply` — input 3D body mesh
- `data/seamlines/upper/` — seam correspondence files (`seam-*.txt`) written by geometry pipeline and read by Stage2 solver
- `results/pattern/latest/upper/patch_*/` — 2D patch PLY files written by geometry pipeline

### Output

Each geometry run writes:
- `results/pattern/latest/upper/patch_N/optim_final-seams.ply` — 2D patch with seam vertices
- `data/seamlines/upper/seam-*.txt` — seam vertex correspondence pairs

### Configuration (run_ga.py)

Key parameters in `RealEvaluatorConfig`:
- `period_u_mm`, `period_v_mm` — stripe texture period
- `K` — number of discrete phase bins (kappa range is 0..K-1)
- `fabric_width_mm` — roll width constraint
- `w1`, `w2` — weights for f1 (height) and f2 (phase mismatch) in fitness sum; currently `w1=0, w2=1`

Key parameters in `GAConfig`:
- `population_size`, `generations`, `elite_count`
- `prob_flip_kappa` — mutation probability for kappa genes
- `weight_sigma` — std dev for Gaussian mutation of seam weights `w`
