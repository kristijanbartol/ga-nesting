# Supplementary Materials Plan (PPSN 2026)

## Tier 1: Essential (addresses likely reviewer concerns)

1. **Full pants results table** — all 5 N values (5, 10, 25, 50, 100), 20 seeds each, mean +/- std
2. **Quantitative wallpaper group comparison** — f_align per group on onesie or sleeveless shirt (all 8 groups), showing the method generalises beyond stripes
3. **Multiple body shapes** — results on at least 2-3 SMPL body shapes (e.g. male, different poses) to show generalisation
4. **Ablation: adaptive sigma vs fixed sigma** — convergence curves + final f_align for both variants (the landscape-aware operator contribution)
5. **Per-seed convergence plots** — all 20 seeds overlaid (not just mean), showing the step-like structure and variance across seeds
6. **f_fit distribution analysis** — histogram of f_fit values across all seeds/body counts, demonstrating the guard constraint works

## Tier 2: Strengthens completeness

7. **Full population figure (qualitative)** — visualise all individuals in one generation (e.g. gen 0 and final gen) showing diversity of seam placements and nesting layouts
8. **Seam alignment graphs** — per-seam mismatch breakdown showing which seams benefit most from GA optimisation; visualise phase residuals along seam arcs for best GA vs B0
9. **sigma trajectory plots** — how adaptive sigma evolves over generations (per-gene or global, depending on operator choice)
10. **Geometry failure rate** — fraction of evaluations that crash (geodesic/LSCM failure) per body count, showing effective population utilisation
11. **Sensitivity to K** — K=4, 8, 16 on pants (where B2 is exact), showing effect of phase bin resolution
12. **Runtime breakdown** — per-component timing (geometry / Stage 2 / nesting) per evaluation

## Tier 3: Nice to have

13. **Sensitivity to sigma_delta** — fixed sigma = 0.1, 0.2, 0.3 showing the landscape is not trivially navigable
14. **Population diversity** — mean pairwise distance in delta-space over generations
15. **Best vs worst individual** — f_sum gap over generations showing selection pressure
16. **Sleeveless shirt results** — the third garment type (M=2, simpler topology)
17. **Full parameter specification** — all hyperparameters in one table

## Notes
- Items 1-6 are critical for addressing W1 (algorithmic contribution), W4 (incomplete evaluation), and W11 (wallpaper groups untested)
- Items 7-8 directly requested by the authors
- Multiple body shapes (item 3) requires running experiments on additional SMPL meshes — check data/SMPL_*.ply availability
- Wallpaper group experiments (item 2) require running GA with each group — moderate compute cost
