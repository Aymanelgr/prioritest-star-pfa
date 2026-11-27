# Phase 1 — Fair, tuned, significance-tested comparison (Ant)

Run on 2026-11-27, CPU only (2 cores), PyTorch 2.15, scikit-learn 1.9. Command:

```bash
python compare_fair.py --project ant          # 6 min 33 s
```

Outputs: `results/phase1/ant/{tuning.json, table.csv, tests.json, test_scores.npz}`.

## Protocol (identical for every model)

1. **Validation search** on train 1.5 → validation 1.6. The test release is not loaded at this stage.
   - **Neural variants (A2–A7):** grid lr {1e-3, 3e-3} × weight decay {1e-4, 1e-3} × Δ-input {no, yes}, plus λc {0.1, 0.25} / λr {0.25, 1} when the variant uses them. Each point is trained with 3 seeds and the epoch is the argmax of the seed-averaged, 5-epoch-smoothed validation-AP curve (≥ 10 epochs).
   - **Tabular models:** a small grid on their main regularisation parameter.
2. **Refit** on 1.5 + 1.6 with the chosen settings, 10 seeds.
3. **One test evaluation** on 1.7 (292 classes, 102 defective).

The preprocessing is the same for all models: train-fitted z-scores, then signed-log, with `pos_weight` / balanced class weights. Tabular models use last-release metrics plus Δ from the previous release.

## Results on Ant 1.7

AP is the mean ± std over 10 training seeds. The CI is a 2 000-resample class bootstrap of the 10-seed ensemble. The ensemble averages ranks for rank metrics and averages scores for score / LOC metrics, because dividing a rank by LOC would distort the inspection order.
"dens" means classes are inspected by score / LOC; "risk" means by raw score.

| Model | AP (seeds) | AP ens. [95 % CI] | Popt (dens) | R@20 % LOC (dens) | Popt (risk) |
|---|---|---|---|---|---|
| A1 HGB | 0.748 | 0.748 [0.655, 0.840] | 0.738 | 0.333 | 0.424 |
| A2 TCN + BCE | 0.801 ± 0.004 | 0.802 [0.725, 0.865] | 0.708 | 0.287 | 0.377 |
| A3 + graph, classic fusion | 0.799 ± 0.004 | 0.802 [0.728, 0.866] | 0.711 | 0.287 | 0.381 |
| A4 + similarity fusion | 0.801 ± 0.005 | 0.806 [0.734, 0.866] | 0.713 | 0.306 | 0.376 |
| A5 + InfoNCE | **0.802 ± 0.009** | 0.805 [0.734, 0.866] | 0.717 | 0.300 | 0.376 |
| A6 + RankNet = STAR | 0.794 ± 0.008 | 0.798 [0.717, 0.866] | 0.711 | 0.300 | 0.379 |
| A7 STAR+ | 0.644 ± 0.035 | 0.677 [0.580, 0.767] | 0.633 | 0.291 | 0.434 |
| Logistic regression | 0.798 | 0.798 [0.721, 0.865] | 0.714 | 0.324 | 0.345 |
| Random Forest | **0.809 ± 0.002** | 0.809 [0.725, 0.880] | **0.750** | 0.351 | 0.418 |
| ExtraTrees | 0.808 ± 0.007 | 0.807 [0.723, 0.878] | 0.738 | **0.363** | 0.399 |
| kNN cosine (Rhazi-style, N=50) | 0.789 | 0.789 [0.702, 0.864] | 0.710 | 0.294 | 0.413 |
| STAR + LR + ET blend | 0.806 | 0.806 [0.724, 0.874] | 0.721 | 0.333 | 0.370 |

Random ordering gives Popt ≈ 0.50 and R@20 % LOC ≈ 0.20.

### Ablation ladder: one mechanism per step (Wilcoxon over the 10 seeds, Cliff's δ)

| Step | ΔAP | p | ΔPopt (dens) | p |
|---|---|---|---|---|
| A1 HGB → A2 TCN | +0.054 | 0.002 | −0.031 | 0.002 |
| A2 → A3 (+ graph) | −0.002 | 0.13 | +0.003 | 0.16 |
| A3 → A4 (+ similarity gate) | +0.002 | 0.32 | +0.002 | 0.63 |
| A4 → A5 (+ InfoNCE) | +0.001 | 0.49 | +0.004 | 0.19 |
| A5 → A6 (+ RankNet) | −0.008 | 0.049 | −0.006 | 0.049 |

### Each model against logistic regression (paired class bootstrap)

| Model vs LR | ΔAP [95 % CI] | p | ΔPopt (dens) [95 % CI] | p |
|---|---|---|---|---|
| A5 + InfoNCE | +0.007 [−0.015, +0.030] | 0.57 | +0.004 [−0.014, +0.023] | 0.67 |
| A6 STAR | −0.000 [−0.022, +0.022] | 1.00 | −0.002 [−0.013, +0.008] | 0.68 |
| A7 STAR+ | −0.121 [−0.177, −0.068] | <0.001 | −0.061 [−0.104, −0.019] | 0.009 |
| Random Forest | +0.011 [−0.024, +0.044] | 0.51 | **+0.036 [+0.011, +0.063]** | **0.007** |
| ExtraTrees | +0.009 [−0.020, +0.039] | 0.52 | **+0.024 [+0.012, +0.038]** | **<0.001** |
| STAR + LR + ET blend | +0.008 [−0.012, +0.028] | 0.39 | **+0.008 [+0.002, +0.014]** | **0.006** |

On AP, nothing differs from LR except A7, which is worse. On effort-aware ranking (Popt by score / LOC), tree ensembles and the STAR blend are significantly better than LR.

## Reading

1. **No STAR mechanism gives a measurable gain on Ant under a fair protocol.**
   - The graph, the similarity gate and InfoNCE each move AP by ≤ 0.002 and are not significant.
   - RankNet is slightly *harmful* (p = 0.049 over seeds).
   - Validation AP tells the same story: every model sits between 0.68 and 0.70.
2. **The test set is the limiting factor.** With 292 classes, the AP CI is ±0.07 wide, so any true effect below about 0.03 cannot be detected on one release. This supports Phase 3 (more data, more projects) as the way forward.
3. **Tree ensembles are the strongest effort-aware models, and the gap is significant**: RF reaches Popt 0.750 (+0.036 vs LR, p = 0.007) and ET reaches R@20 % LOC 0.363, against 0.711 / 0.300 for STAR. The STAR + LR + ET blend also beats LR on Popt (+0.008, p = 0.006), so STAR contributes when it is combined with the tabular models. STAR's ranking is still size-driven: Popt by raw risk is about 0.38, worse than random. This supports Phase 2 (effort-aware training).
4. **A7 STAR+ collapses on AP.** Its effort-weighted RankNet pushes small classes up, which gives the best Popt (risk) among the neural models (0.434) but destroys AP. It should be reported as an AP / effort trade-off, not as an improvement.
5. **Seed-level Wilcoxon tests only measure training-seed variance**, not data variance. The class bootstrap is the honest uncertainty, and conclusions in the report should be based on it.
6. **The A6 configuration chosen here matches `config_star_r.json`**, which independently confirms the earlier STAR-R search.
