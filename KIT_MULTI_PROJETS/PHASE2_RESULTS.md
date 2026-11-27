# Phase 2 — STAR-E: effort-aware ranking learned during training (Ant)

Run on 2026-11-27, CPU only. Command: `python effort_star.py --project ant` (4 min 03 s).
Outputs: `results/phase2/ant/{tuning.json, results.json, test_scores.npz}`.

## Idea

STAR learns a risk score and only becomes effort-aware afterwards, by inspecting in score / LOC order.
STAR-E puts the inspection cost inside the model:

    s_i = rank(z_i) − γ · log LOC_i          (the inspection order is the order of s)

| Variant | γ | Ranking pairs |
|---|---|---|
| E1 | fixed = 1 | (defective, clean) |
| E2 | learned (init 1) | (defective, clean) |
| E3 | learned | + (small defective, large defective), i.e. the Popt oracle order |

The BCE detection head is kept. The search grid is lr × weight decay × Δ-input × λc {0, 0.1} × λr {0.25, 1}, with 3 seeds.
The epoch is chosen on the seed-averaged, smoothed **validation Popt** curve. The model is then refit on 1.5 + 1.6 with 10 seeds and tested once on 1.7.

## Results on Ant 1.7

Each model is a 10-seed ensemble; brackets are the class-bootstrap 95 % CI.

| Model | Popt | R@20 % LOC | IFA |
|---|---|---|---|
| E1 fixed γ | 0.712 [0.647, 0.763] | 0.304 | 28 |
| E2 learned γ (→ 0.98) | 0.711 [0.647, 0.763] | 0.304 | 28 |
| E3 + oracle pairs | 0.711 [0.646, 0.763] | 0.304 | 28 |
| LR + density | 0.714 [0.648, 0.766] | 0.324 | 26 |
| STAR A5 + density | 0.718 [0.652, 0.769] | 0.294 | 25 |
| **RF + density** | **0.750 [0.687, 0.796]** | 0.363 | 28 |
| ExtraTrees + density | 0.738 [0.677, 0.786] | **0.372** | 29 |

Paired bootstrap: E1/E2/E3 vs LR + density give ΔPopt ≈ −0.002 (p ≈ 0.7); vs STAR A5 + density ≈ −0.006 (p ≈ 0.64); **vs RF + density −0.038 [−0.068, −0.009], p = 0.01**.
The detection head of STAR-E keeps an AP of 0.779 (STAR A5: 0.802).

## Reading

1. **Learning the effort term brings nothing over the post-hoc rule.** The learned γ converges to ≈ 1, which is exactly the post-hoc score / LOC rule: once LOC enters the score with weight 1, the network has nothing extra to learn from it. The oracle pairs change nothing either.
2. **The early stopping epoch is very small (16)**: validation Popt peaks early and then degrades. This is consistent with the ranking signal being dominated by LOC.
3. **The gap to RF is a feature-interaction effect, not an effort effect.** RF is better under the same score / LOC rule, so its probability estimates rank classes of similar size better. A natural next experiment is to give STAR the tree ensemble's view, for example by distilling RF probabilities or adding RF leaf embeddings as input.
4. For the report: Phase 2 is a **documented null result**. The FICHE explicitly values this ("interpret the failures").
