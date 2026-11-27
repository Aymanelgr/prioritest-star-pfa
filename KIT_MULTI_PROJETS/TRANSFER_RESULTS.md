# Step 6.3 — Do the Ant findings hold on other projects?

Run on 2026-11-27, CPU only. Command: `python transfer_study.py` (5 min 01 s; Camel, POI and JEdit).
Outputs: `results/transfer/{camel,poi,jedit}.json`.

**Everything is frozen from Ant**: architectures, hyperparameters, epochs, tabular parameters and the blend recipe come from
`results/phase1`, `phase3` and `phase4`. Nothing is tuned on these projects, so their test releases are genuinely held out.

| Project | Train | Test | Test classes / defective |
|---|---|---|---|
| Camel | 1.4 | 1.6 | 565 / 127 (22 %) |
| POI | 2.5 | 3.0 | 311 / 202 (65 %) |
| JEdit | 4.1 + 4.2 | 4.3 | 179 / **5** (3 %) |

"Joint" adds every other project of the kit as source data, with 50 % of the loss on the target project, as in Phase 3.

## AP on the test release (10 seeds; ensemble 95 % CI)

| Model | Ant 1.7 (Phase 1) | Camel 1.6 | POI 3.0 | JEdit 4.3 |
|---|---|---|---|---|
| LR | 0.798 | 0.420 [0.34, 0.51] | 0.872 [0.82, 0.92] | 0.277 [0.05, 0.71] |
| RF | **0.809** | 0.435 | 0.918 | 0.284 |
| ExtraTrees | 0.808 | **0.486** | 0.911 | 0.337 |
| HGB | 0.748 | 0.430 | 0.917 | 0.124 |
| A2 TCN | 0.801 | 0.438 | 0.877 | 0.261 |
| A3 + graph | 0.799 | 0.475 | 0.743 | 0.243 |
| A4 + similarity gate | 0.801 | 0.462 | 0.739 | 0.218 |
| A5 + InfoNCE | 0.802 | 0.439 | 0.857 | 0.242 |
| A6 STAR full | 0.794 | 0.458 | 0.811 | 0.260 |
| G2 (A5 + gate entropy) | 0.802 | 0.438 | 0.858 | 0.243 |
| STAR + LR + ET blend | 0.806 | 0.459 | 0.889 | 0.395 |
| STAR A5 joint | 0.795 | 0.440 | **0.920** | 0.290 |
| LR joint | 0.768 | 0.412 | 0.896 | 0.278 |

## The four claims from the Ant studies (paired class bootstrap)

| Claim | Ant | Camel | POI | JEdit |
|---|---|---|---|---|
| **C1** STAR ≈ LR on AP | holds (all n.s.) | **no — STAR better**: A3 +0.066 (p < 0.001), A4 +0.050 (p = 0.003), A6 +0.045 (p = 0.01) | mixed: A5/G2 ≈ LR; A3/A4 −0.14 (p < 0.001) | holds (n.s.) |
| **C2** trees beat LR on Popt (score / LOC) | RF +0.036 (p = 0.007), ET +0.024 (p < 0.001) | ET +0.012 (p = 0.01); RF n.s. | ET +0.005 (p = 0.02); RF +0.005 (p = 0.06) | RF +0.042 (p = 0.03) |
| **C3** STAR blend beats LR on Popt | +0.008 (p = 0.006) | **+0.009 (p < 0.001)**, also AP +0.038 (p = 0.003) | n.s. | n.s. |
| **C4** STAR-joint beats LR-joint | +0.029 AP (p = 0.04) | +0.029 (p = 0.07) | **+0.025 (p = 0.02)** | n.s. |
| Joint data helps STAR | −0.008 (n.s.) | +0.002 (n.s.) | **+0.059 (p = 0.016)** | n.s. |

## Reading

1. **The value of the graph view depends on the project.**
   - On **Camel** it is the largest single gain in the study: A2 → A3 = +0.037 AP, and every graph variant beats LR significantly.
   - On **POI** the same frozen graph variants collapse.
   - On **Ant** they are neutral.

   So the Ant conclusion "no STAR mechanism helps" does **not** generalise. The report should present the graph as *project-dependent*, and step 7.3 (performance by node degree, per project) becomes the key analysis to explain why.
2. **Frozen Ant hyperparameters do not transfer to POI.** POI's prevalence is 65 % against 17 % for Ant 1.5, and the graph variants were tuned at 22–64 epochs on Ant. The collapse of A3 / A4 on POI is at least partly a tuning-transfer effect, not proof that the graph is harmful there. A per-project validation window would be needed; POI has none.
3. **C4 is the most consistent STAR-specific result.** Joint STAR ≥ joint LR on every informative project (significant on Ant and POI, a trend on Camel). On POI, cross-project data even makes STAR the best model overall (0.920 AP). STAR uses heterogeneous training data better than a linear model.
4. **C2 holds broadly.** Tree ensembles, mostly ExtraTrees, are the safest effort-aware baseline across projects.
5. **C3 (the blend) holds on Ant and Camel**, the two projects with the most training data (≥ 118 and 252 classes) and moderate prevalence.
6. **JEdit 4.3 is uninformative**: with 5 defective classes, the AP CIs span 0.05–0.70. Report it with that caveat only.
