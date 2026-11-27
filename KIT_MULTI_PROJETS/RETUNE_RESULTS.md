# Step 5.5 — Per-project tuning for projects without a validation release (IN PROGRESS)

Command: `python retune_project.py --projects camel poi`. Outputs: `results/retune/`.
**Status:** Camel is complete (5 min 51 s). The POI run was stopped during tuning, so POI has no result yet. Rerun with `--projects poi` to finish it.

## Protocol

Camel and POI have only two target windows (train, test). Step 6.3 reused the settings frozen from Ant; here every model is tuned **inside the training release**:
1. **Search.** Use 3 stratified folds over the classes of the training release. The import graph stays whole; the loss only sees the fold's training classes, and AP is measured on the held-out classes. The grids are the same as in Phase 1, and the epoch is the argmax of the fold-averaged, smoothed AP curve.
2. **Refit.** Train on all classes of the training release with 10 seeds.
3. **Test.** Evaluate once on the later release. The test release is loaded only after the tuning is saved.

## Camel (train 1.4 → test 1.6, 565 classes, 127 defective)

| Model | AP (10 seeds) | AP ens. [95 % CI] | AP frozen from Ant (6.3) | Popt (dens) |
|---|---|---|---|---|
| A1 HGB | 0.437 | 0.437 [0.354, 0.522] | 0.430 | 0.775 |
| A2 TCN | 0.490 ± 0.009 | 0.508 [0.424, 0.594] | 0.438 | 0.810 |
| A3 + graph | 0.498 ± 0.026 | 0.538 [0.447, 0.627] | 0.475 | 0.796 |
| A4 + similarity gate | 0.500 ± 0.019 | 0.542 [0.453, 0.629] | 0.462 | 0.777 |
| **A5 + InfoNCE** | **0.507 ± 0.013** | 0.539 [0.453, 0.629] | 0.439 | **0.814** |
| A6 + RankNet | 0.462 ± 0.023 | 0.498 [0.412, 0.594] | 0.458 | 0.706 |
| LR | 0.452 | 0.452 [0.365, 0.542] | 0.420 | 0.783 |
| RF | 0.475 ± 0.006 | 0.478 [0.395, 0.563] | 0.435 | 0.781 |
| ExtraTrees | 0.486 ± 0.004 | 0.486 [0.402, 0.566] | 0.486 | 0.790 |
| STAR + LR + ET blend | 0.517 | 0.517 [0.428, 0.607] | 0.459 | 0.803 |

Paired class bootstrap tests:

| Comparison | ΔAP [95 % CI] | p | ΔPopt (dens) | p |
|---|---|---|---|---|
| A2 TCN vs LR | +0.056 [+0.001, +0.111] | 0.049 | +0.027 | 0.09 |
| A3 + graph vs LR | **+0.086 [+0.021, +0.151]** | **0.009** | +0.012 | 0.57 |
| A4 vs LR | **+0.090 [+0.024, +0.157]** | **0.010** | −0.006 | 0.77 |
| A5 + InfoNCE vs LR | **+0.088 [+0.028, +0.149]** | **0.003** | +0.031 | 0.056 |
| A6 + RankNet vs LR | +0.047 | 0.14 | **−0.077** | 0.006 |
| Blend vs LR | **+0.065 [+0.025, +0.108]** | **0.002** | **+0.020** | **0.010** |
| A6 vs A5 (adding RankNet) | **−0.041 [−0.069, −0.008]** | **0.011** | | |

## Reading so far

1. **Tuning on the project's own data helps every neural model on Camel** (A5: 0.439 → 0.507 AP). The tabular models barely move. Freezing Ant's settings was handicapping STAR.
2. With proper tuning, **STAR A3–A5 beat logistic regression by about +0.09 AP (p ≤ 0.01)**, which is the largest margin in the study. The STAR blend beats LR on both AP and effort-aware ranking.
3. **RankNet hurts again** (A6 vs A5: −0.041 AP, p = 0.011; −0.077 Popt), as on Ant. The final STAR should drop the RankNet term (A5 + gate entropy).
4. POI is still to run. It is the decisive check of whether the collapse in 6.3 came from the frozen settings.
