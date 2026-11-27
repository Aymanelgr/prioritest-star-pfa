# Phase 3 — More data: cross-project training for Ant

Run on 2026-11-27, CPU only. Command: `python more_data.py --project ant` (3 min 49 s).
Outputs: `results/phase3/ant/{selection.json, results.json, test_scores.npz}`.

## Why cross-project, and not "new classes"

The roadmap first planned to train on the classes that the strict 3-release window drops (Ant 1.5: 293 nodes in the
temporal graph against 118 labelled classes). The kit only ships labels for the 118 strict-window classes: the 175 others
have features and edges but **no label**, because labels live in the PROMISE archive (282 MB, not included).
That study stays open until the archive is added (`prepare_multi.py --archive PROMISE-dataset-main.zip`).

The data lever that *is* available: the other 9 projects of the kit hold **13 windows, 3 399 labelled classes, 1 261 defective**
(camel, jedit, log4j, lucene, poi, synapse, velocity, xalan, xerces), against 118 / 20 for Ant 1.5. Ant data is never used in
"others". Each project is standardised on its own first window (project-wise normalisation), then signed-log.

## Settings

All settings use STAR A5 with its Phase-1 hyperparameters. Epochs are chosen on the seed-averaged, smoothed Ant-1.6 AP. Each setting is refit and tested once on Ant 1.7 with 10 seeds.

| Setting | Training data | Val AP (Ant 1.6) |
|---|---|---|
| X0 Ant only | Ant 1.5 (refit 1.5 + 1.6) | 0.688 |
| X1 Others only (zero-shot) | 13 other windows | 0.688 |
| X2 Joint | others + Ant, Ant cohorts = 50 % of the loss | **0.713** |
| X3 Pre-train → fine-tune | 46 epochs others, then 31 epochs Ant (lr × 0.3) | 0.682 |

Tabular LR / RF get the same three regimes. In the joint regime, Ant samples carry 50 % of the total sample weight.

## Results on Ant 1.7

10 seeds (RF: 5). The CI is a class bootstrap of the ensemble.

| Model | AP | AP ens. [95 % CI] | ROC-AUC | Popt (dens) | R@20 % LOC (dens) |
|---|---|---|---|---|---|
| STAR X0 Ant only | **0.802 ± 0.009** | 0.805 [0.734, 0.866] | 0.864 | 0.718 | 0.294 |
| STAR X1 others only | 0.734 ± 0.008 | 0.737 [0.652, 0.821] | 0.827 | 0.699 | 0.255 |
| STAR X2 joint | 0.795 ± 0.006 | 0.797 [0.722, 0.860] | 0.857 | 0.711 | 0.275 |
| STAR X3 pre-train + fine-tune | 0.774 ± 0.009 | 0.780 [0.700, 0.851] | 0.846 | 0.704 | 0.304 |
| LR Ant only | 0.798 | 0.798 [0.721, 0.865] | 0.864 | 0.714 | 0.324 |
| LR others only | 0.718 | 0.718 [0.626, 0.810] | 0.827 | 0.706 | 0.275 |
| LR joint | 0.768 | 0.768 [0.686, 0.842] | 0.843 | 0.706 | 0.304 |
| RF Ant only | **0.809 ± 0.002** | 0.810 [0.726, 0.880] | **0.879** | **0.751** | **0.363** |
| RF others only | 0.736 ± 0.010 | 0.739 [0.651, 0.829] | 0.825 | 0.710 | 0.314 |
| RF joint | 0.781 ± 0.004 | 0.779 [0.689, 0.868] | 0.867 | 0.736 | 0.324 |

Paired class bootstrap tests:

| Comparison | ΔAP [95 % CI] | p | ΔPopt (dens) | p |
|---|---|---|---|---|
| STAR X1 vs X0 | −0.068 [−0.119, −0.023] | <0.001 | −0.019 | 0.05 |
| STAR X2 vs X0 | −0.008 [−0.021, +0.005] | 0.19 | −0.007 | 0.22 |
| STAR X3 vs X0 | −0.025 [−0.051, −0.002] | 0.03 | −0.014 | 0.36 |
| LR joint vs LR Ant only | −0.030 [−0.054, −0.008] | 0.002 | −0.008 | 0.08 |
| RF joint vs RF Ant only | −0.031 [−0.060, +0.007] | 0.13 | −0.014 | 0.22 |
| **STAR X2 joint vs LR joint** | **+0.029 [+0.001, +0.061]** | **0.04** | +0.005 | 0.44 |

## Reading

1. **More data from other projects does not help on Ant 1.7.** No regime beats Ant-only for any model. This is project-level **distribution shift**: the defect-proneness of a CK-metric profile differs between projects, even after project-wise normalisation.
2. **The joint setting's validation gain (+0.025 on Ant 1.6) did not carry over to the test.** This is a textbook case for selecting on validation and reporting the test once, and also a reminder that one validation release is a noisy selector.
3. **STAR is the model most robust to mixing projects.** Joint training costs STAR only −0.008 AP (n.s.) but costs LR −0.030 (p = 0.002), and STAR-joint beats LR-joint significantly. The structural and temporal views seem to transfer better than a linear decision rule on raw metrics. This is the first result where a STAR-specific property shows up, and it belongs in the report.
4. **Zero-shot cross-project prediction** (FICHE bonus): STAR 0.734 AP, LR 0.718, RF 0.736, against 0.80 within-project. That is usable but clearly worse, consistent with the cross-project defect prediction literature.
5. Next data levers, in order of expected value:
   - add the PROMISE archive to get the **labels of new classes** (cold-start study, R5);
   - **domain adaptation** for cross-project data, for example per-project batch-norm or adversarial project discriminator, or weighting source windows by similarity to Ant;
   - repeat the full study on **Camel and POI** as second target projects.
