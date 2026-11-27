# STAR-R — audit, baseline and improved training (Ant)

All numbers below come from runs executed on 2026-11-27 (CPU only, 2 cores, PyTorch 2.15, scikit-learn 1.9).

## 1. What the project is
- **Task**: rank / detect defect-prone Java classes (PROMISE). Label = at least one bug in the **target** release. This is current-version detection, not next-version forecasting (see the kit README).
- **Data (Ant)**: windows of 3 releases × 20 CK/size metrics, plus a Java-import graph of the target release.
  Train = 1.5 (118 classes, 20 defective, 17 %), validation = 1.6 (167, 60 defective, 36 %), test = 1.7 (292, 102 defective, 35 %).
- **Model (fiche §3)**: causal TCN (2 Conv1D, kernel 2, dilations 1/2) → zT; 2-layer mean GraphSAGE → zS; cos(zT,zS) gate → fusion;
  loss = BCE + λc·InfoNCE + λr·RankNet. Ablations A1–A8 in `run_star.py`.
- **Evaluation**: AP (main), ROC-AUC, Recall@20 % classes / LOC, effort-aware Recall@20 % LOC (score/LOC);
  Accuracy/Precision/Recall/F1/MCC use a flag rate chosen on validation.

## 2. Audit findings
| Check | Result |
|---|---|
| Label leakage | None found: features only come from the 3 window releases; test labels are never used for fitting, early stopping, tuning or thresholds. |
| Normalisation | z-scores fit on the first (train) window only. OK. Values clipped at ±10, and heavy tails remain (LOC, RFC…). |
| Entity overlap | 117/118 train classes and all 167 validation classes reappear in the test window (same classes, later releases). Not label leakage, but test scores are **not** from unseen code. Report this. |
| Class imbalance | Train is 17 % defective, val/test 35 %. Prevalence shift → `pos_weight` and a validation-chosen flag rate are needed; raw accuracy must be read against the majority baseline (0.651 on test). |
| Small data | 118 training classes → high seed variance (A6 AP std 0.024; per-seed early-stop epoch ranged 10–48). |
| Code | TCN padding is causal (checked), no NaNs, 19/19 unit tests pass. `run_star.py --tune` tunes on a single seed (noisy). A7 (`STAR_plus`) is hurt by dropout + bidirectional SAGE on this data size. |
| JEdit | Test release 4.3 has **5 positives / 179** → metrics are not interpretable (all models score below the 0.972 majority accuracy). |
| GPU | No GPU on your VM or in the cloud workspace. The models train in seconds on CPU. |

## 3. What changed (STAR-R)
1. Kit `--improved` preprocessing kept: signed-log compression of the z-scores, `pos_weight` BCE, refit on train+validation.
2. **Validation-only grid** (`explore_val.py`, 192 configs × 3 seeds, test release never loaded). The epoch is chosen on the
   **seed-averaged, smoothed** validation-AP curve instead of each seed's noisy argmax.
   Retained: lr 1e-3, weight decay 1e-3, λc 0.1, λr 0.25, τ 0.2, Δ-metrics input, no dropout, unidirectional SAGE, 64 epochs (`config_star_r.json`).
3. **10-seed ensemble** of the STAR model (rank average) to cut variance.
4. Tabular partners (logistic regression C=0.03, ExtraTrees 500) on last-release + Δ features. The final blend is chosen **on validation**
   among STAR-containing candidates; it picked STAR+LR+ET in most groups.
5. Whole procedure repeated with 5 independent seed groups → mean ± std.

## 4. Results — Ant test release 1.7 (evaluated once per group)
| Model | AP | ROC-AUC | Accuracy | Bal. acc | Precision | Recall | F1 | MCC | EA-R@20 %LOC |
|---|---|---|---|---|---|---|---|---|---|
| Baseline A1 boosting | 0.752 | 0.843 | 0.770 | 0.701 | 0.787 | 0.471 | 0.589 | 0.472 | 0.314 |
| Baseline A6 STAR full (default) | 0.760±0.024 | 0.832±0.015 | 0.751±0.017 | 0.747 | 0.623 | 0.733 | 0.673 | 0.480±0.027 | 0.312 |
| Baseline A8 ensemble (default) | 0.797±0.013 | 0.865 | 0.801±0.034 | 0.758 | 0.785 | 0.614 | 0.671±0.114 | 0.554±0.081 | 0.316 |
| Kit `--improved` A6 | 0.789±0.011 | 0.852 | 0.797±0.006 | 0.773 | 0.717 | 0.696 | 0.705 | 0.552 | 0.300 |
| Kit `--improved` A8 (best existing) | 0.799±0.009 | 0.863 | 0.815±0.008 | 0.788 | 0.755 | 0.698 | 0.725±0.005 | 0.588±0.015 | 0.308 |
| **STAR-R (STAR ensemble only)** | 0.794±0.003 | 0.862 | 0.816±0.005 | 0.786 | 0.766 | 0.686 | 0.723 | 0.590±0.013 | 0.308 |
| **STAR-R FINAL (val-chosen blend)** | **0.799±0.003** | **0.870** | **0.823±0.004** | **0.804** | 0.748 | **0.743** | **0.745±0.005** | **0.610±0.008** | 0.320 |
| Reference: logistic regression alone | 0.798 | 0.864 | 0.819 | 0.804 | 0.733 | 0.755 | 0.744 | 0.604 | 0.304 |

Majority-class accuracy on the test release = 0.651.

**Reading the results honestly.**
- Against the default STAR (A6), STAR-R gains +0.039 AP, +0.072 accuracy and +0.13 MCC, and seed std drops ~8×.
- Against the best configuration the kit already had (`--improved` A8), the gain is small: AP is equal, accuracy +0.008, F1 +0.020, MCC +0.022. The main gain is stability.
- A regularised logistic regression is about as good. On this data the graph/contrastive machinery adds little measurable signal beyond the metrics themselves.
  That is a scientific finding to report, not to hide.

## 5. Reproduce
```bash
pip install -r requirements.txt
python run_star.py --project ant --seeds 42 43 44 45 46 --out results/baseline_default      # baseline, ~20 s
python run_star.py --project ant --seeds 42 43 44 45 46 --improved --out results/baseline_improved   # ~23 s
python explore_val.py --project ant --out results/explore_ant.json                             # validation grid, ~12.5 min
python train_final.py --project ant --groups 5 --ens 10                                        # final, ~61 s
```
Saved artefacts: `results/star_r/ant/star_r_ensemble.pt` (10 STAR state_dicts + config, trained on 1.5+1.6),
`tabular_models.joblib`, `final_test_scores.csv`, `final_metrics.json` (per-group validation choices and test metrics).
