# Phase 4 — Personal improvement of the fusion / cross-view contrast (GUIDE §3)

Run on 2026-11-27, CPU only. Command: `python improve_star.py --project ant` (4 min 47 s).
Outputs: `results/phase4/ant/{tuning.json, results.json, test_scores.npz}`. Code: `src/model.py`
(`supcon_cross_view`, `gate_entropy`, `gate_absdiff`), wired into `run_star.cohort_loss` (all off by default).

## Variants

Each variant changes one thing relative to A5 (graph + similarity gate + InfoNCE, the best ladder step of Phase 1).

| ID | Change | Motivation |
|---|---|---|
| C1 | Label-aware InfoNCE: defective/defective pairs removed from the negatives | Fix InfoNCE false negatives (FICHE §3) |
| C2 | SupCon hybrid: C1 + other defective classes as soft positives across the views | Pull all defective classes into one cluster |
| G1 | The gate also sees \|zT − zS\| | Let fusion react to *where* the views disagree, not only how much (cos) |
| G2 | Gate-entropy regulariser (weight 0.05) | STAR-R uses the temporal view at α ≈ 0.7; keep both views in use |
| C2G1 | C2 + G1 | Combination |

Same protocol as Phase 1: equal validation grid (lr × weight decay × Δ × λc), refit on 1.5 + 1.6, 10 seeds, one test pass.
A5 is re-run with the same seeds for paired tests.

## Results on Ant 1.7

| Variant | Val AP | Test AP | Popt (dens) | R@20 % LOC (dens) | ROC-AUC | mean α (temporal) |
|---|---|---|---|---|---|---|
| A5 reference | 0.688 | 0.802 ± 0.009 | 0.717 | 0.300 | 0.864 | 0.625 |
| C1 label-aware InfoNCE | 0.688 | 0.802 ± 0.009 | 0.718 | 0.305 | 0.864 | 0.621 |
| C2 SupCon hybrid | 0.674 | 0.802 ± 0.008 | 0.717 | 0.298 | 0.864 | 0.703 |
| G1 \|zT − zS\| gate | 0.688 | 0.791 ± 0.015 | 0.711 | 0.300 | 0.854 | 0.755 |
| **G2 gate entropy** | 0.688 | **0.802 ± 0.009** | 0.717 | 0.298 | 0.864 | **0.554** |
| C2G1 | 0.674 | 0.799 ± 0.005 | 0.713 | 0.294 | 0.860 | 0.682 |

Tests against A5. Δens uses the paired class bootstrap of the ensembles; Δseeds uses Wilcoxon over the 10 seeds.

| Variant | ΔAP ens [95 % CI] | p | ΔAP seeds | p | ΔPopt seeds | p |
|---|---|---|---|---|---|---|
| C1 | +0.001 [−0.001, +0.005] | 0.33 | −0.000 | 0.70 | +0.000 | 0.28 |
| C2 | +0.000 [−0.007, +0.006] | 0.90 | −0.000 | 0.85 | −0.000 | 0.85 |
| G1 | −0.006 [−0.014, +0.001] | 0.09 | −0.011 | 0.049 | −0.006 | 0.014 |
| G2 | +0.000 [−0.001, +0.003] | 0.83 | +0.000 | 0.92 | −0.000 | 0.43 |
| C2G1 | −0.003 [−0.011, +0.004] | 0.36 | −0.004 | 0.32 | −0.005 | 0.08 |

## Reading

1. **False negatives are not what limits STAR on Ant.** With 20 defective classes in training, InfoNCE rarely pushes two defective classes apart, so masking them (C1) or pulling them together (C2) changes nothing.
2. **G2 is the improvement retained.** It reaches its stated goal: the fusion now uses both views almost equally (α 0.63 → 0.55) at no cost in AP, Popt or ROC. Two consequences:
   - the analysis of α / β and of view agreement becomes meaningful;
   - the model's performance **does not depend on how the two views are weighted**. That is consistent with Phase 1, where the graph view added no measurable signal.
3. **G1 is harmful.** Given |zT − zS|, the gate falls back on the temporal view (α 0.755) and overfits slightly (−0.011 AP over seeds, p = 0.049).
4. Taken with Phases 1–3: on Ant the bottleneck is **data size and the information content of the metrics**, not the fusion or contrastive design.
