# Step 7.3 — Why does the import graph help on Camel but not on Ant or POI?

Command: `python graph_analysis.py` (15 s). Output: `results/graph_analysis/graph_analysis.json`.
For each project's test window it computes:
- graph statistics on the import graph;
- AP of A2 (TCN, no graph) vs A3 (+ graph, classic fusion), separately on isolated and connected classes.

Both models use the configurations **frozen from Ant**, with a 5-seed ensemble. Labels are used only for evaluation.

## Graph statistics (test window)

| | Ant 1.7 | Camel 1.6 | POI 3.0 | JEdit 4.3 |
|---|---|---|---|---|
| Classes / defect rate | 292 / 35 % | 565 / 22 % | 311 / 65 % | 179 / 3 % |
| Import edges / mean degree | 867 / 5.9 | 1 608 / 5.7 | 472 / 3.0 | 233 / 2.6 |
| Isolated classes | 12.7 % | 14.2 % | 25.4 % | 26.3 % |
| Defect rate: isolated → connected | 21.6 % → 36.9 % (×1.7) | **1.3 % → 26.0 % (×20)** | 55.7 % → 68.1 % (×1.2) | 0 % → 3.8 % |
| Neighbour defect rate: defective vs clean (homophily gap) | 0.48 vs 0.45 (+0.03) | 0.46 vs 0.47 (−0.01) | 0.87 vs 0.75 (**+0.12**) | 0.05 vs 0.16 (−0.12) |
| Label assortativity | 0.006 | 0.002 | 0.090 | −0.089 |

## Where the graph changes AP (A3 − A2)

| | Ant | Camel | POI | JEdit |
|---|---|---|---|---|
| All classes | +0.002 | **+0.043** | −0.149 | +0.016 |
| Isolated classes | +0.005 | −0.001 | −0.000 | (no defects) |
| Connected classes | +0.001 | **+0.052** | **−0.152** | +0.013 |

## Reading

1. **On Camel, the graph helps through connectivity, not through defect clustering.**
   - Isolated Camel classes are almost never defective: 1 out of 80.
   - Defective classes do *not* import each other more than clean ones do (homophily gap ≈ 0).
   - The GNN's mean aggregation gives connected classes a different representation from isolated ones, so whether and how a class is embedded in the import graph becomes a strong signal (×20 in defect rate).
   - The CK coupling metrics (CBO, Ca, Ce) apparently capture this only partially.
2. **On Ant, the same contrast is weak** (×1.7), so the graph has little to add beyond the metrics. That matches Phase 1, where no graph step was significant.
3. **On POI the graph hurts even though defects cluster most there** (homophily +0.12). The losses are entirely on connected classes. The likely cause is the frozen Ant hyperparameters. The graph variants were tuned on Ant (17 % training prevalence, 22–52 epochs), while POI 2.5 has 68 % prevalence and no validation window. This is why step 5.5 (per-project retuning) is in the roadmap.
4. **Practical rule for the report:** before relying on the graph view, check two numbers on the *training* release:
   - the defect-rate contrast between isolated and connected classes;
   - the homophily gap.

   Both can be computed without touching the test set. Camel's large contrast is the regime where STAR's structural view pays off.
5. Caveat: three informative projects make this a **correlational explanation**, not a proof. JEdit 4.3 has too few defects to say anything.
