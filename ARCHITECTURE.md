# PRIORITEST-STAR — Architecture & Roadmap (v2, evidence-based)

> Effort-aware prioritisation of defect-prone Java classes (PROMISE): a temporal view (causal TCN over 3 releases),
> a structural view (GraphSAGE on Java imports), similarity-guided fusion, cross-view InfoNCE and RankNet.
> EMSI Marrakech — PFA IA, 5th year.

This file is the master plan. It records:
1. the **ground rules** of the project;
2. **what is built** and **what each experiment showed**;
3. the **target architecture** of the repo and the model;
4. **every remaining step**, with the command to run and the proof that it is done.

v1 (27 Nov) planned Phases 0–8. Phases 0–4 have since been executed; v2 rewrites the rest of the plan around their results.

---

## 0. Ground rules (never broken)

| # | Rule | How it is enforced |
|---|---|---|
| R1 | **Detection of defects in the observed release**, never "prediction of the next release". | README, report, app caption |
| R2 | **The test release is used once**, after every choice is frozen. No epoch, λ, threshold, blend or seed is chosen on test. | Every script searches on window 2 with the test not loaded, then evaluates window 3 once |
| R3 | Normalisation fitted on the **train window only** (per project). | `prepare_multi.py` + signed-log |
| R4 | The import graph is a **partial proxy**, not a full dependency graph. | Data chapter + limits |
| R5 | New classes (no 3-release history) are **excluded** → selection bias must be analysed. | Step 5.3 (needs the PROMISE archive) |
| R6 | 10 seeds, mean ± std, **class-bootstrap 95 % CI**, paired tests, measured times. | `src/stats.py` |
| R7 | Each ablation adds **one** mechanism, retrained from scratch on identical splits, **same tuning budget**. | `compare_fair.py` |
| R8 | Rhazi et al. (2026) cited for the *principle* of similarity learning; STAR is **not** a reproduction. | State of the art |

---

## 1. Status (27 Nov 2026)

### 1.1 Built

| Component | Files |
|---|---|
| Data kit: 10 projects, 16 windows, temporal-graph extension (verified) | `data/`, `prepare_multi.py`, `check_graph_temporal.py` |
| Model: causal TCN, MeanSAGE, similarity gate, Δ-input, \|zT − zS\| gate option | `src/model.py` |
| Losses: BCE + pos_weight, InfoNCE, label-aware InfoNCE, **SupCon hybrid**, RankNet, effort RankNet, **gate entropy** | `src/model.py`, `run_star.py` |
| Metrics: AP, ROC, Popt, IFA, Recall@LOC, flag-rate Acc/MCC/F1 | `src/train.py`, `src/stats.py` |
| Statistics: class bootstrap CI, paired bootstrap, Wilcoxon, Cliff's δ | `src/stats.py` |
| Ablation ladder A1–A8, STAR-R final model, validation search | `run_star.py`, `explore_val.py`, `train_final.py` |
| Phase 1 fair comparison, Phase 2 STAR-E, Phase 3 cross-project, Phase 4 improvements | `compare_fair.py`, `effort_star.py`, `more_data.py`, `improve_star.py` |
| Analysis figures, Streamlit app, LaTeX report draft | `analyse_star.py`, `app.py`, `rapport/` |
| 34 unit tests (incl. 4 no-leakage tests) | `tests/` |

### 1.2 What the experiments showed (Ant: train 1.5 → val 1.6 → test 1.7)

| Phase | Question | Answer | Write-up |
|---|---|---|---|
| 1 | Does each STAR mechanism help, under an equal tuning budget? | **No significant gain** at any step. A2–A6 ≈ LR (AP ≈ 0.80, CI ± 0.07). RankNet is slightly harmful (p = 0.049). Tree ensembles are significantly better than LR on effort ranking (RF Popt 0.750, +0.036, p = 0.007). The STAR + LR + ET blend also beats LR on Popt (+0.008, p = 0.006). | `PHASE1_RESULTS.md` |
| 2 | Does learning effort-awareness (s = rank − γ·log LOC) beat post-hoc score / LOC? | **No.** γ is learned at ≈ 1, which is the post-hoc rule. Popt 0.711, the same as LR + density, and below RF + density (−0.038, p = 0.01). | `PHASE2_RESULTS.md` |
| 3 | Does more data (3 399 classes from 9 other projects) help? | **No, because of distribution shift.** Joint −0.008 AP (n.s.); pre-training −0.025 (p = 0.03); zero-shot 0.734. **But STAR is the most robust to mixing projects:** STAR-joint beats LR-joint +0.029 (p = 0.04), while LR loses 0.030 with the same data. | `PHASE3_RESULTS.md` |
| 6.3 | Do the Ant findings hold on Camel / POI / JEdit (configs frozen from Ant)? | **Partly.** On **Camel** (565 test classes) STAR with the graph beats LR significantly (A3 +0.066 AP, p < 0.001; blend +0.038 AP and +0.009 Popt, p ≤ 0.003). On **POI** the frozen graph variants collapse, but **joint cross-project STAR is the best model** (AP 0.920, +0.059 from joint data, p = 0.016). Joint STAR ≥ joint LR on every informative project. JEdit 4.3 (5 defects) is uninformative. | `TRANSFER_RESULTS.md` |
| 4 | Personal improvement of fusion / contrast | Label-aware InfoNCE and SupCon are neutral. **Gate entropy** balances the views (α 0.63 → 0.55) at no cost; retained for interpretability. The \|zT − zS\| gate is harmful. | `PHASE4_RESULTS.md` |

**The central finding (across 4 projects).** STAR's value is **project-dependent**:
- On Ant, no mechanism is measurable; one test release of 292 classes cannot resolve effects below about 0.03 AP.
- On Camel, the graph view gives the largest gain of the whole study.
- On POI, STAR wins only when trained with cross-project data.

The STAR-specific results that hold on every informative project:
- STAR **exploits heterogeneous cross-project data better than a linear model** (joint STAR ≥ joint LR on Ant, Camel and POI);
- the **STAR + LR + ET blend** improves effort-aware ranking over LR on the projects with the most training data (Ant, Camel).

Tree ensembles, mostly ExtraTrees, remain the safest effort-aware baseline everywhere.

---

## 2. Target architecture

### 2.1 System view

```
PROMISE archive ──prepare_multi.py──▶ data/<proj>/<proj>-<v>.npz                (strict 3-release windows, labelled)
                                     data/<proj>/<proj>-<v>_graph_temporal.npz  (all nodes, 3 graphs, temporal links)
                                              │
                  src/data.py (step 5.1): window loader · signed-log · masks · multi-project cohorts · domain ids
                                              ▼
 ┌──────────────────────────────────── src/model.py ─────────────────────────────────────┐
 │ x[N,3,F] → causal TCN (k=2, dil. 1,2) → zT[N,32]                                      │
 │ x[N,-1,F] + import edges → MeanSAGE ×2 → zS[N,32]                                     │
 │ a = cos(zT,zS) → gate MLP([zT,zS,a]) → α,β (+ entropy reg.) → z = αzT + βzS           │
 │ heads: risk(z) → BCE (pos_weight)   rank(z) → RankNet        inspection order: p / LOC│
 │ loss = BCE + λc·InfoNCE(zT,zS) [+ λr·RankNet] − λg·H(α,β)                             │
 └───────────────────────────────────────────┬───────────────────────────────────────────┘
                                             ▼
         compare_fair · effort_star · more_data · improve_star  (validation search → refit → single test)
                                             ▼
                  src/stats.py: bootstrap CI · paired bootstrap · Wilcoxon · Cliff's δ (+ Scott-Knott, step 6.2)
                                             ▼
         results/<phase>/<project>/ → analyse_star.py (figures) → app.py (Streamlit) → rapport/ (LaTeX)
```

### 2.2 Final model (for the app and the report's headline)

Chosen on **validation**, as the FICHE requires (STAR must be a component):

- **Detection, AP:** STAR A5 with gate entropy, 10-seed ensemble. AP 0.802, equal to LR and RF within the CI.
- **Effort-aware inspection:** the **STAR + LR + ET blend** inspected by score / LOC. Significantly better than LR (+0.008 Popt, p = 0.006). RF + density is reported as the strongest single baseline (Popt 0.750).

Step 6.1 re-confirms this choice with one final run.

### 2.3 Repository layout

```
prioritest-star/
├── ARCHITECTURE.md                ← this file
├── FICHE / GUIDE / Contrastive.pdf / LIRE_EN_PREMIER.txt
├── rapport/                       ← LaTeX, figures, compiled PDF
└── KIT_MULTI_PROJETS/
    ├── README.md                  ← one command per experiment + CPU times
    ├── requirements.txt           ← numpy, scipy, scikit-learn, torch, pandas, matplotlib, joblib, streamlit, altair
    ├── data/                      ← kit data (unchanged)
    ├── src/  model.py · train.py · stats.py · data.py (5.1)
    ├── run_star.py · explore_val.py · train_final.py                       ← ablations, search, STAR-R
    ├── compare_fair.py · effort_star.py · more_data.py · improve_star.py   ← Phases 1–4
    ├── transfer_study.py (6.3, done) · domain_adapt.py (5.2) · final_model.py (6.1)
    ├── analyse_star.py · app.py
    ├── PHASE{1..4}_RESULTS.md · TRANSFER_RESULTS.md · RESULTS_STAR_R.md
    ├── tests/                     ← 34 tests, incl. no-leakage tests
    ├── notebooks/PRIORITEST_STAR_Colab.ipynb
    └── results/                   ← committed; every number in the report comes from here
```

---

## 3. Completed phases (reference)

| Phase | Done | Command |
|---|---|---|
| 0 Consolidation | `.gitignore` keeps results and drops `.venv`; requirements completed; STAR-R reproduced (AP 0.798) | — |
| 1 Fair comparison | equal-budget search for A1–A7 + LR/RF/ET/HGB/kNN, bootstrap and Wilcoxon | `python compare_fair.py --project ant` (6.5 min) |
| 2 STAR-E | effort term inside the model, 3 variants | `python effort_star.py --project ant` (4 min) |
| 3 More data | cross-project zero-shot / joint / pre-train, STAR vs LR / RF | `python more_data.py --project ant` (4 min) |
| 4 Improvement | C1, C2, G1, G2, C2G1 vs A5 | `python improve_star.py --project ant` (5 min) |

Run Phase 1 before Phases 2 and 3, which read `results/phase1/`.

---

## 4. Remaining roadmap

Ordered by **expected value for the defence**. Each step lists the goal, what to do, and when it counts as done.

### Phase 5 — The experiments most likely to change the conclusions (3–4 days)

- [ ] **5.1 `src/data.py`.** One loader for strict windows, temporal-graph files and multi-project cohorts, with a project/domain id per cohort. Refactor the four phase scripts onto it; no change in results (check against the committed `results/`).
- [ ] **5.2 Domain adaptation (follow-up to Phase 3).** STAR is the model that transfers best, so make cross-project data actually help:
  - per-project normalisation layers (domain-specific BatchNorm/LayerNorm statistics);
  - **source-window weighting** by similarity to the target (MMD on metric distributions, computed on train features only);
  - optional adversarial project discriminator (gradient reversal) on z.
  - Select on Ant 1.6 and test once. **Done when** joint STAR is compared with Ant-only STAR with a CI. Success = ΔAP > 0 with a CI excluding 0.
- [ ] **5.3 Cold-start classes (R5).** Add `PROMISE-dataset-main.zip`, rerun `prepare_multi.py` and also export labels for the classes outside the strict window. Train with a presence mask (missing releases zeroed, mask fed to the TCN), then evaluate separately on old and new classes. **Done when** the report has a "selection bias" table: defect rate and recall for new vs old classes.
- [~] **5.5 Per-project validation for 2-window projects (follow-up to 6.3).** *(in progress: `retune_project.py`, see `RETUNE_RESULTS.md`. Camel done: STAR A3–A5 beat LR by +0.09 AP (p ≤ 0.01) and RankNet hurts (−0.041, p = 0.011). POI not run yet: `python retune_project.py --projects poi`)* Frozen Ant settings hurt the graph variants on POI. Retune epochs per project with an inner chronological check (train on the first window's first two releases, validate on its third) or with cross-project validation, then re-test.
- [ ] **5.4 Give STAR the trees' strength (follow-up to Phase 2).** RF wins on effort ranking through feature interactions, so:
  - (a) **distil** out-of-fold RF probabilities into STAR's risk head as a soft target;
  - (b) add a **third view**, a tabular MLP on [last metrics, Δ], into the similarity-guided gate.
  - Same protocol. **Done when** both are tested against RF + density on Popt.

### Phase 6 — Robustness of the claims (2 days)

- [ ] **6.1 `final_model.py`.** Freeze the final choice of §2.2 on validation and run it once with 10 seeds. Save the checkpoints plus the blend, and regenerate `results/analysis/` and the app's data from it.
- [~] **6.2 Scott-Knott ESD** *(code ready: `stats.scott_knott_esd` + `sk_rank.py`, unit-tested; to run once POI retuning is done)* ranking of all models (Phase 1 table) in `src/stats.py`, as in the SDP literature.
- [x] **6.3 Second and third target projects.** *(done: `python transfer_study.py`, see `TRANSFER_RESULTS.md`)* `transfer_study.py` runs the Phase-1 protocol with **configurations frozen from Ant** (no tuning on the new test) on:
  - **Camel** 1.4 → 1.6 (565 test classes, 127 defective; a bigger test set gives tighter CIs);
  - **POI** 2.5 → 3.0.
  - **JEdit** is reported with its bootstrap CI and the 5-positive caveat.
  - **Done when** the claims "STAR ≈ LR", "trees win Popt" and "STAR robust to mixing" are checked on ≥ 2 projects.
- [x] **6.4 Leakage tests** *(done: 4 tests, all pass)* (`tests/test_no_leakage.py`):
  - the search code never opens the test file (monkeypatch `np.load`);
  - normalisation statistics come from window 1 only;
  - the flag rate and the blend are computed from validation scores only.

### Phase 7 — Analysis & application (2 days)

- [ ] 7.1 Budget–recall curves with CI bands for: LR, RF, STAR A5, the blend, and the random and oracle orderings.
- [ ] 7.2 α / β and cos(zT, zS), with and without gate entropy, broken down by defective vs clean, correct vs wrong, unstable (large |Δ|) vs highly coupled (high CBO / in-degree).
- [x] 7.3 *(done: `python graph_analysis.py`, see `GRAPH_ANALYSIS.md`. On Camel isolated classes are 1.3 % defective vs 26 % connected (×20), and the whole graph gain is on connected classes; there is no defect homophily, so the signal is connectivity)* **Key analysis after 6.3:** performance by node-degree bucket (isolated, low, high) **per project**, plus graph statistics (density, coverage, defect assortativity: do defective classes import defective classes?). This explains why the graph helps on Camel and not on Ant or POI.
- [ ] 7.4 Error analysis plus SHAP on RF/LR: false positives are large coupled classes; missed defects are small, bland classes.
- [ ] 7.5 `app.py`:
  - selectors for project and model (LR, RF, STAR, blend) and for risk vs density ordering, plus a LOC budget slider;
  - a per-class panel showing α / β, agreement and the top metric deviations;
  - the app loads `results/` only.

### Phase 8 — Deliverables (2–3 days, in parallel)

- [ ] 8.1 Colab notebook runs everything end to end: tests → Phases 1–4 → final model → analysis → zip.
- [ ] 8.2 GitHub Actions: unit tests + 2-epoch smoke runs of each phase script.
- [ ] 8.3 Report update:

| Chapter | Content to add |
|---|---|
| Résumé / Abstract | Headline per §1.2: STAR's value is project-dependent (no gain on Ant, graph gain on Camel); STAR exploits cross-project data better than LR; the blend helps effort ranking |
| État de l'art | Critical reading of Rhazi et al. (weighted F1, random split, no effort metrics, Top-N); SAINT, FT-Transformer, bjCnet, DP-CCL; effort-aware SDP (Kamei 2013, Mende & Koschke 2010); cross-project SDP; TCN, GraphSAGE, InfoNCE / SupCon |
| Architecture | Final model (§2.2), loss equations, gate entropy, STAR-E |
| Données | Windows, coverage, prevalence shift (17 % → 35 %), the 37 % isolated nodes, cross-project pool |
| Expérimentations | Phase 1 table with CIs and tests; Phases 2–4 as documented null / positive results; the 4-project claims table from `TRANSFER_RESULTS.md` |
| Analyse | Phase 7 figures |
| Limites | R1, R4, R5, one main project, one test release, CI width |
| Perspectives | Steps 5.2–5.4 not yet done; prospective t → t+1 protocol with a label-availability audit; richer dependency extraction |

- [ ] 8.4 Defence slides: problem → protocol → 4 findings → app demo → limits.

---

## 5. Timeline

| Week | Work | FICHE §10 milestone |
|---|---|---|
| W1 (done) | Phases 0–4 | Data, baselines, STAR, ablations, protocol without leakage |
| W2 | 5.1, 5.2, 5.4, 5.5 | Experiments that can change the conclusions |
| W3 | 5.3 (if archive), 5.5, 6.1, 6.2 (6.3, 6.4, 7.3 done) | Robust, multi-project claims |
| W4 | 7, 8.1–8.2 | Explorable Top-k, figures, reproducibility |
| W5 | 8.3–8.4 | Report and defence |

---

## 6. Pushing to GitHub

The project lives at **https://github.com/Aymanelgr/prioritest-star-pfa** (private), which Claude can push to. The original team repo is `mohamedaoui/prioritest-star`. To update it from `C:\Users\ayman\OneDrive\Desktop\prioritest-star-main`, run in PowerShell:

```powershell
git init -b main
git remote add origin https://github.com/mohamedaoui/prioritest-star.git
git fetch origin main
git reset origin/main          # adopt GitHub history, keep local files as changes
git status                     # .venv/ and __pycache__/ must NOT be listed
git add -A
git commit -m "Phases 0-4: fair comparison, STAR-E, cross-project, fusion/contrast improvements, roadmap v2"
git push origin main
```

---

## 7. What "the best possible result" means here

The FICHE grades **correct reproduction, a leak-free protocol and interpretable ablations**, not a record score
("aucune performance minimale n'est garantie"). The strongest defensible outcome is therefore:

1. A fair, equal-budget, significance-tested ablation showing that, on Ant, **no STAR mechanism gives a measurable gain**, with the reason: 118 training classes and a test CI of ± 0.07. *(done)*
2. The **positive STAR-specific findings**: the graph gain on Camel, the better use of cross-project data than LR (Ant, Camel, POI), and the blend's significant gain on effort ranking (Ant, Camel). *(done, 6.3)*
3. One more serious attempt to make STAR's design pay off (5.2 domain adaptation, 5.4 tree distillation / third view), reported whatever the outcome.
4. A clear app and report that never overclaim (R1, R4, R5).
