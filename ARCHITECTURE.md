# PRIORITEST-STAR — Architecture & Roadmap

> Effort-aware prioritisation of defect-prone Java classes (PROMISE) with a temporal view (TCN),
> a structural view (GraphSAGE on Java imports), similarity-guided fusion, cross-view InfoNCE and RankNet.
> EMSI Marrakech — PFA IA, 5th year.

This file is the master plan. It says **where the project stands**, **what the final system looks like**,
and **every step still needed**, in order, with the command to run and the proof that the step is done.

---

## 0. Ground rules (never broken)

These come from the FICHE and the GUIDE. Every step below must respect them, and the report must state them.

| # | Rule | How it is enforced |
|---|---|---|
| R1 | **Detection of defects in the observed release**, never "prediction of the next release". | Wording in README, report, app caption. |
| R2 | **Test release is touched once**, after every choice is frozen. No epoch, λ, threshold, blend or seed picked on test. | `explore_val.py` never loads the test; `train_final.py` evaluates it once per group. |
| R3 | Normalisation fitted on the **train window only**. | `prepare_multi.py` + signed-log on top. |
| R4 | The import graph is a **partial proxy**, not a full dependency graph. | Stated in data chapter + limits. |
| R5 | New classes (no 3-release history) are **excluded** → selection bias must be analysed. | Phase 3 cold-start study. |
| R6 | ≥ 3 seeds (target 5–10), report **mean ± std**, parameter counts and measured times. | `summary.csv`, `final_metrics.json`. |
| R7 | Each ablation adds **one** mechanism and is retrained from scratch on identical splits. | `run_star.py` A1→A6. |
| R8 | Rhazi et al. (2026) is cited for the *principle* of similarity learning; STAR is **not** a reproduction of it. | State of the art + positioning table. |

---

## 1. Where the project stands (audit, 27 Nov 2026)

### 1.1 Done

| Area | Status | Files |
|---|---|---|
| Data kit (10 projects, 16 windows, temporal-graph extension) | ✅ verified (`check_graph_temporal.py` OK) | `data/`, `prepare_multi.py` |
| Model: causal TCN, MeanSAGE (uni/bi-directional), similarity gate, Δ-metrics, dropout/LayerNorm | ✅ | `src/model.py` |
| Losses: BCE (+pos_weight), InfoNCE, label-aware InfoNCE, RankNet, effort-aware RankNet | ✅ | `src/model.py` |
| Metrics: AP, ROC-AUC, Recall@20% classes / LOC, effort-aware Recall@20% LOC, Acc/BalAcc/MCC/F1 with validation-chosen flag rate | ✅ | `src/train.py` |
| Ablation ladder A1→A6 + A7 STAR+ + A8 ensemble, tuning on validation, 5 seeds | ✅ | `run_star.py` |
| Validation-only grid (192 configs × 3 seeds, smoothed seed-averaged epoch) | ✅ | `explore_val.py`, `config_star_r.json` |
| STAR-R final: 10-seed STAR ensemble + LR/ET blend chosen on validation, refit train+val, 5 groups | ✅ | `train_final.py`, `results/star_r/` |
| Analysis: budget–recall curve, Popt, α/β, agreement, error cases | ✅ | `analyse_star.py`, `results/analysis/` |
| Streamlit app (Top-k, LOC budget, density vs risk ordering) | ✅ | `app.py` |
| Unit tests (19) | ✅ | `tests/` |
| Colab notebook | ✅ (does not yet run STAR-R / analysis / app) | `notebooks/PRIORITEST_STAR_Colab.ipynb` |
| LaTeX report (8 chapters) | ✅ first full draft | `rapport/` |

### 1.2 Current numbers (from `RESULTS_STAR_R.md`, Ant 1.5 → 1.6 → 1.7)

| Model | AP | ROC-AUC | MCC | EA-R@20%LOC |
|---|---|---|---|---|
| A1 boosting | 0.752 | 0.843 | 0.472 | 0.314 |
| A6 STAR full (default) | 0.760 ± 0.024 | 0.832 | 0.480 | 0.312 |
| STAR-R FINAL (val-chosen blend) | **0.799 ± 0.003** | **0.870** | **0.610** | 0.320 |
| Logistic regression alone | 0.798 | 0.864 | 0.604 | 0.304 |

`analysis.json` adds: **Popt** STAR-R sorted by risk = 0.369 (< random 0.502), sorted by risk/LOC = 0.719.
JEdit 4.3 test = 5 positives / 179 → AP ≈ 0.26, not interpretable on its own.

### 1.3 What these numbers tell us (the real gaps)

1. **The network ≈ a regularised logistic regression.** Graph + contrastive add little measurable signal on Ant
   (118 training classes, 20 positives). This is the central scientific finding to *test harder*, not hide.
2. **The model is size-driven.** Sorting by raw risk is *worse than random* under a LOC budget (Popt 0.37).
   Effort-awareness is only applied *after* training (score/LOC). The ranking objective never sees effort.
3. **Training data is tiny.** The strict 3-release intersection keeps 118 of 293 Ant-1.5 nodes. The
   `_graph_temporal.npz` files (all nodes, all 3 graphs, temporal links) are **not used yet**.
4. **No statistical tests.** Differences of 0.001–0.01 AP are reported without CIs or significance.
5. **One real study project.** JEdit test is degenerate; Camel/POI (2 windows) are run only exploratorily.
6. **Ablations are not tuned equally.** A2–A6 use defaults; STAR-R uses a 192-config grid → unfair comparison.
7. **Repo hygiene.** Results are git-ignored, `requirements.txt` lacks `streamlit`, `altair`, `joblib`;
   `.venv/` and `__pycache__/` are in the working folder; local work is not pushed; the notebook stops before STAR-R.
8. **GUIDE inconsistency.** GUIDE says "4 variants", FICHE says a 6-step ladder → keep the ladder, also report the 4 kit variants for traceability.

---

## 2. Target architecture

### 2.1 System view

```
PROMISE archive ──prepare_multi.py──▶ data/<proj>/<proj>-<v>.npz          (strict 3-release windows)
                                     data/<proj>/<proj>-<v>_graph_temporal.npz (all nodes, 3 graphs, temporal links)
                                              │
                                              ▼
                     ┌──────────── src/data.py (NEW) ─────────────┐
                     │ load window · signed-log · masks · cohorts │
                     │ cold-start nodes (mask) · multi-project    │
                     └────────────────────┬───────────────────────┘
                                          ▼
         ┌──────────────────────── src/model.py ───────────────────────────┐
         │  x[N,3,F] ─▶ TCN causal (k=2, d=1,2) ─▶ zT[N,32]                │
         │  x[N,-1,F] + edges ─▶ MeanSAGE ×2 ─▶ zS[N,32]                   │
         │  a = cos(zT,zS) ─▶ gate MLP([zT,zS,a]) ─▶ α,β ─▶ z = αzT + βzS  │
         │  heads: risk(z) → BCE      rank(z, log LOC) → RankNet/LambdaRank│
         │  loss = BCE + λc·InfoNCE(zT,zS) + λr·Rank                       │
         └───────────────────────────────┬─────────────────────────────────┘
                                         ▼
   run_star.py (ablations A1–A8)   explore_val.py (val-only search)   train_final.py (STAR-R, single test)
                                         ▼
          src/stats.py (NEW): bootstrap CI · Wilcoxon · Cliff's δ · Scott-Knott ranking
                                         ▼
     results/<exp>/… (json, csv, .pt)  ─▶  analyse_star.py (figures)  ─▶  app.py (Streamlit)  ─▶  rapport/
```

### 2.2 Target repository layout

```
prioritest-star/
├── ARCHITECTURE.md            ← this file
├── README.md                  ← install + one command per experiment
├── FICHE / GUIDE / Contrastive.pdf / LIRE_EN_PREMIER.txt
├── rapport/                   ← LaTeX + figures + compiled PDF
└── KIT_MULTI_PROJETS/
    ├── requirements.txt       ← + streamlit, altair, joblib (pinned)
    ├── configs/               ← one JSON per model (A1…A8, STAR-R, STAR-E, STAR-T) (NEW)
    ├── data/                  ← untouched kit data
    ├── src/
    │   ├── model.py           ← + effort-aware rank head, LambdaRank, cold-start mask
    │   ├── train.py           ← metrics (+ Popt, IFA, Recall@k LOC curve)
    │   ├── data.py            ← (NEW) loaders for windows, temporal graphs, multi-project
    │   └── stats.py           ← (NEW) significance + CIs
    ├── run_star.py            ← ablations (now reads configs/, equal tuning budget)
    ├── explore_val.py         ← validation-only search (now per variant)
    ├── train_final.py         ← final protocol, single test pass
    ├── analyse_star.py        ← figures + tables for the report
    ├── app.py                 ← Streamlit
    ├── tests/                 ← + test_data.py, test_stats.py, test_no_leakage.py
    ├── notebooks/PRIORITEST_STAR_Colab.ipynb   ← runs everything end-to-end
    └── results/               ← COMMITTED (small files) — numbers in the report must be reproducible
```

---

## 3. Roadmap

Each step has: **goal · tasks · command · done when**. Phases are ordered by
(1) what protects the validity of results, (2) what is most likely to improve them, (3) presentation.

### Phase 0 — Consolidate the repo (½ day)

**Goal:** one clean, pushed, reproducible baseline before any new experiment.

- [ ] 0.1 Push the local folder (`prioritest-star-main`) to `mohamedaoui/prioritest-star` (commands in §6).
- [ ] 0.2 `.gitignore`: ignore `.venv/`, `__pycache__/`, `*.pyc`; **stop ignoring `results/`** (keep files < 5 MB; LFS or a release asset for bigger ones like `tabular_models.joblib`).
- [ ] 0.3 `requirements.txt`: add `streamlit`, `altair`, `joblib`; pin versions that were used (`pip freeze` from `.venv`).
- [ ] 0.4 Remove duplicates: `Claude outputs/` vs `rapport/`, two copies of the PDF report, `to do.txt` (move to `docs/` or delete).
- [ ] 0.5 Tag the current state: `git tag v0.1-star-r`.

**Done when:** a fresh clone + `pip install -r requirements.txt` + `python -m unittest discover -s tests` passes, and `python train_final.py --project ant --groups 5 --ens 10` reproduces AP 0.799 ± 0.003.

---

### Phase 1 — Make the comparison fair and statistically sound (1–2 days)

**Goal:** every claim in the report backed by a CI or a test. This matters more than any +0.01 AP.

- [ ] 1.1 **Equal tuning budget.** Run `explore_val.py` for **each** neural variant A2–A7 (same grid size, same seeds, same smoothed-epoch rule) and save `configs/A*.json`. Tabular baselines (HGB, LR, ET, RF) get a small grid on validation too.
- [ ] 1.2 **Re-run the ladder with tuned configs**, 10 seeds, refit train+val, single test pass:
  `python run_star.py --project ant --seeds 42..51 --config-dir configs --improved`
- [ ] 1.3 `src/stats.py`:
  - class-level **bootstrap 95% CI** on test AP / Popt / Recall@20%LOC (2 000 resamples);
  - **Wilcoxon signed-rank** over seeds between consecutive ablation steps (A_k vs A_{k-1});
  - **Cliff's δ** effect size; **Scott-Knott ESD** ranking of all models.
- [ ] 1.4 Add strong, standard baselines to the table: **LR**, **Random Forest**, **HGB**, and a
  **Rhazi-style kNN** baseline (cosine similarity on standardised metrics, Top-N weighted vote, N chosen on validation) — a cheap, honest proxy of the reference paper's decision rule.
- [ ] 1.5 Test `tests/test_no_leakage.py`: assert the test file is never opened by `explore_val.py`; assert normalisation stats come from window 1 only; assert flag rate is computed from validation scores.

**Command:** `python run_star.py … && python -m src.stats results/ant_tuned`
**Done when:** the report's ablation table has mean ± std, 95% CI and a significance mark per step.

---

### Phase 2 — Make effort-awareness native (main expected gain) (2–3 days)

**Why:** today the model ranks by size (Popt 0.37 < random) and only becomes effort-aware through a post-hoc
score/LOC sort. The FICHE's research question is *effort-sensitive ranking*. Training for it directly is the
most promising lever.

- [ ] 2.1 **Effort-aware rank head.** Score `s_i = rank(z_i) − γ·log LOC_i` with γ learnable (or fixed = 1). Inference ranks by `s_i` directly. Ablate γ = 0 / 1 / learned.
- [ ] 2.2 **LambdaRank with ΔPopt gain.** Weight each (defective i, clean j) pair by |ΔPopt| obtained by swapping them in the current ranking (density-based). Compare to the existing `effort_aware_rank_loss`.
- [ ] 2.3 **Target choice.** Also try ranking on *defect density* (`bug>0` / LOC) as the target of the rank head; keep BCE on the binary label.
- [ ] 2.4 Add metrics: **Popt**, **IFA** (initial false alarms before the first defect), **Recall@{10,20,30,50}% LOC**, curve area — all from `src/train.py`.
- [ ] 2.5 Select on **validation Popt** (not AP) for the effort-aware variants. Name the variant **STAR-E**.

**Done when:** STAR-E Popt / Recall@20%LOC on Ant 1.7 is compared against STAR-R + post-hoc density and LR + density, with CIs. Expect a gain on effort metrics and possibly a small loss on AP: report both.

---

### Phase 3 — Use more data: all nodes, temporal edges, other projects (3–4 days)

**Why:** 118 training classes / 20 positives is the bottleneck for any neural model.

- [ ] 3.1 **Cold-start nodes (fixes R5).** Load `_graph_temporal.npz`: build x[N_all, 3, F] with a presence **mask** (missing versions zeroed, mask fed to the TCN, Δ = 0 when absent). Train on all target-release nodes (Ant 1.5 window: 293 nodes instead of 118). Labels exist only for classes in the target CSV → mask the loss.
  Evaluate **separately** on (a) strict-intersection classes (comparable to before) and (b) new classes.
- [ ] 3.2 **Full graph.** Message passing on the complete import graph of the target release (more neighbours per node), not only on the intersection.
- [ ] 3.3 **Spatio-temporal GNN (variant STAR-T).** Global graph = 3 version graphs + `temporal_edges_global`; GraphSAGE over spatial + temporal edges; read out nodes of the last version. Keep temporal edges past→present only (causal).
- [ ] 3.4 **Cross-project pre-training (FICHE bonus).** Pre-train encoders on other projects' windows (Camel, POI, Xalan, Lucene, Synapse, Velocity, Xerces, Log4j — **never Ant**), per-project normalisation, then fine-tune on Ant 1.5 and select on Ant 1.6. Ablate: no pre-training / pre-training / pre-training + fine-tuning.
- [ ] 3.5 Second study project: **Camel** (train 1.4 → test 1.6) and **POI** (2.5 → 3.0) with configs frozen from Ant (no tuning on their test). JEdit reported with bootstrap CI and the "5 positives" caveat.

**Done when:** a table "data regime × model" (strict / +cold-start / +temporal graph / +pre-training) on Ant, plus Camel & POI transfer results.

---

### Phase 4 — Improve the fusion / contrast (the GUIDE's required personal contribution) (1–2 days)

Pick 1–2, implement, ablate against A6 with the Phase-1 protocol.

- [ ] 4.1 **Label-aware InfoNCE done right**: mask same-label negatives (already coded) **plus** add supervised positives (other defective classes) with a small weight — a SupCon/InfoNCE hybrid. Report false-negative rate before/after.
- [ ] 4.2 **Hard-negative mining** in InfoNCE: negatives sampled among structurally close but differently-labelled classes (graph neighbours).
- [ ] 4.3 **Agreement-aware gating**: feed |zT − zS| and a learnable temperature on `a` to the gate; regularise α towards 0.5 when views agree (entropy penalty) to avoid the observed α≈0.66–0.72 collapse to the temporal view.
- [ ] 4.4 **Multi-task stability**: auxiliary head predicting Δmetrics (churn) of the target release from the history — self-supervision with no label leakage.

**Done when:** one improvement shows a significant (Wilcoxon p < 0.05) gain on at least one primary metric, or a clearly documented null result.

---

### Phase 5 — Analysis & explanation (1–2 days)

- [ ] 5.1 Budget–recall curves for all final models (risk order and density order), with CI bands.
- [ ] 5.2 α/β and cos(zT,zS) distributions: defective vs clean, correct vs wrong, historically unstable (large |Δ|) vs highly coupled (high CBO / in-degree).
- [ ] 5.3 Error analysis: top false positives (large, coupled classes) vs missed defects (small, bland metrics) — already started; add per-feature SHAP on LR/HGB to explain *why*.
- [ ] 5.4 Graph contribution: performance by node degree bucket (isolated / low / high degree).
- [ ] 5.5 Selection-bias study (from 3.1): new classes vs old classes — defect rate, model recall.

---

### Phase 6 — Application (½–1 day)

- [ ] 6.1 `app.py`: project selector (Ant / Camel / POI / JEdit), model selector (A1, LR, STAR-R, STAR-E), budget slider, Top-k table with metrics, α/β and agreement per class, budget–recall curve, "why ranked here" panel (top metric deviations).
- [ ] 6.2 Load from `results/` only (no training in the app). Add `streamlit run app.py` to README and a screenshot to the report.

---

### Phase 7 — Reproducibility & deliverables (1 day)

- [ ] 7.1 README: one command per experiment (baseline, ablations, search, final, analysis, app) with measured CPU time.
- [ ] 7.2 Notebook: clone → install → tests → ablations → STAR-R → STAR-E → analysis → zip results.
- [ ] 7.3 `configs/` complete; every result folder contains the config + git commit hash + seeds + times (`metrics.json`).
- [ ] 7.4 CI (GitHub Actions): unit tests + a 2-epoch smoke run on CPU.
- [ ] 7.5 Checkpoints for final models in `results/` (or GitHub release if > 50 MB).

---

### Phase 8 — Report (2–3 days, in parallel with 5–7)

Update `rapport/rapport_prioritest_star.tex` chapter by chapter:

| Chapter | To add / fix |
|---|---|
| Résumé / Abstract | final numbers with CIs; honest headline (STAR-R ≈ strong LR on Ant; gains on effort metrics if Phase 2 confirms) |
| État de l'art | Rhazi et al. critical reading (weighted F1, random split, no effort metrics, Top-N); SAINT, FT-Transformer, bjCnet, DP-CCL; effort-aware SDP (Kamei 2013, Mende & Koschke 2010); TCN; GraphSAGE; InfoNCE/SupCon |
| Architecture | STAR-E / STAR-T diagrams, losses with equations |
| Données | cold-start counts, graph coverage, prevalence shift table |
| Expérimentations | tuned ablation ladder + stats; data-regime table; Camel/POI transfer |
| Analyse | Phase 5 figures |
| Limites | R1, R4, R5, small N, JEdit, single main project |
| Perspectives | prospective (t → t+1) protocol with label-availability audit; richer dependency extraction |

---

## 4. Experiment matrix (what goes in the final tables)

| ID | Model | Data regime | Selected on | Main metrics |
|---|---|---|---|---|
| B1–B4 | HGB, LR, RF, kNN-cosine (Rhazi-style) | strict | val AP | AP, Popt, R@20%LOC |
| A2–A6 | ablation ladder (tuned) | strict | val AP | AP, ROC-AUC, MCC, R@20% |
| A7 | STAR+ | strict | val AP | idem |
| STAR-R | 10-seed ensemble + val-chosen blend | strict | val AP | all |
| STAR-E | effort-aware rank head / LambdaRank | strict | val Popt | Popt, R@20%LOC, IFA |
| STAR-C | STAR-R/E + cold-start nodes | all nodes | val | split old/new classes |
| STAR-T | spatio-temporal GNN | all nodes + temporal edges | val | all |
| STAR-P | + cross-project pre-training | other projects → Ant | val | all |
| X-Camel / X-POI | frozen configs transferred | 2-window projects | none (fixed) | exploratory |

All with ≥ 10 seeds (or 5 groups × 10-seed ensembles), mean ± std, bootstrap CI, Wilcoxon vs the previous row.

---

## 5. Timeline (suggested)

| Week | Phases | Milestone (FICHE §10) |
|---|---|---|
| W1 | 0, 1 | Data + first models reproducible, no leakage, fair baselines |
| W2 | 2, 4 | STAR-E + personal improvement, ablations complete |
| W3 | 3 | More data: cold-start, temporal graph, pre-training, Camel/POI |
| W4 | 5, 6, 7 | Analysis, app, reproducibility |
| W5 | 8 | Report + defence slides |

---

## 6. Pushing the local folder to GitHub (Phase 0.1)

Run in PowerShell from `C:\Users\ayman\OneDrive\Desktop\prioritest-star-main` (the folder is not a git repo yet):

```powershell
git init -b main
git remote add origin https://github.com/mohamedaoui/prioritest-star.git
git fetch origin main
git reset origin/main          # adopt GitHub history, keep your local files as changes
git status                     # check .venv/ and __pycache__/ are NOT listed
git add -A
git commit -m "STAR-R final training, validation search, analysis, Streamlit app, report, roadmap"
git push origin main
```

---

## 7. Definition of "best possible result" for this project

Per the FICHE, success = **correct reproduction + leak-free protocol + interpretable ablations**, not a record score.
The strongest defensible outcome is therefore:

1. A fair, tuned, significance-tested ablation showing **which** mechanism helps (or doesn't).
2. A **native effort-aware** model that beats LR + post-hoc density on Popt / Recall@20%LOC with a CI that excludes zero — this is where STAR can add something a linear model can't.
3. Evidence that **more data** (cold-start nodes, temporal graph, cross-project pre-training) is what unlocks the graph and contrastive parts.
4. An honest report: if LR still ties, that is a result, explained by data size and metric redundancy.
