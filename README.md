# PRIORITEST-STAR

**Which Java classes should a tester inspect first, when there is only time to read a fraction of the code?**

PRIORITEST-STAR is a deep-learning model that ranks the classes of a Java project by their risk of containing a defect.
Inspecting the top of the ranking finds as many defects as possible within a fixed inspection budget, measured in lines of code.
It is the PFA (capstone) project of the 5th-year Data & AI track at EMSI Marrakech.

> Master plan and progress: [`ARCHITECTURE.md`](ARCHITECTURE.md) · Detailed results: `KIT_MULTI_PROJETS/*RESULTS*.md`

---

## 1. The problem in plain words

Testing and code review are expensive, and teams never have time to inspect everything.
**Software defect prediction** uses measurements of the code to guess which parts are likely to be buggy, so effort goes there first.

Two things make this project more demanding than a plain "buggy / not buggy" classifier:

1. **It is a ranking problem under a budget.** Big classes are more often buggy, but they also cost more to read. A useful model must find many defects *per line inspected*, not only many defects. This is called **effort-aware** evaluation.
2. **Code has history and structure.** A class is not an isolated row in a table. It evolves across releases (its *history*) and it imports other classes (its *dependencies*). STAR tries to use both.

## 2. The data

The data is the **PROMISE** benchmark: open-source Java projects (Ant, Camel, POI, JEdit, Xalan…) measured over several releases.

| Element | Meaning |
|---|---|
| **20 metrics per class** | Classic object-oriented measures: size (LOC), complexity (WMC, max_cc), coupling (CBO, Ca, Ce, RFC), cohesion (LCOM, CAM), inheritance (DIT, NOC)… |
| **Window of 3 releases** | For a *target* release t, the metrics of the same class at t−2, t−1 and t, giving a tensor `X[N, 3, 20]`. Only classes present in all 3 releases are kept. |
| **Label** | 1 if the class has at least one recorded bug in release t. |
| **Import graph** | An edge A → B when class A imports class B in release t. It is extracted from the source code and is a *partial* view of the real dependencies. |
| **LOC** | Lines of code of each class, used as the inspection cost. |

The protocol is **chronological**. On Ant, the model is trained on release 1.5, settings are chosen on 1.6, and it is tested **once** on 1.7.
The model *detects* defects in the observed release; it does **not** predict the next release (rule R1 in `ARCHITECTURE.md`).

## 3. The model: STAR

```
 metrics over 3 releases ──► causal TCN ───────────────► zT  (temporal view: how the class evolved)
                                                            │
 metrics of release t + import graph ──► GraphSAGE ────► zS  (structural view: where the class sits)
                                                            │
                     a = cos(zT, zS)   ◄── how much the two views agree
                     (α, β) = softmax(MLP[zT, zS, a])        similarity-guided fusion
                     z = α·zT + β·zS
                                                            │
             risk head ─► defect probability (BCE loss)     rank head ─► pairwise ranking (RankNet loss)
```

| Block | What it does | Why |
|---|---|---|
| **Causal TCN** (temporal convolution, kernel 2, dilations 1 and 2) | Reads the 3 releases in order, never looking ahead | Unstable or growing classes are riskier |
| **GraphSAGE** (2 mean-aggregation layers) | Mixes each class with the classes it imports | Coupling and position in the architecture |
| **Similarity-guided fusion** | Weighs the two views using their agreement cos(zT, zS) | Trust a view more when the two agree |
| **Cross-view InfoNCE** (contrastive loss) | Pulls zT and zS of the *same* class together, pushes other classes apart | Aligns the two views |
| **RankNet** | Learns that a defective class should score above a clean one | The final goal is a ranking |

Total loss: `BCE + λc·InfoNCE + λr·RankNet` (+ an optional gate-entropy term that keeps both views in use).

**Positioning.** Rhazi et al. (2026, *Science of Computer Programming*) learn similarities *between different classes* with inter-sample attention and a kNN vote. STAR borrows the idea of learning similarities, but between *two views of the same class*. It is not a reproduction of that paper.

## 4. How results are measured

| Metric | Question it answers |
|---|---|
| **AP** (average precision) | How well are defective classes ranked above clean ones? (main detection metric) |
| ROC-AUC | Same idea, less sensitive to the defect rate |
| **Popt** | Area under the "defects found vs % of code inspected" curve, normalised: 1 = perfect, 0.5 ≈ random, 0 = worst |
| **Recall@20 % LOC** | Share of defects found after reading 20 % of the lines of code |
| IFA | Clean classes inspected before the first defect |

For the effort metrics, classes are inspected in order of **risk ÷ LOC** (risk per line).

Every number is reported as a **mean ± std over 10 seeds** with a **95 % bootstrap confidence interval**, and models are compared with **paired statistical tests**.
Nothing is tuned on the test release: four automated tests check this (`tests/test_no_leakage.py`).

## 5. What was found

| Study | Result |
|---|---|
| **Ablation on Ant** (Phase 1) | Each STAR mechanism was added one at a time, with the same tuning budget for every variant. **None gives a significant gain on Ant**: all variants tie logistic regression (AP ≈ 0.80). Random Forest / ExtraTrees are the best effort-aware models (Popt 0.75). |
| **Effort learned inside the model** (Phase 2) | No gain. The model converges to the simple "risk ÷ LOC" rule. |
| **Training with 9 other projects** (Phase 3) | Does not help Ant (projects differ), but **STAR copes with mixed data better than a linear model**. |
| **Improvements to fusion / contrast** (Phase 4) | Label-aware and supervised contrastive losses: neutral. **Gate entropy** balances the two views at no cost (kept). |
| **Other projects** (step 6.3) | **On Camel, STAR with the graph beats logistic regression significantly (+0.066 AP)**. On POI, STAR trained with cross-project data is the best model (AP 0.920). JEdit's test release has only 5 defects and is uninformative. |
| **Why the graph helps on Camel** (step 7.3) | On Camel, isolated classes are almost never defective (1.3 % vs 26 % for connected ones). The graph captures this; on Ant the contrast is weak (×1.7 vs ×20). |

**Take-away.** STAR's value is **project-dependent**:
- the structural view pays off when connectivity separates risky from safe classes (Camel);
- the model makes better use of cross-project data than a linear model;
- blended with simple tabular models, it improves effort-aware ranking (Ant, Camel).

On a small project like Ant (118 training classes), simple models are just as good. That is reported honestly rather than hidden.

## 6. Repository map

```
ARCHITECTURE.md              master plan: rules, status, architecture, remaining steps
README.md                    this file
rapport/                     LaTeX report (+ PDF, figures)
FICHE / GUIDE / LIRE_EN_PREMIER   assignment documents from the supervisor
KIT_MULTI_PROJETS/
├── data/                    PROMISE windows (.npz) for 10 projects + temporal graphs + manifests
├── prepare_multi.py         rebuilds data/ from the PROMISE archive
├── src/model.py             STAR: TCN, GraphSAGE, fusion gate, InfoNCE / SupCon, RankNet, gate entropy
├── src/train.py             data loading and metrics (AP, Recall@LOC, accuracy/MCC with validation-chosen threshold)
├── src/stats.py             Popt, IFA, bootstrap CIs, paired bootstrap, Wilcoxon, Cliff's δ
├── run_star.py              ablation ladder A1–A8 (one mechanism per step)
├── explore_val.py           hyper-parameter search on the validation release only
├── train_final.py           STAR-R: 10-seed ensemble + blend chosen on validation, single test pass
├── compare_fair.py          Phase 1  — fair, equally tuned comparison with statistics
├── effort_star.py           Phase 2  — STAR-E, effort term learned in the model
├── more_data.py             Phase 3  — cross-project training
├── improve_star.py          Phase 4  — fusion / contrast improvements
├── transfer_study.py        Step 6.3 — Camel, POI, JEdit with settings frozen from Ant
├── graph_analysis.py        Step 7.3 — graph statistics and where the graph helps
├── analyse_star.py          figures: budget–recall curve, fusion weights, ranked class list
├── app.py                   Streamlit app: explore the Top-k classes under a LOC budget
├── tests/                   34 unit tests (model, metrics, statistics, no-leakage)
├── notebooks/               Google Colab notebook
├── results/                 every number of the report (json / csv / checkpoints)
└── *_RESULTS.md, GRAPH_ANALYSIS.md   one write-up per study
```

## 7. Tools and why each is used

| Tool | Role in the project |
|---|---|
| **Python 3** | Everything |
| **PyTorch** | The STAR network (TCN, GraphSAGE written by hand, losses), training on CPU or GPU |
| **scikit-learn** | Baselines (logistic regression, Random Forest, ExtraTrees, gradient boosting) and metrics (AP, ROC-AUC, MCC) |
| **SciPy** | Wilcoxon signed-rank test |
| **NumPy / pandas** | Data handling, bootstrap resampling |
| **Matplotlib** | Report figures |
| **Streamlit + Altair** | Interactive app for testers |
| **joblib** | Saving tabular models |
| **unittest** | Automated tests, including protocol tests |
| **Google Colab** | Running the notebook without a local setup |
| **LaTeX (Overleaf)** | The report |
| **Git / GitHub** | Versioning; every result in the report is committed in `results/` |

## 8. Run it

```bash
cd KIT_MULTI_PROJETS
pip install -r requirements.txt
python -m unittest discover -s tests -v       # 34 tests
python compare_fair.py --project ant          # Phase 1 (~6.5 min CPU)  → results/phase1, PHASE1_RESULTS.md
python effort_star.py --project ant           # Phase 2 (~4 min)
python more_data.py --project ant             # Phase 3 (~4 min)
python improve_star.py --project ant          # Phase 4 (~5 min)
python transfer_study.py                      # step 6.3 (~5 min)
python graph_analysis.py                      # step 7.3 (15 s)
python train_final.py --project ant --groups 5 --ens 10 && python analyse_star.py --project ant
streamlit run app.py                          # the application
```

Run Phase 1 first: Phases 2, 3 and 6.3 read its frozen settings. No GPU is needed; every study runs in minutes on 2 CPU cores.

## 9. Limits

- **Detection, not forecasting**: the metrics of release t are used to find the defects of release t.
- The import graph is **partial**: explicit imports only, with no method calls and no inheritance edges.
- **New classes are excluded** (no 3-release history), which biases the sample. Their labels require the full PROMISE archive.
- One main project with one test release: a confidence interval of about ±0.07 AP cannot detect small effects.

## 10. References

- Rhazi, Banouar, Toure, Raghay (2026). *Software defect prediction via contrastive similarity learning and inter-sample attention.* Science of Computer Programming 254, 103534.
- Kamei et al. (2013), Mende & Koschke (2010): effort-aware defect prediction (Popt).
- Bai, Kolter, Koltun (2018): Temporal Convolutional Networks. Hamilton, Ying, Leskovec (2017): GraphSAGE.
- van den Oord et al. (2018): InfoNCE. Khosla et al. (2020): Supervised Contrastive Learning.
- PROMISE / tera-PROMISE defect datasets (Jureczko & Madeyski).
