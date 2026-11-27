# STAR-R — audit, référence et entraînement amélioré (Ant)

Tous les chiffres ci-dessous proviennent d'exécutions du 27/11/2026 (CPU uniquement, 2 cœurs, PyTorch 2.15, scikit-learn 1.9).

## 1. Le projet
- **Tâche** : classer / détecter les classes Java à risque de défaut (PROMISE). Label = au moins un défaut dans la version **cible**. Il s'agit d'une détection sur la version courante, pas d'une prévision de la version suivante (voir le README du kit).
- **Données (Ant)** : fenêtres de 3 versions × 20 métriques CK et de taille, plus un graphe d'imports Java de la version cible.
  Entraînement = 1.5 (118 classes, 20 défectueuses, 17 %), validation = 1.6 (167, 60 défectueuses, 36 %), test = 1.7 (292, 102 défectueuses, 35 %).
- **Modèle (fiche §3)** : TCN causal (2 Conv1D, noyau 2, dilatations 1/2) → zT ; GraphSAGE moyen à 2 couches → zS ; porte cos(zT,zS) → fusion ;
  perte = BCE + λc·InfoNCE + λr·RankNet. Ablations A1–A8 dans `run_star.py`.
- **Évaluation** : AP (principale), ROC-AUC, Recall@20 % des classes / des LOC, Recall@20 % LOC sensible à l'effort (score/LOC) ;
  Accuracy/Précision/Rappel/F1/MCC utilisent un taux de signalement choisi sur la validation.

## 2. Résultats de l'audit
| Vérification | Résultat |
|---|---|
| Fuite de labels | Aucune : les variables ne viennent que des 3 versions de la fenêtre ; les labels de test ne servent jamais à l'apprentissage, à l'arrêt anticipé, au réglage ou aux seuils. |
| Normalisation | Centrage-réduction ajusté sur la seule première fenêtre (entraînement). Correct. Valeurs écrêtées à ±10 ; des queues lourdes subsistent (LOC, RFC…). |
| Recouvrement d'entités | 117 des 118 classes d'entraînement et les 167 classes de validation réapparaissent dans la fenêtre de test (mêmes classes, versions ultérieures). Ce n'est pas une fuite de labels, mais les scores de test ne portent **pas** sur du code inédit. À signaler. |
| Déséquilibre des classes | 17 % de défectueuses à l'entraînement, 35 % en validation et en test. Ce décalage de prévalence impose `pos_weight` et un taux de signalement choisi sur la validation ; l'accuracy brute doit être lue face à la classe majoritaire (0,651 sur le test). |
| Petite taille | 118 classes d'entraînement → forte variance entre graines (écart-type de l'AP de A6 : 0,024 ; l'époque d'arrêt anticipé varie de 10 à 48 selon la graine). |
| Code | Le rembourrage du TCN est causal (vérifié), aucun NaN, 19 tests unitaires sur 19 réussis à ce stade (36 aujourd'hui). `run_star.py --tune` règle sur une seule graine (bruité). A7 (`STAR_plus`) est pénalisé par le dropout et le SAGE bidirectionnel sur si peu de données. |
| JEdit | La version de test 4.3 n'a que **5 positifs sur 179** → métriques non interprétables (tous les modèles sont sous l'accuracy de la classe majoritaire, 0,972). |
| GPU | Pas de GPU sur ta machine virtuelle ni dans l'espace cloud. Les modèles s'entraînent en quelques secondes sur CPU. |

## 3. Ce qui a changé (STAR-R)
1. Prétraitement `--improved` du kit conservé : compression signed-log des métriques centrées-réduites, BCE avec `pos_weight`, réentraînement sur entraînement + validation.
2. **Grille sur la seule validation** (`explore_val.py`, 192 configurations × 3 graines, version de test jamais chargée). L'époque est choisie sur la courbe
   d'AP de validation **moyennée sur les graines et lissée**, au lieu du maximum bruité de chaque graine.
   Retenu : taux d'apprentissage 1e-3, weight decay 1e-3, λc 0,1, λr 0,25, τ 0,2, entrée Δ, sans dropout, SAGE unidirectionnel, 64 époques (`config_star_r.json`).
3. **Ensemble de 10 graines** du modèle STAR (moyenne des rangs) pour réduire la variance.
4. Partenaires tabulaires (régression logistique C = 0,03, ExtraTrees à 500 arbres) sur les métriques de la dernière version + Δ. Le mélange final est choisi **sur la validation**
   parmi les candidats contenant STAR ; STAR+LR+ET a été choisi dans la plupart des groupes.
5. Toute la procédure est répétée sur 5 groupes de graines indépendants → moyenne ± écart-type.

## 4. Résultats — version de test Ant 1.7 (évaluée une fois par groupe)
| Modèle | AP | ROC-AUC | Accuracy | Acc. équil. | Précision | Rappel | F1 | MCC | EA-R@20 %LOC |
|---|---|---|---|---|---|---|---|---|---|
| Référence A1 boosting | 0,752 | 0,843 | 0,770 | 0,701 | 0,787 | 0,471 | 0,589 | 0,472 | 0,314 |
| Référence A6 STAR complet (par défaut) | 0,760±0,024 | 0,832±0,015 | 0,751±0,017 | 0,747 | 0,623 | 0,733 | 0,673 | 0,480±0,027 | 0,312 |
| Référence A8 ensemble (par défaut) | 0,797±0,013 | 0,865 | 0,801±0,034 | 0,758 | 0,785 | 0,614 | 0,671±0,114 | 0,554±0,081 | 0,316 |
| Kit `--improved` A6 | 0,789±0,011 | 0,852 | 0,797±0,006 | 0,773 | 0,717 | 0,696 | 0,705 | 0,552 | 0,300 |
| Kit `--improved` A8 (meilleur existant) | 0,799±0,009 | 0,863 | 0,815±0,008 | 0,788 | 0,755 | 0,698 | 0,725±0,005 | 0,588±0,015 | 0,308 |
| **STAR-R (ensemble STAR seul)** | 0,794±0,003 | 0,862 | 0,816±0,005 | 0,786 | 0,766 | 0,686 | 0,723 | 0,590±0,013 | 0,308 |
| **STAR-R FINAL (mélange choisi sur la validation)** | **0,799±0,003** | **0,870** | **0,823±0,004** | **0,804** | 0,748 | **0,743** | **0,745±0,005** | **0,610±0,008** | 0,320 |
| Référence : régression logistique seule | 0,798 | 0,864 | 0,819 | 0,804 | 0,733 | 0,755 | 0,744 | 0,604 | 0,304 |

Accuracy de la classe majoritaire sur la version de test = 0,651.

**Lecture honnête des résultats.**
- Par rapport au STAR par défaut (A6), STAR-R gagne +0,039 d'AP, +0,072 d'accuracy et +0,13 de MCC, et l'écart-type entre graines est divisé par environ 8.
- Par rapport à la meilleure configuration que le kit avait déjà (A8 `--improved`), le gain est faible : AP identique, accuracy +0,008, F1 +0,020, MCC +0,022. L'apport principal est la stabilité.
- Une régression logistique régularisée fait à peu près aussi bien. Sur ces données, le graphe et le contraste apportent peu de signal mesurable au-delà des métriques elles-mêmes.
  C'est un résultat scientifique à présenter, pas à cacher. (La comparaison équitable de la phase 1 et l'étude sur Camel précisent ce constat : voir `PHASE1_RESULTS.md` et `RETUNE_RESULTS.md`.)

## 5. Reproduire
```bash
pip install -r requirements.txt
python run_star.py --project ant --seeds 42 43 44 45 46 --out results/baseline_default      # référence, ~20 s
python run_star.py --project ant --seeds 42 43 44 45 46 --improved --out results/baseline_improved   # ~23 s
python explore_val.py --project ant --out results/explore_ant.json                             # grille de validation, ~12,5 min
python train_final.py --project ant --groups 5 --ens 10                                        # final, ~61 s
```
Artefacts sauvegardés : `results/star_r/ant/star_r_ensemble.pt` (10 state_dicts STAR + configuration, entraînés sur 1.5+1.6),
`tabular_models.joblib`, `final_test_scores.csv`, `final_metrics.json` (choix de validation et métriques de test par groupe).
