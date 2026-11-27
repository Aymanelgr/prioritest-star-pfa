# PRIORITEST-STAR — Architecture et feuille de route (v2, fondée sur les résultats)

> Priorisation sensible à l'effort des classes Java à risque de défaut (PROMISE) : une vue temporelle (TCN causal sur
> 3 versions), une vue structurelle (GraphSAGE sur les imports Java), une fusion guidée par similarité, un InfoNCE
> inter-vues et RankNet. EMSI Marrakech — PFA IA, 5e année.

Ce fichier est le plan directeur du projet. Il présente :
1. les **règles de base** du projet ;
2. **ce qui est construit** et **ce que chaque expérience a montré** ;
3. l'**architecture cible** du dépôt et du modèle ;
4. **chaque étape restante**, avec la commande à lancer et la preuve qu'elle est terminée.

La v1 (27 nov.) prévoyait les phases 0 à 8. Les phases 0 à 4 ont été exécutées ; la v2 réécrit la suite du plan à
partir de leurs résultats.

---

## 0. Règles de base (jamais enfreintes)

| # | Règle | Comment elle est garantie |
|---|---|---|
| R1 | **Détection des défauts de la version observée**, jamais « prédiction de la version suivante ». | README, rapport, légende de l'application |
| R2 | **La version de test n'est utilisée qu'une fois**, après que tous les choix sont figés. Aucune époque, aucun λ, seuil, mélange ou graine n'est choisi sur le test. | Chaque script cherche sur la fenêtre 2 sans charger le test, puis évalue la fenêtre 3 une seule fois |
| R3 | Normalisation ajustée sur la **seule fenêtre d'entraînement** (par projet). | `prepare_multi.py` + compression signed-log |
| R4 | Le graphe d'imports est une **approximation partielle**, pas un graphe de dépendances complet. | Chapitre données + limites |
| R5 | Les classes nouvelles (sans historique sur 3 versions) sont **exclues** → le biais de sélection doit être analysé. | Étape 5.3 (nécessite l'archive PROMISE) |
| R6 | 10 graines, moyenne ± écart-type, **IC à 95 % par bootstrap sur les classes**, tests appariés, temps mesurés. | `src/stats.py` |
| R7 | Chaque ablation ajoute **un seul** mécanisme, réentraîné de zéro sur les mêmes partitions, **avec le même budget de réglage**. | `compare_fair.py` |
| R8 | Rhazi et al. (2026) sont cités pour le *principe* d'apprentissage des similarités ; STAR n'en est **pas** une reproduction. | État de l'art |

---

## 1. État d'avancement (27 nov. 2026)

### 1.1 Ce qui est construit

| Composant | Fichiers |
|---|---|
| Kit de données : 10 projets, 16 fenêtres, extension graphes temporels (vérifiée) | `data/`, `prepare_multi.py`, `check_graph_temporal.py` |
| Modèle : TCN causal, MeanSAGE, porte de similarité, entrée Δ, option de porte \|zT − zS\| | `src/model.py` |
| Pertes : BCE + pos_weight, InfoNCE, InfoNCE sans faux négatifs, **contraste supervisé hybride**, RankNet, RankNet pondéré par l'effort, **entropie de la porte** | `src/model.py`, `run_star.py` |
| Métriques : AP, ROC, Popt, IFA, Recall@LOC, Accuracy/MCC/F1 avec taux de signalement choisi sur la validation | `src/train.py`, `src/stats.py` |
| Statistiques : IC par bootstrap, bootstrap apparié, Wilcoxon, delta de Cliff, Scott-Knott ESD | `src/stats.py` |
| Ablations A1–A8, modèle final STAR-R, recherche sur la validation | `run_star.py`, `explore_val.py`, `train_final.py` |
| Phase 1 comparaison équitable, phase 2 STAR-E, phase 3 inter-projets, phase 4 améliorations | `compare_fair.py`, `effort_star.py`, `more_data.py`, `improve_star.py` |
| Étapes 6.3 (transfert), 5.5 (réglage par projet), 7.3 (analyse du graphe) | `transfer_study.py`, `retune_project.py`, `graph_analysis.py` |
| Figures d'analyse, application Streamlit, rapport LaTeX à jour | `analyse_star.py`, `app.py`, `rapport/` |
| 36 tests unitaires (dont 4 tests anti-fuite) | `tests/` |

### 1.2 Ce que les expériences ont montré (Ant : entraînement 1.5 → validation 1.6 → test 1.7)

| Phase | Question | Réponse | Compte rendu |
|---|---|---|---|
| 1 | Chaque mécanisme de STAR aide-t-il, à budget de réglage égal ? | **Aucun gain significatif**, à aucune étape. A2–A6 ≈ LR (AP ≈ 0,80, IC ± 0,07). RankNet est légèrement nuisible (p = 0,049). Les ensembles d'arbres battent significativement LR en classement sensible à l'effort (RF Popt 0,750, +0,036, p = 0,007). Le mélange STAR + LR + ET bat aussi LR en Popt (+0,008, p = 0,006). | `PHASE1_RESULTS.md` |
| 2 | Apprendre l'effort dans le modèle (s = rank − γ·log LOC) bat-il le tri score / LOC appliqué après coup ? | **Non.** γ est appris à ≈ 1, soit la règle appliquée après coup. Popt 0,711, comme LR + densité, et sous RF + densité (−0,038, p = 0,01). | `PHASE2_RESULTS.md` |
| 3 | Plus de données (3 399 classes de 9 autres projets) aident-elles ? | **Non, à cause du décalage de distribution.** Conjoint −0,008 d'AP (n.s.) ; pré-entraînement −0,025 (p = 0,03) ; sans données d'Ant 0,734. **Mais STAR est le plus robuste au mélange de projets :** STAR conjoint bat LR conjointe de +0,029 (p = 0,04), alors que LR perd 0,030 avec les mêmes données. | `PHASE3_RESULTS.md` |
| 6.3 | Les conclusions d'Ant tiennent-elles sur Camel / POI / JEdit (réglages figés depuis Ant) ? | **En partie.** Sur **Camel** (565 classes de test), STAR avec graphe bat LR significativement (A3 +0,066 d'AP, p < 0,001 ; mélange +0,038 d'AP et +0,009 de Popt, p ≤ 0,003). Sur **POI**, les variantes avec graphe s'effondrent, mais **STAR conjoint inter-projets est le meilleur modèle** (AP 0,920, +0,059 grâce aux données conjointes, p = 0,016). STAR conjoint ≥ LR conjointe sur tous les projets informatifs. JEdit 4.3 (5 défauts) n'est pas interprétable. | `TRANSFER_RESULTS.md` |
| 5.5 | Régler sur les données du projet change-t-il le résultat (Camel) ? | **Oui.** Tous les réseaux progressent (A5 : 0,439 → 0,507). A3–A5 battent LR d'environ +0,09 d'AP (p ≤ 0,01) ; RankNet dégrade encore (−0,041, p = 0,011). POI reste à exécuter. | `RETUNE_RESULTS.md` |
| 4 | Amélioration personnelle de la fusion / du contraste | InfoNCE sans faux négatifs et contraste supervisé : neutres. **L'entropie de la porte** équilibre les vues (α 0,63 → 0,55) sans coût ; retenue pour l'interprétabilité. La porte \|zT − zS\| est nuisible. | `PHASE4_RESULTS.md` |
| 7.3 | Pourquoi le graphe aide-t-il sur Camel ? | Sur Camel, une classe isolée n'est presque jamais défectueuse (1,3 % contre 26 % pour les connectées, ×20) et tout le gain du graphe porte sur les classes connectées. Pas d'homophilie des défauts : le signal est la connectivité. | `GRAPH_ANALYSIS.md` |

**Le résultat central (sur 4 projets).** La valeur de STAR **dépend du projet** :
- sur Ant, aucun mécanisme n'est mesurable ; une version de test de 292 classes ne permet pas de détecter des effets inférieurs à environ 0,03 d'AP ;
- sur Camel, la vue structurelle donne le plus grand gain de toute l'étude ;
- sur POI, STAR ne gagne qu'entraîné avec des données d'autres projets.

Les résultats propres à STAR qui tiennent sur tous les projets informatifs :
- STAR **exploite mieux qu'un modèle linéaire des données hétérogènes d'autres projets** (STAR conjoint ≥ LR conjointe sur Ant, Camel et POI) ;
- le **mélange STAR + LR + ET** améliore le classement sensible à l'effort par rapport à LR sur les projets qui ont le plus de données d'entraînement (Ant, Camel) ;
- **RankNet dégrade les résultats** sur Ant comme sur Camel.

Les ensembles d'arbres, surtout ExtraTrees, restent partout la référence sensible à l'effort la plus sûre.

---

## 2. Architecture cible

### 2.1 Vue d'ensemble du système

```
Archive PROMISE ──prepare_multi.py──▶ data/<proj>/<proj>-<v>.npz                (fenêtres strictes de 3 versions, étiquetées)
                                     data/<proj>/<proj>-<v>_graph_temporal.npz  (tous les nœuds, 3 graphes, liens temporels)
                                              │
               src/data.py (étape 5.1) : chargement des fenêtres · signed-log · masques · cohortes multi-projets · id de domaine
                                              ▼
 ┌──────────────────────────────────── src/model.py ──────────────────────────────────────┐
 │ x[N,3,F] → TCN causal (k=2, dil. 1,2) → zT[N,32]                                       │
 │ x[N,-1,F] + arêtes d'import → MeanSAGE ×2 → zS[N,32]                                   │
 │ a = cos(zT,zS) → MLP de porte([zT,zS,a]) → α,β (+ rég. d'entropie) → z = αzT + βzS     │
 │ têtes : risque(z) → BCE (pos_weight)   rang(z) → RankNet   ordre d'inspection : p / LOC│
 │ perte = BCE + λc·InfoNCE(zT,zS) [+ λr·RankNet] − λg·H(α,β)                             │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             ▼
   compare_fair · effort_star · more_data · improve_star · transfer_study · retune_project
                       (recherche sur la validation → réentraînement → test unique)
                                             ▼
          src/stats.py : IC par bootstrap · bootstrap apparié · Wilcoxon · delta de Cliff · Scott-Knott ESD
                                             ▼
   results/<phase>/<projet>/ → analyse_star.py (figures) → app.py (Streamlit) → rapport/ (LaTeX)
```

### 2.2 Modèle final (pour l'application et le message principal du rapport)

Choisi sur la **validation**, comme l'exige la fiche (STAR doit en faire partie) :

- **Détection (AP) :** STAR A5 avec entropie de la porte, ensemble de 10 graines. AP 0,802 sur Ant, égal à LR et RF dans l'IC ; sur Camel réglé, A5 bat LR de +0,088 (p = 0,003). Sans RankNet, qui dégrade les résultats.
- **Inspection sensible à l'effort :** le **mélange STAR + LR + ET** trié par score / LOC. Significativement meilleur que LR sur Ant (+0,008 de Popt, p = 0,006) et sur Camel (+0,020, p = 0,010). RF + densité est présenté comme la meilleure référence individuelle (Popt 0,750 sur Ant).

L'étape 6.1 confirme ce choix par une exécution finale.

### 2.3 Organisation du dépôt

```
prioritest-star/
├── ARCHITECTURE.md                ← ce fichier
├── README.md                      ← présentation complète du projet
├── FICHE / GUIDE / Contrastive.pdf / LIRE_EN_PREMIER.txt
├── rapport/                       ← LaTeX, figures, PDF compilé
└── KIT_MULTI_PROJETS/
    ├── README.md                  ← une commande par expérience + temps CPU
    ├── requirements.txt           ← numpy, scipy, scikit-learn, torch, pandas, matplotlib, joblib, streamlit, altair
    ├── data/                      ← données du kit (inchangées)
    ├── src/  model.py · train.py · stats.py · data.py (5.1)
    ├── run_star.py · explore_val.py · train_final.py                       ← ablations, recherche, STAR-R
    ├── compare_fair.py · effort_star.py · more_data.py · improve_star.py   ← phases 1 à 4
    ├── transfer_study.py · retune_project.py · graph_analysis.py · sk_rank.py ← étapes 6.3, 5.5, 7.3, 6.2
    ├── domain_adapt.py (5.2) · final_model.py (6.1)                        ← à écrire
    ├── analyse_star.py · app.py
    ├── PHASE{1..4}_RESULTS.md · TRANSFER_RESULTS.md · RETUNE_RESULTS.md · GRAPH_ANALYSIS.md · RESULTS_STAR_R.md
    ├── tests/                     ← 36 tests, dont les tests anti-fuite
    ├── notebooks/PRIORITEST_STAR_Colab.ipynb
    └── results/                   ← versionné ; tous les chiffres du rapport en proviennent
```

---

## 3. Phases terminées (référence)

| Phase | Réalisé | Commande |
|---|---|---|
| 0 Consolidation | `.gitignore` conserve les résultats et exclut `.venv` ; dépendances complétées ; STAR-R reproduit (AP 0,798) | — |
| 1 Comparaison équitable | recherche à budget égal pour A1–A7 + LR/RF/ET/HGB/kNN, bootstrap et Wilcoxon | `python compare_fair.py --project ant` (6,5 min) |
| 2 STAR-E | terme d'effort dans le modèle, 3 variantes | `python effort_star.py --project ant` (4 min) |
| 3 Plus de données | inter-projets sans données cibles / conjoint / pré-entraînement, STAR contre LR / RF | `python more_data.py --project ant` (4 min) |
| 4 Amélioration | C1, C2, G1, G2, C2G1 contre A5 | `python improve_star.py --project ant` (5 min) |
| 6.3 Transfert | Camel, POI, JEdit avec réglages figés depuis Ant | `python transfer_study.py` (5 min) |
| 6.4 Anti-fuite | 4 tests de protocole | `python -m unittest tests.test_no_leakage` |
| 7.3 Graphe | statistiques des graphes et apport selon la connectivité | `python graph_analysis.py` (15 s) |

Lancer la phase 1 avant les phases 2, 3 et 6.3, qui lisent `results/phase1/` (et 6.3 lit aussi `phase3/` et `phase4/`).

---

## 4. Feuille de route restante

Classée par **valeur attendue pour la soutenance**. Chaque étape indique l'objectif, ce qu'il faut faire et quand elle
est considérée comme terminée.

### Phase 5 — Les expériences les plus susceptibles de changer les conclusions (3–4 jours)

- [ ] **5.1 `src/data.py`.** Un chargeur unique pour les fenêtres strictes, les fichiers de graphes temporels et les cohortes multi-projets, avec un identifiant de projet/domaine par cohorte. Faire reposer les scripts de phase dessus ; aucun changement de résultat (vérifier contre les `results/` versionnés).
- [ ] **5.2 Adaptation de domaine (suite de la phase 3).** STAR est le modèle qui se transfère le mieux ; il faut que les données d'autres projets aident vraiment :
  - couches de normalisation propres à chaque projet (statistiques BatchNorm/LayerNorm par domaine) ;
  - **pondération des fenêtres sources** selon leur ressemblance avec la cible (MMD sur les distributions de métriques, calculée sur les seules données d'entraînement) ;
  - en option, un discriminateur de projet adverse (inversion de gradient) sur z.
  - Sélection sur Ant 1.6, test unique. **Terminé quand** STAR conjoint est comparé à STAR sur Ant seul avec un IC. Succès = ΔAP > 0 avec un IC qui exclut 0.
- [ ] **5.3 Classes nouvelles (R5).** Ajouter `PROMISE-dataset-main.zip`, relancer `prepare_multi.py` en exportant aussi les labels des classes hors fenêtre stricte. Entraîner avec un masque de présence (versions absentes mises à zéro, masque fourni au TCN), puis évaluer séparément anciennes et nouvelles classes. **Terminé quand** le rapport contient un tableau « biais de sélection » : taux de défauts et rappel, nouvelles contre anciennes classes.
- [~] **5.5 Réglage par projet pour les projets à 2 fenêtres (suite de 6.3).** *(en cours : `retune_project.py`, voir `RETUNE_RESULTS.md`. Camel terminé : STAR A3–A5 bat LR de +0,09 d'AP (p ≤ 0,01) et RankNet dégrade (−0,041, p = 0,011). POI pas encore exécuté : `python retune_project.py --projects poi`)* Les réglages figés d'Ant pénalisent les variantes avec graphe sur POI ; chaque modèle est donc réglé par 3 partitions des classes de la version d'entraînement, puis testé à nouveau.
- [ ] **5.4 Donner à STAR la force des arbres (suite de la phase 2).** RF gagne en classement sensible à l'effort grâce aux interactions entre variables ; donc :
  - (a) **distiller** les probabilités hors-échantillon de RF dans la tête de risque de STAR, comme cible souple ;
  - (b) ajouter une **troisième vue**, un MLP tabulaire sur [dernières métriques, Δ], dans la porte guidée par similarité.
  - Même protocole. **Terminé quand** les deux sont testées contre RF + densité en Popt.

### Phase 6 — Robustesse des conclusions (2 jours)

- [ ] **6.1 `final_model.py`.** Figer le choix final du §2.2 sur la validation et l'exécuter une fois avec 10 graines. Sauvegarder les points de contrôle et le mélange, et régénérer `results/analysis/` et les données de l'application.
- [~] **6.2 Scott-Knott ESD** *(code prêt : `stats.scott_knott_esd` + `sk_rank.py`, testé ; à exécuter une fois le réglage de POI terminé)* : classement de tous les modèles par groupes statistiquement équivalents, comme dans la littérature sur la prédiction de défauts.
- [x] **6.3 Deuxième et troisième projets cibles.** *(terminé : `python transfer_study.py`, voir `TRANSFER_RESULTS.md`)* Protocole de la phase 1 avec **configurations figées depuis Ant** (aucun réglage sur le nouveau test) sur Camel 1.4 → 1.6, POI 2.5 → 3.0 et JEdit (présenté avec son IC et la réserve des 5 défauts).
- [x] **6.4 Tests anti-fuite** *(terminé : 4 tests, tous réussis)* (`tests/test_no_leakage.py`) :
  - le code de recherche n'ouvre jamais le fichier de test avant que le réglage soit sauvegardé ;
  - les statistiques de normalisation viennent de la seule fenêtre 1 ;
  - le taux de signalement est calculé à partir des seuls scores de validation.

### Phase 7 — Analyse et application (2 jours)

- [ ] 7.1 Courbes effort–rappel avec bandes d'IC pour LR, RF, STAR A5, le mélange, et les ordres aléatoire et oracle.
- [ ] 7.2 α / β et cos(zT, zS), avec et sans entropie de la porte, selon défectueuse ou saine, bien ou mal classée, instable (|Δ| élevé) ou très couplée (CBO / degré entrant élevés).
- [x] 7.3 *(terminé : `python graph_analysis.py`, voir `GRAPH_ANALYSIS.md`)* Statistiques des graphes et apport du graphe selon la connectivité, par projet. Sur Camel, une classe isolée est défectueuse à 1,3 % contre 26 % pour une connectée (×20), et tout le gain du graphe porte sur les classes connectées.
- [ ] 7.4 Analyse des erreurs avec SHAP sur RF/LR : les faux positifs sont de grandes classes couplées, les défauts manqués de petites classes banales.
- [ ] 7.5 `app.py` :
  - choix du projet et du modèle (LR, RF, STAR, mélange) et de l'ordre (risque ou densité), budget en LOC ;
  - panneau par classe montrant α / β, l'accord et les métriques les plus éloignées de la médiane ;
  - l'application ne lit que `results/`.

### Phase 8 — Livrables (2–3 jours, en parallèle)

- [ ] 8.1 Le notebook Colab exécute tout de bout en bout : tests → phases 1 à 4 → modèle final → analyse → archive zip.
- [ ] 8.2 GitHub Actions : tests unitaires + exécutions rapides (2 époques) de chaque script de phase.
- [x] 8.3 **Mise à jour du rapport** *(terminé le 27 nov. : résumé, abstract, sections 5.4 à 5.8 et 6.4, conclusion, limites et perspectives ; tous les chiffres vérifiés)*. Reste à y ajouter POI réglé et le classement Scott-Knott quand ils seront exécutés, et à compléter la page de titre (noms).
- [ ] 8.4 Diapositives de soutenance : problème → protocole → résultats principaux → démonstration de l'application → limites.

---

## 5. Calendrier

| Semaine | Travail | Jalon de la fiche (§10) |
|---|---|---|
| S1 (terminée) | Phases 0 à 4, 6.3, 6.4, 7.3, rapport | Données, références, STAR, ablations, protocole sans fuite |
| S2 | 5.5 (POI), 6.2, 5.1, 5.2, 5.4 | Expériences pouvant changer les conclusions |
| S3 | 5.3 (si archive), 6.1 | Conclusions robustes et multi-projets |
| S4 | 7, 8.1–8.2 | Top-k explorable, figures, reproductibilité |
| S5 | 8.4 | Soutenance |

---

## 6. Dépôts GitHub

Le projet est sur **https://github.com/Aymanelgr/prioritest-star-pfa** (privé), où Claude peut pousser. Le dépôt
d'origine de l'équipe est `mohamedaoui/prioritest-star`. Pour le mettre à jour depuis
`C:\Users\ayman\OneDrive\Desktop\prioritest-star-main`, lancer dans PowerShell :

```powershell
git init -b main
git remote add origin https://github.com/mohamedaoui/prioritest-star.git
git fetch origin main
git reset origin/main          # reprend l'historique GitHub, les fichiers locaux deviennent des modifications
git status                     # .venv/ et __pycache__/ ne doivent PAS apparaître
git add -A
git commit -m "Phases 0-4, transfert, réglage par projet, analyse du graphe, rapport à jour"
git push origin main
```

---

## 7. Ce que signifie « le meilleur résultat possible » ici

La fiche évalue **une reproduction correcte, un protocole sans fuite et des ablations interprétables**, pas un score
record (« aucune performance minimale n'est garantie »). Le résultat le plus défendable est donc :

1. Une ablation équitable, à budget égal et testée statistiquement, montrant que sur Ant **aucun mécanisme de STAR ne donne de gain mesurable**, avec la raison : 118 classes d'entraînement et un IC de test de ± 0,07. *(fait)*
2. Les **résultats positifs propres à STAR** : le gain du graphe sur Camel (+0,09 d'AP une fois réglé sur Camel), la meilleure utilisation des données d'autres projets que LR (Ant, Camel, POI), et le gain significatif du mélange en classement sensible à l'effort (Ant, Camel). *(fait)*
3. Une tentative sérieuse de plus pour rentabiliser la conception de STAR (5.2 adaptation de domaine, 5.4 distillation des arbres / troisième vue), présentée quel que soit le résultat.
4. Une application et un rapport clairs, qui ne surestiment jamais les résultats (R1, R4, R5).
