# PRIORITEST-STAR

**Quelles classes Java un testeur doit-il inspecter en premier, quand il n'a le temps de lire qu'une partie du code ?**

PRIORITEST-STAR est un modèle d'apprentissage profond qui classe les classes d'un projet Java selon leur risque de
contenir un défaut. En inspectant le haut du classement, on trouve le plus de défauts possible avec un budget
d'inspection fixé, mesuré en lignes de code.
C'est le projet PFA (capstone) de 5e année, option Data & IA, à l'EMSI Marrakech.

> Plan directeur et avancement : [`ARCHITECTURE.md`](ARCHITECTURE.md) · Résultats détaillés : `KIT_MULTI_PROJETS/*RESULTS*.md`
> · Rapport : `rapport/rapport_prioritest_star.pdf`

---

## 1. Le problème, simplement

Tester et relire du code coûte cher, et une équipe n'a jamais le temps de tout inspecter.
La **prédiction de défauts** utilise des mesures du code pour deviner quelles parties ont le plus de chances d'être
défectueuses, afin d'y consacrer l'effort en priorité.

Deux choses rendent ce projet plus exigeant qu'un simple classifieur « défectueuse / saine » :

1. **C'est un problème de classement sous budget.** Les grandes classes sont plus souvent défectueuses, mais elles
   coûtent aussi plus cher à lire. Un bon modèle doit trouver beaucoup de défauts *par ligne inspectée*, pas seulement
   beaucoup de défauts. C'est l'évaluation **sensible à l'effort**.
2. **Le code a une histoire et une structure.** Une classe n'est pas une ligne isolée dans un tableau. Elle évolue
   d'une version à l'autre (son *historique*) et elle importe d'autres classes (ses *dépendances*). STAR cherche à
   exploiter les deux.

## 2. Les données

Les données proviennent du jeu **PROMISE** : des projets Java open source (Ant, Camel, POI, JEdit, Xalan…) mesurés sur
plusieurs versions.

| Élément | Signification |
|---|---|
| **20 métriques par classe** | Mesures orientées objet classiques : taille (LOC), complexité (WMC, max_cc), couplage (CBO, Ca, Ce, RFC), cohésion (LCOM, CAM), héritage (DIT, NOC)… |
| **Fenêtre de 3 versions** | Pour une version *cible* t, les métriques de la même classe en t−2, t−1 et t, d'où un tenseur `X[N, 3, 20]`. Seules les classes présentes dans les 3 versions sont gardées. |
| **Label** | 1 si la classe a au moins un défaut enregistré dans la version t. |
| **Graphe d'imports** | Une arête A → B quand la classe A importe la classe B dans la version t. Il est extrait du code source et ne donne qu'une vue *partielle* des dépendances réelles. |
| **LOC** | Nombre de lignes de code de chaque classe, utilisé comme coût d'inspection. |

Le protocole est **chronologique**. Sur Ant, le modèle est entraîné sur la version 1.5, les réglages sont choisis sur
la 1.6, et il est testé **une seule fois** sur la 1.7.
Le modèle *détecte* les défauts de la version observée ; il ne **prédit pas** la version suivante (règle R1 de
`ARCHITECTURE.md`).

## 3. Le modèle : STAR

```
 métriques sur 3 versions ──► TCN causal ──────────────► zT  (vue temporelle : comment la classe a évolué)
                                                            │
 métriques de la version t + graphe d'imports ──► GraphSAGE ──► zS  (vue structurelle : où se situe la classe)
                                                            │
                     a = cos(zT, zS)   ◄── degré d'accord entre les deux vues
                     (α, β) = softmax(MLP[zT, zS, a])        fusion guidée par la similarité
                     z = α·zT + β·zS
                                                            │
     tête de risque ─► probabilité de défaut (perte BCE)     tête de classement ─► classement par paires (RankNet)
```

| Bloc | Rôle | Pourquoi |
|---|---|---|
| **TCN causal** (convolution temporelle, noyau 2, dilatations 1 et 2) | Lit les 3 versions dans l'ordre, sans jamais regarder vers l'avant | Les classes instables ou en croissance sont plus risquées |
| **GraphSAGE** (2 couches d'agrégation par moyenne) | Combine chaque classe avec les classes qu'elle importe | Couplage et position dans l'architecture |
| **Fusion guidée par la similarité** | Pondère les deux vues selon leur accord cos(zT, zS) | Faire davantage confiance à une vue quand les deux concordent |
| **InfoNCE inter-vues** (perte contrastive) | Rapproche zT et zS d'une *même* classe, éloigne les autres classes | Aligne les deux vues |
| **RankNet** | Apprend qu'une classe défectueuse doit être mieux notée qu'une classe saine | Le but final est un classement |

Perte totale : `BCE + λc·InfoNCE + λr·RankNet`, plus éventuellement un terme d'entropie de la porte qui maintient les
deux vues en usage.

**Positionnement.** Rhazi et al. (2026, *Science of Computer Programming*) apprennent des similarités *entre classes
différentes* avec une attention inter-échantillons et un vote des plus proches voisins. STAR reprend l'idée
d'apprendre des similarités, mais entre *deux vues d'une même classe*. Ce n'est pas une reproduction de cet article.

## 4. Comment les résultats sont mesurés

| Métrique | Question à laquelle elle répond |
|---|---|
| **AP** (précision moyenne) | Les classes défectueuses sont-elles bien classées au-dessus des saines ? (métrique principale de détection) |
| ROC-AUC | Même idée, moins sensible à la proportion de défauts |
| **Popt** | Aire sous la courbe « défauts trouvés en fonction du % de code inspecté », normalisée : 1 = parfait, 0,5 ≈ hasard, 0 = pire cas |
| **Recall@20 % LOC** | Part des défauts trouvés après avoir lu 20 % des lignes de code |
| IFA | Nombre de classes saines inspectées avant le premier défaut |

Pour les métriques d'effort, les classes sont inspectées par ordre de **risque ÷ LOC** (risque par ligne).

Chaque chiffre est donné en **moyenne ± écart-type sur 10 graines**, avec un **intervalle de confiance à 95 % par
bootstrap**, et les modèles sont comparés par des **tests statistiques appariés**.
Rien n'est réglé sur la version de test : quatre tests automatiques le vérifient (`tests/test_no_leakage.py`).

## 5. Ce qui a été trouvé

| Étude | Résultat |
|---|---|
| **Ablation sur Ant** (phase 1) | Chaque mécanisme de STAR a été ajouté un par un, avec le même budget de réglage pour chaque variante. **Aucun n'apporte de gain significatif sur Ant** : toutes les variantes font jeu égal avec la régression logistique (AP ≈ 0,80). Random Forest et ExtraTrees sont les meilleurs modèles sensibles à l'effort (Popt 0,75). |
| **Effort appris dans le modèle** (phase 2) | Aucun gain. Le modèle converge vers la règle simple « risque ÷ LOC ». |
| **Entraînement avec 9 autres projets** (phase 3) | N'aide pas Ant (les projets diffèrent), mais **STAR supporte bien mieux les données mélangées qu'un modèle linéaire**. |
| **Améliorations de la fusion / du contraste** (phase 4) | Pertes contrastives sans faux négatifs et supervisées : neutres. **L'entropie de la porte** équilibre les deux vues sans coût (retenue). |
| **Autres projets** (étape 6.3) | **Sur Camel, STAR avec le graphe bat significativement la régression logistique (+0,066 d'AP)**. Sur POI, STAR entraîné avec des données d'autres projets est le meilleur modèle (AP 0,920). La version de test de JEdit n'a que 5 défauts et n'est pas interprétable. |
| **Réglage sur les données du projet** (étape 5.5, Camel) | Tous les réseaux progressent ; STAR bat alors la régression logistique d'environ **+0,09 d'AP** (p ≤ 0,01). RankNet dégrade de nouveau les résultats. POI reste à exécuter. |
| **Pourquoi le graphe aide sur Camel** (étape 7.3) | Sur Camel, les classes isolées ne sont presque jamais défectueuses (1,3 % contre 26 % pour les connectées). Le graphe capte ce signal ; sur Ant, le contraste est faible (×1,7 contre ×20). |

**À retenir.** La valeur de STAR **dépend du projet** :
- la vue structurelle est rentable quand la connectivité sépare les classes risquées des classes sûres (Camel) ;
- le modèle exploite mieux qu'un modèle linéaire les données d'autres projets ;
- combiné à des modèles tabulaires simples, il améliore le classement sensible à l'effort (Ant, Camel).

Sur un petit projet comme Ant (118 classes d'entraînement), des modèles simples font aussi bien. C'est présenté
honnêtement plutôt que caché.

## 6. Plan du dépôt

```
ARCHITECTURE.md              plan directeur : règles, avancement, architecture, étapes restantes
README.md                    ce fichier
rapport/                     rapport LaTeX (+ PDF, figures)
FICHE / GUIDE / LIRE_EN_PREMIER   documents du sujet fournis par l'encadrant
KIT_MULTI_PROJETS/
├── data/                    fenêtres PROMISE (.npz) de 10 projets + graphes temporels + manifestes
├── prepare_multi.py         reconstruit data/ à partir de l'archive PROMISE
├── src/model.py             STAR : TCN, GraphSAGE, porte de fusion, InfoNCE / contraste supervisé, RankNet, entropie de la porte
├── src/train.py             chargement des données et métriques (AP, Recall@LOC, accuracy/MCC avec seuil choisi sur la validation)
├── src/stats.py             Popt, IFA, IC par bootstrap, bootstrap apparié, Wilcoxon, delta de Cliff, Scott-Knott ESD
├── run_star.py              ablations A1–A8 (un mécanisme par étape)
├── explore_val.py           recherche d'hyperparamètres sur la seule version de validation
├── train_final.py           STAR-R : ensemble de 10 graines + mélange choisi sur la validation, test unique
├── compare_fair.py          phase 1  — comparaison équitable, à réglage égal, avec statistiques
├── effort_star.py           phase 2  — STAR-E, terme d'effort appris dans le modèle
├── more_data.py             phase 3  — entraînement inter-projets
├── improve_star.py          phase 4  — améliorations de la fusion / du contraste
├── transfer_study.py        étape 6.3 — Camel, POI, JEdit avec réglages figés depuis Ant
├── retune_project.py        étape 5.5 — réglage à l'intérieur de la version d'entraînement de chaque projet
├── graph_analysis.py        étape 7.3 — statistiques des graphes et apport du graphe
├── sk_rank.py               étape 6.2 — classement Scott-Knott ESD (à exécuter)
├── analyse_star.py          figures : courbe effort–rappel, poids de fusion, liste priorisée
├── app.py                   application Streamlit : explorer le Top-k sous un budget en LOC
├── tests/                   36 tests unitaires (modèle, métriques, statistiques, absence de fuite)
├── notebooks/               notebook Google Colab
├── results/                 tous les chiffres du rapport (json / csv / points de contrôle)
└── *_RESULTS.md, GRAPH_ANALYSIS.md   un compte rendu par étude
```

## 7. Les outils et leur rôle

| Outil | Rôle dans le projet |
|---|---|
| **Python 3** | Tout |
| **PyTorch** | Le réseau STAR (TCN, GraphSAGE écrit à la main, pertes), entraînement sur CPU ou GPU |
| **scikit-learn** | Références (régression logistique, Random Forest, ExtraTrees, gradient boosting) et métriques (AP, ROC-AUC, MCC) |
| **SciPy** | Test des rangs signés de Wilcoxon, ANOVA de Scott-Knott |
| **NumPy / pandas** | Manipulation des données, rééchantillonnage bootstrap |
| **Matplotlib** | Figures du rapport |
| **Streamlit + Altair** | Application interactive pour les testeurs |
| **joblib** | Sauvegarde des modèles tabulaires |
| **unittest** | Tests automatiques, y compris les tests de protocole |
| **Google Colab** | Exécuter le notebook sans installation locale |
| **LaTeX (Overleaf)** | Le rapport |
| **Git / GitHub** | Gestion de versions ; chaque résultat du rapport est versionné dans `results/` |

## 8. Lancer le projet

```bash
cd KIT_MULTI_PROJETS
pip install -r requirements.txt
python -m unittest discover -s tests -v       # 36 tests
python compare_fair.py --project ant          # phase 1 (~6,5 min CPU)  → results/phase1, PHASE1_RESULTS.md
python effort_star.py --project ant           # phase 2 (~4 min)
python more_data.py --project ant             # phase 3 (~4 min)
python improve_star.py --project ant          # phase 4 (~5 min)
python transfer_study.py                      # étape 6.3 (~5 min)
python retune_project.py --projects camel poi # étape 5.5 (~6 min par projet)
python graph_analysis.py                      # étape 7.3 (15 s)
python sk_rank.py                             # étape 6.2 (quelques secondes, après 5.5)
python train_final.py --project ant --groups 5 --ens 10 && python analyse_star.py --project ant
streamlit run app.py                          # l'application
```

Lancer la phase 1 en premier : les phases 2, 3 et 6.3 lisent ses réglages figés. Aucun GPU n'est nécessaire ; chaque
étude tourne en quelques minutes sur 2 cœurs CPU.

## 9. Limites

- **Détection, pas prévision** : les métriques de la version t servent à trouver les défauts de la version t.
- Le graphe d'imports est **partiel** : imports explicites seulement, sans appels de méthodes ni arêtes d'héritage.
- **Les classes nouvelles sont exclues** (pas d'historique sur 3 versions), ce qui biaise l'échantillon. Leurs labels
  nécessitent l'archive PROMISE complète.
- Un seul projet principal et une seule version de test par projet : avec un intervalle de confiance d'environ
  ±0,07 d'AP, les petits effets ne sont pas détectables.

## 10. Références

- Rhazi, Banouar, Toure, Raghay (2026). *Software defect prediction via contrastive similarity learning and inter-sample attention.* Science of Computer Programming 254, 103534.
- Kamei et al. (2013), Mende et Koschke (2010) : prédiction de défauts sensible à l'effort (Popt).
- Bai, Kolter, Koltun (2018) : réseaux convolutifs temporels. Hamilton, Ying, Leskovec (2017) : GraphSAGE.
- van den Oord et al. (2018) : InfoNCE. Khosla et al. (2020) : apprentissage contrastif supervisé.
- Jeux de données PROMISE / tera-PROMISE (Jureczko et Madeyski).
