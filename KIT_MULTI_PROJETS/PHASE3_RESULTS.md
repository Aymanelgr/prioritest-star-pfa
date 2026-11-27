# Phase 3 — Plus de données : entraînement inter-projets pour Ant

Exécutée le 27/11/2026, sur CPU uniquement. Commande : `python more_data.py --project ant` (3 min 49 s).
Sorties : `results/phase3/ant/{selection.json, results.json, test_scores.npz}`.

## Pourquoi des données inter-projets, et pas les « classes nouvelles »

La feuille de route prévoyait d'abord d'entraîner sur les classes que la fenêtre stricte de 3 versions écarte (Ant 1.5 :
293 nœuds dans le graphe temporel contre 118 classes étiquetées). Mais le kit ne fournit les labels que pour les 118
classes de la fenêtre stricte : les 175 autres ont des métriques et des arêtes, mais **aucun label**, car les labels
sont dans l'archive PROMISE (282 Mo, non incluse).
Cette étude reste ouverte tant que l'archive n'est pas ajoutée (`prepare_multi.py --archive PROMISE-dataset-main.zip`).

Le levier de données qui *est* disponible : les 9 autres projets du kit comptent **13 fenêtres, 3 399 classes
étiquetées, dont 1 261 défectueuses** (camel, jedit, log4j, lucene, poi, synapse, velocity, xalan, xerces), contre
118 / 20 pour Ant 1.5. Les données d'Ant ne font jamais partie des « autres ». Chaque projet est centré-réduit sur sa
propre première fenêtre (normalisation par projet), puis compressé en signed-log.

## Régimes comparés

Tous utilisent STAR A5 avec ses hyperparamètres de la phase 1. Les époques sont choisies sur l'AP d'Ant 1.6 moyennée
sur les graines et lissée. Chaque régime est réentraîné et testé une fois sur Ant 1.7 avec 10 graines.

| Régime | Données d'entraînement | AP de validation (Ant 1.6) |
|---|---|---|
| X0 Ant seul | Ant 1.5 (réentraînement 1.5 + 1.6) | 0,688 |
| X1 Autres projets seuls (sans donnée d'Ant) | 13 autres fenêtres | 0,688 |
| X2 Conjoint | autres + Ant, les cohortes d'Ant pesant 50 % de la perte | **0,713** |
| X3 Pré-entraînement → ajustement | 46 époques sur les autres, puis 31 époques sur Ant (taux × 0,3) | 0,682 |

LR et RF sont testés dans les trois mêmes régimes. Dans le régime conjoint, les exemples d'Ant pèsent 50 % du poids total.

## Résultats sur Ant 1.7

10 graines (RF : 5). L'IC est un bootstrap sur les classes de l'ensemble.

| Modèle | AP | AP ens. [IC 95 %] | ROC-AUC | Popt (dens) | R@20 % LOC (dens) |
|---|---|---|---|---|---|
| STAR X0 Ant seul | **0,802 ± 0,009** | 0,805 [0,734 ; 0,866] | 0,864 | 0,718 | 0,294 |
| STAR X1 autres seuls | 0,734 ± 0,008 | 0,737 [0,652 ; 0,821] | 0,827 | 0,699 | 0,255 |
| STAR X2 conjoint | 0,795 ± 0,006 | 0,797 [0,722 ; 0,860] | 0,857 | 0,711 | 0,275 |
| STAR X3 pré-entraînement + ajustement | 0,774 ± 0,009 | 0,780 [0,700 ; 0,851] | 0,846 | 0,704 | 0,304 |
| LR Ant seul | 0,798 | 0,798 [0,721 ; 0,865] | 0,864 | 0,714 | 0,324 |
| LR autres seuls | 0,718 | 0,718 [0,626 ; 0,810] | 0,827 | 0,706 | 0,275 |
| LR conjointe | 0,768 | 0,768 [0,686 ; 0,842] | 0,843 | 0,706 | 0,304 |
| RF Ant seul | **0,809 ± 0,002** | 0,810 [0,726 ; 0,880] | **0,879** | **0,751** | **0,363** |
| RF autres seuls | 0,736 ± 0,010 | 0,739 [0,651 ; 0,829] | 0,825 | 0,710 | 0,314 |
| RF conjoint | 0,781 ± 0,004 | 0,779 [0,689 ; 0,868] | 0,867 | 0,736 | 0,324 |

Tests par bootstrap apparié sur les classes :

| Comparaison | ΔAP [IC 95 %] | p | ΔPopt (dens) | p |
|---|---|---|---|---|
| STAR X1 contre X0 | −0,068 [−0,119 ; −0,023] | < 0,001 | −0,019 | 0,05 |
| STAR X2 contre X0 | −0,008 [−0,021 ; +0,005] | 0,19 | −0,007 | 0,22 |
| STAR X3 contre X0 | −0,025 [−0,051 ; −0,002] | 0,03 | −0,014 | 0,36 |
| LR conjointe contre LR Ant seul | −0,030 [−0,054 ; −0,008] | 0,002 | −0,008 | 0,08 |
| RF conjoint contre RF Ant seul | −0,031 [−0,060 ; +0,007] | 0,13 | −0,014 | 0,22 |
| **STAR X2 conjoint contre LR conjointe** | **+0,029 [+0,001 ; +0,061]** | **0,04** | +0,005 | 0,44 |

## Lecture

1. **Les données d'autres projets n'aident pas sur Ant 1.7.** Aucun régime ne bat l'entraînement sur Ant seul, pour aucun modèle. C'est un **décalage de distribution** entre projets : le lien entre un profil de métriques CK et le risque de défaut change d'un projet à l'autre, même après normalisation par projet.
2. **Le gain de validation du régime conjoint (+0,025 sur Ant 1.6) ne s'est pas retrouvé sur le test.** C'est un cas d'école pour choisir sur la validation et rapporter le test une seule fois, et un rappel qu'une seule version de validation est un sélecteur bruité.
3. **STAR est le modèle le plus robuste au mélange de projets.** L'entraînement conjoint ne coûte à STAR que −0,008 d'AP (non significatif), mais coûte −0,030 à LR (p = 0,002), et STAR conjoint bat significativement LR conjointe. Les vues structurelle et temporelle semblent mieux se transférer qu'une règle de décision linéaire sur les métriques brutes. C'est le premier résultat où une propriété propre à STAR apparaît, et il a sa place dans le rapport.
4. **Prédiction inter-projets sans données cibles** (bonus de la fiche) : STAR 0,734 d'AP, LR 0,718, RF 0,736, contre 0,80 en intra-projet. Utilisable, mais nettement moins bon, ce qui est cohérent avec la littérature sur la prédiction de défauts inter-projets.
5. Prochains leviers de données, par valeur attendue :
   - ajouter l'archive PROMISE pour obtenir les **labels des classes nouvelles** (étude des classes nouvelles, R5) ;
   - **adaptation de domaine** pour les données inter-projets, par exemple une normalisation par projet apprise, un discriminateur de projet adverse, ou une pondération des fenêtres sources selon leur ressemblance avec Ant ;
   - refaire toute l'étude sur **Camel et POI** comme projets cibles (fait à l'étape 6.3, voir `TRANSFER_RESULTS.md`).
