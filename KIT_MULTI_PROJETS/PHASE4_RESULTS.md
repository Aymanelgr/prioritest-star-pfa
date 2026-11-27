# Phase 4 — Amélioration personnelle de la fusion / du contraste inter-vues (guide §3)

Exécutée le 27/11/2026, sur CPU uniquement. Commande : `python improve_star.py --project ant` (4 min 47 s).
Sorties : `results/phase4/ant/{tuning.json, results.json, test_scores.npz}`. Code : `src/model.py`
(`supcon_cross_view`, `gate_entropy`, `gate_absdiff`), branché dans `run_star.cohort_loss` (tout est désactivé par défaut).

## Variantes

Chaque variante change une seule chose par rapport à A5 (graphe + porte de similarité + InfoNCE, la meilleure étape de
l'ablation de la phase 1).

| Id | Modification | Motivation |
|---|---|---|
| C1 | InfoNCE sans faux négatifs : paires défectueuse/défectueuse retirées des négatifs | Corriger les faux négatifs d'InfoNCE (fiche §3) |
| C2 | Contraste supervisé hybride : C1 + les autres classes défectueuses comme positifs souples entre les vues | Regrouper toutes les classes défectueuses |
| G1 | La porte voit aussi \|zT − zS\| | Laisser la fusion réagir à *l'endroit* où les vues divergent, pas seulement à leur degré d'accord (cos) |
| G2 | Régularisation par l'entropie de la porte (poids 0,05) | STAR-R utilise la vue temporelle à α ≈ 0,7 ; garder les deux vues en usage |
| C2G1 | C2 + G1 | Combinaison |

Même protocole qu'en phase 1 : grille de validation identique (taux d'apprentissage × weight decay × Δ × λc),
réentraînement sur 1.5 + 1.6, 10 graines, un seul test. A5 est réexécuté avec les mêmes graines pour les tests appariés.

## Résultats sur Ant 1.7

| Variante | AP de validation | AP de test | Popt (dens) | R@20 % LOC (dens) | ROC-AUC | α moyen (temporel) |
|---|---|---|---|---|---|---|
| A5 référence | 0,688 | 0,802 ± 0,009 | 0,717 | 0,300 | 0,864 | 0,625 |
| C1 InfoNCE sans faux négatifs | 0,688 | 0,802 ± 0,009 | 0,718 | 0,305 | 0,864 | 0,621 |
| C2 contraste supervisé hybride | 0,674 | 0,802 ± 0,008 | 0,717 | 0,298 | 0,864 | 0,703 |
| G1 porte \|zT − zS\| | 0,688 | 0,791 ± 0,015 | 0,711 | 0,300 | 0,854 | 0,755 |
| **G2 entropie de la porte** | 0,688 | **0,802 ± 0,009** | 0,717 | 0,298 | 0,864 | **0,554** |
| C2G1 | 0,674 | 0,799 ± 0,005 | 0,713 | 0,294 | 0,860 | 0,682 |

Tests contre A5. Δens utilise le bootstrap apparié sur les classes des ensembles ; Δgraines utilise Wilcoxon sur les
10 graines.

| Variante | ΔAP ens [IC 95 %] | p | ΔAP graines | p | ΔPopt graines | p |
|---|---|---|---|---|---|---|
| C1 | +0,001 [−0,001 ; +0,005] | 0,33 | −0,000 | 0,70 | +0,000 | 0,28 |
| C2 | +0,000 [−0,007 ; +0,006] | 0,90 | −0,000 | 0,85 | −0,000 | 0,85 |
| G1 | −0,006 [−0,014 ; +0,001] | 0,09 | −0,011 | 0,049 | −0,006 | 0,014 |
| G2 | +0,000 [−0,001 ; +0,003] | 0,83 | +0,000 | 0,92 | −0,000 | 0,43 |
| C2G1 | −0,003 [−0,011 ; +0,004] | 0,36 | −0,004 | 0,32 | −0,005 | 0,08 |

## Lecture

1. **Les faux négatifs ne limitent pas STAR sur Ant.** Avec 20 classes défectueuses à l'entraînement, InfoNCE repousse rarement deux classes défectueuses l'une de l'autre : les masquer (C1) ou les rapprocher (C2) ne change rien.
2. **G2 est l'amélioration retenue.** Elle atteint son objectif : la fusion utilise désormais les deux vues presque à égalité (α 0,63 → 0,55), sans perte d'AP, de Popt ou de ROC. Deux conséquences :
   - l'analyse de α / β et de l'accord entre les vues devient plus informative ;
   - les performances du modèle **ne dépendent pas de la pondération des deux vues**, ce qui est cohérent avec la phase 1, où la vue graphe n'apportait aucun signal mesurable.
3. **G1 est nuisible.** Quand elle voit \|zT − zS\|, la porte se replie sur la vue temporelle (α 0,755) et surapprend légèrement (−0,011 d'AP sur les graines, p = 0,049).
4. Avec les phases 1 à 3 : sur Ant, la limite vient de **la taille des données et de l'information contenue dans les métriques**, pas de la conception de la fusion ou du contraste.
