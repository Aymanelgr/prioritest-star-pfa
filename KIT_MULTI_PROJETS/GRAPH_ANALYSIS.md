# Étape 7.3 — Pourquoi le graphe d'imports aide-t-il sur Camel mais pas sur Ant ni POI ?

Commande : `python graph_analysis.py` (15 s). Sortie : `results/graph_analysis/graph_analysis.json`.
Pour la version de test de chaque projet, le script calcule :
- des statistiques sur le graphe d'imports ;
- l'AP de A2 (TCN, sans graphe) et de A3 (+ graphe, fusion classique), séparément sur les classes isolées et connectées.

Les deux modèles utilisent les configurations **figées depuis Ant**, avec un ensemble de 5 graines. Les labels ne servent
qu'à l'évaluation.

## Statistiques des graphes (version de test)

| | Ant 1.7 | Camel 1.6 | POI 3.0 | JEdit 4.3 |
|---|---|---|---|---|
| Classes / taux de défauts | 292 / 35 % | 565 / 22 % | 311 / 65 % | 179 / 3 % |
| Arêtes d'import / degré moyen | 867 / 5,9 | 1 608 / 5,7 | 472 / 3,0 | 233 / 2,6 |
| Classes isolées | 12,7 % | 14,2 % | 25,4 % | 26,3 % |
| Taux de défauts : isolées → connectées | 21,6 % → 36,9 % (×1,7) | **1,3 % → 26,0 % (×20)** | 55,7 % → 68,1 % (×1,2) | 0 % → 3,8 % |
| Taux de voisins défectueux : défectueuses contre saines (écart d'homophilie) | 0,48 contre 0,45 (+0,03) | 0,46 contre 0,47 (−0,01) | 0,87 contre 0,75 (**+0,12**) | 0,05 contre 0,16 (−0,12) |
| Assortativité des labels | 0,006 | 0,002 | 0,090 | −0,089 |

## Où le graphe change l'AP (A3 − A2)

| | Ant | Camel | POI | JEdit |
|---|---|---|---|---|
| Toutes les classes | +0,002 | **+0,043** | −0,149 | +0,016 |
| Classes isolées | +0,005 | −0,001 | −0,000 | (aucun défaut) |
| Classes connectées | +0,001 | **+0,052** | **−0,152** | +0,013 |

## Lecture

1. **Sur Camel, le graphe aide par la connectivité, pas par le regroupement des défauts.**
   - Les classes isolées de Camel ne sont presque jamais défectueuses : 1 sur 80.
   - Les classes défectueuses ne s'importent *pas* davantage entre elles que les classes saines (écart d'homophilie ≈ 0).
   - L'agrégation par moyenne du GNN donne aux classes connectées une représentation différente de celle des classes isolées : le fait qu'une classe soit intégrée au graphe d'imports, et la façon dont elle l'est, devient un signal fort (×20 sur le taux de défauts).
   - Les métriques de couplage CK (CBO, Ca, Ce) ne capturent apparemment ce signal qu'en partie.
2. **Sur Ant, le même contraste est faible** (×1,7) : le graphe n'a presque rien à ajouter aux métriques, ce qui correspond à la phase 1, où aucune étape liée au graphe n'était significative.
3. **Sur POI, le graphe dégrade les résultats alors que les défauts y sont les plus regroupés** (homophilie +0,12). Les pertes portent entièrement sur les classes connectées. La cause la plus probable est le transfert des hyperparamètres d'Ant : les variantes avec graphe ont été réglées sur Ant (17 % de défauts à l'entraînement, 22–52 époques), alors que POI 2.5 compte 68 % de défauts et n'a pas de fenêtre de validation. C'est la raison de l'étape 5.5 (réglage par projet).
4. **Règle pratique pour le rapport :** avant de compter sur la vue graphe, vérifier deux chiffres sur la version d'*entraînement* :
   - le contraste de taux de défauts entre classes isolées et connectées ;
   - l'écart d'homophilie.

   Les deux se calculent sans toucher au test. Le fort contraste de Camel correspond à la situation où la vue structurelle de STAR est rentable.
5. Réserve : trois projets informatifs font de ceci une **explication corrélationnelle**, pas une preuve. JEdit 4.3 a trop peu de défauts pour conclure quoi que ce soit.
