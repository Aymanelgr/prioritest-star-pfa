# Étape 6.3 — Les conclusions d'Ant tiennent-elles sur d'autres projets ?

Exécutée le 27/11/2026, sur CPU uniquement. Commande : `python transfer_study.py` (5 min 01 s ; Camel, POI et JEdit).
Sorties : `results/transfer/{camel,poi,jedit}.json`.

**Tout est figé depuis Ant** : les architectures, hyperparamètres, nombres d'époques, paramètres tabulaires et la
recette du mélange viennent de `results/phase1`, `phase3` et `phase4`. Rien n'est réglé sur ces projets : leurs versions
de test restent réellement inédites.

| Projet | Entraînement | Test | Classes de test / défectueuses |
|---|---|---|---|
| Camel | 1.4 | 1.6 | 565 / 127 (22 %) |
| POI | 2.5 | 3.0 | 311 / 202 (65 %) |
| JEdit | 4.1 + 4.2 | 4.3 | 179 / **5** (3 %) |

« Conjoint » ajoute comme données sources tous les autres projets du kit, avec 50 % de la perte sur le projet cible,
comme en phase 3.

## AP sur la version de test (10 graines ; IC à 95 % de l'ensemble)

| Modèle | Ant 1.7 (phase 1) | Camel 1.6 | POI 3.0 | JEdit 4.3 |
|---|---|---|---|---|
| LR | 0,798 | 0,420 [0,34 ; 0,51] | 0,872 [0,82 ; 0,92] | 0,277 [0,05 ; 0,71] |
| RF | **0,809** | 0,435 | 0,918 | 0,284 |
| ExtraTrees | 0,808 | **0,486** | 0,911 | 0,337 |
| HGB | 0,748 | 0,430 | 0,917 | 0,124 |
| A2 TCN | 0,801 | 0,438 | 0,877 | 0,261 |
| A3 + graphe | 0,799 | 0,475 | 0,743 | 0,243 |
| A4 + porte de similarité | 0,801 | 0,462 | 0,739 | 0,218 |
| A5 + InfoNCE | 0,802 | 0,439 | 0,857 | 0,242 |
| A6 STAR complet | 0,794 | 0,458 | 0,811 | 0,260 |
| G2 (A5 + entropie de la porte) | 0,802 | 0,438 | 0,858 | 0,243 |
| Mélange STAR + LR + ET | 0,806 | 0,459 | 0,889 | 0,395 |
| STAR A5 conjoint | 0,795 | 0,440 | **0,920** | 0,290 |
| LR conjointe | 0,768 | 0,412 | 0,896 | 0,278 |

## Les quatre conclusions des études sur Ant (bootstrap apparié sur les classes)

| Conclusion | Ant | Camel | POI | JEdit |
|---|---|---|---|---|
| **C1** STAR ≈ LR en AP | tient (tout n.s.) | **non — STAR meilleur** : A3 +0,066 (p < 0,001), A4 +0,050 (p = 0,003), A6 +0,045 (p = 0,01) | mitigée : A5/G2 ≈ LR ; A3/A4 −0,14 (p < 0,001) | tient (n.s.) |
| **C2** les arbres battent LR en Popt (score / LOC) | RF +0,036 (p = 0,007), ET +0,024 (p < 0,001) | ET +0,012 (p = 0,01) ; RF n.s. | ET +0,005 (p = 0,02) ; RF +0,005 (p = 0,06) | RF +0,042 (p = 0,03) |
| **C3** le mélange STAR bat LR en Popt | +0,008 (p = 0,006) | **+0,009 (p < 0,001)**, et AP +0,038 (p = 0,003) | n.s. | n.s. |
| **C4** STAR conjoint bat LR conjointe | +0,029 d'AP (p = 0,04) | +0,029 (p = 0,07) | **+0,025 (p = 0,02)** | n.s. |
| Les données conjointes aident STAR | −0,008 (n.s.) | +0,002 (n.s.) | **+0,059 (p = 0,016)** | n.s. |

## Lecture

1. **L'intérêt de la vue graphe dépend du projet.**
   - Sur **Camel**, c'est le plus grand gain individuel de l'étude : A2 → A3 = +0,037 d'AP, et toutes les variantes avec graphe battent LR significativement.
   - Sur **POI**, les mêmes variantes figées s'effondrent.
   - Sur **Ant**, elles sont neutres.

   La conclusion d'Ant « aucun mécanisme de STAR n'aide » ne se **généralise donc pas**. Le rapport doit présenter le graphe comme *dépendant du projet*, et l'étape 7.3 (performance selon le degré des nœuds, par projet) devient l'analyse clé pour l'expliquer.
2. **Les hyperparamètres figés d'Ant ne se transfèrent pas à POI.** La proportion de défauts de POI est de 65 %, contre 17 % pour Ant 1.5, et les variantes avec graphe ont été réglées sur Ant à 22–64 époques. L'effondrement de A3 / A4 sur POI est donc au moins en partie un effet du transfert des réglages, pas la preuve que le graphe y est nuisible. Il faudrait une validation propre au projet ; POI n'en a pas.
3. **C4 est le résultat propre à STAR le plus constant.** STAR conjoint ≥ LR conjointe sur tous les projets informatifs (significatif sur Ant et POI, tendance sur Camel). Sur POI, les données inter-projets font même de STAR le meilleur modèle (0,920 d'AP). STAR exploite mieux qu'un modèle linéaire des données d'entraînement hétérogènes.
4. **C2 tient largement.** Les ensembles d'arbres, surtout ExtraTrees, sont la référence sensible à l'effort la plus sûre d'un projet à l'autre.
5. **C3 (le mélange) tient sur Ant et Camel**, les deux projets qui ont le plus de données d'entraînement (≥ 118 et 252 classes) et une proportion de défauts modérée.
6. **JEdit 4.3 n'est pas interprétable** : avec 5 classes défectueuses, les IC de l'AP vont de 0,05 à 0,70. À présenter avec cette réserve uniquement.
