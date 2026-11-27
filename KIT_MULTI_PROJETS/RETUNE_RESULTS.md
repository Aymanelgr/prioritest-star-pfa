# Étape 5.5 — Réglage par projet pour les projets sans version de validation (EN COURS)

Commande : `python retune_project.py --projects camel poi`. Sorties : `results/retune/`.
**État :** Camel est terminé (5 min 51 s). L'exécution sur POI a été arrêtée pendant le réglage, donc POI n'a pas
encore de résultat. Relancer avec `--projects poi` pour la terminer.

## Protocole

Camel et POI n'ont que deux fenêtres cibles (entraînement, test). L'étape 6.3 réutilisait les réglages figés depuis Ant ;
ici, chaque modèle est réglé **à l'intérieur de la version d'entraînement** :
1. **Recherche.** 3 partitions stratifiées des classes de la version d'entraînement. Le graphe d'imports reste entier ; la perte ne voit que les classes d'entraînement de la partition, et l'AP est mesurée sur les classes mises de côté. Les grilles sont les mêmes qu'en phase 1, et l'époque maximise la courbe d'AP moyennée sur les partitions et lissée.
2. **Réentraînement** sur toutes les classes de la version d'entraînement, avec 10 graines.
3. **Test** : une seule évaluation sur la version suivante. La version de test n'est chargée qu'après la sauvegarde du réglage.

## Camel (entraînement 1.4 → test 1.6, 565 classes, 127 défectueuses)

| Modèle | AP (10 graines) | AP ens. [IC 95 %] | AP avec réglages d'Ant (6.3) | Popt (dens) |
|---|---|---|---|---|
| A1 HGB | 0,437 | 0,437 [0,354 ; 0,522] | 0,430 | 0,775 |
| A2 TCN | 0,490 ± 0,009 | 0,508 [0,424 ; 0,594] | 0,438 | 0,810 |
| A3 + graphe | 0,498 ± 0,026 | 0,538 [0,447 ; 0,627] | 0,475 | 0,796 |
| A4 + porte de similarité | 0,500 ± 0,019 | 0,542 [0,453 ; 0,629] | 0,462 | 0,777 |
| **A5 + InfoNCE** | **0,507 ± 0,013** | 0,539 [0,453 ; 0,629] | 0,439 | **0,814** |
| A6 + RankNet | 0,462 ± 0,023 | 0,498 [0,412 ; 0,594] | 0,458 | 0,706 |
| LR | 0,452 | 0,452 [0,365 ; 0,542] | 0,420 | 0,783 |
| RF | 0,475 ± 0,006 | 0,478 [0,395 ; 0,563] | 0,435 | 0,781 |
| ExtraTrees | 0,486 ± 0,004 | 0,486 [0,402 ; 0,566] | 0,486 | 0,790 |
| Mélange STAR + LR + ET | 0,517 | 0,517 [0,428 ; 0,607] | 0,459 | 0,803 |

Tests par bootstrap apparié sur les classes :

| Comparaison | ΔAP [IC 95 %] | p | ΔPopt (dens) | p |
|---|---|---|---|---|
| A2 TCN contre LR | +0,056 [+0,001 ; +0,111] | 0,049 | +0,027 | 0,09 |
| A3 + graphe contre LR | **+0,086 [+0,021 ; +0,151]** | **0,009** | +0,012 | 0,57 |
| A4 contre LR | **+0,090 [+0,024 ; +0,157]** | **0,010** | −0,006 | 0,77 |
| A5 + InfoNCE contre LR | **+0,088 [+0,028 ; +0,149]** | **0,003** | +0,031 | 0,056 |
| A6 + RankNet contre LR | +0,047 | 0,14 | **−0,077** | 0,006 |
| Mélange contre LR | **+0,065 [+0,025 ; +0,108]** | **0,002** | **+0,020** | **0,010** |
| A6 contre A5 (ajout de RankNet) | **−0,041 [−0,069 ; −0,008]** | **0,011** | | |

## Lecture à ce stade

1. **Régler sur les données du projet profite à tous les réseaux sur Camel** (A5 : 0,439 → 0,507 d'AP). Les modèles tabulaires bougent à peine. Les réglages figés d'Ant handicapaient STAR.
2. Correctement réglés, **STAR A3–A5 battent la régression logistique d'environ +0,09 d'AP (p ≤ 0,01)**, le plus grand écart de l'étude. Le mélange STAR bat LR à la fois en AP et en classement sensible à l'effort.
3. **RankNet dégrade de nouveau** (A6 contre A5 : −0,041 d'AP, p = 0,011 ; −0,077 de Popt), comme sur Ant. La version finale de STAR doit abandonner le terme RankNet (A5 + entropie de la porte).
4. POI reste à exécuter. C'est la vérification décisive pour savoir si l'effondrement observé en 6.3 venait des réglages figés.
