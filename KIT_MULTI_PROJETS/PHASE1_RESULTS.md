# Phase 1 — Comparaison équitable, à réglage égal et testée statistiquement (Ant)

Exécutée le 27/11/2026, sur CPU uniquement (2 cœurs), PyTorch 2.15, scikit-learn 1.9. Commande :

```bash
python compare_fair.py --project ant          # 6 min 33 s
```

Sorties : `results/phase1/ant/{tuning.json, table.csv, tests.json, test_scores.npz}`.

## Protocole (identique pour chaque modèle)

1. **Recherche sur la validation** : entraînement sur 1.5 → validation sur 1.6. La version de test n'est pas chargée à ce stade.
   - **Variantes neuronales (A2–A7) :** grille taux d'apprentissage {1e-3, 3e-3} × weight decay {1e-4, 1e-3} × entrée Δ {non, oui}, plus λc {0,1 ; 0,25} / λr {0,25 ; 1} quand la variante les utilise. Chaque point est entraîné avec 3 graines ; l'époque retenue maximise la courbe d'AP de validation moyennée sur les graines et lissée sur 5 époques (au moins 10 époques).
   - **Modèles tabulaires :** petite grille sur leur principal paramètre de régularisation.
2. **Réentraînement** sur 1.5 + 1.6 avec les réglages choisis, 10 graines.
3. **Une seule évaluation** sur la version de test 1.7 (292 classes, 102 défectueuses).

Le prétraitement est le même pour tous les modèles : centrage-réduction ajusté sur l'entraînement, puis compression
signed-log, avec `pos_weight` ou des poids de classes équilibrés. Les modèles tabulaires utilisent les métriques de la
dernière version et leurs variations Δ par rapport à la version précédente.

## Résultats sur Ant 1.7

L'AP est la moyenne ± écart-type sur 10 graines d'entraînement. L'IC est un bootstrap sur les classes (2 000 tirages)
de l'ensemble des 10 graines. L'ensemble moyenne les rangs pour les métriques de rang et les scores pour les métriques
score / LOC, car diviser un rang par le LOC fausserait l'ordre d'inspection.
« dens » signifie que les classes sont inspectées par score / LOC ; « risque », par score brut.

| Modèle | AP (graines) | AP ens. [IC 95 %] | Popt (dens) | R@20 % LOC (dens) | Popt (risque) |
|---|---|---|---|---|---|
| A1 HGB | 0,748 | 0,748 [0,655 ; 0,840] | 0,738 | 0,333 | 0,424 |
| A2 TCN + BCE | 0,801 ± 0,004 | 0,802 [0,725 ; 0,865] | 0,708 | 0,287 | 0,377 |
| A3 + graphe, fusion classique | 0,799 ± 0,004 | 0,802 [0,728 ; 0,866] | 0,711 | 0,287 | 0,381 |
| A4 + fusion par similarité | 0,801 ± 0,005 | 0,806 [0,734 ; 0,866] | 0,713 | 0,306 | 0,376 |
| A5 + InfoNCE | **0,802 ± 0,009** | 0,805 [0,734 ; 0,866] | 0,717 | 0,300 | 0,376 |
| A6 + RankNet = STAR | 0,794 ± 0,008 | 0,798 [0,717 ; 0,866] | 0,711 | 0,300 | 0,379 |
| A7 STAR+ | 0,644 ± 0,035 | 0,677 [0,580 ; 0,767] | 0,633 | 0,291 | 0,434 |
| Régression logistique | 0,798 | 0,798 [0,721 ; 0,865] | 0,714 | 0,324 | 0,345 |
| Random Forest | **0,809 ± 0,002** | 0,809 [0,725 ; 0,880] | **0,750** | 0,351 | 0,418 |
| ExtraTrees | 0,808 ± 0,007 | 0,807 [0,723 ; 0,878] | 0,738 | **0,363** | 0,399 |
| kNN cosinus (à la Rhazi, N = 50) | 0,789 | 0,789 [0,702 ; 0,864] | 0,710 | 0,294 | 0,413 |
| Mélange STAR + LR + ET | 0,806 | 0,806 [0,724 ; 0,874] | 0,721 | 0,333 | 0,370 |

Un ordre aléatoire donne Popt ≈ 0,50 et R@20 % LOC ≈ 0,20.

### Ablation : un mécanisme par étape (Wilcoxon sur les 10 graines, delta de Cliff)

| Étape | ΔAP | p | ΔPopt (dens) | p |
|---|---|---|---|---|
| A1 HGB → A2 TCN | +0,054 | 0,002 | −0,031 | 0,002 |
| A2 → A3 (+ graphe) | −0,002 | 0,13 | +0,003 | 0,16 |
| A3 → A4 (+ porte de similarité) | +0,002 | 0,32 | +0,002 | 0,63 |
| A4 → A5 (+ InfoNCE) | +0,001 | 0,49 | +0,004 | 0,19 |
| A5 → A6 (+ RankNet) | −0,008 | 0,049 | −0,006 | 0,049 |

### Chaque modèle contre la régression logistique (bootstrap apparié sur les classes)

| Modèle contre LR | ΔAP [IC 95 %] | p | ΔPopt (dens) [IC 95 %] | p |
|---|---|---|---|---|
| A5 + InfoNCE | +0,007 [−0,015 ; +0,030] | 0,57 | +0,004 [−0,014 ; +0,023] | 0,67 |
| A6 STAR | −0,000 [−0,022 ; +0,022] | 1,00 | −0,002 [−0,013 ; +0,008] | 0,68 |
| A7 STAR+ | −0,121 [−0,177 ; −0,068] | < 0,001 | −0,061 [−0,104 ; −0,019] | 0,009 |
| Random Forest | +0,011 [−0,024 ; +0,044] | 0,51 | **+0,036 [+0,011 ; +0,063]** | **0,007** |
| ExtraTrees | +0,009 [−0,020 ; +0,039] | 0,52 | **+0,024 [+0,012 ; +0,038]** | **< 0,001** |
| Mélange STAR + LR + ET | +0,008 [−0,012 ; +0,028] | 0,39 | **+0,008 [+0,002 ; +0,014]** | **0,006** |

En AP, aucun modèle ne diffère de LR, sauf A7 qui est moins bon. En classement sensible à l'effort (Popt par score /
LOC), les ensembles d'arbres et le mélange STAR sont significativement meilleurs que LR.

## Lecture

1. **Aucun mécanisme de STAR ne donne de gain mesurable sur Ant avec un protocole équitable.**
   - Le graphe, la porte de similarité et InfoNCE font chacun varier l'AP de 0,002 au plus, sans significativité.
   - RankNet est légèrement *nuisible* (p = 0,049 sur les graines).
   - L'AP de validation raconte la même chose : tous les modèles sont entre 0,68 et 0,70.
2. **La taille du test est le facteur limitant.** Avec 292 classes, l'IC de l'AP fait ±0,07 : un effet réel inférieur à environ 0,03 n'est pas détectable sur une seule version. Cela justifie la phase 3 (plus de données, plus de projets).
3. **Les ensembles d'arbres sont les meilleurs modèles sensibles à l'effort, et l'écart est significatif** : RF atteint un Popt de 0,750 (+0,036 contre LR, p = 0,007) et ET un R@20 % LOC de 0,363, contre 0,711 / 0,300 pour STAR. Le mélange STAR + LR + ET bat aussi LR en Popt (+0,008, p = 0,006) : STAR apporte quelque chose quand il est combiné aux modèles tabulaires. Le classement de STAR reste dominé par la taille : le Popt par risque brut vaut environ 0,38, moins bien que le hasard. Cela justifie la phase 2 (effort appris).
4. **A7 STAR+ s'effondre en AP.** Son RankNet pondéré par l'effort fait remonter les petites classes, ce qui lui donne le meilleur Popt (risque) des modèles neuronaux (0,434) mais détruit l'AP. Il faut le présenter comme un compromis AP / effort, pas comme une amélioration.
5. **Les tests de Wilcoxon sur les graines ne mesurent que la variance d'entraînement**, pas celle des données. Le bootstrap sur les classes est l'incertitude honnête, et c'est sur lui que doivent reposer les conclusions du rapport.
6. **La configuration A6 retenue ici est la même que `config_star_r.json`**, ce qui confirme de façon indépendante la recherche STAR-R antérieure.
