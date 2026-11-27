# Phase 2 — STAR-E : classement sensible à l'effort appris pendant l'entraînement (Ant)

Exécutée le 27/11/2026, sur CPU uniquement. Commande : `python effort_star.py --project ant` (4 min 03 s).
Sorties : `results/phase2/ant/{tuning.json, results.json, test_scores.npz}`.

## Idée

STAR apprend un score de risque et ne devient sensible à l'effort qu'après coup, en inspectant les classes par ordre de
score / LOC. STAR-E intègre le coût d'inspection dans le modèle lui-même :

    s_i = rank(z_i) − γ · log LOC_i          (l'ordre d'inspection est l'ordre de s)

| Variante | γ | Paires de classement |
|---|---|---|
| E1 | fixé à 1 | (défectueuse, saine) |
| E2 | appris (initialisé à 1) | (défectueuse, saine) |
| E3 | appris | + (petite défectueuse, grande défectueuse), soit l'ordre de l'oracle de Popt |

La tête de détection BCE est conservée. La grille de recherche est taux d'apprentissage × weight decay × entrée Δ ×
λc {0 ; 0,1} × λr {0,25 ; 1}, avec 3 graines. L'époque est choisie sur la courbe de **Popt de validation** moyennée sur
les graines et lissée. Le modèle est ensuite réentraîné sur 1.5 + 1.6 avec 10 graines et testé une fois sur 1.7.

## Résultats sur Ant 1.7

Chaque modèle est un ensemble de 10 graines ; les crochets donnent l'IC à 95 % par bootstrap sur les classes.

| Modèle | Popt | R@20 % LOC | IFA |
|---|---|---|---|
| E1 γ fixé | 0,712 [0,647 ; 0,763] | 0,304 | 28 |
| E2 γ appris (→ 0,98) | 0,711 [0,647 ; 0,763] | 0,304 | 28 |
| E3 + paires de l'oracle | 0,711 [0,646 ; 0,763] | 0,304 | 28 |
| LR + densité | 0,714 [0,648 ; 0,766] | 0,324 | 26 |
| STAR A5 + densité | 0,718 [0,652 ; 0,769] | 0,294 | 25 |
| **RF + densité** | **0,750 [0,687 ; 0,796]** | 0,363 | 28 |
| ExtraTrees + densité | 0,738 [0,677 ; 0,786] | **0,372** | 29 |

Bootstrap apparié : E1/E2/E3 contre LR + densité donnent ΔPopt ≈ −0,002 (p ≈ 0,7) ; contre STAR A5 + densité
≈ −0,006 (p ≈ 0,64) ; **contre RF + densité −0,038 [−0,068 ; −0,009], p = 0,01**.
La tête de détection de STAR-E garde une AP de 0,779 (STAR A5 : 0,802).

## Lecture

1. **Apprendre le terme d'effort n'apporte rien de plus que la règle appliquée après coup.** Le γ appris converge vers ≈ 1, ce qui est exactement la règle score / LOC : une fois le LOC intégré au score avec un poids de 1, le réseau n'a plus rien à en apprendre. Les paires de l'oracle ne changent rien non plus.
2. **L'époque retenue est très précoce (16)** : le Popt de validation culmine tôt puis se dégrade, ce qui est cohérent avec un signal de classement dominé par le LOC.
3. **L'écart avec RF tient aux interactions entre variables, pas à l'effort.** RF est meilleur avec la même règle score / LOC : ses probabilités classent mieux des classes de taille comparable. Une expérience naturelle serait de donner à STAR le point de vue des arbres, par exemple en distillant les probabilités de RF ou en ajoutant des représentations issues des feuilles de RF en entrée.
4. Pour le rapport : la phase 2 est un **résultat négatif documenté**. La fiche valorise explicitement ce type de résultat (« interprétation des échecs »).
