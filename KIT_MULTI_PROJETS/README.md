# PRIORITEST-STAR — Données PROMISE multi-projets

Cet ensemble remplace **les données Ant seules** du kit étudiant précédent. Les fichiers `data/<projet>/<projet>-<version>.npz` suivent exactement le format attendu (`X`, `y`, `edges`, `ids`, `loc`).

## Répartition
- **Ant et JEdit** : trois fenêtres cibles consécutives, permettant un entraînement, une validation et un test de version distincts.
- **Camel et POI** : deux fenêtres cibles, permettant entraînement / test exploratoire **sans sélection des hyperparamètres sur le test**.
- **Xalan, Log4j, Lucene, Synapse, Velocity, Xerces** : une fenêtre cible, pour description, prédiction inter-projets dans un protocole à construire, ou étude complémentaire ; **pas de train/test chronologique interne**.
- **Ivy** : aucune classe commune sur trois versions selon l'appariement strict exact (non préparé).
- **PBeans** : deux versions originales, insuffisantes pour la fenêtre imposée de trois (non préparé).

## Exécution
```bash
pip install -r requirements.txt
python run_project.py --project jedit --epochs 35 --seed 42
python run_project.py --project ant --epochs 35 --seed 42
python run_project.py --project poi --epochs 35 --seed 42
```
Ne pas exécuter `run_project.py` sur les projets à une seule fenêtre : ce script les refuse explicitement.

## Régénération
```bash
python prepare_multi.py --archive PROMISE-dataset-main.zip --out data
```
L'archive originale n'est pas incluse (282 Mo) ; seuls les jeux normalisés/filtrés sont distribués. Voir `data/manifest_multi.json` pour chaque décompte et les couvertures de sources.

## Limites essentielles
- Échantillon : seulement les classes communes à trois versions successives, les nouvelles classes sont exclues.
- Graphe : seuls les `import` Java explicites exactement appariés sont retenus, donc ce n'est PAS un graphe de dépendances complet.
- Les métriques de la version cible sont utilisées pour le label de cette version : il s'agit d'une **détection de défauts de la version observée**, pas d'une prédiction des défauts d'une future version.
- Standardisation entraînement seulement par projet, ajustée sur la première fenêtre ; les résultats inter-projets nécessitent une nouvelle politique de normalisation.
- Les versions restent fortement apparentées : les labels et les classements ne doivent pas être mélangés au hasard entre versions.
- Les résultats expérimentaux et conclusions scientifiques restent à établir par les étudiants.

## Extension graphes temporels (nouvelle livraison)
Chaque fichier `.npz` original est maintenant accompagné d'un fichier `_graph_temporal.npz` avec les trois graphes, les correspondances entre versions, les arêtes spatiales et temporelles en indices globaux. Les projets et fenêtres restent identiques; les modèles TCN+GraphSAGE existants continuent d'utiliser les anciens fichiers. Consulter **GRAPHE_TEMPOREL_MODE_EMPLOI.md** et exécuter `python check_graph_temporal.py`.
