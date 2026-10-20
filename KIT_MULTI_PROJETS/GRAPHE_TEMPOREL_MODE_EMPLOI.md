# Extension du dataset : graphes multi-versions et arêtes temporelles

Chaque fenêtre `<projet>-<version>.npz` reste **inchangée** pour garantir le fonctionnement du code initial. Le nouveau fichier `<projet>-<version>_graph_temporal.npz` fournit des données supplémentaires :

- `versions`: noms des trois versions, dans l'ordre.
- `ids_v0`, `ids_v1`, `ids_v2`: identifiants complets des classes dans **chaque version** (y compris nouvelles classes hors intersection stricte).
- `features_raw_v0`, `features_raw_v1`, `features_raw_v2`: métriques brutes `[nombre_noeuds, 20]`, sans standardisation. Les entrées invalides sont mises à zéro; leur emplacement est conservé dans `feature_valid_v*`.
- `edges_v0`, `edges_v1`, `edges_v2`: arêtes orientées d'imports internes `[nombre_aretes, 2]`; chaque indice est local à sa version.
- `temporal_links`: `[nombre_liens,4]` avec `(indice_version_source, indice_classe_source, indice_version_destination, indice_classe_destination)`; seuls les passages v0→v1 et v1→v2 sont enregistrés.
- `graph_offsets`: décalages d'indices des nœuds des trois versions dans le graphe global.
- `spatial_edges_global`: arêtes de chaque version avec indices globaux.
- `temporal_edges_global`: arêtes entre versions avec indices globaux.
- `target_ids`: identifiants des classes utilisées dans l'ancien `.npz` à la version cible.
- `target_indices_last`: correspondance entre `target_ids` et les nœuds de `ids_v2`.

## Test simple

```python
import numpy as np
from pathlib import Path
p = Path('data/ant/ant-1.7_graph_temporal.npz')
d = np.load(p)
for i in range(3):
    print(d['versions'][i],len(d[f'ids_v{i}']),len(d[f'edges_v{i}']))
print('Liens temporels :',len(d['temporal_links']))
# Les arêtes globales pourront servir à un futur modèle spatio-temporel.
```

## Exploitation proposée

- **TCN + GraphSAGE de base** : garder `X`, `y`, `edges` de l'ancien fichier; aucune modification nécessaire.
- **Analyse de l'évolution des dépendances** : comparer `edges_v*` après conversion en noms des classes et observer apparition/disparition des relations.
- **Extension GNN temporel** : assembler les trois ensembles de nœuds et concaténer `spatial_edges_global` et `temporal_edges_global`; ne pas confondre les features brutes et celles normalisées.
- **Gestion des classes nouvelles** : les nœuds supplémentaires sont présents, mais aucune imputation temporelle ni architecture dédiée aux historiques incomplets n'est encore fournie.

## Limites et bonnes pratiques

Les arêtes structurelles sont des imports Java *explicitement détectés* via analyse légère du texte. Elles ne capturent ni toutes les invocations, ni les héritages, ni les imports génériques, ni tous les types de dépendances. Les liens temporels ne détectent pas les renommages : une classe correspond seulement si son nom pleinement qualifié est identique à deux versions consécutives.

Les liens temporels sont **orientés vers l'avenir** et utilisables pour un graphe spatio-temporel, mais ils n'introduisent pas automatiquement une validation de la prédiction future. Le protocole d'entraînement fourni antérieurement reste une détection des défauts de la version observée. Les masques de train/validation/test devront être traités proprement avant un apprentissage sur graphe global pour exclure toute fuite entre versions.

Les features des graphes sont brutes : calculer une normalisation avec les seules données d'entraînement dans toute expérience utilisant les nouveaux fichiers.

Voir `data/graph_temporal_manifest.json` pour la couverture par fenêtre.
