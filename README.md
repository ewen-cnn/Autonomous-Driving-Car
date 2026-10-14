# Autonomous Driving Car

Vision embarquée pour un véhicule autonome miniature : à partir d'une image caméra, un CNN multi-tâche prédit l'action de conduite (`left`, `forward`, `right`) **et** les consignes moteur continues (`speedA`, `speedB`).

Le dépôt couvre toute la chaîne : alignement des enregistrements bruts (images + commandes GPIO/moteur), construction d'un dataset équilibré, entraînement multi-tâche, évaluation par classe et inférence en ligne de commande.

<!-- À compléter après le prochain entraînement : accuracy sur le split test, matrice de confusion (image), GIF d'inférence sur le circuit. -->

## Points techniques

- **Labellisation automatique** : les états GPIO et vitesses des deux moteurs sont convertis en directions fines (`pivot_left`, `light_right`, …), puis ramenées à 3 classes (`actions.py`).
- **Rééchantillonnage temporel** des enregistrements à période fixe (250 ms) pour aligner images et commandes (`Traitement/resample_dataset.py`).
- **Dataset équilibré** : 1 000 images par classe en entraînement. Le découpage train / valid / test est fait **avant** l'augmentation, donc aucune image dérivée ne passe d'un split à l'autre.
- **Augmentation cohérente avec la physique** : une image retournée horizontalement change de classe (`left` ↔ `right`) et ses consignes `speedA` / `speedB` sont échangées.
- **Modèle multi-tâche** : encodeur convolutionnel partagé (4 blocs Conv-BN-ReLU, ~0,46 M paramètres au total), une tête de classification et une tête de régression bornée par `tanh`.
- **Critère de sauvegarde par classe** : un checkpoint n'est conservé que si chaque classe dépasse son propre seuil d'accuracy sur la validation, pour éviter un modèle bon en moyenne mais aveugle sur un virage.
- **Prétraitement unique** entre entraînement et inférence (redimensionnement + normalisation), vérifié par un test de non-régression.

## Démarrage rapide

Depuis la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_action_model.py
```

## Organisation

```text
.
|- actions.py                       # Commandes moteur/GPIO -> labels de direction
|- config.py                        # Chemins, tailles et constantes partagées
|- model.py                         # Architectures CNN (classification, régression, multi-tâche)
|- train_action_model.py            # Entraînement et évaluation multi-tâche
|- inference.py                     # Chargement du modèle et prédiction
|- predict_action.py                # Interface en ligne de commande
|- Traitement/
|  |- transforms.py                 # Transformations d'images (entraînement et inférence)
|  |- resample_dataset.py           # Alignement commandes/images brutes
|  |- build_balanced_dataset.py     # Splits et augmentation
|  `- class_repartition_analysis.py # Répartition des classes
|- tests_unitaire/                  # Tests automatisés (pytest)
`- dataset_augmente_equilibre/      # Dataset (non versionné, voir ci-dessous)
```

`dataset.py` et le dossier `Entrainement/` sont des modules historiques conservés pour compatibilité. Le pipeline actif est piloté par `train_action_model.py` et `Traitement/`.

## Dataset

Le dataset et les checkpoints `.pth` ne sont pas versionnés (taille). L'entraînement attend la structure suivante :

```text
dataset_augmente_equilibre/
|- train/
|  |- Images/
|  `- labels/labels.csv
|- valid/
|  |- Images/
|  `- labels/labels.csv
`- test/
   |- Images/
   `- labels/labels.csv
```

Les CSV utilisent le séparateur `;`. Ils contiennent une colonne image (par exemple `image_filename`), une colonne de label (`direction_class`) et les cibles moteur `speedA` et `speedB`.

Les classes sont toujours interprétées dans cet ordre : `left`, `forward`, `right`. Les labels fins sont normalisés dans `actions.py` : `light_left`, `pivot_left` et `sharp_left` deviennent `left`. Les commandes `stop`, `backward` et `other` ne sont pas des classes du modèle.

| Split | Images | Répartition |
| --- | ---: | --- |
| `train` | 3 000 | 1 000 par classe |
| `valid` | 600 | 200 par classe |
| `test` | 180 | 60 par classe |

## Entraînement

```powershell
python train_action_model.py
```

Le script :

- charge les trois splits et applique le même prétraitement que l'inférence ;
- entraîne `DrivingCNN` (alias de `DrivingMultiTaskCNN`) ;
- optimise une loss de classification et une loss de régression des vitesses moteur, pondérée par `REGRESSION_WEIGHT` ;
- évalue chaque époque sur `valid` (accuracy globale et par classe, MAE et RMSE moteur) ;
- sauvegarde le meilleur modèle dans `MULTITASK_DRIVING_CNN.pth` si les seuils par classe sont atteints ;
- recharge ce checkpoint puis l'évalue sur `test` (matrice de confusion et rapport de classification).

Les hyperparamètres sont en haut de `train_action_model.py`. Les chemins et le nom du checkpoint sont centralisés dans `config.py`.

### Prétraitement

Entraînement et inférence utilisent `build_eval_transform(DEFAULT_IMAGE_SIZE)` de `Traitement/transforms.py` : redimensionnement en `120x160` puis normalisation ImageNet. La taille est enregistrée dans le checkpoint et relue à l'inférence. `build_train_transform` (jitter couleur, flou, affine, bruit) est disponible pour une augmentation à la volée.

> Les checkpoints produits avant cette unification (entraînés sans redimensionnement ni normalisation) ne sont pas compatibles avec `inference.py` : il faut les régénérer.

## Inférence sur une image

```powershell
python predict_action.py chemin\vers\image.png
```

Par défaut, le script charge `MULTITASK_DRIVING_CNN.pth` ; `--model-path` permet d'en indiquer un autre. La commande affiche la classe prédite et sa probabilité softmax. Cette probabilité est une confiance du modèle, pas une garantie de sécurité pour piloter un véhicule réel.

## Reconstruction du dataset

Les scripts de `Traitement/` reconstruisent le dataset depuis des enregistrements bruts :

```text
raw_records/
`- record_001/
   |- labels.csv
   `- Images/
      |- 1710000000000.png
      `- ...
```

Configurez d'abord trois dossiers **distincts** dans `config.py` :

```python
dataset_DIR = "raw_records"
OUTPUT_DIR = "segmentation"
BALANCED_dataset_DIR = "dataset_augmente_equilibre"
```

Puis, depuis la racine du projet :

```powershell
python -m Traitement.resample_dataset
python -m Traitement.build_balanced_dataset
python -m Traitement.class_repartition_analysis segmentation
```

> **Attention** : `build_balanced_dataset` supprime intégralement son dossier de sortie avant de le reconstruire. Ne le lancez jamais sur un dossier qui contient les seules données à conserver.

## Tests

```powershell
python -m pytest tests_unitaire
```

Les tests couvrent la conversion des commandes moteur en directions, les transformations d'images, l'ancien sampler pondéré et la cohérence du prétraitement entre entraînement et inférence.

## Licence

[MIT](LICENSE)
