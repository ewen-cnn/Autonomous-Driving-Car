# Autonomous Driving Car

Projet de vision par ordinateur pour un véhicule autonome miniature. À partir d’une image de la caméra, le modèle apprend à reconnaître une direction de conduite parmi trois classes : `left`, `forward` et `right`.

Le dépôt contient déjà un jeu de données équilibré dans `dataset_augmente_equilibre/`, ainsi que des checkpoints de modèles. Le point d’entrée principal est donc l’entraînement sur les splits existants.

## Contenu du projet

- `dataset_augmente_equilibre/` : jeu de données prêt à l’emploi, organisé en `train`, `valid` et `test`.
- `train_action_model.py` : entraîne et évalue le modèle sur ces trois splits.
- `model.py` : extracteur CNN partagé et têtes de classification, régression moteur et multi-tâche.
- `dataset.py` : chargement des images et des labels depuis les CSV.
- `transforms.py` : prétraitement et augmentations d’images.
- `inference.py` et `predict_action.py` : chargement d’un checkpoint et prédiction sur une image.
- `Traitement/` : rééchantillonnage des traces brutes, équilibrage du jeu de données et analyse des classes.
- `tests_unitaire/` : tests de la logique des actions et des transformations.

## Installation

Le projet nécessite Python, PyTorch et les dépendances listées dans `requirements.txt`. Le script d’entraînement utilise également `scikit-learn` pour la matrice de confusion et le rapport de classification.

```bash
python -m venv .venv
```

Activez ensuite l’environnement, puis installez les dépendances :

```bash
pip install -r requirements.txt
pip install scikit-learn
```

Sous Windows PowerShell, l’activation est généralement :

```powershell
.\.venv\Scripts\Activate.ps1
```

> Si l’environnement virtuel déjà présent dans le dépôt pointe vers une installation Python disparue, recréez-le avec les commandes ci-dessus.

## Jeu de données utilisé par l’entraînement

Chaque split suit cette structure :

```text
dataset_augmente_equilibre/
├── train/
│   ├── Images/
│   └── labels/labels.csv
├── valid/
│   ├── Images/
│   └── labels/labels.csv
└── test/
    ├── Images/
    └── labels/labels.csv
```

Le fichier `labels.csv` est séparé par des points-virgules. Il doit contenir une référence à l’image (`image_filename`, `image_path` ou `filename`) et, de préférence, `direction_class`. En l’absence de ce dernier, le projet déduit la direction à partir de `speedA`, `speedB` et de `GPIO1` à `GPIO4`.

Les labels fins tels que `light_left`, `pivot_right` ou `sharp_left` sont normalisés vers les trois classes du modèle. Les commandes `stop`, `backward` et `other` ne sont pas utilisées pour l’apprentissage.

## Entraîner le modèle

```bash
python train_action_model.py
```

Le script entraîne un `DrivingCNN`, qui est actuellement l’alias du modèle multi-tâche : sa sortie contient des logits de classification et deux commandes moteur continues. La boucle d’entraînement active pour l’instant uniquement la perte de classification.

Configuration par défaut :

- 20 époques ;
- batch de 32 images ;
- taille d’entrée de 120 × 160 ;
- GPU CUDA utilisé automatiquement s’il est disponible ;
- modèle sauvegardé dans `MULTITASK_DRIVING_CNN.pth` uniquement si les seuils de validation par classe sont atteints : `left ≥ 0,90`, `forward ≥ 0,60`, `right ≥ 0,92`.

En fin d’entraînement, le script recharge le meilleur checkpoint, l’évalue sur le split `test` et affiche une matrice de confusion ainsi qu’un rapport de classification.

## Comprendre le modèle et son apprentissage

Le [guide utilisateur](GUIDE_UTILISATEUR.md) contient désormais une explication détaillée : rôle de chaque couche convolutionnelle, dimensions des tenseurs, calcul de la loss, rétropropagation, mise à jour Adam, validation, contenu des checkpoints et chemin d’inférence jusqu’à `predict_action.py`.

> La tête de régression moteur fait partie de l’architecture multi-tâche, mais l’entraînement actuel ne lui applique pas encore de loss dédiée : seules les logits de classification sont optimisées.

## Préparer à nouveau des données brutes

Le pipeline de préparation est disponible dans `Traitement/` :

```bash
python Traitement/resample_dataset.py
python Traitement/build_balanced_dataset.py
```

Avant de l’exécuter, lisez le [guide utilisateur](GUIDE_UTILISATEUR.md#régénérer-le-jeu-de-données-depuis-des-traces-brutes) : les chemins sont centralisés dans `config.py` et `build_balanced_dataset.py` supprime entièrement son dossier de sortie avant de le reconstruire.

## Tests

```bash
pytest tests_unitaire
```

Les tests couvrent notamment la conversion des commandes moteur en directions, les transformations d’images et le rééquilibrage par `WeightedRandomSampler`.

## Documentation détaillée

Le [guide utilisateur](GUIDE_UTILISATEUR.md) décrit le format des données, le pipeline de préparation, l’entraînement, l’inférence et les limites connues des scripts actuels.
