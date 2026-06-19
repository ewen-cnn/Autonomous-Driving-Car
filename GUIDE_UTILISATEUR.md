# Guide utilisateur du projet Autonomous Driving Car

## 1. Objectif du projet

Ce depot sert a preparer un jeu de donnees de conduite, entrainer un modele de classification d'action, puis faire de l'inference sur une image unique.

Le projet suit ce flux :

1. lire les enregistrements bruts dans `dataset/`,
2. reechantillonner les commandes et les aligner avec les images dans `segmentation/`,
3. construire un dataset dans `dataset_augmente_equilibre/`,
4. entrainer un CNN pour predire `left`, `forward` ou `right`,
5. ré-utiliser le modele entraine pour classer une nouvelle image.

## 2. Architecture generale

Arborescence logique :

```text
.
|-- actions.py
|-- config.py
|-- dataset.py
|-- transforms.py
|-- model.py
|-- training.py
|-- inference.py
|-- train_action_model.py
|-- predict_action.py
|-- Traitement/
|   |-- resample_dataset.py
|   |-- build_balanced_dataset.py
|   |-- class_repartition_analysis.py
|   `-- dataset_preparation.md
|-- Entrainement/
|   |-- Acquisition.py
|   |-- analyse_dataset.py
|   |-- data_processing_techniques.py
|   `-- segmentation_commandes_csv.py
`-- tests_unitaire/
    |-- test_actions.py
    `-- test_data_processing_techniques.py
```

## 3. Flux de donnees

### Etape 1 - Donnees brutes

`dataset/` contient les enregistrements bruts. Chaque enregistrement doit en pratique contenir :

- un `labels.csv` avec les vitesses moteurs, etats GPIO et timestamps,
- un dossier `Images/` contenant les images nommees par timestamp.

### Etape 2 - Reechantillonnage et alignement

`Traitement/resample_dataset.py` lit les CSV bruts, reechantillonne les commandes sur une grille temporelle fixe, puis associe a chaque ligne l'image la plus recente disponible au moment de la commande.

Sortie :

- dossier `segmentation/`,
- un `labels.csv` par enregistrement,
- les images effectivement associees,
- `resampling_analysis.csv` pour un resume global.

### Etape 3 - Construction du dataset augmente

`Traitement/build_balanced_dataset.py` lit `segmentation/`, convertit les labels fins en trois classes (`left`, `forward`, `right`), cree les splits `train`, `valid`, `test` et complete les classes rares par augmentation.

Sortie :

- `dataset_augmente_equilibre/train/`
- `dataset_augmente_equilibre/valid/`
- `dataset_augmente_equilibre/test/`

Chaque split contient :

- `Images/`
- `labels/labels.csv`

### Etape 4 - Entrainement

`train_action_model.py` appelle `training.py` pour lancer l'entrainement du reseau `DrivingCNN`.

Sortie :

- un checkpoint `driving_cnn.pth` par defaut.

### Etape 5 - Inference

`predict_action.py` charge le checkpoint et classe une image unique via `inference.py`.

## 4. Fichiers racine

### `config.py`

Role : centraliser les constantes globales du projet.

Ce fichier definit notamment :

- les chemins des dossiers (`dataset__DIR`, `OUTPUT_DIR`, `BALANCED_dataset_DIR`),
- le chemin du modele (`DEFAULT_MODEL_PATH`),
- la taille d'image par defaut (`DEFAULT_IMAGE_SIZE`),
- la periode de reechantillonnage (`SAMPLE_PERIOD_MS`),
- les labels fins et les labels simplifies utilises par le modele.

Notion importante :

- `CLASSES` contient les labels fins presents dans les CSV.
- `MODEL_CLASSES` contient les 3 classes apprises par le modele.
- `CLASS_TO_IDX` et `IDX_TO_CLASS` fixent l'ordre des classes dans les tenseurs PyTorch.

### `actions.py`

Role : convertir les signaux moteurs bruts en labels semantiques.

Fonctions importantes :

- `normalize_model_action(label)` : reduit les labels fins vers `left`, `forward`, `right`. Les actions `stop`, `backward` et `other` sont ignorees pour l'entrainement.
- `motor_direction(gpio_a, gpio_b, side)` : decode le sens d'un moteur a partir de deux GPIO. La logique depend du cote gauche/droite car le cablage n'est pas symetrique.
- `classify_direction(row)` : reconstruit l'action fine a partir de `speedA`, `speedB`, `GPIO1..GPIO4`.
- `row_to_model_action(row)` : chaine courte qui combine `classify_direction` puis `normalize_model_action`.
- `motor_to_action(...)` : heuristique simplifiee utile quand on n'a que des vitesses signees sans etats GPIO.

Notions a bien comprendre :

- Le projet ne predit pas directement les labels fins comme `sharp_left` ou `pivot_right`.
- Les labels fins servent surtout a convertir les traces brutes en un espace de decision a 3 classes.
- Quand les deux moteurs sont en marche avant, la comparaison `speedA` vs `speedB` sert a distinguer `forward` d'un virage leger.

### `dataset.py`

Role : fournir un `dataset` PyTorch reutilisable.

Classe principale :

- `AutonomousCardataset`

Fonctions importantes :

- `_detect_image_column()` : accepte plusieurs conventions de colonnes (`image_filename`, `image_path`, `filename`).
- `_resolve_labels()` : utilise d'abord une colonne explicite de labels, sinon reconstruit les labels a partir des commandes moteur.
- `_resolve_image_path()` : gere plusieurs formes de chemins d'image, absolus ou relatifs.

Notion importante :

- Le dataset_augmente_equilibre filtre des la construction les echantillons inutilisables : image absente, label inconnu ou label ignore.

### `transforms.py`

Role : definir les transformations d'images pour l'entrainement et l'evaluation.

Elements importants :

- `AddGaussianNoise` : ajoute un bruit gaussien borne apres conversion en tenseur.
- `build_train_transform(...)` : resize, jitter de couleur, flou occasionnel, affine leger, bruit puis normalisation.
- `build_eval_transform(...)` : pipeline deterministe pour validation et inference.

Notion importante :

- Les augmentations d'entrainement restent legeres pour ne pas casser la semantique de conduite.

### `model.py`

Role : definir l'architecture du reseau de neurones.

Classe principale :

- `DrivingCNN`

Architecture :

- 4 blocs convolution + batch norm + ReLU + pooling,
- `AdaptiveAvgPool2d((1, 1))` pour rendre la fin du reseau moins dependante de la taille spatiale restante,
- un classifieur final `Dropout + Linear`.

Notion importante :

- Le modele est volontairement compact et simple a entrainer sur un dataset de taille moderee.

### `training.py`

Role : orchestrer l'entrainement et la validation.

Elements importants :

- `TrainingConfig` : configuration de l'entrainement.
- `compute_accuracy(...)` : calcule l'accuracy globale et par classe.
- `_build_dataloaders(...)` : construit un split train/validation a partir d'un seul CSV.
- `train_model(...)` : boucle d'entrainement principale et sauvegarde du meilleur checkpoint.

Notions a bien comprendre :

- Le code cree deux instances du dataset_augmente_equilibre : une pour le train avec augmentations aleatoires, une pour la validation avec transformations deterministes.
- Le checkpoint sauvegarde non seulement les poids, mais aussi l'ordre des classes et la taille d'image.

### `inference.py`

Role : charger un modele entraine et predire une classe sur une image.

Fonctions importantes :

- `load_model(...)` : charge le checkpoint, verifie la compatibilite des classes et restaure le reseau.
- `predict_image(...)` : applique la transformation d'evaluation, lance le modele et retourne le label avec sa confiance.

Notion importante :

- La taille d'image utilisee pour l'inference est lue depuis le checkpoint afin de rester coherente avec l'entrainement.

### `train_action_model.py`

Role : point d'entree en ligne de commande pour l'entrainement.

Ce script :

- parse les arguments CLI,
- construit un `TrainingConfig`,
- appelle `train_model(...)`,
- affiche la meilleure accuracy de validation.

Commande type :

```bash
python train_action_model.py
```

### `predict_action.py`

Role : point d'entree en ligne de commande pour l'inference.

Ce script :

- prend le chemin d'une image,
- prend optionnellement un chemin de modele,
- appelle `predict_image(...)`,
- affiche la prediction et la confiance.

Commande type :

```bash
python predict_action.py path/to/image.png
```

### `README.md`

Role : vue d'ensemble courte du projet, du workflow et des commandes principales.

Usage :

- bon point d'entree rapide,
- moins detaille que ce guide.

### `requirements.txt`

Role : dependances Python minimales du projet.

Contenu actuel :

- `pandas`
- `Pillow`
- `pytest`
- `torch`
- `torchvision`

### `.gitignore`

Role : eviter de versionner les dataset_augmente_equilibres generes, les checkpoints, les environnements virtuels et les fichiers temporaires.

### `notes_classification.txt`

Role : memo manuel sur la logique de classification et l'organisation des sorties d'analyse.

Remarque :

- c'est un document informel, utile pour comprendre l'intention metier qui a precede la version actuelle de `actions.py`.

### `DrivingActions.py`

Role : wrapper de compatibilite qui re-exporte le contenu de `actions.py`.

Pourquoi il existe :

- il permet de conserver d'anciens imports sans dupliquer la logique metier.

### `Acquisition.py`

Role : wrapper de compatibilite qui re-exporte `Entrainement/Acquisition.py`.

### `analyse_dataset.py`

Role : wrapper de compatibilite qui re-exporte `Entrainement/analyse_dataset.py`.

### `data_processing_techniques.py`

Role : wrapper de compatibilite qui re-exporte `Entrainement/data_processing_techniques.py`.

## 5. Dossier `Traitement/`

### `Traitement/resample_dataset.py`

Role : transformer les enregistrements bruts en echantillons images + labels alignes temporellement.

Fonctions importantes :

- `resample_commands(df, sample_period_ms)` : recree une grille temporelle reguliere.
- `get_image_files(image_dir)` : recupere les images nommees par timestamp.
- `attach_unique_previous_images(...)` : associe a chaque commande l'image la plus recente disponible.
- `analyze_record(...)` : produit des statistiques pour un enregistrement.
- `process_dataset(config)` : applique tout le pipeline a tous les enregistrements.

Notions a bien comprendre :

- Les vitesses `speedA/speedB` sont interpolees, car ce sont des valeurs quasi continues.
- Les GPIO sont propages par `forward fill`, car on suppose qu'un etat reste valide jusqu'a la commande suivante.
- L'image choisie est volontairement l'image precedente la plus proche, jamais une image future.

### `Traitement/build_balanced.py`

Role : fabriquer un dataset_augmente_equilibre d'apprentissage equilibre.

Fonctions importantes :

- `collect_samples(input_root)` : rassemble tous les echantillons exploitables.
- `split_by_class(df, config)` : cree les splits par classe pour garder des proportions stables.
- `ensure_clean_output(output_root)` : recree proprement l'arborescence de sortie.
- `build_split(...)` : complete chaque classe jusqu'a une cible fixee.

Notions a bien comprendre :

- Pour `left` et `right`, le script peut creer de nouveaux exemples en retournant horizontalement la classe opposee.
- Pour les autres cas, l'augmentation repose surtout sur la luminosite et le contraste.
- Le script fixe un nombre cible d'echantillons par classe et par split, ce qui simplifie l'entrainement.

### `Traitement/class_repartition_analysis.py`

Role : mesurer la distribution des classes dans un dataset_augmente_equilibre prepare.

Fonctions importantes :

- `_load_labels(csv_path)` : normalise les labels dans l'espace du modele.
- `collect_label_files(dataset_root)` : trouve tous les `labels.csv`.
- `summarize_dataset(dataset_root)` : retourne un resume par enregistrement et un resume global.

Usage :

```bash
python Traitement/class_repartition_analysis.py segmentation
```

### `Traitement/dataset_preparation.md`

Role : mini mode d'emploi des etapes de preparation de donnees.

### `Traitement/__init__.py`

Role : marque le dossier comme package Python et documente son but.

## 6. Dossier `Entrainement/`

Ce dossier contient des points d'entree historiques et des re-exports utiles pour conserver la compatibilite avec l'ancienne organisation du code.

### `Entrainement/Acquisition.py`

Role : re-exporte `AutonomousCardataset` depuis `dataset.py`.

### `Entrainement/analyse_dataset.py`

Role : script CLI qui appelle `Traitement.class_repartition_analysis.summarize_dataset`.

### `Entrainement/data_processing_techniques.py`

Role : expose les transforms image et un `WeightedRandomSampler` pour les usages d'entrainement.

Fonction importante :

- `weighted_sampler(df_train, label_column="pseudo_class")` : construit un sur-echantillonnage inversement proportionnel a la frequence de classe.

### `Entrainement/segmentation_commandes_csv.py`

Role : point d'entree historique vers `Traitement.resample_dataset`.

### `Entrainement/__init__.py`

Role : expose `AutonomousCardataset` au niveau du package.

## 7. Dossier `tests_unitaire/`

### `tests_unitaire/test_actions.py`

Role : verifier la logique de conversion des actions.

Ce qui est teste :

- la normalisation des labels,
- l'heuristique `motor_to_action`,
- la classification a partir d'une ligne CSV.

### `tests_unitaire/test_data_processing_techniques.py`

Role : verifier les transformations d'image et le `WeightedRandomSampler`.

Ce qui est teste :

- la forme et la plage des tenseurs,
- le caractere deterministe du transform de test,
- le comportement du bruit gaussien,
- le reequilibrage statistique du sampler.

## 8. Par ou commencer si tu reprends le projet

Ordre conseille de lecture :

1. `README.md` pour la vue d'ensemble rapide.
2. `config.py` pour les chemins et labels.
3. `actions.py` pour comprendre la logique metier.
4. `Traitement/resample_dataset.py` puis `Traitement/build_balanced_dataset.py` pour suivre la preparation des donnees.
5. `dataset.py`, `transforms.py`, `model.py`, `training.py` pour la partie ML.
6. `inference.py` et `predict_action.py` pour l'exploitation du modele.

## 9. Commandes utiles

Preparation des donnees :

```bash
python Traitement/resample_dataset.py
python Traitement/build_balanced_dataset.py
python Traitement/class_repartition_analysis.py segmentation
```

Entrainement :

```bash
python train_action_model.py
```

Inference :

```bash
python predict_action.py path/to/image.png
```

Tests :

```bash
pytest tests_unitaire
```

## 10. Resume rapide

Si tu veux retenir une seule idee, la voici :

- `actions.py` convertit la telemetrie en labels,
- `Traitement/` fabrique un dataset_augmente_equilibre propre et equilibre,
- `training.py` entraine le CNN,
- `inference.py` recharge le modele pour predire sur une image.
