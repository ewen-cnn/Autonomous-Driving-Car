# Autonomous Driving Car

Projet de vision par ordinateur pour un vehicule autonome miniature. A partir d'une image de camera, le modele predit une action parmi trois classes : `left`, `forward` et `right`.

Le projet contient un dataset equilibre, un modele CNN multi-tache, des scripts d'entrainement et d'inference, ainsi que des outils pour reconstruire le dataset depuis des enregistrements bruts.

## Demarrage rapide

Depuis la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install scikit-learn
python train_action_model.py
```

`scikit-learn` est necessaire pour produire la matrice de confusion et le rapport de classification a la fin de l'entrainement.

## Organisation

```text
.
|- actions.py                       # Commandes moteur/GPIO vers labels de direction
|- config.py                        # Chemins, tailles et constantes partagees
|- model.py                         # Architectures CNN
|- train_action_model.py            # Entrainement et evaluation multi-tache
|- inference.py                     # Chargement du modele et prediction Python
|- predict_action.py                # Interface en ligne de commande
|- Traitement/
|  |- transforms.py                 # Transformations d'images
|  |- resample_dataset.py           # Alignement commandes/images brutes
|  |- build_balanced_dataset.py     # Splits et augmentation
|  `- class_repartition_analysis.py # Analyse de la repartition des classes
|- dataset_augmente_equilibre/      # Dataset pret pour l'entrainement
`- tests_unitaire/                  # Tests automatises
```

Les fichiers `dataset.py` et le dossier `Entrainement/` sont des modules de compatibilite historiques. Le pipeline actif est pilote par `train_action_model.py` et `Traitement/`.

## Dataset utilise

L'entrainement attend la structure suivante :

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

Les CSV utilisent le separateur `;`. Ils doivent contenir une colonne image, telle que `image_filename`, une colonne de label (`direction_class`) et, pour l'entrainement multi-tache, les cibles moteur `speedA` et `speedB`.

Les classes sont toujours interpretees dans cet ordre :

```text
left, forward, right
```

Les labels fins sont normalises dans `actions.py` : par exemple `light_left`, `pivot_left` et `sharp_left` deviennent `left`. Les commandes `stop`, `backward` et `other` ne sont pas utilisees comme classes du modele.

Le dataset actuellement present contient :

| Split | Images | Repartition |
| --- | ---: | --- |
| `train` | 3 000 | 1 000 par classe |
| `valid` | 600 | 200 par classe |
| `test` | 180 | 60 par classe |

## Entrainement

Lancez :

```powershell
python train_action_model.py
```

Le script :

- charge les trois splits du dataset ;
- entraine `DrivingCNN`, actuellement un alias de `DrivingMultiTaskCNN` ;
- optimise une loss de classification et une loss de regression des vitesses moteur ;
- evalue chaque epoque sur `valid` ;
- sauvegarde le meilleur modele dans `MULTITASK_DRIVING_CNN.pth` si les seuils par classe sont atteints ;
- recharge ce checkpoint puis l'evalue sur `test`.

Les parametres principaux sont en haut de `train_action_model.py` : batch size, nombre d'epoques, learning rate, ponderation de regression et seuils de validation. Les chemins et le nom du checkpoint sont centralises dans `config.py`.

### Note sur le pretraitement

Le dataset fourni contient des images en `224x224`, mais `DEFAULT_IMAGE_SIZE` vaut actuellement `(120, 160)`. L'entrainement applique seulement `ToTensor()`, tandis que l'inference redimensionne et normalise l'image. Avant de produire un nouveau modele utilisable, il est recommande d'unifier ces transformations dans les deux flux avec les fonctions de `Traitement/transforms.py`.

## Inference sur une image

Pour predire une action :

```powershell
python predict_action.py chemin\vers\image.png --model-path MULTITASK_DRIVING_CNN.pth
```

La commande affiche la classe predite et sa probabilite softmax. Cette probabilite est une confiance du modele, pas une garantie de securite pour piloter un vehicule reel.

Sans `--model-path`, le script utilise `driving_cnn.pth`, defini par `DEFAULT_MODEL_PATH`. Il est preferable de fournir explicitement le checkpoint genere par l'entrainement actuel (`MULTITASK_DRIVING_CNN.pth`) tant que ces deux fichiers coexistent.

## Reconstruction du dataset

Les scripts de `Traitement/` permettent de repartir d'enregistrements bruts. Un enregistrement brut doit avoir cette forme :

```text
raw_records/
`- record_001/
   |- labels.csv
   `- Images/
      |- 1710000000000.png
      `- ...
```

Avant toute execution, configurez trois dossiers distincts dans `config.py` :

```python
dataset_DIR = "raw_records"
OUTPUT_DIR = "segmentation"
BALANCED_dataset_DIR = "dataset_augmente_equilibre"
```

Puis executez :

```powershell
python Traitement/resample_dataset.py
python Traitement/build_balanced_dataset.py
```

Attention : `build_balanced_dataset.py` supprime integralement son dossier de sortie avant de le reconstruire. Ne lancez jamais cette commande avec un dossier de sortie qui contient les seules donnees que vous souhaitez conserver.

`class_repartition_analysis.py` contient actuellement une incoherence dans le nom de son argument CLI. En attendant sa correction, utilisez directement sa fonction Python :

```python
from pathlib import Path
from Traitement.class_repartition_analysis import summarize_dataset

per_record, global_summary = summarize_dataset(Path("segmentation"))
print(per_record)
print(global_summary)
```

## Tests

```powershell
pytest tests_unitaire
```

Les tests couvrent la conversion des commandes moteur en directions, les transformations d'images et l'ancien sampler pondere.

## Fichiers locaux et versionnement

Le dataset, les checkpoints `.pth`, les environnements virtuels et les caches sont ignores par Git. Conservez une copie des donnees et des checkpoints importants avant de nettoyer le repertoire ou de reconstruire le dataset.
