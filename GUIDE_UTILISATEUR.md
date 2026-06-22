# Guide utilisateur — Autonomous Driving Car

## Objectif

Ce projet entraîne un réseau de neurones à partir d’images de conduite afin de prédire une direction :

- `left`
- `forward`
- `right`

Il propose aussi les outils qui permettent de transformer des enregistrements bruts — commandes moteur, états GPIO et images horodatées — en jeu de données équilibré.

## État actuel du dépôt

Le dépôt contient déjà :

- un jeu de données équilibré dans `dataset_augmente_equilibre/` ;
- trois splits : `train`, `valid` et `test` ;
- des checkpoints `driving_cnn.pth` et `MULTITASK_DRIVING_CNN.pth` ;
- un script d’entraînement, des scripts de préparation et des tests unitaires.

Pour entraîner à partir du jeu déjà disponible, il n’est pas nécessaire d’exécuter le pipeline de préparation.

## Installation

Installez une version de Python compatible avec PyTorch, puis créez un environnement virtuel à la racine du projet :

```bash
python -m venv .venv
```

Activez-le et installez les dépendances :

```bash
pip install -r requirements.txt
pip install scikit-learn
```

`scikit-learn` est requis par `train_action_model.py` pour calculer la matrice de confusion et le rapport de classification ; il n’est pas encore déclaré dans `requirements.txt`.

Sous PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Si `.venv` a été créé avec une installation de Python qui n’existe plus, supprimez uniquement ce dossier local puis recréez l’environnement avec la commande ci-dessus.

## Organisation du projet

```text
.
├── actions.py                         # Télémetrie moteur → direction
├── config.py                          # Chemins et constantes partagés
├── dataset.py                         # Dataset PyTorch
├── transforms.py                      # Prétraitement et augmentations
├── model.py                           # CNN classification / régression / multi-tâche
├── train_action_model.py              # Entraînement et évaluation
├── inference.py                       # Chargement et prédiction d’une image
├── predict_action.py                  # Interface en ligne de commande d’inférence
├── dataset_augmente_equilibre/        # Splits prêts à l’emploi
├── Traitement/                        # Préparation et analyse de données
├── Entrainement/                      # Modules de compatibilité et utilitaires
└── tests_unitaire/                    # Tests automatisés
```

## Jeu de données prêt à l’emploi

L’entraînement lit les chemins suivants :

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

Les CSV sont séparés par `;`. Le chargeur `AutonomousCardataset` attend :

- une colonne image parmi `image_filename`, `image_path` ou `filename` ;
- une colonne `direction_class` quand elle est disponible.

Sans `direction_class`, il reconstruit la direction depuis `speedA`, `speedB`, `GPIO1`, `GPIO2`, `GPIO3` et `GPIO4`. Les références d’images peuvent être relatives ou absolues ; le chargeur recherche aussi les fichiers dans un sous-dossier `Images/`.

### Labels et normalisation

Le modèle utilise l’ordre fixe suivant : `left`, `forward`, `right`.

Les actions détaillées sont converties ainsi :

| Action source | Classe modèle |
| --- | --- |
| `left`, `light_left`, `pivot_left`, `sharp_left` | `left` |
| `right`, `light_right`, `pivot_right`, `sharp_right` | `right` |
| `forward` | `forward` |
| `stop`, `backward`, `other`, `ignored` | échantillon ignoré |

La fonction `classify_direction()` applique la logique GPIO propre au câblage du véhicule. Lorsque les deux moteurs avancent, la comparaison des vitesses permet de distinguer la marche avant d’un virage léger.

## Entraîner le modèle

Depuis la racine du dépôt :

```bash
python train_action_model.py
```

Les valeurs par défaut, définies dans `train_action_model.py` et `config.py`, sont les suivantes :

| Paramètre | Valeur |
| --- | --- |
| Taille d’image | `120 × 160` |
| Batch size | `32` |
| Époques | `20` |
| Learning rate | `0.001` |
| Seuil `left` | `0.90` |
| Seuil `forward` | `0.60` |
| Seuil `right` | `0.92` |
| Checkpoint de sortie | `MULTITASK_DRIVING_CNN.pth` |

### Déroulement complet d’un entraînement

Au lancement, `main()` fixe la graine PyTorch à `42`, choisit `cuda` s’il est disponible (sinon `cpu`) et vérifie l’existence des six chemins indispensables : les CSV et dossiers `Images/` des splits `train`, `valid` et `test`.

Le script supprime ensuite `MULTITASK_DRIVING_CNN.pth` s’il existe déjà. C’est important : le fichier créé par une exécution précédente n’est donc pas conservé comme repli si le nouvel entraînement n’atteint aucun seuil de sauvegarde.

`AutonomousCardataset` lit les CSV, élimine les lignes dont le label ne fait pas partie de `left`, `forward`, `right` ou dont l’image est introuvable, puis renvoie à PyTorch une paire :

```text
(image, label)
image : tenseur float32 de forme [3, 120, 160]
label : entier long, 0 pour left, 1 pour forward, 2 pour right
```

Le `DataLoader` assemble ces paires en mini-lots. Avec la configuration par défaut, un lot complet a donc la forme suivante :

```text
images : [32, 3, 120, 160]
labels : [32]
```

Les lots du train sont mélangés à chaque époque (`shuffle=True`). Ceux de validation et de test ne le sont pas : leur ordre n’influence pas le résultat, ce qui rend les mesures plus faciles à reproduire.

#### Préparation des images

Avant d’entrer dans le réseau, chaque image est ouverte en RGB, puis redimensionnée à 120 pixels de haut et 160 pixels de large.

Pour le train, `build_train_transform()` ajoute volontairement de petites variations :

1. `ColorJitter` modifie légèrement luminosité, contraste, saturation et teinte ;
2. `GaussianBlur(3)` est appliqué dans 20 % des cas ;
3. `RandomAffine` effectue une petite rotation (au plus 5°), translation et mise à l’échelle ;
4. l’image est convertie en tenseur dont les valeurs sont d’abord dans `[0, 1]` ;
5. un bruit gaussien borné est ajouté dans 15 % des cas ;
6. la normalisation ImageNet soustrait les moyennes RGB `[0.485, 0.456, 0.406]` et divise par les écarts-types `[0.229, 0.224, 0.225]`.

Ces augmentations ne créent pas de nouveaux fichiers : elles sont tirées à la volée à chaque lecture d’image. Deux passages de la même image dans des époques différentes peuvent donc produire deux tenseurs légèrement différents. L’objectif est d’empêcher le modèle de mémoriser les pixels exacts au lieu d’apprendre des repères de conduite robustes.

Pour `valid` et `test`, le pipeline est volontairement déterministe : redimensionnement, conversion en tenseur et normalisation uniquement. Ainsi, une variation d’accuracy entre deux époques vient du modèle et non d’une augmentation aléatoire appliquée au jeu d’évaluation.

#### Ce qui se passe dans un mini-lot

Pour chaque mini-lot, `train_one_epoch()` exécute précisément les opérations suivantes :

1. `model.train()` place le réseau en mode entraînement. Les couches `Dropout` deviennent actives et les `BatchNorm2d` utilisent les statistiques du lot courant.
2. Les images et les labels sont déplacés sur le CPU ou GPU choisi.
3. `optimizer.zero_grad()` remet à zéro les gradients conservés depuis le lot précédent. Sans cette ligne, PyTorch additionnerait les gradients successifs.
4. `outputs = model(images)` réalise le passage avant. Comme `DrivingCNN` hérite actuellement de `DrivingMultiTaskCNN`, `outputs` est un dictionnaire ; le script retient `outputs["logits"]` pour la classification.
5. `criterion(logits, labels)` calcule la loss avec `nn.CrossEntropyLoss()`.
6. `loss.backward()` propage l’erreur de la sortie vers toutes les couches ayant contribué aux logits ; PyTorch calcule alors un gradient pour chaque poids concerné.
7. `optimizer.step()` applique une mise à jour Adam aux paramètres qui ont reçu un gradient.

Cette séquence est répétée pour tous les lots de l’époque. La valeur affichée sous `Train loss` est la moyenne arithmétique des losses de lots (`running_loss / len(dataloader)`). Le dernier lot, s’il est plus petit, a le même poids dans cette moyenne que les lots complets.

### Loss de classification : ce qui est réellement calculé

La dernière couche de classification produit trois **logits** par image. Ce sont des scores bruts, non bornés et non normalisés. Pour une image, on peut imaginer :

```text
logits = [2.1, 0.4, -0.8]     # left, forward, right
```

`CrossEntropyLoss` applique en interne un softmax stable numériquement, puis calcule l’opposé du logarithme de la probabilité de la vraie classe. Pour une image `i`, si `yᵢ` est son vrai index et `zᵢ` ses logits :

```text
pᵢ,c = exp(zᵢ,c) / Σₖ exp(zᵢ,k)
lossᵢ = -log(pᵢ,yᵢ)
```

La loss du lot est la moyenne des `lossᵢ`. Il ne faut donc pas ajouter un `softmax` dans `model.py` avant `CrossEntropyLoss` : cette fonction attend les logits et effectue elle-même l’opération nécessaire de manière plus stable.

Avec trois classes et une prédiction totalement uniforme, la loss théorique est environ `-log(1/3) = 1.0986`. Une loss faible indique que le modèle attribue une forte probabilité à la bonne classe ; une loss élevée indique qu’il se trompe, particulièrement s’il est très confiant dans une mauvaise classe.

La loss de train ne diminue pas forcément à chaque lot ni même à chaque époque. Les images sont mélangées, les augmentations changent et Adam effectue des pas successifs : de petites oscillations sont normales. Il faut surtout observer une tendance sur plusieurs époques et la comparer à l’accuracy de validation. Une loss de train qui baisse pendant que l’accuracy de validation stagne ou baisse peut signaler du surapprentissage.

### Comment les poids sont ajustés

Chaque couche possède des poids et, le plus souvent, des biais. Après le passage avant, `loss.backward()` applique la règle de dérivation en chaîne :

- la couche finale reçoit directement l’erreur entre ses logits et les labels ;
- cette erreur est propagée vers la couche dense précédente ;
- puis vers les cartes de caractéristiques convolutives ;
- enfin vers chaque filtre de convolution de l’encodeur.

Le gradient d’un poids indique de combien la loss augmenterait ou diminuerait si ce poids changeait très légèrement. Le rôle de l’optimiseur est de déplacer chaque poids dans une direction qui réduit la loss sur les exemples observés.

L’optimiseur utilisé est `torch.optim.Adam` avec `lr=0.001`. Adam conserve, pour chaque paramètre, une moyenne mobile du gradient (`m`) et une moyenne mobile de son carré (`v`). Sous ses valeurs par défaut, les coefficients sont β₁ = 0,9 et β₂ = 0,999. Schématiquement, à l’itération `t` :

```text
mₜ = β₁ mₜ₋₁ + (1 - β₁) gₜ
vₜ = β₂ vₜ₋₁ + (1 - β₂) gₜ²
θₜ = θₜ₋₁ - lr × m̂ₜ / (√v̂ₜ + ε)
```

`gₜ` est le gradient calculé par rétropropagation et `θ` représente un poids ou un biais. Les versions corrigées `m̂` et `v̂` compensent le fait que les moyennes sont initialement nulles. Adam adapte ainsi la taille effective du pas à chaque paramètre : une direction dont les gradients varient beaucoup reçoit en général des mises à jour plus prudentes.

### Entraînement multi-tâche : comportement actuel

L’architecture retourne à la fois :

```python
{
    "logits": logits_de_classification,
    "motor_commands": commandes_speedA_speedB,
}
```

Cependant, la boucle actuelle calcule uniquement :

```python
loss = CrossEntropyLoss(outputs["logits"], labels)
```

La tête `regressor_head` ne participe donc pas à la loss, ne reçoit pas de gradient et ses poids ne sont pas entraînés. L’encodeur partagé apprend des caractéristiques utiles à la classification, mais les deux commandes continues retournées par `motor_commands` ne sont pas exploitables comme commandes de conduite tant qu’une loss de régression, des cibles `speedA`/`speedB` et une pondération de loss multi-tâche n’ont pas été ajoutées.

### Validation, sélection et test

Après chaque époque, `compute_accuracy()` appelle `model.eval()` et englobe les calculs dans `torch.no_grad()` :

- le dropout est désactivé ;
- les batch normalizations utilisent leurs statistiques accumulées ;
- aucun gradient n’est stocké, ce qui réduit l’usage mémoire ;
- la classe prédite est l’index du plus grand logit (`argmax`).

L’accuracy globale est la proportion de labels correctement prédits. Le script calcule également une accuracy séparée pour `left`, `forward` et `right` ; c’est essentiel pour éviter qu’une bonne performance sur la classe majoritaire masque une classe de virage médiocre.

Un checkpoint est sauvegardé seulement si, durant la même époque, les trois seuils sont respectés (`left ≥ 0.90`, `forward ≥ 0.60`, `right ≥ 0.92`) **et** si l’accuracy globale de validation est supérieure à celle du meilleur checkpoint déjà sauvegardé. Le fichier contient l’époque, les poids (`model_state_dict`), l’ordre des classes, la taille d’image et l’accuracy de validation.

Si au moins un checkpoint a été enregistré, le script recharge le meilleur modèle, le passe à nouveau en mode évaluation et mesure une dernière fois l’accuracy par classe sur `test`. Il affiche ensuite une matrice de confusion — lignes : vraies classes, colonnes : classes prédites — et un rapport `precision` / `recall` / `f1-score` issu de `scikit-learn`.

## Architecture du modèle

`model.py` sépare l’architecture en un extracteur visuel partagé (`DrivingFeatureExtractor`) et une ou deux têtes de prédiction. Le cheminement ci-dessous utilise la taille effectivement configurée par défaut : une image RGB de forme `[batch, 3, 120, 160]`, où 120 est la hauteur et 160 la largeur.

### Extracteur commun : du pixel au vecteur de 256 valeurs

Chaque convolution utilise un noyau 3 × 3, un stride de 1 et un padding de 1. Dans cette configuration, la convolution conserve la hauteur et la largeur ; elle modifie seulement le nombre de canaux. Les 32, 64, 128 et 256 canaux correspondent au nombre de filtres appris : chaque filtre cherche un motif visuel différent.

| Étape | Couche et sortie pour une image de 120 × 160 | Rôle à cet emplacement |
| --- | --- | --- |
| Entrée | `[3, 120, 160]` | Trois canaux RGB normalisés. |
| Bloc 1 — convolution | `Conv2d(3, 32, 3, padding=1)` → `[32, 120, 160]` | Chaque filtre 3 × 3 observe de très petits voisinages de pixels. À ce niveau, le réseau apprend typiquement des contours, contrastes, lignes, transitions sol/obstacle et couleurs locales. Conserver la résolution permet de ne pas perdre prématurément les détails. |
| Bloc 1 — normalisation | `BatchNorm2d(32)` → `[32, 120, 160]` | Normalise chaque canal à l’aide des statistiques du lot pendant l’entraînement. Cela stabilise l’échelle des activations reçues par les couches suivantes et rend l’optimisation moins sensible à leur amplitude. |
| Bloc 1 — activation | `ReLU(inplace=True)` → `[32, 120, 160]` | Remplace les valeurs négatives par zéro. Sans non-linéarité, l’empilement de convolutions resterait essentiellement une transformation linéaire. `inplace=True` économise de la mémoire ; il ne change pas le calcul conceptuel. |
| Bloc 1 — réduction | `MaxPool2d(2)` → `[32, 60, 80]` | Conserve le maximum de chaque fenêtre 2 × 2 et divise hauteur et largeur par deux. Les motifs simples deviennent moins sensibles à un petit décalage dans l’image et le coût des blocs suivants diminue. |
| Bloc 2 — convolution | `Conv2d(32, 64, 3, padding=1)` → `[64, 60, 80]` | Combine les contours et textures du premier bloc pour former des motifs plus structurés : segments de route, bords, bandes, zones de végétation ou portions d’obstacles. Le nombre de filtres double parce que la représentation devient plus riche. |
| Bloc 2 — normalisation + ReLU | `BatchNorm2d(64)` puis `ReLU` → `[64, 60, 80]` | Même rôle que dans le premier bloc : activations stables puis sélection non linéaire des motifs utiles. |
| Bloc 2 — réduction | `MaxPool2d(2)` → `[64, 30, 40]` | Réduit encore la résolution tout en agrandissant le champ visuel effectif de chaque neurone des couches suivantes. |
| Bloc 3 — convolution | `Conv2d(64, 128, 3, padding=1)` → `[128, 30, 40]` | Assemble les motifs intermédiaires en indices de conduite plus étendus : géométrie locale de la piste, direction d’une ligne, ouverture d’un virage ou présence d’une zone libre. |
| Bloc 3 — normalisation + ReLU | `BatchNorm2d(128)` puis `ReLU` → `[128, 30, 40]` | Maintient une distribution d’activation plus stable tandis que le réseau devient plus profond. |
| Bloc 3 — réduction | `MaxPool2d(2)` → `[128, 15, 20]` | Les neurones suivants reçoivent une vue plus large du contenu de l’image, avec un coût calculatoire réduit. |
| Bloc 4 — convolution | `Conv2d(128, 256, 3, padding=1)` → `[256, 15, 20]` | Produit 256 cartes de caractéristiques de haut niveau. À cette profondeur, les filtres peuvent combiner la structure de la scène et des indices de direction plutôt que de simples pixels voisins. |
| Bloc 4 — normalisation + ReLU | `BatchNorm2d(256)` puis `ReLU` → `[256, 15, 20]` | Stabilise les 256 cartes finales et conserve leurs réponses positives. Il n’y a pas de quatrième max-pooling : la représentation garde encore une grille 15 × 20 avant l’agrégation globale. |
| Agrégation globale | `AdaptiveAvgPool2d((1, 1))` → `[256, 1, 1]` | Calcule une moyenne spatiale par carte de caractéristiques. Chaque canal est résumé par un nombre : « à quel point ce motif est-il présent dans l’image ? ». La couche s’adapte à toute taille spatiale entrante ; elle évite de fixer une grande couche dense dépendante de 15 × 20. |
| Aplatissement | `Flatten()` → `[256]` | Transforme `[256, 1, 1]` en un vecteur de 256 caractéristiques, attendu par les couches linéaires des têtes. |

Le champ réceptif augmente progressivement : un neurone profond dépend indirectement d’une région plus large de l’image qu’un filtre du premier bloc. C’est précisément ce qui permet de passer de « cette zone contient un bord » à « la scène suggère un virage à gauche ».

### Tête de classification

`DrivingClassifierCNN` et la tête de classification de `DrivingMultiTaskCNN` ont la même structure :

```text
vecteur 256
  → Linear(256, 128)
  → ReLU
  → Dropout(p=0.30, seulement pendant train)
  → Linear(128, 3)
  → logits [left, forward, right]
```

`Linear(256, 128)` apprend à recombiner les 256 indices visuels globaux en 128 représentations plus directement utiles à la décision. Après la `ReLU`, `Dropout(p=0.30)` met aléatoirement à zéro environ 30 % des activations durant l’entraînement. Les activations restantes sont automatiquement mises à l’échelle par PyTorch ; le modèle ne peut donc pas dépendre systématiquement d’un seul neurone. En évaluation et en inférence, le dropout est désactivé.

La dernière couche `Linear(128, 3)` possède trois sorties, une par classe, dans l’ordre `left`, `forward`, `right`. Elle ne contient ni `ReLU` ni `softmax`, car il faut pouvoir produire des scores négatifs ou positifs et parce que `CrossEntropyLoss` applique le softmax nécessaire pendant l’entraînement.

### Tête de régression moteur

`DrivingRegressorCNN` possède une tête distincte :

```text
vecteur 256
  → Linear(256, 128) → ReLU → Dropout(p=0.20)
  → Linear(128, 64)  → ReLU
  → Linear(64, 2)
  → tanh × 100
  → [speedA, speedB]
```

La couche intermédiaire de 64 unités permet une transformation supplémentaire avant les deux sorties numériques. `tanh` borne chaque sortie entre -1 et 1 ; la multiplication par `max_abs_speed=100` produit donc des commandes comprises entre -100 et +100. Cette borne évite qu’une sortie du réseau devienne arbitrairement grande.

`DrivingMultiTaskCNN` partage exactement le même extracteur pour les deux têtes, puis retourne un dictionnaire avec les logits et ces deux valeurs moteur. `DrivingCNN` hérite actuellement de cette classe : c’est donc cette architecture multi-tâche qui est créée par `train_action_model.py` et `inference.py`.

Les trois variantes présentes dans le code sont donc `DrivingClassifierCNN`, `DrivingRegressorCNN` et `DrivingMultiTaskCNN`. La dernière est l’architecture effectivement utilisée par l’alias `DrivingCNN`.

## Régénérer le jeu de données depuis des traces brutes

Cette partie est facultative et ne doit être lancée que si vous souhaitez reconstruire le dataset.

### 1. Vérifier les chemins et sauvegarder les données existantes

Les chemins par défaut sont dans `config.py` :

```python
dataset_DIR = "dataset_augmente_equilibre"
OUTPUT_DIR = "segmentation"
BALANCED_dataset_DIR = "dataset_augmente_equilibre"
```

Pour des traces brutes, `dataset_DIR` doit désigner un dossier contenant un sous-dossier par enregistrement. Chaque enregistrement doit contenir :

```text
<enregistrement>/
├── labels.csv
└── Images/
    ├── <timestamp>.png
    └── ...
```

> Attention : `Traitement/build_balanced_dataset.py` appelle `shutil.rmtree()` sur son dossier de sortie. Avec les valeurs actuelles, il supprimerait `dataset_augmente_equilibre/` avant de le reconstruire. Sauvegardez le dataset et vérifiez `BuildConfig.output_root` / `BALANCED_dataset_DIR` avant de l’exécuter.

### 2. Rééchantillonner et aligner les images

Après avoir configuré `dataset_DIR` vers les traces brutes :

```bash
python Traitement/resample_dataset.py
```

Le script :

- crée une grille de commandes à 250 ms par défaut ;
- interpole `speedA` et `speedB` ;
- propage les GPIO par maintien de la dernière valeur ;
- associe chaque commande à l’image disponible la plus récente, jamais à une image future ;
- copie les images retenues et écrit `segmentation/<enregistrement>/labels/labels.csv` ;
- écrit un résumé global dans `segmentation/resampling_analysis.csv`.

### 3. Construire les splits équilibrés

Lorsque `segmentation/` est prêt et que la destination configurée est sûre :

```bash
python Traitement/build_balanced_dataset.py
```

Par défaut, le script sépare les données par classe selon les proportions 70 % / 15 % / 15 %, puis cible :

| Split | Cible par classe |
| --- | ---: |
| `train` | 1 000 |
| `valid` | 200 |
| `test` | 200 |

Les exemples manquants de `left` et `right` peuvent être créés par retournement horizontal d’exemples de la classe opposée. Pour les autres cas, le script applique des variations de luminosité ou de contraste. Les lignes CSV générées gardent la trace du type d’augmentation.

### 4. Analyser une segmentation

La fonction `summarize_dataset()` de `Traitement/class_repartition_analysis.py` produit une distribution par enregistrement et une distribution globale. L’entrée CLI directe de ce fichier est actuellement à corriger avant utilisation : son argument est déclaré sous un nom différent de celui relu dans `main()`.

Vous pouvez l’utiliser depuis un interpréteur Python une fois l’environnement configuré :

```python
from pathlib import Path
from Traitement.class_repartition_analysis import summarize_dataset

per_record, global_summary = summarize_dataset(Path("segmentation"))
print(per_record)
print(global_summary)
```

## Inférence sur une image

L’objectif de l’inférence est de reprendre les poids appris et de classer une image sans modifier aucun paramètre. Les deux fichiers concernés ont des responsabilités distinctes :

- `inference.py` contient la logique Python de chargement et de prédiction ;
- `predict_action.py` est une interface en ligne de commande qui parse les arguments, appelle `predict_image()` puis affiche le résultat.

### Chargement d’un checkpoint par `load_model()`

`load_model()` choisit le GPU CUDA s’il est disponible, sinon le CPU, puis lit le fichier `.pth` avec `torch.load(..., map_location=device)`. Il attend un dictionnaire comprenant au moins `model_state_dict`. Le checkpoint de l’entraînement enregistre aussi l’ordre des classes et la taille d’image.

Avant de charger les poids, la fonction vérifie que les classes du checkpoint correspondent exactement à `actions.CLASSES`, soit `left`, `forward`, `right`. Cette vérification évite d’interpréter la sortie d’un modèle avec un ordre de classes différent — par exemple traiter l’index 0 comme `left` alors qu’il aurait été entraîné pour signifier `forward`.

La fonction crée ensuite un `DrivingCNN(num_classes=3)`, copie les poids du checkpoint avec `load_state_dict()`, place le modèle sur le bon périphérique et appelle `model.eval()`. Ce dernier appel désactive le dropout et fige le comportement de BatchNorm pour une prédiction stable.

### Prétraitement et décision dans `predict_image()`

Une fois le modèle chargé, `predict_image()` :

1. récupère `image_size` dans le checkpoint ; si elle est absente, il essaie `img_size`, puis utilise la valeur de `config.py` (`120, 160`) ;
2. ouvre l’image avec Pillow et la force en RGB ;
3. applique `build_eval_transform()` : redimensionnement, conversion en tenseur et normalisation ImageNet ;
4. ajoute une dimension de batch avec `unsqueeze(0)`, ce qui transforme `[3, H, W]` en `[1, 3, H, W]` ;
5. exécute le passage avant sous `torch.no_grad()` : aucun gradient n’est calculé ni stocké ;
6. transforme les logits en probabilités par softmax, sélectionne l’index maximal avec `argmax` et le convertit en nom de classe grâce à `IDX_TO_CLASS`.

La confiance affichée est la plus grande probabilité softmax. Elle répond à « quelle part de la masse de probabilité interne le modèle attribue-t-il à la classe choisie ? » et non à « quelle est la garantie que la voiture peut exécuter l’action sans risque ». Elle n’est pas calibrée pour une décision de sécurité réelle.

### Utilisation prévue de `predict_action.py`

L’interface prévue est :

```bash
python predict_action.py chemin/vers/image.png --model-path MULTITASK_DRIVING_CNN.pth
```

L’argument positionnel est le chemin de l’image. `--model-path` est optionnel : sans lui, le script utilise `driving_cnn.pth`, défini par `DEFAULT_MODEL_PATH`. Après une prédiction valide, l’affichage attendu est de cette forme :

```text
Predicted action: left
Confidence: 87.42%
```

### Limite actuelle à corriger avant exécution

Le dernier passage de `predict_image()` appelle actuellement :

```python
outputs = model(tensor)
probabilities = torch.softmax(outputs, dim=1)
```

Cela fonctionnait avec un modèle qui renvoyait directement un tenseur de logits. Or `DrivingCNN` renvoie maintenant un dictionnaire multi-tâche :

```python
{
    "logits": tensor_de_forme_[1, 3],
    "motor_commands": tensor_de_forme_[1, 2],
}
```

`torch.softmax()` ne peut pas être appliqué à ce dictionnaire. La commande CLI n’est donc pas fonctionnelle avec l’architecture actuelle tant que les logits ne sont pas extraits. Le comportement attendu après la correction est conceptuellement :

```python
outputs = model(tensor)
logits = outputs["logits"] if isinstance(outputs, dict) else outputs
probabilities = torch.softmax(logits, dim=1)
predicted_index = torch.argmax(probabilities, dim=1).item()
```

Cette correction ne transforme pas le script en pilote autonome : il renvoie seulement une classe et une confiance pour une image isolée. Il ne lit pas de flux caméra continu, ne commande aucun moteur et ne vérifie pas les conditions de sécurité.

## Tests

Exécutez les tests depuis la racine du dépôt :

```bash
pytest tests_unitaire
```

Ils couvrent :

- la normalisation et la classification des actions moteur ;
- le bruit gaussien et les transformations d’images ;
- le sampler pondéré utilisé par les outils historiques d’entraînement.

## Dépannage rapide

| Symptôme | Vérification |
| --- | --- |
| `FileNotFoundError` au lancement de l’entraînement | Vérifier les trois CSV et les dossiers `Images/` sous `dataset_augmente_equilibre/`. |
| `No usable samples found` | Vérifier les noms d’images, le séparateur `;` et les valeurs de `direction_class`. |
| Aucun modèle sauvegardé | Lire les accuracies par classe : les trois seuils doivent être atteints durant la même époque. |
| `ModuleNotFoundError: sklearn` | Installer `scikit-learn` dans l’environnement actif. |
| Erreur pendant la prédiction | Voir la limitation de compatibilité multi-tâche décrite dans la section Inférence. |
