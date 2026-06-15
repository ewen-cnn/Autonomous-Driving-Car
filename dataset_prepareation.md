# Préparation du dataset de conduite autonome

## Objectif

Le dataset initial était constitué d'enregistrements réalisés lors de plusieurs essais de conduite du véhicule autonome. Chaque essai contenait :

* un fichier `labels.csv` contenant les commandes moteur et les vitesses,
* un dossier `Images` contenant les images acquises par la caméra.

L'objectif était de construire un dataset exploitable pour entraîner un modèle de vision capable de :

1. prédire la direction du véhicule (`left`, `right`, `forward`),
2. prédire les vitesses des moteurs (`speedA` et `speedB`).

---

# 1. Rééchantillonnage temporel des commandes

Script utilisé :

`resample_dataset.py`

Les commandes moteur n'étaient pas enregistrées à fréquence fixe. Afin d'obtenir un jeu de données homogène, les fichiers CSV ont été rééchantillonnés à une période constante de :

```text
250 ms
```

Pour chaque nouveau timestamp :

* `speedA` et `speedB` sont interpolées linéairement,
* les états GPIO sont conservés par maintien de la dernière valeur connue (forward fill).

Cette étape a permis :

* d'uniformiser les données temporelles,
* de créer des transitions de vitesse progressives,
* d'obtenir des vitesses intermédiaires (par exemple 57, 63, 48, etc.).

---

# 2. Association image / commande

Script utilisé :

`resample_dataset.py`

Après rééchantillonnage, chaque ligne du CSV est associée à une image.

Pour un timestamp CSV donné :

* l'image choisie est la dernière image dont le timestamp est inférieur ou égal au timestamp de la commande,
* une image n'est utilisée qu'une seule fois,
* les lignes ne possédant aucune image antérieure sont supprimées.

Les informations suivantes sont ajoutées au CSV :

* `image_filename`
* `image_time_ms`
* `image_delta_ms`

Cette méthode garantit que chaque commande est associée à une image réellement disponible au moment de l'action.

---

# 3. Analyse de la répartition des commandes

Scripts utilisés :

`analyse_dataset.py`

`class_repartition_analysis.py`

Une analyse statistique du dataset a été réalisée afin d'identifier les classes réellement présentes.

Les commandes d'origine étaient :

```text
forward
light_left
light_right
pivot_left
pivot_right
sharp_left
sharp_right
backward
stop
other
```

L'analyse a montré une forte domination des virages à gauche.

Par exemple :

```text
left     : très majoritaire
forward  : minoritaire
right    : extrêmement rare
```

Cette situation provenait principalement du circuit utilisé lors des acquisitions.

---

# 4. Simplification du problème

Afin de réduire la complexité du problème de classification, les classes ont été regroupées.

Les nouvelles classes sont :

```text
light_left
sharp_left
pivot_left
    ↓
left
```

```text
light_right
sharp_right
pivot_right
    ↓
right
```

```text
forward
    ↓
forward
```

Les classes suivantes ont été supprimées :

```text
stop
backward
other
```

Le problème est ainsi transformé en une classification à trois classes :

```text
left
right
forward
```

---

# 5. Vérification des augmentations miroir

Script utilisé :

`test_flip_left_to_right.py`

Avant de générer le dataset final, une vérification a été réalisée sur une image unique.

Le principe était le suivant :

* sélection d'une image de classe `left`,
* inversion horizontale de l'image,
* création d'une nouvelle ligne CSV.

Les modifications appliquées étaient :

```text
left  -> right
```

et

```text
speedA <-> speedB
```

Cette étape a permis de valider la cohérence des augmentations avant leur application au dataset complet.

---

# 6. Construction du dataset final

Script utilisé :

`generate_dataset_augmente_equilibre.py`

Le dataset final est construit en plusieurs étapes.

## 6.1 Création du train et de la validation

Le dossier `train` du dataset initial est d'abord séparé en :

```text
train_source
valid_source
```

selon la répartition :

```text
80 % train_source
20 % valid_source
```

Cette séparation est effectuée avant toute augmentation afin d'éviter toute fuite de données entre apprentissage et validation.

Ainsi :

* aucune image d'origine présente dans `train_source` ne peut se retrouver dans `valid_source`,
* aucune augmentation issue d'une image de train ne peut apparaître dans la validation.

---

## 6.2 Création du jeu de test

Le jeu de test est constitué à partir des anciens dossiers :

```text
valid
+
test
```

du dataset initial.

Cette stratégie permet de conserver un jeu d'essai totalement indépendant des données d'entraînement.

---

## 6.3 Augmentations utilisées

### Inversion horizontale

Transformation :

```text
left -> right
right -> left
```

Modifications appliquées au CSV :

```text
speedA <-> speedB
```

Cette augmentation permet de générer de nombreux exemples de virages à droite malgré le faible nombre d'images réellement acquises dans cette configuration.

---

### Modification de luminosité

Facteurs utilisés :

```text
0.80
1.20
```

Effets :

* simulation d'éclairages plus sombres,
* simulation d'éclairages plus lumineux.

Les labels ne sont pas modifiés.

---

### Modification de contraste

Facteurs utilisés :

```text
0.85
1.15
```

Effets :

* réduction du contraste,
* augmentation du contraste.

Les labels ne sont pas modifiés.

---

# 7. Répartition finale

Le dataset final est équilibré.

## Entraînement

```text
1000 left
1000 right
1000 forward
```

soit :

```text
3000 images
```

---

## Validation

```text
200 left
200 right
200 forward
```

soit :

```text
600 images
```

---

## Test indépendant

```text
60 left
60 right
60 forward
```

soit :

```text
180 images
```

---

# Conclusion

Le dataset final obtenu présente plusieurs avantages :

* classes équilibrées,
* séparation stricte entre apprentissage et validation,
* test indépendant provenant d'essais différents,
* augmentation cohérente avec la symétrie du véhicule,
* conservation des vitesses interpolées issues des commandes réelles.

Cette préparation permet d'entraîner et d'évaluer les modèles dans des conditions beaucoup plus robustes que le dataset brut initial.
    