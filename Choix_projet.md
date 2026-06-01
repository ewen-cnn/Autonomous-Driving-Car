## Justification des choix d’architecture et de modèle

Le projet consiste à développer un module de décision pour une voiture miniature autonome à partir d’images issues d’une caméra embarquée. Le dataset fourni contient des images associées à des instructions de commande moteur, sous forme de GPIO et/ou de coefficients moteurs. Le problème est donc formulé comme une tâche d’apprentissage supervisé par imitation : le système apprend à reproduire les commandes moteur associées à chaque situation visuelle.

### Choix d’une architecture modulaire

L’architecture du projet a été découpée en plusieurs modules afin de séparer clairement les responsabilités du système :

* un module de chargement du dataset ;
* un module de prétraitement des images ;
* un modèle d’apprentissage basé sur un réseau de neurones convolutif ;
* un module d’entraînement ;
* un module d’évaluation ;
* un module de conversion des sorties du modèle en commandes moteur exploitables.

Cette séparation permet de rendre le projet plus lisible, plus maintenable et plus facilement testable. Par exemple, le chargement des données peut être modifié sans changer le modèle, et le modèle peut être remplacé sans réécrire toute la chaîne d’entraînement. Cette approche est cohérente avec une logique d’ingénierie logicielle où chaque composant possède une responsabilité précise.

### Choix du dataset personnalisé

Un dataset personnalisé a été implémenté afin de faire le lien entre les images et les labels moteurs associés. Dans PyTorch, cette approche permet de contrôler précisément la manière dont les images sont chargées, transformées et associées à leurs commandes.

Le fichier CSV sert de table de correspondance entre chaque image et ses valeurs de commande. Les images sont chargées depuis un dossier dédié, puis converties en RGB afin de garantir un format uniforme. Cette conversion évite les erreurs liées à la présence éventuelle d’images en niveaux de gris ou avec un nombre de canaux différent.

Les commandes GPIO et les coefficients moteurs sont ensuite transformés en valeurs numériques exploitables par le réseau. Cette étape est importante, car un réseau de neurones ne manipule pas directement des états matériels bruts, mais des valeurs numériques normalisées.

### Choix d’une formulation en régression

Le problème a été formulé comme une tâche de régression plutôt que comme une tâche de classification. En effet, les commandes moteur ne correspondent pas uniquement à des classes discrètes telles que “gauche”, “droite” ou “tout droit”. Les coefficients moteurs peuvent prendre des valeurs continues, ce qui permet de représenter différents niveaux d’intensité.

Le modèle prédit donc deux valeurs continues correspondant aux commandes du moteur gauche et du moteur droit. Cette formulation est plus fine qu’une simple classification, car elle permet de produire des commandes progressives. Par exemple, un léger virage et un virage fort ne doivent pas nécessairement produire la même commande.

La sortie du modèle est donc de la forme :

```text
image → [commande_moteur_gauche, commande_moteur_droit]
```

Cette représentation est adaptée à une voiture miniature utilisant une commande différentielle, où la trajectoire dépend de la différence de vitesse entre les deux moteurs.

### Choix de commandes moteur normalisées

Les commandes moteur ont été représentées sous forme de valeurs normalisées dans un intervalle borné, généralement entre -1 et 1. Cette normalisation présente plusieurs avantages.

Premièrement, elle rend l’apprentissage plus stable, car les valeurs manipulées par le réseau restent dans un ordre de grandeur limité. Deuxièmement, elle permet de représenter à la fois le sens et l’intensité de la commande. Une valeur positive peut représenter une marche avant, une valeur négative une marche arrière, et une valeur proche de zéro un arrêt ou une commande très faible.

Cette représentation est plus exploitable qu’une prédiction directe des GPIO, car elle permet de séparer la phase d’apprentissage de la phase de commande matérielle. Le réseau prédit une commande abstraite, puis une fonction dédiée transforme cette commande en GPIO et PWM réels.

### Choix d’un réseau de neurones convolutif

Un réseau de neurones convolutif a été choisi car les données d’entrée sont des images. Les CNN sont adaptés au traitement d’images car ils exploitent la structure spatiale des pixels. Contrairement à un réseau entièrement connecté classique, un CNN peut détecter des motifs locaux tels que des lignes, des bords, des zones de contraste ou des formes de trajectoire.

Dans le contexte du projet, ces motifs visuels sont directement liés aux décisions de conduite. Par exemple, la position de la route, la courbure de la trajectoire ou la présence d’un virage peuvent être extraites progressivement par les couches convolutives.

L’architecture utilisée reste volontairement légère afin d’être cohérente avec les contraintes d’un projet embarqué ou semi-embarqué. Un modèle trop complexe pourrait améliorer les performances sur le dataset d’entraînement, mais augmenterait le risque de surapprentissage et serait plus difficile à exécuter sur une plateforme limitée en ressources.

### Choix d’une architecture CNN légère

Le modèle utilise plusieurs couches de convolution suivies de fonctions d’activation non linéaires. Les premières couches extraient des caractéristiques simples comme les contours et les contrastes. Les couches plus profondes combinent ces informations pour extraire des caractéristiques plus abstraites liées à la direction de la route.

L’utilisation de couches de normalisation, comme BatchNorm, permet de stabiliser l’apprentissage. Le Dropout peut être utilisé pour limiter le surapprentissage en empêchant le réseau de dépendre excessivement de certains neurones.

La dernière partie du modèle est un régresseur qui transforme les caractéristiques visuelles extraites en deux valeurs de commande moteur. Une fonction d’activation bornée, comme Tanh, peut être utilisée en sortie afin de contraindre les prédictions dans l’intervalle [-1, 1].

### Choix de la fonction de perte

Comme le modèle prédit des valeurs continues, une fonction de perte de régression est nécessaire. Une fonction telle que SmoothL1Loss est pertinente car elle combine les avantages de l’erreur absolue et de l’erreur quadratique. Elle est moins sensible aux valeurs aberrantes qu’une MSE classique tout en conservant une bonne capacité d’optimisation.

Ce choix est adapté à un dataset réel, où certaines commandes peuvent être bruitées ou légèrement incohérentes. La SmoothL1Loss permet donc d’entraîner le modèle de manière plus robuste.

### Choix des transformations et de l’augmentation d’images

Les images sont redimensionnées afin d’obtenir une taille fixe en entrée du modèle. Cette étape est nécessaire car un réseau de neurones attend des tenseurs de dimensions constantes.

Des techniques d’augmentation d’images peuvent être appliquées pendant l’entraînement afin d’améliorer la robustesse du modèle. Les variations de luminosité, de contraste, de saturation ou l’ajout d’un léger flou permettent de simuler des conditions visuelles différentes. Cela force le modèle à apprendre des caractéristiques plus générales plutôt que de mémoriser précisément les images du dataset.

Certaines augmentations doivent toutefois être utilisées avec prudence. Par exemple, un retournement horizontal inverse la signification de la commande : un virage à gauche devient un virage à droite. Dans ce cas, les labels moteurs doivent également être inversés. Sinon, le modèle apprendrait des associations fausses.

### Choix du rééquilibrage des données

Dans un dataset de conduite, les situations “tout droit” sont souvent beaucoup plus nombreuses que les virages ou les arrêts. Sans correction, le modèle risque d’être biaisé vers la commande majoritaire et de prédire trop souvent une trajectoire rectiligne.

Comme le problème est formulé en régression, le rééquilibrage ne peut pas être appliqué directement sur des classes explicites. Une solution consiste à créer des pseudo-classes à partir des commandes moteur. Par exemple, la différence entre la commande du moteur droit et celle du moteur gauche permet d’identifier les situations de virage à gauche, virage à droite, ligne droite ou arrêt.

Ces pseudo-classes peuvent ensuite être utilisées pour équilibrer l’échantillonnage pendant l’entraînement. Cette méthode permet de conserver une sortie continue tout en réduisant le biais du dataset.

### Choix des métriques d’évaluation

La loss seule ne suffit pas pour évaluer correctement le modèle. Il est nécessaire d’utiliser des métriques interprétables par rapport au comportement de la voiture.

L’erreur moyenne absolue sur les moteurs gauche et droit permet de mesurer l’écart entre les commandes prédites et les commandes réelles. L’erreur de direction, calculée à partir de la différence entre les deux moteurs, est également importante car elle indique si le modèle prédit correctement la trajectoire.

Il est aussi possible de convertir les commandes continues en pseudo-classes comme “gauche”, “droite”, “tout droit” et “stop”, puis de construire une matrice de confusion. Cette analyse permet d’identifier les erreurs critiques, par exemple lorsqu’un virage est confondu avec une trajectoire droite.

### Choix d’un module de sécurité

Un module de sécurité est nécessaire afin d’éviter d’envoyer des commandes dangereuses lorsque la prédiction est incertaine ou incohérente. Par exemple, si les valeurs prédites sont instables ou si l’image d’entrée est mauvaise, le système peut réduire la vitesse ou envoyer une commande d’arrêt.

Ce choix est essentiel dans un système autonome, même miniature. La décision ne doit pas seulement être performante en moyenne ; elle doit également être robuste face aux cas dégradés.

### Limites de l’approche

L’approche proposée repose uniquement sur la vision. Elle peut donc être sensible aux variations de lumière, aux ombres, à la qualité de la caméra ou à des situations non représentées dans le dataset. De plus, le modèle apprend à imiter les commandes présentes dans les données, mais il ne comprend pas explicitement l’environnement.

Le comportement final dépend donc fortement de la qualité du dataset. Si certaines situations sont absentes ou mal représentées, le modèle risque de mal généraliser. Pour améliorer le système, il serait possible d’ajouter d’autres capteurs, comme des capteurs de distance, une IMU ou des encodeurs moteurs, afin de rendre la décision plus robuste.

### Conclusion

Les choix d’architecture, de modèle et de traitement des données sont cohérents avec la nature du problème. Le dataset associe des images à des commandes moteur, ce qui justifie une approche supervisée par imitation. Le choix d’un CNN est adapté au traitement visuel, tandis que la formulation en régression permet de prédire des commandes continues plus précises que de simples classes de direction. La modularité du code, la normalisation des commandes, l’augmentation des données, le rééquilibrage et les métriques d’évaluation permettent de construire une chaîne d’apprentissage plus robuste, plus lisible et plus défendable techniquement.
