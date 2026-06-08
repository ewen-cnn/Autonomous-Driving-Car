=========================================================
TRAITEMENT DU DATASET AUTONOMOUS DRIVING CAR
=========================================================

Objectif
---------

Le dataset original contient :

- un fichier labels.csv contenant les commandes envoyées au véhicule
- un dossier Images contenant les images acquises par la caméra

Problème :

Les commandes et les images ne sont pas enregistrées à la même fréquence.

Dans la version V2 du dataset :

- les images sont enregistrées à haute fréquence
- les commandes GPIO sont enregistrées uniquement lorsqu'un changement est détecté

Il est donc nécessaire de reconstruire une séquence de commandes régulière afin de pouvoir associer chaque image à une commande.


=========================================================
ÉTAPE 1 : RÉÉCHANTILLONNAGE DES COMMANDES
=========================================================

Les lignes du fichier labels.csv sont rééchantillonnées à période fixe.

Période utilisée :

    250 ms

Exemple :

CSV original :

    t=1000 ms
    t=1350 ms
    t=1800 ms

CSV rééchantillonné :

    t=1000 ms
    t=1250 ms
    t=1500 ms
    t=1750 ms

Le nouveau CSV possède donc une commande toutes les 250 ms.


=========================================================
ÉTAPE 2 : INTERPOLATION DES VITESSES
=========================================================

Les colonnes :

    speedA
    speedB

sont interpolées linéairement.

Exemple :

Commande 1 :

    t=1000 ms
    speedA=0

Commande 2 :

    t=1500 ms
    speedA=50

Après interpolation :

    t=1250 ms
    speedA=25

L'objectif est de reconstruire une évolution progressive de la vitesse entre deux commandes.


=========================================================
ÉTAPE 3 : TRAITEMENT DES GPIO
=========================================================

Les GPIO représentent des états logiques.

Ils ne sont donc PAS interpolés.

Le script conserve simplement la dernière valeur connue.

Exemple :

    t=1000 ms
    GPIO1=1

    t=1500 ms
    GPIO1=0

Après rééchantillonnage :

    t=1250 ms
    GPIO1=1

La valeur reste identique jusqu'au prochain changement.


=========================================================
ÉTAPE 4 : ASSOCIATION DES IMAGES
=========================================================

Chaque image possède un timestamp dans son nom.

Exemple :

    1754930112983.png

Le timestamp de l'image est :

    1754930112983

Pour chaque ligne du CSV rééchantillonné, le script cherche :

    l'image la plus proche dans le temps
    mais STRICTEMENT antérieure ou égale au timestamp CSV

Exemple :

Images :

    1000.png
    1100.png
    1200.png
    1300.png

Commande :

    t = 1280 ms

Image retenue :

    1200.png

L'image 1300.png est rejetée car elle est postérieure à la commande.


=========================================================
ÉTAPE 5 : CONTRAINTE D'UNICITÉ
=========================================================

Une image ne peut être utilisée qu'une seule fois.

Une commande ne peut être associée qu'à une seule image.

La relation est donc :

    1 image <-> 1 ligne CSV

Si une image a déjà été utilisée, elle ne pourra plus être réutilisée pour une autre commande.

Cette contrainte garantit un dataset propre pour
l'entraînement des réseaux de neurones.


=========================================================
ÉTAPE 6 : COPIE DES IMAGES
=========================================================

Les images sélectionnées sont copiées dans :

    resampled_dataset/
        Record_xxx/
            Images/

Le CSV correspondant est enregistré dans :

    segmentation/
        Record_xxx/
            labels/
                labels.csv


=========================================================
SORTIE FINALE
=========================================================

Structure produite :

segmentation/
│
├── Record_2025-08-11_18-34-54/
│   │
│   ├── Images/
│   │   ├── 1754930112983.png
│   │   ├── 1754930113377.png
│   │   └── ...
│   │
│   └── labels/
│       └── labels.csv
│
└── resampling_analysis.csv


=========================================================
ANALYSE PRODUITE
=========================================================

Le fichier :

    resampling_analysis.csv

contient pour chaque enregistrement :

- nombre de lignes CSV originales
- nombre de lignes rééchantillonnées
- nombre de lignes supprimées
- nombre d'images utilisées
- nombre d'images non utilisées
- vitesse min / max / moyenne
- répartition des classes de direction

Cela permet de vérifier la qualité du traitement avant
l'entraînement du modèle.
=========================================================



baan@PortabHP2024:/mnt/c/Users/33658/Documents/enib/s9/ias/projet_final/Autonomous-Driving-Car$ python3 resample_dataset.py 

==============================
RÉÉCHANTILLONNAGE + ASSOCIATION IMAGES
==============================
Dossier source : Record_V2
Dossier sortie : resampled_dataset
Période        : 250 ms

------------------------------
Record : Record_2025-08-08_19-36-02
Lignes CSV originales          : 2741
Lignes CSV rééchantillonnées   : 5867
Lignes gardées avec image      : 5197
Lignes supprimées sans image   : 670
Images copiées                 : 5197
Images non utilisées             : 19320
CSV généré                     : resampled_dataset/Record_2025-08-08_19-36-02/labels/labels.csv
Dossier images                 : resampled_dataset/Record_2025-08-08_19-36-02/Images

------------------------------
Record : Record_2025-08-11_18-31-58
Lignes CSV originales          : 46
Lignes CSV rééchantillonnées   : 304
Lignes gardées avec image      : 71
Lignes supprimées sans image   : 233
Images copiées                 : 71
Images non utilisées             : 157
CSV généré                     : resampled_dataset/Record_2025-08-11_18-31-58/labels/labels.csv
Dossier images                 : resampled_dataset/Record_2025-08-11_18-31-58/Images

------------------------------
Record : Record_2025-08-11_18-34-54
Lignes CSV originales          : 72
Lignes CSV rééchantillonnées   : 318
Lignes gardées avec image      : 120
Lignes supprimées sans image   : 198
Images copiées                 : 120
Images non utilisées             : 362
CSV généré                     : resampled_dataset/Record_2025-08-11_18-34-54/labels/labels.csv
Dossier images                 : resampled_dataset/Record_2025-08-11_18-34-54/Images

==============================
TERMINÉ
==============================
Analyse globale : resampled_dataset/dataset_analysis.csv
baan@PortabHP2024:/mnt/c/Users/33658/Documents/enib/s9/ias/projet_final/Autonomous-Driving-Car$ 