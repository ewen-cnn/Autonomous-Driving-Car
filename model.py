from __future__ import annotations

import torch
import torch.nn as nn


# =============================================================================
# BLOC COMMUN : EXTRACTION DE CARACTÉRISTIQUES IMAGE
# =============================================================================

class DrivingFeatureExtractor(nn.Module):
    """
    Extracteur de caractéristiques visuelles.

    Entrée :
        image RGB : (batch_size, 3, H, W)

    Sortie :
        vecteur de caractéristiques : (batch_size, 256)

    Ce bloc est commun aux modèles de classification, régression et multi-tâche.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            # Bloc 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Bloc 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Bloc 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Bloc 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Rend la sortie indépendante de la taille exacte de l'image
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        return x


# =============================================================================
# MODÈLE 1 : CLASSIFICATION IMAGE -> ACTION
# =============================================================================

class DrivingClassifierCNN(nn.Module):
    """
    Modèle de classification.

    Objectif :
        image caméra -> classe d'action

    Exemple de classes :
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

    Sortie :
        logits de taille (batch_size, num_classes)

    Important :
        On n'applique PAS softmax dans le modèle.
        CrossEntropyLoss le fait implicitement pendant l'entraînement.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.encoder = DrivingFeatureExtractor(in_channels=3)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),

            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


# =============================================================================
# MODÈLE 2 : RÉGRESSION IMAGE -> COMMANDE MOTEUR
# =============================================================================

class DrivingRegressorCNN(nn.Module):
    """
    Modèle de régression.

    Objectif :
        image caméra -> commande moteur continue

    Sortie :
        speedA, speedB

    Exemple :
        image -> (50.0, 35.0)

    Ce modèle ne prédit pas une classe comme "left" ou "right".
    Il prédit directement les vitesses moteur.

    max_abs_speed :
        limite la sortie dans l'intervalle [-max_abs_speed, +max_abs_speed]
        grâce à tanh.

    Exemple :
        max_abs_speed = 100.0
        sortie dans [-100, 100]
    """

    def __init__(self, output_dim: int = 2, max_abs_speed: float = 100.0):
        super().__init__()

        self.encoder = DrivingFeatureExtractor(in_channels=3)
        self.max_abs_speed = float(max_abs_speed)

        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        raw_output = self.regressor(features)

        # Contraint les commandes moteur dans une plage réaliste.
        motor_commands = torch.tanh(raw_output) * self.max_abs_speed

        return motor_commands


# =============================================================================
# MODÈLE 3 : MULTI-TÂCHE IMAGE -> ACTION + COMMANDE MOTEUR
# =============================================================================

class DrivingMultiTaskCNN(nn.Module):
    """
    Modèle hybride classification + régression.

    Objectif :
        image caméra -> action discrète
                     -> commandes moteur continues

    Sorties :
        {
            "logits": scores de classification,
            "motor_commands": [speedA, speedB]
        }

    Intérêt :
        - la classification donne une décision lisible : forward, sharp_left...
        - la régression donne une commande moteur continue
        - le modèle apprend une représentation visuelle partagée
    """

    def __init__(
        self,
        num_classes: int,
        output_dim: int = 2,
        max_abs_speed: float = 100.0,
    ):
        super().__init__()

        self.encoder = DrivingFeatureExtractor(in_channels=3)
        self.max_abs_speed = float(max_abs_speed)

        self.classifier_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(128, num_classes),
        )

        self.regressor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)

        logits = self.classifier_head(features)

        raw_motor_commands = self.regressor_head(features)
        motor_commands = torch.tanh(raw_motor_commands) * self.max_abs_speed

        return {
            "logits": logits,
            "motor_commands": motor_commands,
        }


# =============================================================================
# COMPATIBILITÉ AVEC TON CODE EXISTANT
# =============================================================================

class DrivingCNN(DrivingMultiTaskCNN):
    """
    Alias pour garder la compatibilité avec ton code actuel.

    Si ton train_action_model.py fait :
        model = DrivingCNN(num_classes=len(CLASSES))

    alors il utilisera automatiquement le modèle de classification.
    """

    pass