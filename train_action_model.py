from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from actions import CLASSES, IDX_TO_CLASS
from config import DEFAULT_IMAGE_SIZE, MULTITASK_MODEL_PATH, BALANCED_dataset_DIR
from dataset import AutonomousCardataset
from model import DrivingCNN
from transforms import build_eval_transform, build_train_transform


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_ROOT = Path("dataset_augmente_equilibre")



BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
SEED = 42
MIN_CLASS_ACCURACY = {
    "left": 0.90,
    "forward": 0.60,
    "right": 0.92,
}
LABEL_ROOT = Path("dataset_augmente_equilibre")
IMAGE_ROOT = Path("dataset_augmente_equilibre")

TRAIN_CSV = LABEL_ROOT / "train" / "labels" / "labels.csv"
VALID_CSV = LABEL_ROOT / "valid" / "labels" / "labels.csv"
TEST_CSV = LABEL_ROOT / "test" / "labels" / "labels.csv"

TRAIN_IMAGE_DIR = IMAGE_ROOT / "train" / "Images"
VALID_IMAGE_DIR = IMAGE_ROOT / "valid" / "Images"
TEST_IMAGE_DIR = IMAGE_ROOT / "test" / "Images"

# =============================================================================
# OUTILS
# =============================================================================


def check_path_exists(path: Path, description: str) -> None:
    """
    Vérifie qu'un chemin existe.
    Arrête proprement le programme si un fichier/dossier est introuvable.
    """
    if not path.exists():
        raise FileNotFoundError(f"{description} introuvable : {path}")


def check_dataset_paths() -> None:
    """
    Vérifie que les CSV et les dossiers d'images existent.
    """
    check_path_exists(TRAIN_CSV, "CSV d'entraînement")
    check_path_exists(VALID_CSV, "CSV de validation")
    check_path_exists(TEST_CSV, "CSV de test")

    check_path_exists(TRAIN_IMAGE_DIR, "Dossier images d'entraînement")
    check_path_exists(VALID_IMAGE_DIR, "Dossier images de validation")
    check_path_exists(TEST_IMAGE_DIR, "Dossier images de test")


def create_train_transform():
    """
    Crée la transformation d'entraînement.

    Cette fonction est volontairement tolérante :
    elle fonctionne que build_train_transform attende :
    - image_size=...
    - DEFAULT_IMAGE_SIZE en argument positionnel
    - aucun argument
    """
    try:
        return build_train_transform(image_size=DEFAULT_IMAGE_SIZE)
    except TypeError:
        try:
            return build_train_transform(DEFAULT_IMAGE_SIZE)
        except TypeError:
            return build_train_transform()


def create_eval_transform():
    """
    Crée la transformation de validation/test.
    """
    try:
        return build_eval_transform(image_size=DEFAULT_IMAGE_SIZE)
    except TypeError:
        try:
            return build_eval_transform(DEFAULT_IMAGE_SIZE)
        except TypeError:
            return build_eval_transform()


def create_model(device: torch.device) -> nn.Module:
    """
    Crée le modèle CNN.

    Cette fonction gère plusieurs signatures possibles de DrivingCNN :
    - DrivingCNN(num_classes=...)
    - DrivingCNN(...)
    - DrivingCNN()
    """
    try:
        model = DrivingCNN(num_classes=len(CLASSES))
    except TypeError:
        try:
            model = DrivingCNN(len(CLASSES))
        except TypeError:
            model = DrivingCNN()

    return model.to(device)


def compute_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float | None]]:
    """
    Calcule :
    - l'accuracy globale
    - l'accuracy par classe
    """
    model.eval()

    total = 0
    correct = 0

    class_total = {class_name: 0 for class_name in CLASSES}
    class_correct = {class_name: 0 for class_name in CLASSES}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            #outputs = model(images)
            #predictions = torch.argmax(outputs, dim=1)

            outputs = model(images)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            predictions = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            for prediction, label in zip(predictions, labels):
                label_idx = label.item()
                pred_idx = prediction.item()

                label_name = IDX_TO_CLASS[label_idx]

                class_total[label_name] += 1

                if pred_idx == label_idx:
                    class_correct[label_name] += 1

    global_accuracy = correct / total if total > 0 else 0.0

    class_accuracy: dict[str, float | None] = {}

    for class_name in CLASSES:
        if class_total[class_name] == 0:
            class_accuracy[class_name] = None
        else:
            class_accuracy[class_name] = (
                class_correct[class_name] / class_total[class_name]
            )

    return global_accuracy, class_accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Entraîne le modèle pendant une époque.
    """
    model.train()

    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        #outputs = model(images)
        #loss = criterion(outputs, labels)

        outputs = model(images)

        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def print_class_accuracy(class_accuracy: dict[str, float | None]) -> None:
    """
    Affiche l'accuracy par classe.
    """
    print("Accuracy par classe :")

    for class_name, accuracy in class_accuracy.items():
        if accuracy is None:
            print(f"  {class_name:15s} : aucune donnée")
        else:
            print(f"  {class_name:15s} : {accuracy:.4f}")

def class_accuracy_thresholds_are_met(
    class_accuracy: dict[str, float | None],
    thresholds: dict[str, float],
) -> bool:
    """
    Vérifie que les accuracy minimales par classe sont atteintes.

    Exemple :
        left    >= 0.92
        forward >= 0.60
        right   >= 0.92
    """
    for class_name, min_accuracy in thresholds.items():
        accuracy = class_accuracy.get(class_name)

        if accuracy is None:
            return False

        if accuracy < min_accuracy:
            return False

    return True

def save_checkpoint(
    model: nn.Module,
    epoch: int,
    validation_accuracy: float,
    path: str | Path,
) -> None:
    """
    Sauvegarde le meilleur modèle.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "image_size": DEFAULT_IMAGE_SIZE,
        "validation_accuracy": validation_accuracy,
    }

    torch.save(checkpoint, path)


def load_best_model(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
) -> nn.Module:
    """
    Recharge le meilleur modèle sauvegardé.
    """
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model

# =============================================================================
# Matrice de confusion pour visualisation erreurs
# =============================================================================

def evaluate_with_confusion_matrix(model, dataloader, device):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).long()

            #outputs = model(images)
            #predictions = torch.argmax(outputs, dim=1)

            outputs = model(images)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASSES)))
    )

    print("\nMatrice de confusion :")
    print(cm)

    print("\nRapport de classification :")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=CLASSES,
            digits=4
        )
    )

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    model_path = Path(MULTITASK_MODEL_PATH)

    if model_path.exists():
        model_path.unlink()
        print(f"Ancien modèle supprimé : {model_path}")

    check_dataset_paths()

    train_transform = create_train_transform()
    eval_transform = create_eval_transform()

    train_dataset = AutonomousCardataset(
        csv_file=str(TRAIN_CSV),
        image_dir=str(TRAIN_IMAGE_DIR),
        transform=train_transform,
    )

    valid_dataset = AutonomousCardataset(
        csv_file=str(VALID_CSV),
        image_dir=str(VALID_IMAGE_DIR),
        transform=eval_transform,
    )

    test_dataset = AutonomousCardataset(
        csv_file=str(TEST_CSV),
        image_dir=str(TEST_IMAGE_DIR),
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Images entraînement : {len(train_dataset)}")
    print(f"Images validation   : {len(valid_dataset)}")
    print(f"Images test         : {len(test_dataset)}")
    print(f"Classes             : {CLASSES}")

    model = create_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_accuracy = -1.0
    model_was_saved = False

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_accuracy, validation_class_accuracy = compute_accuracy(
            model=model,
            dataloader=valid_loader,
            device=device,
        )

        print("\n" + "=" * 70)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train loss          : {train_loss:.4f}")
        print(f"Validation accuracy : {validation_accuracy:.4f}")

        print_class_accuracy(validation_class_accuracy)

        thresholds_ok = class_accuracy_thresholds_are_met(
            class_accuracy=validation_class_accuracy,
            thresholds=MIN_CLASS_ACCURACY,
        )

        if thresholds_ok:
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                model_was_saved = True

                save_checkpoint(
                    model=model,
                    epoch=epoch + 1,
                    validation_accuracy=validation_accuracy,
                    path=MULTITASK_MODEL_PATH,
                )

                print(
                    f"Modèle sauvegardé : {MULTITASK_MODEL_PATH} "
                    f"| val_acc={validation_accuracy:.4f} "
                    f"| left={validation_class_accuracy['left']:.4f} "
                    f"| forward={validation_class_accuracy['forward']:.4f} "
                    f"| right={validation_class_accuracy['right']:.4f}"
                )
            else:
                print(
                    "Seuils respectés, mais modèle non sauvegardé : "
                    f"val_acc={validation_accuracy:.4f} "
                    f"<= best_val_acc={best_validation_accuracy:.4f}"
                )
        else:
            print(
                "Modèle non sauvegardé : seuils non respectés "
                f"| left={validation_class_accuracy.get('left')} "
                f"| forward={validation_class_accuracy.get('forward')} "
                f"| right={validation_class_accuracy.get('right')}"
            )

    print("\n" + "=" * 70)

    if not model_was_saved:
        print(
            "Aucun modèle sauvegardé : "
            f"aucune epoch n'a respecté les seuils {MIN_CLASS_ACCURACY}."
        )
        return

    print(f"Meilleure accuracy validation sauvegardée : {best_validation_accuracy:.4f}")

    print("\nÉvaluation finale sur le jeu de test...")

    model = load_best_model(
        model=model,
        path=MULTITASK_MODEL_PATH,
        device=device,
    )

    test_accuracy, test_class_accuracy = compute_accuracy(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    print(f"Test accuracy : {test_accuracy:.4f}")
    print_class_accuracy(test_class_accuracy)
    
    evaluate_with_confusion_matrix(
    model=model,
    dataloader=test_loader,
    device=device
    )

if __name__ == "__main__":
    main()