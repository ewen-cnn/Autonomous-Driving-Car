from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from actions import CLASSES, IDX_TO_CLASS
from config import MULTITASK_MODEL_PATH
from dataset import AutonomousCardataset
from model import DrivingCNN
from torchvision import transforms


# =============================================================================
# CONFIGURATION
# =============================================================================

BATCH_SIZE = 32
NUM_WORKERS = 0

# Modifie ici si tu veux évaluer les labels nettoyés.
# Version originale :
LABEL_ROOT = Path("dataset_augmente_equilibre")

# Version nettoyée, si tu veux évaluer sur les CSV nettoyés :
# LABEL_ROOT = Path("dataset_augmente_equilibre_clean_labels")

IMAGE_ROOT = Path("dataset_augmente_equilibre")

TRAIN_CSV = LABEL_ROOT / "train" / "labels" / "labels.csv"
VALID_CSV = LABEL_ROOT / "valid" / "labels" / "labels.csv"
TEST_CSV = LABEL_ROOT / "test" / "labels" / "labels.csv"

TRAIN_IMAGE_DIR = IMAGE_ROOT / "train" / "Images"
VALID_IMAGE_DIR = IMAGE_ROOT / "valid" / "Images"
TEST_IMAGE_DIR = IMAGE_ROOT / "test" / "Images"

MODEL_PATH = Path(MULTITASK_MODEL_PATH)


# =============================================================================
# TRANSFORMATION MINIMALE
# =============================================================================
# Ici, on ne réapplique PAS d'augmentation.
# On fait seulement la conversion technique PIL.Image -> torch.Tensor.
# C'est obligatoire pour PyTorch.

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
])


# =============================================================================
# OUTILS
# =============================================================================

def check_path_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} introuvable : {path}")


def check_paths() -> None:
    check_path_exists(MODEL_PATH, "Fichier modèle .pth")

    check_path_exists(TRAIN_CSV, "CSV train")
    check_path_exists(VALID_CSV, "CSV valid")
    check_path_exists(TEST_CSV, "CSV test")

    check_path_exists(TRAIN_IMAGE_DIR, "Dossier images train")
    check_path_exists(VALID_IMAGE_DIR, "Dossier images valid")
    check_path_exists(TEST_IMAGE_DIR, "Dossier images test")


def get_logits(outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compatible avec :
    - modèle classification simple : outputs est un Tensor
    - modèle multi-tâche : outputs est un dict avec la clé 'logits'
    """
    if isinstance(outputs, dict):
        return outputs["logits"]

    return outputs


def create_model(device: torch.device) -> nn.Module:
    """
    Reconstruit le modèle avec le même nombre de classes.
    """
    model = DrivingCNN(num_classes=len(CLASSES))
    return model.to(device)


def load_saved_model(
    model: nn.Module,
    model_path: Path,
    device: torch.device,
) -> nn.Module:
    """
    Charge un checkpoint .pth sauvegardé par torch.save.
    Compatible avec :
    - checkpoint dict contenant 'model_state_dict'
    - fichier contenant directement le state_dict
    """
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("Checkpoint détecté avec model_state_dict.")

        if "epoch" in checkpoint:
            print(f"Epoch sauvegardée : {checkpoint['epoch']}")

        if "validation_accuracy" in checkpoint:
            print(f"Validation accuracy sauvegardée : {checkpoint['validation_accuracy']:.4f}")

        if "classes" in checkpoint:
            print(f"Classes sauvegardées : {checkpoint['classes']}")

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("State_dict direct détecté.")
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def create_dataloader(csv_path: Path, image_dir: Path) -> DataLoader:
    dataset = AutonomousCardataset(
        csv_file=str(csv_path),
        image_dir=str(image_dir),
        transform=tensor_transform,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


def evaluate_loss_and_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Calcule :
    - loss moyenne
    - accuracy globale
    - accuracy par classe
    - matrice de confusion
    - rapport de classification
    """
    model.eval()

    running_loss = 0.0
    total = 0
    correct = 0

    class_total = {class_name: 0 for class_name in CLASSES}
    class_correct = {class_name: 0 for class_name in CLASSES}

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)
            logits = get_logits(outputs)

            loss = criterion(logits, labels)
            running_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            for prediction, label in zip(predictions, labels):
                label_idx = int(label.item())
                pred_idx = int(prediction.item())

                label_name = IDX_TO_CLASS[label_idx]

                class_total[label_name] += 1

                if pred_idx == label_idx:
                    class_correct[label_name] += 1

    average_loss = running_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0

    class_accuracy = {}

    for class_name in CLASSES:
        if class_total[class_name] == 0:
            class_accuracy[class_name] = None
        else:
            class_accuracy[class_name] = (
                class_correct[class_name] / class_total[class_name]
            )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASSES))),
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASSES,
        digits=4,
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "class_accuracy": class_accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def print_results(split_name: str, results: dict) -> None:
    print("\n" + "=" * 80)
    print(f"Résultats sur {split_name}")
    print("=" * 80)

    print(f"Loss moyenne : {results['loss']:.4f}")
    print(f"Accuracy     : {results['accuracy']:.4f}")

    print("\nAccuracy par classe :")
    for class_name, accuracy in results["class_accuracy"].items():
        if accuracy is None:
            print(f"  {class_name:15s} : aucune donnée")
        else:
            print(f"  {class_name:15s} : {accuracy:.4f}")

    print("\nMatrice de confusion :")
    print(results["confusion_matrix"])

    print("\nRapport de classification :")
    print(results["classification_report"])


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    check_paths()

    criterion = nn.CrossEntropyLoss()

    model = create_model(device)
    model = load_saved_model(
        model=model,
        model_path=MODEL_PATH,
        device=device,
    )

    train_loader = create_dataloader(TRAIN_CSV, TRAIN_IMAGE_DIR)
    valid_loader = create_dataloader(VALID_CSV, VALID_IMAGE_DIR)
    test_loader = create_dataloader(TEST_CSV, TEST_IMAGE_DIR)

    train_results = evaluate_loss_and_accuracy(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
    )

    valid_results = evaluate_loss_and_accuracy(
        model=model,
        dataloader=valid_loader,
        criterion=criterion,
        device=device,
    )

    test_results = evaluate_loss_and_accuracy(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print_results("TRAIN", train_results)
    print_results("VALID", valid_results)
    print_results("TEST", test_results)


if __name__ == "__main__":
    main()