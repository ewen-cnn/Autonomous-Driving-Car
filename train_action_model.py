from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms

from actions import CLASSES, IDX_TO_CLASS
from config import DEFAULT_IMAGE_SIZE, MULTITASK_MODEL_PATH
from model import DrivingCNN


# =============================================================================
# CONFIGURATION
# =============================================================================

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
SEED = 42

# Poids de la loss régression dans la loss totale.
# Plus il est grand, plus le modèle privilégie speedA/speedB.
REGRESSION_WEIGHT = 0.3

# Utilisé pour normaliser speedA/speedB dans la loss.
MAX_ABS_SPEED = 100.0

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

CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CLASSES)}


# =============================================================================
# DATASET MULTI-TÂCHE
# =============================================================================

class MultiTaskAutonomousCarDataset(Dataset):
    """
    Dataset pour entraînement multi-tâche.

    Retourne :
        image         : Tensor [3, H, W]
        action_label  : Tensor long, classe left/forward/right
        motor_target  : Tensor float [speedA, speedB]
    """

    POSSIBLE_IMAGE_COLUMNS = [
        "image_filename",
        "image_path",
        "image",
        "filename",
        "file_name",
        "image_name",
        "path",
    ]

    POSSIBLE_LABEL_COLUMNS = [
        "direction_class",
        "label",
        "class",
        "action",
    ]

    def __init__(
        self,
        csv_file: str | Path,
        image_dir: str | Path,
        transform=None,
    ):
        self.csv_file = Path(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

        sep = self.detect_separator(self.csv_file)
        self.data = pd.read_csv(self.csv_file, sep=sep, encoding="utf-8-sig")

        self.image_col = self.find_column(
            self.data,
            self.POSSIBLE_IMAGE_COLUMNS,
            "image",
        )

        self.label_col = self.find_column(
            self.data,
            self.POSSIBLE_LABEL_COLUMNS,
            "label",
        )

        self.check_required_columns()

    @staticmethod
    def detect_separator(csv_path: Path) -> str:
        first_line = csv_path.read_text(encoding="utf-8-sig").splitlines()[0]
        return ";" if ";" in first_line else ","

    @staticmethod
    def find_column(
        df: pd.DataFrame,
        possible_columns: list[str],
        description: str,
    ) -> str:
        for col in possible_columns:
            if col in df.columns:
                return col

        raise ValueError(
            f"Impossible de détecter la colonne {description}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    def check_required_columns(self) -> None:
        required_columns = ["speedA", "speedB"]

        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(
                    f"Colonne obligatoire absente : {col}. "
                    f"Colonnes disponibles : {list(self.data.columns)}"
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        image_name = Path(str(row[self.image_col])).name
        image_path = self.image_dir / image_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_name = str(row[self.label_col]).strip()

        if label_name not in CLASS_TO_IDX:
            raise ValueError(
                f"Classe inconnue : {label_name}. "
                f"Classes attendues : {CLASSES}"
            )

        action_label = torch.tensor(
            CLASS_TO_IDX[label_name],
            dtype=torch.long,
        )

        motor_target = torch.tensor(
            [
                float(row["speedA"]),
                float(row["speedB"]),
            ],
            dtype=torch.float32,
        )

        return image, action_label, motor_target


# =============================================================================
# LOSS MULTI-TÂCHE
# =============================================================================

class MultiTaskDrivingLoss(nn.Module):
    """
    Loss multi-tâche :
    - classification : CrossEntropyLoss
    - régression moteur : SmoothL1Loss

    La régression est normalisée par MAX_ABS_SPEED pour éviter que
    les valeurs moteur dominent trop fortement la loss totale.
    """

    def __init__(
        self,
        regression_weight: float = 0.3,
        max_abs_speed: float = 100.0,
    ):
        super().__init__()

        self.regression_weight = regression_weight
        self.max_abs_speed = max_abs_speed

        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.SmoothL1Loss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        action_labels: torch.Tensor,
        motor_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["logits"]
        motor_predictions = outputs["motor_commands"]

        loss_cls = self.classification_loss(
            logits,
            action_labels.long(),
        )

        loss_reg = self.regression_loss(
            motor_predictions / self.max_abs_speed,
            motor_targets / self.max_abs_speed,
        )

        loss_total = loss_cls + self.regression_weight * loss_reg

        metrics = {
            "loss_total": float(loss_total.item()),
            "loss_classification": float(loss_cls.item()),
            "loss_regression": float(loss_reg.item()),
        }

        return loss_total, metrics


# =============================================================================
# OUTILS
# =============================================================================

def check_path_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} introuvable : {path}")


def check_dataset_paths() -> None:
    check_path_exists(TRAIN_CSV, "CSV d'entraînement")
    check_path_exists(VALID_CSV, "CSV de validation")
    check_path_exists(TEST_CSV, "CSV de test")

    check_path_exists(TRAIN_IMAGE_DIR, "Dossier images d'entraînement")
    check_path_exists(VALID_IMAGE_DIR, "Dossier images de validation")
    check_path_exists(TEST_IMAGE_DIR, "Dossier images de test")


def create_model(device: torch.device) -> nn.Module:
    model = DrivingCNN(
        num_classes=len(CLASSES),
        output_dim=2,
        max_abs_speed=MAX_ABS_SPEED,
    )

    return model.to(device)


def class_accuracy_thresholds_are_met(
    class_accuracy: dict[str, float | None],
    thresholds: dict[str, float],
) -> bool:
    for class_name, min_accuracy in thresholds.items():
        accuracy = class_accuracy.get(class_name)

        if accuracy is None:
            return False

        if accuracy < min_accuracy:
            return False

    return True


def print_class_accuracy(class_accuracy: dict[str, float | None]) -> None:
    print("Accuracy par classe :")

    for class_name, accuracy in class_accuracy.items():
        if accuracy is None:
            print(f"  {class_name:15s} : aucune donnée")
        else:
            print(f"  {class_name:15s} : {accuracy:.4f}")


def print_multitask_metrics(prefix: str, metrics: dict[str, Any]) -> None:
    print(f"{prefix} loss totale        : {metrics['loss_total']:.4f}")
    print(f"{prefix} loss classification: {metrics['loss_classification']:.4f}")
    print(f"{prefix} loss régression    : {metrics['loss_regression']:.4f}")
    print(f"{prefix} accuracy           : {metrics['classification_accuracy']:.4f}")

    if "mae_speedA" in metrics:
        print(f"{prefix} MAE speedA         : {metrics['mae_speedA']:.2f}")
        print(f"{prefix} MAE speedB         : {metrics['mae_speedB']:.2f}")
        print(f"{prefix} RMSE speedA        : {metrics['rmse_speedA']:.2f}")
        print(f"{prefix} RMSE speedB        : {metrics['rmse_speedB']:.2f}")


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    validation_metrics: dict[str, Any],
    path: str | Path,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "image_size": DEFAULT_IMAGE_SIZE,
        "validation_accuracy": validation_metrics["classification_accuracy"],
        "validation_loss_total": validation_metrics["loss_total"],
        "validation_loss_classification": validation_metrics["loss_classification"],
        "validation_loss_regression": validation_metrics["loss_regression"],
        "validation_mae_speedA": validation_metrics["mae_speedA"],
        "validation_mae_speedB": validation_metrics["mae_speedB"],
        "max_abs_speed": MAX_ABS_SPEED,
        "regression_weight": REGRESSION_WEIGHT,
    }

    torch.save(checkpoint, path)


def load_best_model(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


# =============================================================================
# ENTRAÎNEMENT MULTI-TÂCHE
# =============================================================================

def train_one_epoch_multitask(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskDrivingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0

    total = 0
    correct = 0

    for images, action_labels, motor_targets in dataloader:
        images = images.to(device)
        action_labels = action_labels.to(device).long()
        motor_targets = motor_targets.to(device).float()

        optimizer.zero_grad()

        outputs = model(images)

        loss, loss_metrics = criterion(
            outputs=outputs,
            action_labels=action_labels,
            motor_targets=motor_targets,
        )

        loss.backward()
        optimizer.step()

        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=1)

        total += action_labels.size(0)
        correct += (predictions == action_labels).sum().item()

        running_total_loss += loss_metrics["loss_total"]
        running_cls_loss += loss_metrics["loss_classification"]
        running_reg_loss += loss_metrics["loss_regression"]

    n_batches = len(dataloader)

    return {
        "loss_total": running_total_loss / n_batches,
        "loss_classification": running_cls_loss / n_batches,
        "loss_regression": running_reg_loss / n_batches,
        "classification_accuracy": correct / total if total > 0 else 0.0,
    }


# =============================================================================
# ÉVALUATION MULTI-TÂCHE
# =============================================================================

def evaluate_multitask(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskDrivingLoss,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0

    total = 0
    correct = 0

    class_total = {class_name: 0 for class_name in CLASSES}
    class_correct = {class_name: 0 for class_name in CLASSES}

    all_labels = []
    all_predictions = []

    all_motor_targets = []
    all_motor_predictions = []

    with torch.no_grad():
        for images, action_labels, motor_targets in dataloader:
            images = images.to(device)
            action_labels = action_labels.to(device).long()
            motor_targets = motor_targets.to(device).float()

            outputs = model(images)

            loss, loss_metrics = criterion(
                outputs=outputs,
                action_labels=action_labels,
                motor_targets=motor_targets,
            )

            logits = outputs["logits"]
            motor_predictions = outputs["motor_commands"]

            predictions = torch.argmax(logits, dim=1)

            total += action_labels.size(0)
            correct += (predictions == action_labels).sum().item()

            running_total_loss += loss_metrics["loss_total"]
            running_cls_loss += loss_metrics["loss_classification"]
            running_reg_loss += loss_metrics["loss_regression"]

            all_labels.extend(action_labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            all_motor_targets.append(motor_targets.cpu())
            all_motor_predictions.append(motor_predictions.cpu())

            for prediction, label in zip(predictions, action_labels):
                label_idx = int(label.item())
                pred_idx = int(prediction.item())

                label_name = IDX_TO_CLASS[label_idx]

                class_total[label_name] += 1

                if pred_idx == label_idx:
                    class_correct[label_name] += 1

    n_batches = len(dataloader)

    motor_targets_np = torch.cat(all_motor_targets, dim=0).numpy()
    motor_predictions_np = torch.cat(all_motor_predictions, dim=0).numpy()

    motor_errors = motor_predictions_np - motor_targets_np

    mae_per_motor = np.mean(np.abs(motor_errors), axis=0)
    rmse_per_motor = np.sqrt(np.mean(motor_errors ** 2, axis=0))

    class_accuracy: dict[str, float | None] = {}

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
        "loss_total": running_total_loss / n_batches,
        "loss_classification": running_cls_loss / n_batches,
        "loss_regression": running_reg_loss / n_batches,
        "classification_accuracy": correct / total if total > 0 else 0.0,
        "class_accuracy": class_accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "mae_speedA": float(mae_per_motor[0]),
        "mae_speedB": float(mae_per_motor[1]),
        "rmse_speedA": float(rmse_per_motor[0]),
        "rmse_speedB": float(rmse_per_motor[1]),
        "mae_global": float(np.mean(np.abs(motor_errors))),
        "rmse_global": float(np.sqrt(np.mean(motor_errors ** 2))),
    }


def print_final_evaluation(metrics: dict[str, Any]) -> None:
    print(f"Test accuracy : {metrics['classification_accuracy']:.4f}")

    print_class_accuracy(metrics["class_accuracy"])

    print("\nMatrice de confusion :")
    print(metrics["confusion_matrix"])

    print("\nRapport de classification :")
    print(metrics["classification_report"])

    print("\nÉvaluation des commandes moteur :")
    print(f"  MAE speedA       : {metrics['mae_speedA']:.2f}")
    print(f"  MAE speedB       : {metrics['mae_speedB']:.2f}")
    print(f"  RMSE speedA      : {metrics['rmse_speedA']:.2f}")
    print(f"  RMSE speedB      : {metrics['rmse_speedB']:.2f}")
    print(f"  MAE globale      : {metrics['mae_global']:.2f}")
    print(f"  RMSE globale     : {metrics['rmse_global']:.2f}")


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    model_path = Path(MULTITASK_MODEL_PATH)

    check_dataset_paths()

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = MultiTaskAutonomousCarDataset(
        csv_file=TRAIN_CSV,
        image_dir=TRAIN_IMAGE_DIR,
        transform=tensor_transform,
    )

    valid_dataset = MultiTaskAutonomousCarDataset(
        csv_file=VALID_CSV,
        image_dir=VALID_IMAGE_DIR,
        transform=tensor_transform,
    )

    test_dataset = MultiTaskAutonomousCarDataset(
        csv_file=TEST_CSV,
        image_dir=TEST_IMAGE_DIR,
        transform=tensor_transform,
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

    criterion = MultiTaskDrivingLoss(
        regression_weight=REGRESSION_WEIGHT,
        max_abs_speed=MAX_ABS_SPEED,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_accuracy = -1.0
    model_was_saved = False

    for epoch in range(EPOCHS):
        train_metrics = train_one_epoch_multitask(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_metrics = evaluate_multitask(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
        )

        validation_accuracy = validation_metrics["classification_accuracy"]
        validation_class_accuracy = validation_metrics["class_accuracy"]

        print("\n" + "=" * 70)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")

        print_multitask_metrics("Train", train_metrics)
        print_multitask_metrics("Valid", validation_metrics)

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
                    validation_metrics=validation_metrics,
                    path=model_path,
                )

                print(
                    f"Modèle sauvegardé : {model_path} "
                    f"| val_acc={validation_accuracy:.4f} "
                    f"| left={validation_class_accuracy['left']:.4f} "
                    f"| forward={validation_class_accuracy['forward']:.4f} "
                    f"| right={validation_class_accuracy['right']:.4f} "
                    f"| MAE_A={validation_metrics['mae_speedA']:.2f} "
                    f"| MAE_B={validation_metrics['mae_speedB']:.2f}"
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
        path=model_path,
        device=device,
    )

    test_metrics = evaluate_multitask(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print_final_evaluation(test_metrics)


if __name__ == "__main__":
    main()