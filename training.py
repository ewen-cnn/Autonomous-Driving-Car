from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from actions import CLASSES, IDX_TO_CLASS
from config import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_PATH
from dataset import AutonomousCardataset
from model import DrivingCNN
from transforms import build_eval_transform, build_train_transform


@dataclass
class TrainingConfig:
    csv_file: str
    image_dir: str
    model_path: str = DEFAULT_MODEL_PATH
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    validation_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 0


def compute_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, dict[str, float | None]]:
    model.eval()
    total = 0
    correct = 0
    class_correct = {class_name: 0 for class_name in CLASSES}
    class_total = {class_name: 0 for class_name in CLASSES}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for predicted, label in zip(predictions.tolist(), labels.tolist()):
                label_name = IDX_TO_CLASS[label]
                class_total[label_name] += 1
                if predicted == label:
                    class_correct[label_name] += 1

    class_accuracy = {
        class_name: (
            class_correct[class_name] / class_total[class_name]
            if class_total[class_name] > 0
            else None
        )
        for class_name in CLASSES
    }
    global_accuracy = correct / total if total > 0 else 0.0
    return global_accuracy, class_accuracy


def _build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    train_dataset = AutonomousCardataset(
        csv_file=config.csv_file,
        image_dir=config.image_dir,
        transform=build_train_transform(config.image_size),
    )
    validation_dataset = AutonomousCardataset(
        csv_file=config.csv_file,
        image_dir=config.image_dir,
        transform=build_eval_transform(config.image_size),
    )

    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    if len(indices) < 2:
        raise ValueError("Need at least 2 usable samples to build train/validation splits.")

    train_size = max(1, int(len(indices) * (1 - config.validation_ratio)))
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:]

    if not validation_indices:
        validation_indices = train_indices[-1:]
        train_indices = train_indices[:-1]

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, validation_loader


def train_model(config: TrainingConfig) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = _build_dataloaders(config)

    model = DrivingCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_validation_accuracy = 0.0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        validation_accuracy, class_accuracy = compute_accuracy(model, validation_loader, device)

        print(f"\nEpoch [{epoch + 1}/{config.epochs}]")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy:.4f}")
        for class_name, accuracy in class_accuracy.items():
            if accuracy is None:
                print(f"  {class_name:8s}: no samples")
            else:
                print(f"  {class_name:8s}: {accuracy:.4f}")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "image_size": config.image_size,
                },
                config.model_path,
            )
            print(f"Saved best checkpoint to {config.model_path}")

    return {"best_validation_accuracy": best_validation_accuracy}
