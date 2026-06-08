from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as transforms


class AutonomousCarDataset(Dataset):
    def __init__(self, csv_file, record_dir, transform=None):
        self.csv_file = Path(csv_file)
        self.record_dir = Path(record_dir)
        self.transform = transform

        self.data = pd.read_csv(
            self.csv_file,
            sep=";",
            encoding="utf-8-sig"
        )

        self.data.columns = self.data.columns.str.strip()

        print("Colonnes détectées :", self.data.columns.tolist())
        print(self.data.head())

        required_columns = ["image_path", "speedA", "speedB"]

        for column in required_columns:
            if column not in self.data.columns:
                raise KeyError(
                    f"Colonne manquante : {column}. "
                    f"Colonnes disponibles : {self.data.columns.tolist()}"
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.data.iloc[idx]

        relative_image_path = str(row["image_path"]).replace("\\", "/")
        image_path = self.record_dir / relative_image_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(
            [
                float(row["speedA"]),
                float(row["speedB"])
            ],
            dtype=torch.float32
        )

        return image, target


# Transformations validation/test : pas d'augmentation
val_transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor()
])


# Transformations entraînement : augmentation réaliste
train_transform = transforms.Compose([
    transforms.Resize((120, 160)),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2
    ),

    transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05)
    ),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.2),

    transforms.ToTensor()
])


RECORD_DIR = "dataSet/#Record_2025-07-02_09-46-48"
CSV_FILE = f"{RECORD_DIR}/aligned_labels.csv"


# Deux datasets identiques en données, mais avec des transformations différentes
full_train_dataset = AutonomousCarDataset(
    csv_file=CSV_FILE,
    record_dir=RECORD_DIR,
    transform=train_transform
)

full_val_dataset = AutonomousCarDataset(
    csv_file=CSV_FILE,
    record_dir=RECORD_DIR,
    transform=val_transform
)


# Split train / validation
dataset_size = len(full_train_dataset)
indices = torch.randperm(dataset_size).tolist()

train_ratio = 0.8
train_size = int(train_ratio * dataset_size)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)


train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=10,
    shuffle=False
)

train_behaviors = full_train_dataset.data.iloc[train_indices]


# Test train loader
for images, targets in train_loader:
    print("TRAIN")
    print("Images :", images.shape)
    print("Targets :", targets.shape)
    print("Premières commandes :", targets[:10])
    break


# Test val loader
for images, targets in val_loader:
    print("VALIDATION")
    print("Images :", images.shape)
    print("Targets :", targets.shape)
    print("Premières commandes :", targets[:10])
    break

for images, targets in train_loader:
    print("Image dtype:", images.dtype)
    print("Image min:", images.min().item())
    print("Image max:", images.max().item())
    print("Targets:", targets[:5])
    break