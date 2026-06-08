from pathlib import Path
from collections import Counter
import random

import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

# ============================================================
# AUGMENTATION
# ============================================================

class AddGaussianNoise:

    """
    Ajoute un bruit gaussien léger à l'image tensorisée.
    Utile pour simuler du bruit caméra.
    """

    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        tensor = tensor + noise
        return torch.clamp(tensor, 0.0, 1.0)
    

    
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05
    ),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ],
    p=0.2),

    transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# REBALANCING
# ============================================================

def weighted_sampler(df_train: pd.DataFrame) ->WeightedRandomSampler:
    """
    Crée un sampler qui sur-échantillonne les pseudo-classes rares.
    """
    classes = df_train["pseudo_class"].tolist()

    class_counts = Counter(classes)

    print ("\n Distribution train avant rebalancing :")
    for class_id, count in sorted(class_counts.items()):
        print(f"{CLASS_NAMES[class_id]:> 8s}: {count}")

        class_weights = {
            class_id: 1.0 / count
            for class_id, count in class_counts.items() 
        }

        sample_weights = [
            class_weights[class_id]
            for class_id in classes
        ]

        sampler = WeightedRandomSampler(
            weights = torch.DoubleTensor(sample_weights),
            num_samples= len (sample_weights),
            replacement= True
        )

        return sampler