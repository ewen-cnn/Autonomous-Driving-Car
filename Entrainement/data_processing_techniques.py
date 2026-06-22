from collections import Counter

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from Traitement.transforms import AddGaussianNoise, build_eval_transform, build_train_transform

train_transform = build_train_transform((224, 224))
test_transform = build_eval_transform((224, 224))


def weighted_sampler(df_train: pd.DataFrame, label_column: str = "pseudo_class") -> WeightedRandomSampler:
    if label_column not in df_train.columns:
        raise KeyError(f"Missing label column: {label_column}")

    classes = df_train[label_column].tolist()
    class_counts = Counter(classes)
    class_weights = {class_id: 1.0 / count for class_id, count in class_counts.items()}
    sample_weights = [class_weights[class_id] for class_id in classes]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


__all__ = [
    "AddGaussianNoise",
    "test_transform",
    "train_transform",
    "weighted_sampler",
]
