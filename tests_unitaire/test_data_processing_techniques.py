from collections import Counter

import pytest
import torch
import pandas as pd
from PIL import Image

from torch.utils.data import WeightedRandomSampler

from Entrainement.data_processing_techniques import (
    AddGaussianNoise,
    train_transform,
    test_transform,
    weighted_sampler,
)


# ============================================================
# OUTILS DE TEST
# ============================================================

def create_dummy_image(width=320, height=240, color=(120, 120, 120)):
    """
    Crée une image RGB synthétique pour tester les transforms
    sans dépendre du dataset réel.
    """
    return Image.new("RGB", (width, height), color=color)


def create_dummy_dataframe():
    """
    Crée un DataFrame synthétique avec une distribution déséquilibrée.
    Classe 0 : 4 échantillons
    Classe 1 : 2 échantillons
    """
    return pd.DataFrame({
        "image_path": [
            "img_0.png",
            "img_1.png",
            "img_2.png",
            "img_3.png",
            "img_4.png",
            "img_5.png",
        ],
        "pseudo_class": [0, 0, 0, 0, 1, 1]
    })


# ============================================================
# TESTS AddGaussianNoise
# ============================================================

def test_add_gaussian_noise_preserves_shape():
    """
    Vérifie que le bruit gaussien ne change pas la taille du tenseur.
    """
    tensor = torch.ones((3, 224, 224)) * 0.5

    noise_transform = AddGaussianNoise(mean=0.0, std=0.02)
    output = noise_transform(tensor)

    assert output.shape == tensor.shape


def test_add_gaussian_noise_preserves_value_range():
    """
    Vérifie que les valeurs restent dans [0, 1].
    C'est important car les images tensorisées avant Normalize doivent rester valides.
    """
    tensor = torch.ones((3, 224, 224)) * 0.5

    noise_transform = AddGaussianNoise(mean=0.0, std=0.2)
    output = noise_transform(tensor)

    assert torch.min(output) >= 0.0
    assert torch.max(output) <= 1.0


def test_add_gaussian_noise_with_zero_std_returns_same_tensor():
    """
    Si std = 0 et mean = 0, l'image ne doit pas changer.
    """
    tensor = torch.ones((3, 224, 224)) * 0.5

    noise_transform = AddGaussianNoise(mean=0.0, std=0.0)
    output = noise_transform(tensor)

    assert torch.allclose(output, tensor)


def test_add_gaussian_noise_changes_tensor_when_std_positive():
    """
    Si std > 0, le tenseur doit être modifié.
    """
    torch.manual_seed(42)

    tensor = torch.ones((3, 224, 224)) * 0.5

    noise_transform = AddGaussianNoise(mean=0.0, std=0.05)
    output = noise_transform(tensor)

    assert not torch.allclose(output, tensor)


# ============================================================
# TESTS transforms
# ============================================================

def test_train_transform_output_shape():
    """
    Vérifie que train_transform sort bien un tenseur image compatible CNN :
    [channels, height, width] = [3, 224, 224].
    """
    image = create_dummy_image()

    output = train_transform(image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([3, 224, 224])


def test_train_transform_output_is_finite():
    """
    Vérifie que train_transform ne produit pas de NaN ou d'infini.
    """
    image = create_dummy_image()

    output = train_transform(image)

    assert torch.isfinite(output).all()


def test_test_transform_output_shape():
    """
    Vérifie que test_transform sort une image normalisée de bonne taille.
    """
    image = create_dummy_image()

    output = test_transform(image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([3, 224, 224])


def test_test_transform_is_deterministic():
    """
    test_transform ne doit pas être aléatoire.
    Deux appels sur la même image doivent donner exactement le même résultat.
    """
    image = create_dummy_image()

    output_1 = test_transform(image)
    output_2 = test_transform(image)

    assert torch.allclose(output_1, output_2)


# ============================================================
# TESTS weighted_sampler
# ============================================================

def test_weighted_sampler_returns_sampler():
    """
    Vérifie que la fonction retourne bien un WeightedRandomSampler.
    """
    df = create_dummy_dataframe()

    sampler = weighted_sampler(df)

    assert isinstance(sampler, WeightedRandomSampler)


def test_weighted_sampler_num_samples_matches_dataframe_length():
    """
    Vérifie que le sampler tire autant d'échantillons qu'il y a de lignes dans le train set.
    """
    df = create_dummy_dataframe()

    sampler = weighted_sampler(df)

    assert sampler.num_samples == len(df)


def test_weighted_sampler_uses_replacement():
    """
    Le rebalancing par sur-échantillonnage doit utiliser replacement=True.
    Sinon les classes rares ne peuvent pas être tirées plusieurs fois.
    """
    df = create_dummy_dataframe()

    sampler = weighted_sampler(df)

    assert sampler.replacement is True


def test_weighted_sampler_inverse_class_weights():
    """
    Vérifie que les poids sont inverses aux fréquences de classes.

    Pour pseudo_class = [0,0,0,0,1,1] :
    - classe 0 : 4 occurrences -> poids = 1/4 = 0.25
    - classe 1 : 2 occurrences -> poids = 1/2 = 0.50
    """
    df = create_dummy_dataframe()

    sampler = weighted_sampler(df)

    weights = sampler.weights.tolist()

    expected_weights = [
        1 / 4,
        1 / 4,
        1 / 4,
        1 / 4,
        1 / 2,
        1 / 2,
    ]

    assert weights == pytest.approx(expected_weights)


def test_weighted_sampler_sampling_reduces_imbalance():
    """
    Test statistique simple :
    On crée un dataset très déséquilibré :
    - classe 0 : 900 exemples
    - classe 1 : 100 exemples

    Après WeightedRandomSampler, les tirages doivent être beaucoup plus équilibrés.
    Ce test ne demande pas un équilibre parfait, car le tirage est aléatoire.
    """
    torch.manual_seed(42)

    df = pd.DataFrame({
        "image_path": [f"img_{i}.png" for i in range(1000)],
        "pseudo_class": [0] * 900 + [1] * 100
    })

    sampler = weighted_sampler(df)

    sampled_indices = list(iter(sampler))
    sampled_classes = df.iloc[sampled_indices]["pseudo_class"].tolist()

    counts = Counter(sampled_classes)

    class_0_count = counts[0]
    class_1_count = counts[1]

    ratio = class_1_count / class_0_count

    # Sans balancing, ratio environ 100/900 = 0.11
    # Avec balancing, on attend quelque chose proche de 1.
    # On tolère une plage large à cause de l'aléatoire.
    assert 0.6 <= ratio <= 1.6