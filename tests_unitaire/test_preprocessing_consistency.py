"""Regression test: training and inference must feed the model identical tensors.

Before this test existed, training used ToTensor() only (native resolution, no
normalisation) while inference resized to DEFAULT_IMAGE_SIZE and applied
ImageNet normalisation, so a trained checkpoint saw different inputs once deployed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DEFAULT_IMAGE_SIZE  # noqa: E402
from Traitement.transforms import build_eval_transform  # noqa: E402
import train_action_model  # noqa: E402


def _write_sample(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_dir = tmp_path / "Images"
    image_dir.mkdir()
    image_path = image_dir / "frame_0001.png"

    generator = torch.Generator().manual_seed(0)
    pixels = (torch.rand(224, 224, 3, generator=generator) * 255).to(torch.uint8)
    Image.fromarray(pixels.numpy()).save(image_path)

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "image_filename;direction_class;speedA;speedB\nframe_0001.png;left;20;60\n",
        encoding="utf-8",
    )
    return csv_path, image_dir, image_path


def test_training_dataset_matches_inference_preprocessing(tmp_path: Path) -> None:
    csv_path, image_dir, image_path = _write_sample(tmp_path)

    dataset = train_action_model.MultiTaskAutonomousCarDataset(
        csv_file=csv_path,
        image_dir=image_dir,
        transform=build_eval_transform(DEFAULT_IMAGE_SIZE),
    )
    training_tensor = dataset[0][0]

    inference_tensor = build_eval_transform(DEFAULT_IMAGE_SIZE)(
        Image.open(image_path).convert("RGB")
    )

    assert training_tensor.shape == (3, *DEFAULT_IMAGE_SIZE)
    assert torch.equal(training_tensor, inference_tensor)
