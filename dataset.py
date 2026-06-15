from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from actions import CLASS_TO_IDX, normalize_model_action, row_to_model_action


class AutonomousCarDataset(Dataset):
    """Dataset that loads images and resolves labels from prepared CSV files."""

    def __init__(
        self,
        csv_file: str | Path,
        image_dir: str | Path,
        transform=None,
        label_column: str = "direction_class",
    ) -> None:
        self.csv_file = Path(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_column = label_column

        self.data = pd.read_csv(self.csv_file, sep=";", encoding="utf-8-sig")
        self.data.columns = self.data.columns.str.strip()

        image_column = self._detect_image_column()
        labels = self._resolve_labels()

        self.samples: list[dict[str, Any]] = []
        # Filter out rows we cannot train on so the rest of the pipeline can
        # assume that every sample has both a valid image and a valid label.
        for row, label in zip(self.data.to_dict("records"), labels):
            if label is None:
                continue

            image_path = self._resolve_image_path(row[image_column])
            if not image_path.exists():
                continue

            self.samples.append(
                {
                    "image_path": image_path,
                    "label_name": label,
                    "label": CLASS_TO_IDX[label],
                }
            )

        if not self.samples:
            raise ValueError(
                f"No usable samples found in {self.csv_file}. "
                "Check image paths and direction labels."
            )

    def _detect_image_column(self) -> str:
        """Locate the CSV column that points to image files."""
        for candidate in ("image_filename", "image_path", "filename"):
            if candidate in self.data.columns:
                return candidate
        raise KeyError(
            "Missing image reference column. Expected one of "
            "'image_filename', 'image_path' or 'filename'."
        )

    def _resolve_labels(self) -> list[str | None]:
        """Prefer explicit labels, otherwise infer them from motor telemetry."""
        if self.label_column in self.data.columns:
            return [normalize_model_action(value) for value in self.data[self.label_column].tolist()]

        required_columns = {"speedA", "speedB", "GPIO1", "GPIO2", "GPIO3", "GPIO4"}
        if not required_columns.issubset(self.data.columns):
            raise KeyError(
                f"Could not infer labels from {self.csv_file}. "
                f"Missing columns: {sorted(required_columns - set(self.data.columns))}"
            )

        return [row_to_model_action(row) for row in self.data.to_dict("records")]

    def _resolve_image_path(self, raw_path: str | Path) -> Path:
        """Resolve image references from either absolute or dataset-relative paths."""
        relative_path = Path(str(raw_path).replace("\\", "/"))
        if relative_path.is_absolute():
            return relative_path

        direct_path = self.image_dir / relative_path
        if direct_path.exists():
            return direct_path

        # Some CSV files point to bare filenames while images actually live in
        # an ``Images`` subfolder.
        nested_path = self.image_dir / "Images" / relative_path.name
        if nested_path.exists():
            return nested_path

        return direct_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(sample["label"], dtype=torch.long)
        return image, label
