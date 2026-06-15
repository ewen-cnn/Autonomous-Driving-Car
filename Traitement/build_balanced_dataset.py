from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance

from actions import CLASSES, normalize_model_action, row_to_model_action
from config import BALANCED_DATASET_DIR, IMAGE_EXTENSIONS


@dataclass
class BuildConfig:
    input_root: Path = Path("segmentation")
    output_root: Path = Path(BALANCED_DATASET_DIR)
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    target_per_class_train: int = 1000
    target_per_class_valid: int = 200
    target_per_class_test: int = 200
    seed: int = 42


def collect_samples(input_root: Path) -> pd.DataFrame:
    """Load every prepared labels file and keep samples usable by the model."""
    frames = []
    for csv_path in sorted(input_root.rglob("labels.csv")):
        image_dir = csv_path.parent.parent / "Images"
        if not image_dir.exists():
            continue

        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        if "direction_class" in df.columns:
            df["model_direction"] = df["direction_class"].map(normalize_model_action)
        else:
            df["model_direction"] = [row_to_model_action(row) for row in df.to_dict("records")]

        image_column = "image_filename" if "image_filename" in df.columns else "image_path"
        df["source_image"] = df[image_column].astype(str).map(lambda name: str(image_dir / Path(name).name))
        frames.append(df[df["model_direction"].isin(CLASSES)].copy())

    if not frames:
        raise FileNotFoundError(f"No labeled samples with images found under {input_root}")

    return pd.concat(frames, ignore_index=True)


def split_by_class(df: pd.DataFrame, config: BuildConfig) -> dict[str, pd.DataFrame]:
    """Split each class independently to preserve class proportions per split."""
    random.seed(config.seed)
    parts = {"train": [], "valid": [], "test": []}

    for class_name in CLASSES:
        class_df = df[df["model_direction"] == class_name].sample(frac=1, random_state=config.seed)
        total = len(class_df)
        train_end = int(total * config.train_ratio)
        valid_end = train_end + int(total * config.valid_ratio)

        parts["train"].append(class_df.iloc[:train_end])
        parts["valid"].append(class_df.iloc[train_end:valid_end])
        parts["test"].append(class_df.iloc[valid_end:])

    return {split_name: pd.concat(frames, ignore_index=True) for split_name, frames in parts.items()}


def ensure_clean_output(output_root: Path) -> None:
    """Recreate the output directory tree from scratch."""
    if output_root.exists():
        shutil.rmtree(output_root)

    for split_name in ("train", "valid", "test"):
        (output_root / split_name / "Images").mkdir(parents=True, exist_ok=True)
        (output_root / split_name / "labels").mkdir(parents=True, exist_ok=True)


def _copy_image(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def _flip_image(src_path: Path, dst_path: Path) -> None:
    image = Image.open(src_path).convert("RGB")
    image.transpose(Image.FLIP_LEFT_RIGHT).save(dst_path)


def _enhance_image(src_path: Path, dst_path: Path, brightness: float | None = None, contrast: float | None = None) -> None:
    image = Image.open(src_path).convert("RGB")
    if brightness is not None:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast is not None:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    image.save(dst_path)


def _make_row(base_row: pd.Series, filename: str, direction: str, augmentation: str) -> dict:
    row = base_row.to_dict()
    row["image_filename"] = filename
    row["direction_class"] = direction
    row["model_direction"] = direction
    row["augmentation_type"] = augmentation
    return row


def build_split(split_name: str, df: pd.DataFrame, output_root: Path, target_per_class: int, seed: int) -> None:
    """Build one split with a fixed target size per class.

    When a class is under-represented, the function first reuses original
    samples, then generates synthetic variety:
    - left/right can be created by horizontally flipping the opposite class,
    - the remaining classes use light brightness/contrast variations.
    """
    output_rows = []
    image_dir = output_root / split_name / "Images"

    for class_name in CLASSES:
        class_df = df[df["model_direction"] == class_name].reset_index(drop=True)
        if class_df.empty:
            continue

        opposite_class = "right" if class_name == "left" else "left" if class_name == "right" else None
        opposite_df = df[df["model_direction"] == opposite_class].reset_index(drop=True) if opposite_class else pd.DataFrame()

        for index in range(target_per_class):
            use_original = index < len(class_df)
            source_df = class_df if use_original or opposite_df.empty else opposite_df
            source_row = source_df.iloc[index % len(source_df)]
            src_path = Path(source_row["source_image"])

            if src_path.suffix.lower() not in IMAGE_EXTENSIONS or not src_path.exists():
                continue

            suffix = src_path.suffix.lower()
            filename = f"{split_name}_{class_name}_{index:05d}{suffix}"
            dst_path = image_dir / filename

            augmentation = "original"
            direction = class_name

            if use_original:
                _copy_image(src_path, dst_path)
            elif class_name in {"left", "right"} and not opposite_df.empty:
                # A horizontal flip swaps left and right semantics while keeping
                # the road scene otherwise realistic.
                _flip_image(src_path, dst_path)
                augmentation = f"flip_from_{opposite_class}"
            else:
                # Non-directional classes cannot be mirrored into a different
                # class, so we cycle through lightweight photometric changes.
                cycle = index % 4
                if cycle == 0:
                    _enhance_image(src_path, dst_path, brightness=0.85)
                    augmentation = "brightness_low"
                elif cycle == 1:
                    _enhance_image(src_path, dst_path, brightness=1.15)
                    augmentation = "brightness_high"
                elif cycle == 2:
                    _enhance_image(src_path, dst_path, contrast=0.9)
                    augmentation = "contrast_low"
                else:
                    _enhance_image(src_path, dst_path, contrast=1.1)
                    augmentation = "contrast_high"

            output_row = _make_row(source_row, filename, direction, augmentation)
            if augmentation.startswith("flip_from") and {"speedA", "speedB"}.issubset(output_row):
                output_row["speedA"], output_row["speedB"] = output_row["speedB"], output_row["speedA"]
            output_rows.append(output_row)

    output_df = pd.DataFrame(output_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    output_df.to_csv(output_root / split_name / "labels" / "labels.csv", sep=";", index=False)
    print(f"[OK] Built {split_name} split with {len(output_df)} rows")


def main() -> None:
    config = BuildConfig()
    ensure_clean_output(config.output_root)

    full_df = collect_samples(config.input_root)
    split_frames = split_by_class(full_df, config)

    build_split("train", split_frames["train"], config.output_root, config.target_per_class_train, config.seed)
    build_split("valid", split_frames["valid"], config.output_root, config.target_per_class_valid, config.seed)
    build_split("test", split_frames["test"], config.output_root, config.target_per_class_test, config.seed)


if __name__ == "__main__":
    main()
