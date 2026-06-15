from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from actions import classify_direction
from config import DATASET_DIR, IMAGE_EXTENSIONS, OUTPUT_DIR, SAMPLE_PERIOD_MS


@dataclass
class ResampleConfig:
    dataset_dir: Path = Path(DATASET_DIR)
    output_dir: Path = Path(OUTPUT_DIR)
    sample_period_ms: int = SAMPLE_PERIOD_MS


def resample_commands(df: pd.DataFrame, sample_period_ms: int) -> pd.DataFrame:
    df = df.sort_values("time_in_ms").reset_index(drop=True)

    start_time = int(df["time_in_ms"].iloc[0])
    end_time = int(df["time_in_ms"].iloc[-1])
    new_times = list(range(start_time, end_time + 1, sample_period_ms))

    original = df.set_index("time_in_ms")
    resampled = original.reindex(original.index.union(new_times)).sort_index()
    resampled["speedA"] = resampled["speedA"].interpolate(method="index")
    resampled["speedB"] = resampled["speedB"].interpolate(method="index")

    gpio_columns = ["GPIO1", "GPIO2", "GPIO3", "GPIO4"]
    resampled[gpio_columns] = resampled[gpio_columns].ffill()

    resampled = resampled.loc[new_times].reset_index()
    resampled = resampled.rename(columns={"index": "time_in_ms"})

    resampled["speedA"] = resampled["speedA"].round().clip(0, 100).astype(int)
    resampled["speedB"] = resampled["speedB"].round().clip(0, 100).astype(int)
    resampled[gpio_columns] = resampled[gpio_columns].astype(int)
    return resampled


def get_image_files(image_dir: Path) -> list[dict[str, Path | int | str]]:
    if not image_dir.exists():
        return []

    images: list[dict[str, Path | int | str]] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            timestamp = int(image_path.stem)
        except ValueError:
            continue
        images.append({"timestamp": timestamp, "filename": image_path.name, "path": image_path})
    return images


def attach_unique_previous_images(df: pd.DataFrame, image_dir: Path, output_image_dir: Path) -> tuple[pd.DataFrame, int, int, int]:
    images = get_image_files(image_dir)
    if not images:
        return df.iloc[0:0].copy(), 0, len(df), 0

    output_image_dir.mkdir(parents=True, exist_ok=True)

    image_index = 0
    copied_images = set()
    matched_rows = []
    no_image_available = 0

    for _, row in df.iterrows():
        csv_time = int(row["time_in_ms"])
        selected_image = None

        while image_index < len(images) and int(images[image_index]["timestamp"]) <= csv_time:
            selected_image = images[image_index]
            image_index += 1

        if selected_image is None:
            no_image_available += 1
            continue

        copied_images.add(str(selected_image["filename"]))
        shutil.copy2(selected_image["path"], output_image_dir / str(selected_image["filename"]))

        enriched_row = row.copy()
        enriched_row["image_filename"] = selected_image["filename"]
        enriched_row["image_time_ms"] = int(selected_image["timestamp"])
        enriched_row["image_delta_ms"] = csv_time - int(selected_image["timestamp"])
        enriched_row["direction_class"] = classify_direction(enriched_row)
        matched_rows.append(enriched_row)

    unused_images = len(images) - len(copied_images)
    return pd.DataFrame(matched_rows), len(copied_images), no_image_available, unused_images


def analyze_record(
    record_name: str,
    original_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    copied_images: int,
    no_image_available: int,
    unused_images: int,
    sample_period_ms: int,
) -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {
        "record": record_name,
        "sample_period_ms": sample_period_ms,
        "original_csv_rows": len(original_df),
        "aligned_rows": len(matched_df),
        "rows_removed_without_image": no_image_available,
        "copied_images": copied_images,
        "unused_images": unused_images,
    }

    if matched_df.empty:
        return result

    counts = matched_df["direction_class"].value_counts()
    for direction, count in counts.items():
        result[f"{direction}_count"] = int(count)
    return result


def process_dataset(config: ResampleConfig) -> pd.DataFrame:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_rows = []

    for record_dir in sorted(config.dataset_dir.iterdir()):
        if not record_dir.is_dir():
            continue

        csv_path = record_dir / "labels.csv"
        image_dir = record_dir / "Images"
        if not csv_path.exists():
            continue

        original_df = pd.read_csv(csv_path, sep=";")
        resampled_df = resample_commands(original_df, config.sample_period_ms)

        output_record_dir = config.output_dir / record_dir.name
        output_label_dir = output_record_dir / "labels"
        output_image_dir = output_record_dir / "Images"
        output_label_dir.mkdir(parents=True, exist_ok=True)

        matched_df, copied_images, no_image_available, unused_images = attach_unique_previous_images(
            resampled_df,
            image_dir=image_dir,
            output_image_dir=output_image_dir,
        )
        matched_df.to_csv(output_label_dir / "labels.csv", sep=";", index=False)

        analysis_rows.append(
            analyze_record(
                record_name=record_dir.name,
                original_df=original_df,
                matched_df=matched_df,
                copied_images=copied_images,
                no_image_available=no_image_available,
                unused_images=unused_images,
                sample_period_ms=config.sample_period_ms,
            )
        )

        print(f"[OK] {record_dir.name}: {len(matched_df)} aligned rows, {copied_images} copied images")

    analysis_df = pd.DataFrame(analysis_rows)
    analysis_df.to_csv(config.output_dir / "resampling_analysis.csv", sep=";", index=False)
    return analysis_df


def main() -> None:
    process_dataset(ResampleConfig())


if __name__ == "__main__":
    main()
