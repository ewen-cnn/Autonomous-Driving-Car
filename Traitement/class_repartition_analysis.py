from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from actions import normalize_model_action, row_to_model_action


def _load_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if "direction_class" in df.columns:
        df["model_direction"] = df["direction_class"].map(normalize_model_action)
    else:
        df["model_direction"] = [row_to_model_action(row) for row in df.to_dict("records")]

    df["record_name"] = csv_path.parent.parent.name
    return df


def collect_label_files(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.rglob("labels.csv"))


def summarize_dataset(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_files = collect_label_files(dataset_root)
    if not label_files:
        raise FileNotFoundError(f"No labels.csv files found under {dataset_root}")

    frames = [_load_labels(csv_path) for csv_path in label_files]
    full_df = pd.concat(frames, ignore_index=True)

    per_record = (
        full_df.groupby(["record_name", "model_direction"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["record_name", "model_direction"])
    )

    global_summary = (
        full_df["model_direction"]
        .fillna("ignored")
        .value_counts()
        .rename_axis("direction")
        .reset_index(name="count")
    )
    return per_record, global_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize direction distribution in a prepared dataset.")
    parser.add_argument("dataset_root", nargs="?", default="segmentation")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    per_record, global_summary = summarize_dataset(dataset_root)

    print("\nPer record summary:")
    print(per_record.to_string(index=False))

    print("\nGlobal summary:")
    print(global_summary.to_string(index=False))


if __name__ == "__main__":
    main()
