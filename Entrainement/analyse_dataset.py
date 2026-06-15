from pathlib import Path

from config import CLASSES
from Traitement.class_repartition_analysis import summarize_dataset


def main(dataset_root: str = "segmentation") -> None:
    per_record, global_summary = summarize_dataset(Path(dataset_root))
    print("\nPer record summary:")
    print(per_record.to_string(index=False))
    print("\nGlobal summary:")
    print(global_summary.to_string(index=False))


if __name__ == "__main__":
    main()
