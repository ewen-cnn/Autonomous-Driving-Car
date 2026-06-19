import argparse

from config import BALANCED_dataset_DIR, DEFAULT_MODEL_PATH
from training import TrainingConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the driving action classifier.")
    parser.add_argument(
        "--csv-file",
        default=f"{BALANCED_dataset_DIR}/train/labels/labels.csv",
        help="CSV file containing image references and direction labels.",
    )
    parser.add_argument(
        "--image-dir",
        default=f"{BALANCED_dataset_DIR}/train/Images",
        help="Directory containing the images referenced by the CSV.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Where to save the best checkpoint.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_model(
        TrainingConfig(
            csv_file=args.csv_file,
            image_dir=args.image_dir,
            model_path=args.model_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            validation_ratio=args.validation_ratio,
        )
    )
    print(f"\nBest validation accuracy: {summary['best_validation_accuracy']:.4f}")


if __name__ == "__main__":
    main()
