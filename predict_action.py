import argparse

from config import DEFAULT_MODEL_PATH
from inference import predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a driving action from a single image.")
    parser.add_argument("image_path", help="Path to the image to classify.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action, confidence = predict_image(args.image_path, model_path=args.model_path)
    print(f"Predicted action: {action}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
