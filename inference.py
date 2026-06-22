from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from actions import CLASSES, IDX_TO_CLASS
from config import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_PATH
from model import DrivingCNN
from Traitement.transforms import build_eval_transform


def load_model(model_path: str = DEFAULT_MODEL_PATH, device: torch.device | None = None) -> tuple[DrivingCNN, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    classes = checkpoint.get("classes", CLASSES)
    if list(classes) != list(CLASSES):
        raise ValueError(
            f"Checkpoint classes {classes} do not match current classes {CLASSES}."
        )

    model = DrivingCNN(num_classes=len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_image(image_path: str | Path, model_path: str = DEFAULT_MODEL_PATH) -> tuple[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(model_path=model_path, device=device)

    image_size = tuple(checkpoint.get("image_size", checkpoint.get("img_size", DEFAULT_IMAGE_SIZE)))
    transform = build_eval_transform(image_size)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_index = torch.argmax(probabilities, dim=1).item()

    predicted_action = IDX_TO_CLASS[predicted_index]
    confidence = probabilities[0, predicted_index].item()
    return predicted_action, confidence
