import os
import shutil
import pandas as pd
from PIL import Image

INPUT_DATASET = "resampled_dataset"
OUTPUT_DIR = "test_1image"

SPLIT = "train"

IMAGE_DIR = "Images"
LABEL_DIR = "labels"
LABEL_FILE = "labels.csv"


def motor_direction(gpio_a, gpio_b, side):
    if side == "left":
        if gpio_a == 0 and gpio_b == 0:
            return "stop"
        if gpio_a == 1 and gpio_b == 0:
            return "forward"
        if gpio_a == 0 and gpio_b == 1:
            return "backward"

    if side == "right":
        if gpio_a == 0 and gpio_b == 0:
            return "stop"
        if gpio_a == 0 and gpio_b == 1:
            return "forward"
        if gpio_a == 1 and gpio_b == 0:
            return "backward"

    return "unknown"


def classify_old_direction(row):
    speedA = row["speedA"]
    speedB = row["speedB"]

    left_dir = motor_direction(row["GPIO1"], row["GPIO2"], "left")
    right_dir = motor_direction(row["GPIO3"], row["GPIO4"], "right")

    if left_dir == "forward" and right_dir == "forward":
        if speedA == speedB:
            return "forward"
        if speedA > speedB:
            return "light_right"
        if speedB > speedA:
            return "light_left"

    if left_dir == "forward" and right_dir == "backward":
        return "sharp_right"

    if left_dir == "backward" and right_dir == "forward":
        return "sharp_left"

    if left_dir == "forward" and right_dir == "stop":
        return "pivot_right"

    if left_dir == "stop" and right_dir == "forward":
        return "pivot_left"

    if left_dir == "stop" and right_dir == "stop":
        return "stop"

    if left_dir == "backward" and right_dir == "backward":
        return "backward"

    return "other"


def map_to_3_classes(old_class):
    if old_class in ["light_left", "sharp_left", "pivot_left"]:
        return "left"
    if old_class in ["light_right", "sharp_right", "pivot_right"]:
        return "right"
    if old_class == "forward":
        return "forward"
    return None


def flip_row_left_to_right(row, new_filename):
    new_row = row.copy()

    new_row["image_filename"] = new_filename
    new_row["direction_class"] = "right"
    new_row["old_direction_class"] = "augmented_from_left_flip"

    speed_a = new_row["speedA"]
    speed_b = new_row["speedB"]

    new_row["speedA"] = speed_b
    new_row["speedB"] = speed_a

    return new_row


def main():
    input_csv = os.path.join(INPUT_DATASET, SPLIT, LABEL_DIR, LABEL_FILE)
    input_img_dir = os.path.join(INPUT_DATASET, SPLIT, IMAGE_DIR)

    output_img_dir = os.path.join(OUTPUT_DIR, IMAGE_DIR)
    output_label_dir = os.path.join(OUTPUT_DIR, LABEL_DIR)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    df = pd.read_csv(input_csv, sep=";")

    df["old_direction_class"] = df.apply(classify_old_direction, axis=1)
    df["direction_class"] = df["old_direction_class"].apply(map_to_3_classes)

    left_df = df[df["direction_class"] == "left"].copy()

    if len(left_df) == 0:
        raise ValueError("Aucune image left trouvée.")

    original_row = left_df.iloc[0].copy()
    original_filename = original_row["image_filename"]

    src_img = os.path.join(input_img_dir, original_filename)

    if not os.path.exists(src_img):
        raise FileNotFoundError(f"Image introuvable : {src_img}")

    name, ext = os.path.splitext(original_filename)

    copied_original_filename = f"{name}_original{ext}"
    flipped_filename = f"{name}_flip_right{ext}"

    dst_original = os.path.join(output_img_dir, copied_original_filename)
    dst_flipped = os.path.join(output_img_dir, flipped_filename)

    shutil.copy2(src_img, dst_original)

    image = Image.open(src_img).convert("RGB")
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped.save(dst_flipped)

    original_row["image_filename"] = copied_original_filename
    original_row["row_type"] = "original"

    flipped_row = flip_row_left_to_right(original_row, flipped_filename)
    flipped_row["row_type"] = "augmented_flip"

    output_df = pd.DataFrame([original_row, flipped_row])

    output_csv = os.path.join(output_label_dir, LABEL_FILE)
    output_df.to_csv(output_csv, sep=";", index=False)

    print("\nTest terminé.")
    print(f"Dossier créé : {OUTPUT_DIR}")
    print(f"Image originale : {copied_original_filename}")
    print(f"Image modifiée  : {flipped_filename}")
    print(f"CSV généré      : {output_csv}")


if __name__ == "__main__":
    main()