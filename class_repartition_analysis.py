import os
import pandas as pd

DATASET_DIR = "resampled_dataset"
SPLITS = ["train", "valid", "test"]

VALID_3_CLASSES = ["left", "right", "forward"]


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

    if left_dir == "stop" and right_dir == "stop":
        return "stop"

    if left_dir == "backward" and right_dir == "backward":
        return "backward"

    if left_dir == "forward" and right_dir == "backward":
        return "sharp_right"

    if left_dir == "backward" and right_dir == "forward":
        return "sharp_left"

    if left_dir == "forward" and right_dir == "stop":
        return "pivot_right"

    if left_dir == "stop" and right_dir == "forward":
        return "pivot_left"

    if left_dir == "forward" and right_dir == "forward":
        if speedA == speedB:
            return "forward"
        if speedA > speedB:
            return "light_right"
        if speedB > speedA:
            return "light_left"

    return "other"


def map_to_3_classes(old_class):
    if old_class in ["light_left", "sharp_left", "pivot_left"]:
        return "left"

    if old_class in ["light_right", "sharp_right", "pivot_right"]:
        return "right"

    if old_class == "forward":
        return "forward"

    return "ignored"


for split in SPLITS:
    csv_path = os.path.join(DATASET_DIR, split, "labels", "labels.csv")
    image_dir = os.path.join(DATASET_DIR, split, "Images")

    print("\n" + "=" * 50)
    print(split.upper())
    print("=" * 50)

    if not os.path.exists(csv_path):
        print("labels.csv introuvable")
        continue

    df = pd.read_csv(csv_path, sep=";")

    df["old_direction_class"] = df.apply(classify_old_direction, axis=1)
    df["direction_class"] = df["old_direction_class"].apply(map_to_3_classes)

    total_raw = len(df)
    total_used = len(df[df["direction_class"].isin(VALID_3_CLASSES)])

    print(f"Nombre d'échantillons total       : {total_raw}")
    print(f"Nombre d'échantillons utilisables : {total_used}")

    if os.path.exists(image_dir):
        image_count = len([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        print(f"Nombre d'images                   : {image_count}")

    print("\nAnciennes classes :")
    old_counts = df["old_direction_class"].value_counts()

    for cls, count in old_counts.items():
        percent = 100 * count / total_raw if total_raw > 0 else 0
        print(f"{cls:<12} {count:>6} ({percent:>6.2f} %)")

    print("\nNouvelles classes utilisées :")
    new_counts = df["direction_class"].value_counts()

    for cls in ["left", "right", "forward", "ignored"]:
        count = new_counts.get(cls, 0)
        percent = 100 * count / total_raw if total_raw > 0 else 0
        print(f"{cls:<10} {count:>6} ({percent:>6.2f} %)")

    print("\nRépartition finale hors ignored :")
    used_df = df[df["direction_class"].isin(VALID_3_CLASSES)]

    for cls in VALID_3_CLASSES:
        count = (used_df["direction_class"] == cls).sum()
        percent = 100 * count / total_used if total_used > 0 else 0
        print(f"{cls:<10} {count:>6} ({percent:>6.2f} %)")

print("\nAnalyse terminée.")