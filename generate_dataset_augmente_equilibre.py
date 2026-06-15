import os
import shutil
import random
import pandas as pd
from PIL import Image, ImageEnhance

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DATASET = "resampled_dataset"
OUTPUT_DATASET = "dataset_augmente_equilibre"

IMAGE_DIR_NAME = "Images"
LABEL_DIR_NAME = "labels"
LABEL_FILE = "labels.csv"

CLASSES = ["left", "right", "forward"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TRAIN_TARGET = 1000
VALID_TARGET = 200
TEST_TARGET = 60

VALID_RATIO_FROM_TRAIN = 0.20


# ============================================================
# CLASSIFICATION DES COMMANDES
# ============================================================

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
    return None


# ============================================================
# AUGMENTATIONS
# ============================================================

def copy_original(src_path, dst_path):
    shutil.copy2(src_path, dst_path)


def save_horizontal_flip(src_path, dst_path):
    """
    Flip horizontal :
    - left devient right
    - right devient left
    - forward reste forward

    Dans le CSV associé, on échange aussi speedA et speedB.
    """
    image = Image.open(src_path).convert("RGB")
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped.save(dst_path)


def save_brightness(src_path, dst_path, factor):
    """
    Augmentation luminosité.
    Ne change pas la classe ni les vitesses.
    """
    image = Image.open(src_path).convert("RGB")
    image = ImageEnhance.Brightness(image).enhance(factor)
    image.save(dst_path)


def save_contrast(src_path, dst_path, factor):
    """
    Augmentation contraste.
    Ne change pas la classe ni les vitesses.
    """
    image = Image.open(src_path).convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(factor)
    image.save(dst_path)


# ============================================================
# OUTILS CSV / IMAGES
# ============================================================

def clean_output():
    if os.path.exists(OUTPUT_DATASET):
        shutil.rmtree(OUTPUT_DATASET)

    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(OUTPUT_DATASET, split, IMAGE_DIR_NAME), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DATASET, split, LABEL_DIR_NAME), exist_ok=True)


def find_image_path(image_dir, filename):
    path = os.path.join(image_dir, filename)

    if os.path.exists(path):
        return path

    name, _ = os.path.splitext(filename)

    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(image_dir, name + ext)
        if os.path.exists(candidate):
            return candidate

    return None


def unique_filename(filename, suffix, index, split_name):
    name, ext = os.path.splitext(filename)
    return f"{split_name}_{name}_{suffix}_{index}{ext}"


def make_output_row(row, new_filename, direction_class, augmentation_type):
    new_row = row.copy()
    new_row["image_filename"] = new_filename
    new_row["direction_class"] = direction_class
    new_row["augmentation_type"] = augmentation_type
    return new_row


def make_flip_row(row, new_filename):
    new_row = row.copy()

    if row["direction_class"] == "left":
        new_class = "right"
    elif row["direction_class"] == "right":
        new_class = "left"
    else:
        new_class = "forward"

    speedA = new_row["speedA"]
    speedB = new_row["speedB"]

    new_row["speedA"] = speedB
    new_row["speedB"] = speedA

    new_row["image_filename"] = new_filename
    new_row["direction_class"] = new_class
    new_row["augmentation_type"] = "horizontal_flip"

    return new_row


def load_split(split):
    csv_path = os.path.join(INPUT_DATASET, split, LABEL_DIR_NAME, LABEL_FILE)
    image_dir = os.path.join(INPUT_DATASET, split, IMAGE_DIR_NAME)

    if not os.path.exists(csv_path):
        print(f"[ATTENTION] {split} ignoré : labels.csv introuvable")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, sep=";")

    df["source_split"] = split
    df["source_image_dir"] = image_dir
    df["old_direction_class"] = df.apply(classify_old_direction, axis=1)
    df["direction_class"] = df["old_direction_class"].apply(map_to_3_classes)

    df = df[df["direction_class"].isin(CLASSES)].copy()
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return df


# ============================================================
# SPLIT TRAIN_SOURCE / VALID_SOURCE AVANT AUGMENTATION
# ============================================================

def split_train_valid_source(train_df):
    train_parts = []
    valid_parts = []

    for cls in CLASSES:
        cls_df = train_df[train_df["direction_class"] == cls].copy()
        cls_df = cls_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        valid_count = max(1, int(len(cls_df) * VALID_RATIO_FROM_TRAIN))

        valid_parts.append(cls_df.iloc[:valid_count])
        train_parts.append(cls_df.iloc[valid_count:])

    train_source = pd.concat(train_parts, ignore_index=True)
    valid_source = pd.concat(valid_parts, ignore_index=True)

    train_source = train_source.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    valid_source = valid_source.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return train_source, valid_source


# ============================================================
# GÉNÉRATION D'UN SPLIT ÉQUILIBRÉ
# ============================================================

def add_originals(df, output_rows, output_image_dir, class_name, target, split_name):
    rows = df[df["direction_class"] == class_name].copy()
    rows = rows.sample(frac=1, random_state=RANDOM_SEED).head(target)

    count = 0

    for i, row in rows.iterrows():
        src_path = find_image_path(row["source_image_dir"], row["image_filename"])

        if src_path is None:
            print(f"[ATTENTION] image introuvable : {row['image_filename']}")
            continue

        new_filename = unique_filename(row["image_filename"], "orig", i, split_name)
        dst_path = os.path.join(output_image_dir, new_filename)

        copy_original(src_path, dst_path)

        output_rows.append(
            make_output_row(row, new_filename, class_name, "original")
        )

        count += 1

    return count


def augment_right_from_left(df, output_rows, output_image_dir, current_count, target, split_name):
    rows = df[df["direction_class"] == "left"].copy()

    if len(rows) == 0:
        print("[ATTENTION] impossible de créer right : aucune image left")
        return current_count

    rows = rows.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    i = 0

    while current_count < target:
        row = rows.iloc[i % len(rows)].copy()
        src_path = find_image_path(row["source_image_dir"], row["image_filename"])

        if src_path is None:
            i += 1
            continue

        new_filename = unique_filename(row["image_filename"], "flip_to_right", i, split_name)
        dst_path = os.path.join(output_image_dir, new_filename)

        save_horizontal_flip(src_path, dst_path)

        output_rows.append(
            make_flip_row(row, new_filename)
        )

        current_count += 1
        i += 1

    return current_count


def augment_left_from_right(df, output_rows, output_image_dir, current_count, target, split_name):
    rows = df[df["direction_class"] == "right"].copy()

    if len(rows) == 0:
        print("[ATTENTION] impossible de créer left : aucune image right")
        return current_count

    rows = rows.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    i = 0

    while current_count < target:
        row = rows.iloc[i % len(rows)].copy()
        src_path = find_image_path(row["source_image_dir"], row["image_filename"])

        if src_path is None:
            i += 1
            continue

        new_filename = unique_filename(row["image_filename"], "flip_to_left", i, split_name)
        dst_path = os.path.join(output_image_dir, new_filename)

        save_horizontal_flip(src_path, dst_path)

        output_rows.append(
            make_flip_row(row, new_filename)
        )

        current_count += 1
        i += 1

    return current_count


def augment_forward(df, output_rows, output_image_dir, current_count, target, split_name):
    rows = df[df["direction_class"] == "forward"].copy()

    if len(rows) == 0:
        print("[ATTENTION] impossible de créer forward : aucune image forward")
        return current_count

    augmentations = [
        ("bright_low", 0.80, save_brightness),
        ("bright_high", 1.20, save_brightness),
        ("contrast_low", 0.85, save_contrast),
        ("contrast_high", 1.15, save_contrast),
    ]

    i = 0

    while current_count < target:
        row = rows.iloc[i % len(rows)].copy()
        src_path = find_image_path(row["source_image_dir"], row["image_filename"])

        if src_path is None:
            i += 1
            continue

        aug_name, factor, aug_func = augmentations[i % len(augmentations)]

        new_filename = unique_filename(row["image_filename"], aug_name, i, split_name)
        dst_path = os.path.join(output_image_dir, new_filename)

        aug_func(src_path, dst_path, factor)

        output_rows.append(
            make_output_row(row, new_filename, "forward", aug_name)
        )

        current_count += 1
        i += 1

    return current_count


def generate_balanced_split(df, split_name, target):
    print("\n" + "=" * 60)
    print(f"GÉNÉRATION {split_name.upper()}")
    print("=" * 60)

    output_image_dir = os.path.join(OUTPUT_DATASET, split_name, IMAGE_DIR_NAME)
    output_label_dir = os.path.join(OUTPUT_DATASET, split_name, LABEL_DIR_NAME)
    output_csv = os.path.join(output_label_dir, LABEL_FILE)

    print("Source utilisable :")
    print(df["direction_class"].value_counts())

    output_rows = []
    counts = {}

    # 1) On met d'abord les vraies images disponibles.
    for cls in CLASSES:
        counts[cls] = add_originals(
            df=df,
            output_rows=output_rows,
            output_image_dir=output_image_dir,
            class_name=cls,
            target=target,
            split_name=split_name
        )

    # 2) On crée right avec des left flipées.
    counts["right"] = augment_right_from_left(
        df=df,
        output_rows=output_rows,
        output_image_dir=output_image_dir,
        current_count=counts["right"],
        target=target,
        split_name=split_name
    )

    # 3) Si nécessaire, on crée left avec des right flipées.
    counts["left"] = augment_left_from_right(
        df=df,
        output_rows=output_rows,
        output_image_dir=output_image_dir,
        current_count=counts["left"],
        target=target,
        split_name=split_name
    )

    # 4) On complète forward avec luminosité / contraste.
    counts["forward"] = augment_forward(
        df=df,
        output_rows=output_rows,
        output_image_dir=output_image_dir,
        current_count=counts["forward"],
        target=target,
        split_name=split_name
    )

    output_df = pd.DataFrame(output_rows)

    final_parts = []

    for cls in CLASSES:
        cls_df = output_df[output_df["direction_class"] == cls].copy()
        cls_df = cls_df.sample(frac=1, random_state=RANDOM_SEED).head(target)
        final_parts.append(cls_df)

    output_df = pd.concat(final_parts, ignore_index=True)
    output_df = output_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    output_df.to_csv(output_csv, sep=";", index=False)

    image_count = len([
        f for f in os.listdir(output_image_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    print("\nRépartition finale :")
    print(output_df["direction_class"].value_counts())
    print(f"Images : {image_count}")
    print(f"Lignes CSV : {len(output_df)}")
    print(f"CSV généré : {output_csv}")


# ============================================================
# MAIN
# ============================================================

def main():
    clean_output()

    # --------------------------------------------------------
    # 1. Ancien train -> train_source / valid_source
    # --------------------------------------------------------
    old_train_df = load_split("train")

    print("\nAncien train utilisable :")
    print(old_train_df["direction_class"].value_counts())

    train_source, valid_source = split_train_valid_source(old_train_df)

    print("\ntrain_source avant augmentation :")
    print(train_source["direction_class"].value_counts())

    print("\nvalid_source avant augmentation :")
    print(valid_source["direction_class"].value_counts())

    # --------------------------------------------------------
    # 2. Test = ancien valid + ancien test
    # --------------------------------------------------------
    old_valid_df = load_split("valid")
    old_test_df = load_split("test")

    test_source = pd.concat([old_valid_df, old_test_df], ignore_index=True)
    test_source = test_source.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # --------------------------------------------------------
    # 3. Génération finale
    # --------------------------------------------------------
    # Test d'abord, puis valid, puis train en dernier.
    generate_balanced_split(test_source, "test", TEST_TARGET)
    generate_balanced_split(valid_source, "valid", VALID_TARGET)
    generate_balanced_split(train_source, "train", TRAIN_TARGET)

    print("\n" + "=" * 60)
    print("DATASET AUGMENTÉ ÉQUILIBRÉ TERMINÉ")
    print("=" * 60)
    print(f"Dossier créé : {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()