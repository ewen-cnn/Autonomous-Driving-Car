import os
import shutil
import bisect
import pandas as pd

DATASET_DIR = "Record_V2"
OUTPUT_DIR = "segmentation"
SAMPLE_PERIOD_MS = 250

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

CLASSES = [
    "forward", "light_left", "light_right",
    "pivot_left", "pivot_right",
    "sharp_left", "sharp_right",
    "backward", "stop", "other"
]


def resample_commands(df, sample_period_ms=250):
    df = df.sort_values("time_in_ms").reset_index(drop=True)

    start_time = int(df["time_in_ms"].iloc[0])
    end_time = int(df["time_in_ms"].iloc[-1])

    new_times = list(range(start_time, end_time + 1, sample_period_ms))

    original = df.set_index("time_in_ms")

    resampled = original.reindex(
        original.index.union(new_times)
    ).sort_index()

    resampled["speedA"] = resampled["speedA"].interpolate(method="index")
    resampled["speedB"] = resampled["speedB"].interpolate(method="index")

    gpio_cols = ["GPIO1", "GPIO2", "GPIO3", "GPIO4"]
    resampled[gpio_cols] = resampled[gpio_cols].ffill()

    resampled = resampled.loc[new_times].reset_index()
    resampled = resampled.rename(columns={"index": "time_in_ms"})

    resampled["speedA"] = resampled["speedA"].round().clip(0, 100).astype(int)
    resampled["speedB"] = resampled["speedB"].round().clip(0, 100).astype(int)
    resampled[gpio_cols] = resampled[gpio_cols].astype(int)

    return resampled


def get_image_files(image_dir):
    images = []

    if not os.path.exists(image_dir):
        return []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        name_without_ext = os.path.splitext(filename)[0]

        try:
            timestamp = int(name_without_ext)
        except ValueError:
            continue

        images.append({
            "timestamp": timestamp,
            "filename": filename,
            "path": os.path.join(image_dir, filename)
        })

    images.sort(key=lambda x: x["timestamp"])
    return images


def attach_previous_images(df, image_dir, output_image_dir):
    images = get_image_files(image_dir)

    if len(images) == 0:
        print("  [ATTENTION] Aucune image trouvée")
        df["image_filename"] = None
        df["image_time_ms"] = None
        df["image_delta_ms"] = None
        return df.iloc[0:0].copy(), 0, 0

    os.makedirs(output_image_dir, exist_ok=True)

    image_times = [img["timestamp"] for img in images]

    matched_rows = []
    copied_images = set()
    no_image_before = 0

    for _, row in df.iterrows():
        csv_time = int(row["time_in_ms"])

        # Image la plus proche temporellement, mais strictement avant ou égale au CSV
        idx = bisect.bisect_right(image_times, csv_time) - 1

        if idx < 0:
            no_image_before += 1
            continue

        selected_image = images[idx]

        new_row = row.copy()
        new_row["image_filename"] = selected_image["filename"]
        new_row["image_time_ms"] = selected_image["timestamp"]
        new_row["image_delta_ms"] = csv_time - selected_image["timestamp"]

        matched_rows.append(new_row)

        if selected_image["filename"] not in copied_images:
            dst = os.path.join(output_image_dir, selected_image["filename"])
            shutil.copy2(selected_image["path"], dst)
            copied_images.add(selected_image["filename"])

    matched_df = pd.DataFrame(matched_rows)

    return matched_df, len(copied_images), no_image_before


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


def classify_direction(row):
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


def analyze_csv(record_name, df_original, df_resampled, copied_images, no_image_before):
    df_tmp = df_resampled.copy()
    df_tmp["direction_class"] = df_tmp.apply(classify_direction, axis=1)

    result = {
        "record": record_name,
        "sample_period_ms": SAMPLE_PERIOD_MS,

        "original_csv_rows": len(df_original),
        "resampled_csv_rows_before_image_filter": len(df_resampled) + no_image_before,
        "resampled_csv_rows_after_image_filter": len(df_resampled),

        "rows_removed_no_previous_image": no_image_before,

        "copied_images": copied_images,

        "speedA_min": int(df_tmp["speedA"].min()) if len(df_tmp) else 0,
        "speedA_max": int(df_tmp["speedA"].max()) if len(df_tmp) else 0,
        "speedA_mean": round(df_tmp["speedA"].mean(), 2) if len(df_tmp) else 0,

        "speedB_min": int(df_tmp["speedB"].min()) if len(df_tmp) else 0,
        "speedB_max": int(df_tmp["speedB"].max()) if len(df_tmp) else 0,
        "speedB_mean": round(df_tmp["speedB"].mean(), 2) if len(df_tmp) else 0,
    }

    counts = df_tmp["direction_class"].value_counts()
    percents = df_tmp["direction_class"].value_counts(normalize=True) * 100

    for cls in CLASSES:
        result[f"{cls}_count"] = int(counts.get(cls, 0))
        result[f"{cls}_percent"] = round(percents.get(cls, 0), 2)

    return result


os.makedirs(OUTPUT_DIR, exist_ok=True)

analysis_rows = []

print("\n==============================")
print("RÉÉCHANTILLONNAGE + ASSOCIATION IMAGES")
print("==============================")
print(f"Dossier source : {DATASET_DIR}")
print(f"Dossier sortie : {OUTPUT_DIR}")
print(f"Période        : {SAMPLE_PERIOD_MS} ms")

for record_name in sorted(os.listdir(DATASET_DIR)):
    record_path = os.path.join(DATASET_DIR, record_name)

    if not os.path.isdir(record_path):
        continue

    csv_path = os.path.join(record_path, "labels.csv")
    image_dir = os.path.join(record_path, "Images")

    if not os.path.exists(csv_path):
        print(f"\n[IGNORÉ] {record_name} : labels.csv introuvable")
        continue

    print("\n------------------------------")
    print(f"Record : {record_name}")

    df_original = pd.read_csv(csv_path, sep=";")
    df_original = df_original.sort_values("time_in_ms").reset_index(drop=True)

    df_resampled = resample_commands(df_original, SAMPLE_PERIOD_MS)

    output_label_dir = os.path.join(OUTPUT_DIR, record_name, "labels")
    output_image_dir = os.path.join(OUTPUT_DIR, record_name, "Images")

    os.makedirs(output_label_dir, exist_ok=True)

    df_matched, copied_images, no_image_before = attach_previous_images(
        df_resampled,
        image_dir,
        output_image_dir
    )

    output_csv = os.path.join(output_label_dir, "labels.csv")
    df_matched.to_csv(output_csv, sep=";", index=False)

    print(f"Lignes CSV originales          : {len(df_original)}")
    print(f"Lignes CSV rééchantillonnées   : {len(df_resampled)}")
    print(f"Lignes gardées avec image      : {len(df_matched)}")
    print(f"Lignes supprimées sans image   : {no_image_before}")
    print(f"Images copiées                 : {copied_images}")
    print(f"CSV généré                     : {output_csv}")
    print(f"Dossier images                 : {output_image_dir}")

    analysis_rows.append(
        analyze_csv(
            record_name,
            df_original,
            df_matched,
            copied_images,
            no_image_before
        )
    )

analysis_df = pd.DataFrame(analysis_rows)

analysis_csv = os.path.join(OUTPUT_DIR, "resampling_analysis.csv")
analysis_df.to_csv(analysis_csv, sep=";", index=False)

print("\n==============================")
print("TERMINÉ")
print("==============================")
print(f"Analyse globale : {analysis_csv}")