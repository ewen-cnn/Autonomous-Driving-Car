# Rééchantillonnage tous les 250 ms
# interpolation linéaire de speedA / speedB
# maintien de l’état précédent pour les GPIO
# sauvegarde dans segmentation/nom_du_record/labels/labels.csv
# un seul fichier d’analyse global : segmentation/resampling_analysis.csv

import os
import pandas as pd

# =========================
# CONFIGURATION
# =========================

DATASET_DIR = "Record_V2"
OUTPUT_DIR = "segmentation"
SAMPLE_PERIOD_MS = 250

CLASSES = [
    "forward",
    "light_left",
    "light_right",
    "pivot_left",
    "pivot_right",
    "sharp_left",
    "sharp_right",
    "backward",
    "stop",
    "other"
]


# =========================
# RÉÉCHANTILLONNAGE
# =========================

def resample_commands(df, sample_period_ms=250):
    df = df.sort_values("time_in_ms").reset_index(drop=True)

    start_time = int(df["time_in_ms"].iloc[0])
    end_time = int(df["time_in_ms"].iloc[-1])

    new_times = list(range(start_time, end_time + 1, sample_period_ms))

    original = df.set_index("time_in_ms")

    resampled = original.reindex(
        original.index.union(new_times)
    ).sort_index()

    # Vitesses : interpolation linéaire
    resampled["speedA"] = resampled["speedA"].interpolate(method="index")
    resampled["speedB"] = resampled["speedB"].interpolate(method="index")

    # GPIO : état logique maintenu
    gpio_cols = ["GPIO1", "GPIO2", "GPIO3", "GPIO4"]
    resampled[gpio_cols] = resampled[gpio_cols].ffill()

    # On garde uniquement les timestamps échantillonnés
    resampled = resampled.loc[new_times].reset_index()
    resampled = resampled.rename(columns={"index": "time_in_ms"})

    # Nettoyage des types
    resampled["speedA"] = resampled["speedA"].round().clip(0, 100).astype(int)
    resampled["speedB"] = resampled["speedB"].round().clip(0, 100).astype(int)
    resampled[gpio_cols] = resampled[gpio_cols].astype(int)

    return resampled


# =========================
# CLASSIFICATION
# =========================

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


# =========================
# ANALYSE
# =========================

def analyze_resampled_csv(record_name, df_original, df_resampled):
    df_resampled = df_resampled.copy()
    df_resampled["direction_class"] = df_resampled.apply(classify_direction, axis=1)

    result = {
        "record": record_name,
        "sample_period_ms": SAMPLE_PERIOD_MS,

        "original_rows": len(df_original),
        "resampled_rows": len(df_resampled),
        "rows_created": max(0, len(df_resampled) - len(df_original)),
        "rows_removed": max(0, len(df_original) - len(df_resampled)),

        "start_time_ms": int(df_original["time_in_ms"].min()),
        "end_time_ms": int(df_original["time_in_ms"].max()),
        "duration_s": round(
            (df_original["time_in_ms"].max() - df_original["time_in_ms"].min()) / 1000,
            2
        ),

        "speedA_min": int(df_resampled["speedA"].min()),
        "speedA_max": int(df_resampled["speedA"].max()),
        "speedA_mean": round(df_resampled["speedA"].mean(), 2),

        "speedB_min": int(df_resampled["speedB"].min()),
        "speedB_max": int(df_resampled["speedB"].max()),
        "speedB_mean": round(df_resampled["speedB"].mean(), 2),
    }

    counts = df_resampled["direction_class"].value_counts()
    percents = df_resampled["direction_class"].value_counts(normalize=True) * 100

    for cls in CLASSES:
        result[f"{cls}_count"] = int(counts.get(cls, 0))
        result[f"{cls}_percent"] = round(percents.get(cls, 0), 2)

    return result


# =========================
# MAIN
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

analysis_rows = []

print("\n==============================")
print("RÉÉCHANTILLONNAGE DES CSV")
print("==============================")
print(f"Dossier source        : {DATASET_DIR}")
print(f"Dossier sortie        : {OUTPUT_DIR}")
print(f"Période échantillonnage : {SAMPLE_PERIOD_MS} ms")

for record_name in sorted(os.listdir(DATASET_DIR)):
    record_path = os.path.join(DATASET_DIR, record_name)

    if not os.path.isdir(record_path):
        continue

    csv_path = os.path.join(record_path, "labels.csv")

    if not os.path.exists(csv_path):
        print(f"\n[IGNORÉ] {record_name} : labels.csv introuvable")
        continue

    print("\n------------------------------")
    print(f"Record : {record_name}")

    df_original = pd.read_csv(csv_path, sep=";")
    df_original = df_original.sort_values("time_in_ms").reset_index(drop=True)

    df_resampled = resample_commands(df_original, SAMPLE_PERIOD_MS)

    output_dir = os.path.join(OUTPUT_DIR, record_name, "labels")
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "labels.csv")

    df_resampled.to_csv(output_csv, sep=";", index=False)

    rows_created = max(0, len(df_resampled) - len(df_original))
    rows_removed = max(0, len(df_original) - len(df_resampled))

    print(f"Lignes originales       : {len(df_original)}")
    print(f"Lignes rééchantillonnées: {len(df_resampled)}")
    print(f"Lignes créées           : {rows_created}")
    print(f"Lignes supprimées       : {rows_removed}")
    print(f"CSV généré              : {output_csv}")

    analysis_rows.append(
        analyze_resampled_csv(record_name, df_original, df_resampled)
    )

analysis_df = pd.DataFrame(analysis_rows)

analysis_csv = os.path.join(OUTPUT_DIR, "resampling_analysis.csv")
analysis_df.to_csv(analysis_csv, sep=";", index=False)

print("\n==============================")
print("TERMINÉ")
print("==============================")
print(f"Fichier d'analyse : {analysis_csv}")