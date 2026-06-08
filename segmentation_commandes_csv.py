import os
import pandas as pd

DATASET_DIR = "Record_V2"
OUTPUT_DIR = "segmentation"
SAMPLE_PERIOD_MS = 250


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

    # GPIO : maintien de l’état précédent
    gpio_cols = ["GPIO1", "GPIO2", "GPIO3", "GPIO4"]
    resampled[gpio_cols] = resampled[gpio_cols].ffill()

    # On ne garde que les temps rééchantillonnés
    resampled = resampled.loc[new_times].reset_index()
    resampled = resampled.rename(columns={"index": "time_in_ms"})

    # Nettoyage des types
    resampled["speedA"] = resampled["speedA"].round().clip(0, 100).astype(int)
    resampled["speedB"] = resampled["speedB"].round().clip(0, 100).astype(int)
    resampled[gpio_cols] = resampled[gpio_cols].astype(int)

    return resampled


os.makedirs(OUTPUT_DIR, exist_ok=True)

for record_name in sorted(os.listdir(DATASET_DIR)):
    record_path = os.path.join(DATASET_DIR, record_name)

    if not os.path.isdir(record_path):
        continue

    csv_path = os.path.join(record_path, "labels.csv")

    if not os.path.exists(csv_path):
        print(f"Pas de labels.csv dans {record_name}")
        continue

    print(f"Traitement : {record_name}")

    df = pd.read_csv(csv_path, sep=";")
    df_resampled = resample_commands(df, SAMPLE_PERIOD_MS)

    output_dir = os.path.join(OUTPUT_DIR, record_name, "labels")
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "labels.csv")

    df_resampled.to_csv(output_csv, sep=";", index=False)

    print(f"  {len(df)} lignes originales -> {len(df_resampled)} lignes rééchantillonnées")

print("\nRééchantillonnage terminé.")