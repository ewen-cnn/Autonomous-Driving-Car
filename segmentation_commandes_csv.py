import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================

DATASET_DIR = "Records"
RECORD_NAME = "#Record_2025-07-02_09-46-48"

OUTPUT_DIR = "analysis_segments"

MIN_STOP_DURATION_MS = 300

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
# SEGMENTS
# =========================

def build_raw_segments(df):
    segments = []

    for i in range(len(df) - 1):
        start = int(df.loc[i, "time_in_ms"])
        end = int(df.loc[i + 1, "time_in_ms"])

        segments.append({
            "start_time": start,
            "end_time": end,
            "duration_ms": end - start,
            "direction_class": df.loc[i, "direction_class"],
            "speedA": df.loc[i, "speedA"],
            "speedB": df.loc[i, "speedB"],
            "GPIO1": df.loc[i, "GPIO1"],
            "GPIO2": df.loc[i, "GPIO2"],
            "GPIO3": df.loc[i, "GPIO3"],
            "GPIO4": df.loc[i, "GPIO4"],
        })

    return pd.DataFrame(segments)


def merge_consecutive_segments(segments_df):
    if segments_df.empty:
        return segments_df

    merged = []

    current = segments_df.iloc[0].to_dict()

    for i in range(1, len(segments_df)):
        row = segments_df.iloc[i].to_dict()

        same_command = (
            row["direction_class"] == current["direction_class"]
            and row["speedA"] == current["speedA"]
            and row["speedB"] == current["speedB"]
            and row["GPIO1"] == current["GPIO1"]
            and row["GPIO2"] == current["GPIO2"]
            and row["GPIO3"] == current["GPIO3"]
            and row["GPIO4"] == current["GPIO4"]
        )

        if same_command:
            current["end_time"] = row["end_time"]
            current["duration_ms"] = current["end_time"] - current["start_time"]
        else:
            merged.append(current)
            current = row

    merged.append(current)

    return pd.DataFrame(merged)


def remove_short_stop_segments(segments_df, min_stop_duration_ms=300):
    """
    Supprime les stop courts en les fusionnant avec le voisin le plus logique.
    Cas le plus propre :
    - classe avant == classe après => on fusionne les trois.
    Sinon :
    - on rattache le stop court au voisin ayant la plus grande durée.
    """
    segments = segments_df.to_dict("records")
    cleaned = []
    i = 0

    while i < len(segments):
        seg = segments[i]

        is_short_stop = (
            seg["direction_class"] == "stop"
            and seg["duration_ms"] < min_stop_duration_ms
        )

        if not is_short_stop:
            cleaned.append(seg)
            i += 1
            continue

        prev_seg = cleaned[-1] if len(cleaned) > 0 else None
        next_seg = segments[i + 1] if i + 1 < len(segments) else None

        # Cas idéal : même classe avant et après
        if (
            prev_seg is not None
            and next_seg is not None
            and prev_seg["direction_class"] == next_seg["direction_class"]
        ):
            prev_seg["end_time"] = next_seg["end_time"]
            prev_seg["duration_ms"] = prev_seg["end_time"] - prev_seg["start_time"]
            i += 2
            continue

        # Sinon, rattacher au segment précédent s'il est plus long
        if prev_seg is not None and next_seg is not None:
            if prev_seg["duration_ms"] >= next_seg["duration_ms"]:
                prev_seg["end_time"] = seg["end_time"]
                prev_seg["duration_ms"] = prev_seg["end_time"] - prev_seg["start_time"]
                i += 1
            else:
                next_seg["start_time"] = seg["start_time"]
                next_seg["duration_ms"] = next_seg["end_time"] - next_seg["start_time"]
                i += 1
            continue

        # Si pas de voisin précédent, rattacher au suivant
        if prev_seg is None and next_seg is not None:
            next_seg["start_time"] = seg["start_time"]
            next_seg["duration_ms"] = next_seg["end_time"] - next_seg["start_time"]
            i += 1
            continue

        # Si pas de voisin suivant, rattacher au précédent
        if prev_seg is not None and next_seg is None:
            prev_seg["end_time"] = seg["end_time"]
            prev_seg["duration_ms"] = prev_seg["end_time"] - prev_seg["start_time"]
            i += 1
            continue

        i += 1

    cleaned_df = pd.DataFrame(cleaned)

    # Après suppression, on refusionne au cas où deux segments identiques se touchent.
    return merge_consecutive_segments(cleaned_df)


def save_duration_outputs(segments_df, prefix, output_dir):
    segments_df.to_csv(
        os.path.join(output_dir, f"{prefix}_segments.csv"),
        sep=";",
        index=False
    )

    summary_df = segments_df.groupby("direction_class").agg(
        count=("direction_class", "count"),
        mean_duration_ms=("duration_ms", "mean"),
        median_duration_ms=("duration_ms", "median"),
        min_duration_ms=("duration_ms", "min"),
        max_duration_ms=("duration_ms", "max")
    ).reset_index().round(2)

    summary_df.to_csv(
        os.path.join(output_dir, f"{prefix}_duration_summary.csv"),
        sep=";",
        index=False
    )

    plt.figure(figsize=(10, 5))
    plt.hist(segments_df["duration_ms"], bins=50)
    plt.title(f"Distribution des durées - {prefix}")
    plt.xlabel("Durée du segment (ms)")
    plt.ylabel("Nombre de segments")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_duration_histogram.png"), dpi=300)
    plt.close()

    for cls in CLASSES:
        subset = segments_df[segments_df["direction_class"] == cls]

        if len(subset) == 0:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(subset["duration_ms"], bins=30)
        plt.title(f"Durées des segments : {cls} - {prefix}")
        plt.xlabel("Durée du segment (ms)")
        plt.ylabel("Nombre de segments")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_duration_{cls}.png"),
            dpi=300
        )
        plt.close()

    return summary_df


# =========================
# MAIN
# =========================

record_path = os.path.join(DATASET_DIR, RECORD_NAME)
csv_path = os.path.join(record_path, "labels.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

safe_record_name = RECORD_NAME.replace("#", "").replace("/", "_")

df = pd.read_csv(csv_path, sep=";")
df = df.sort_values("time_in_ms").reset_index(drop=True)
df["direction_class"] = df.apply(classify_direction, axis=1)

raw_segments = build_raw_segments(df)
merged_segments = merge_consecutive_segments(raw_segments)
cleaned_segments = remove_short_stop_segments(
    merged_segments,
    min_stop_duration_ms=MIN_STOP_DURATION_MS
)

raw_summary = save_duration_outputs(
    raw_segments,
    f"{safe_record_name}_raw",
    OUTPUT_DIR
)

merged_summary = save_duration_outputs(
    merged_segments,
    f"{safe_record_name}_merged",
    OUTPUT_DIR
)

cleaned_summary = save_duration_outputs(
    cleaned_segments,
    f"{safe_record_name}_cleaned_stop_{MIN_STOP_DURATION_MS}ms",
    OUTPUT_DIR
)

short_stops_before = merged_segments[
    (merged_segments["direction_class"] == "stop")
    & (merged_segments["duration_ms"] < MIN_STOP_DURATION_MS)
]

short_stops_before.to_csv(
    os.path.join(
        OUTPUT_DIR,
        f"{safe_record_name}_short_stops_before_cleaning.csv"
    ),
    sep=";",
    index=False
)

print("\n==============================")
print("RÉSUMÉ")
print("==============================")

print(f"Segments bruts : {len(raw_segments)}")
print(f"Segments après fusion commandes identiques : {len(merged_segments)}")
print(f"Segments après suppression stop < {MIN_STOP_DURATION_MS} ms : {len(cleaned_segments)}")

print(f"\nStops courts supprimés/fusionnés : {len(short_stops_before)}")

print("\nRésumé après nettoyage :")
print(cleaned_summary)

print("\nPremiers segments nettoyés :")
print(
    cleaned_segments[
        ["start_time", "end_time", "duration_ms", "direction_class", "speedA", "speedB"]
    ].head(30)
)

print("\nAnalyse terminée.")
print(f"Fichiers générés dans : {OUTPUT_DIR}")