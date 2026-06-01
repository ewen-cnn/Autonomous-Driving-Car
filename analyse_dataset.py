import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt

DATASET_DIR = "Records"
OUTPUT_DIR = "analysis_result"

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


def add_examples(df, dataset_path, output_dataset_dir):
    image_dir = os.path.join(dataset_path, "Images")

    if not os.path.exists(image_dir):
        print(f"Pas de dossier Images dans {dataset_path}")
        return

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    n = min(len(df), len(image_files))

    df_examples = df.iloc[:n].copy()
    df_examples["image_path"] = [
        os.path.join(image_dir, img)
        for img in image_files[:n]
    ]

    examples_dir = os.path.join(output_dataset_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    examples = []

    for cls in CLASSES:
        subset = df_examples[df_examples["direction_class"] == cls]

        if len(subset) == 0:
            continue

        samples = subset.sample(
            n=min(3, len(subset)),
            random_state=42
        )

        for i, (_, row) in enumerate(samples.iterrows(), start=1):
            src = row["image_path"]
            ext = os.path.splitext(src)[1]

            dst_name = f"{cls}_{i}{ext}"
            dst = os.path.join(examples_dir, dst_name)

            shutil.copy(src, dst)

            examples.append({
                "class": cls,
                "source_image": src,
                "copied_image": dst
            })

    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(
        os.path.join(output_dataset_dir, "class_examples.csv"),
        sep=";",
        index=False
    )


os.makedirs(OUTPUT_DIR, exist_ok=True)

global_results = []
all_data = []

for dataset_name in sorted(os.listdir(DATASET_DIR)):
    dataset_path = os.path.join(DATASET_DIR, dataset_name)

    if not os.path.isdir(dataset_path):
        continue

    csv_path = os.path.join(dataset_path, "labels.csv")

    if not os.path.exists(csv_path):
        print(f"Pas de labels.csv dans {dataset_name}")
        continue

    print(f"Analyse de {dataset_name}")

    output_dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(output_dataset_dir, exist_ok=True)

    df = pd.read_csv(csv_path, sep=";")
    df["direction_class"] = df.apply(classify_direction, axis=1)

    add_examples(df, dataset_path, output_dataset_dir)

    all_data.append(df)

    counts = df["direction_class"].value_counts()
    percents = df["direction_class"].value_counts(normalize=True) * 100

    summary_df = pd.DataFrame([
        {
            "class": cls,
            "count": int(counts.get(cls, 0)),
            "percent": round(percents.get(cls, 0), 2)
        }
        for cls in CLASSES
    ])

    summary_df.to_csv(
        os.path.join(output_dataset_dir, "direction_distribution.csv"),
        sep=";",
        index=False
    )

    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["class"], summary_df["percent"])
    plt.title(f"Répartition des directions\n{dataset_name}")
    plt.xlabel("Classe de direction")
    plt.ylabel("Pourcentage (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dataset_dir, "direction_distribution.png"), dpi=300)
    plt.close()

    non_zero = summary_df[summary_df["count"] > 0]

    global_results.append({
        "dataset": dataset_name,
        "nb_samples": len(df),
        "majority_class": non_zero.loc[non_zero["percent"].idxmax(), "class"],
        "majority_percent": non_zero["percent"].max(),
        "minority_class": non_zero.loc[non_zero["percent"].idxmin(), "class"],
        "minority_percent": non_zero["percent"].min(),
        "imbalance_ratio": round(
            non_zero["percent"].max() / non_zero["percent"].min(),
            2
        )
    })


global_summary_df = pd.DataFrame(global_results)
global_summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "global_direction_summary.csv"),
    sep=";",
    index=False
)

df_all = pd.concat(all_data, ignore_index=True)

global_counts = df_all["direction_class"].value_counts()
global_percents = df_all["direction_class"].value_counts(normalize=True) * 100

global_distribution = pd.DataFrame([
    {
        "class": cls,
        "count": int(global_counts.get(cls, 0)),
        "percent": round(global_percents.get(cls, 0), 2)
    }
    for cls in CLASSES
])

global_distribution.to_csv(
    os.path.join(OUTPUT_DIR, "global_direction_distribution.csv"),
    sep=";",
    index=False
)

plt.figure(figsize=(10, 5))
plt.bar(global_distribution["class"], global_distribution["percent"])
plt.title("Répartition globale des directions")
plt.xlabel("Classe de direction")
plt.ylabel("Pourcentage (%)")
plt.ylim(0, 100)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "global_direction_distribution.png"), dpi=300)
plt.close()

print("\nAnalyse terminée.")
print(f"Résultats enregistrés dans : {OUTPUT_DIR}")