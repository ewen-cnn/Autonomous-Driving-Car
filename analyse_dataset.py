import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET_DIR = "Records"
OUTPUT_DIR = "analysis_result"

CLASSES = ["forward", "left", "right", "stop", "backward", "other"]


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


def classify_action(row):
    speedA = row["speedA"]
    speedB = row["speedB"]

    left_dir = motor_direction(row["GPIO1"], row["GPIO2"], "left")
    right_dir = motor_direction(row["GPIO3"], row["GPIO4"], "right")

    if left_dir == "stop" and right_dir == "stop":
        return "stop"

    if left_dir == "backward" or right_dir == "backward":
        return "backward"

    if left_dir == "unknown" or right_dir == "unknown":
        return "other"

    if left_dir == "forward" and right_dir == "forward":
        if speedA > speedB:
            return "right"
        elif speedB > speedA:
            return "left"
        else:
            return "forward"

    if left_dir == "forward" and right_dir == "stop":
        return "right"

    if right_dir == "forward" and left_dir == "stop":
        return "left"

    return "other"


os.makedirs(OUTPUT_DIR, exist_ok=True)

global_results = []

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
    df["action"] = df.apply(classify_action, axis=1)

    counts = df["action"].value_counts()
    percents = df["action"].value_counts(normalize=True) * 100

    rows = []

    for cls in CLASSES:
        rows.append({
            "class": cls,
            "count": int(counts.get(cls, 0)),
            "percent": round(percents.get(cls, 0), 2)
        })

    summary_df = pd.DataFrame(rows)

    csv_output_path = os.path.join(output_dataset_dir, "class_distribution.csv")
    summary_df.to_csv(csv_output_path, sep=";", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["class"], summary_df["percent"])
    plt.title(f"Répartition des classes\n{dataset_name}")
    plt.xlabel("Classe")
    plt.ylabel("Pourcentage (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()

    graph_output_path = os.path.join(output_dataset_dir, "class_distribution.png")
    plt.savefig(graph_output_path, dpi=300)
    plt.close()

    global_results.append({
        "dataset": dataset_name,
        "nb_samples": len(df),
        "majority_class": summary_df.loc[summary_df["percent"].idxmax(), "class"],
        "majority_percent": summary_df["percent"].max(),
        "minority_class": summary_df[summary_df["count"] > 0].loc[
            summary_df[summary_df["count"] > 0]["percent"].idxmin(), "class"
        ],
        "minority_percent": summary_df[summary_df["count"] > 0]["percent"].min(),
        "imbalance_ratio": round(
            summary_df["percent"].max()
            / summary_df[summary_df["count"] > 0]["percent"].min(),
            2
        )
    })


global_summary_df = pd.DataFrame(global_results)
global_summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "global_summary.csv"),
    sep=";",
    index=False
)

print("\nAnalyse terminée.")
print(f"Résultats enregistrés dans : {OUTPUT_DIR}")