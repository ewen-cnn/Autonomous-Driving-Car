# =========================
# CONFIGURATION
# =========================

DATASET_DIR = "dataSet"
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

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}
