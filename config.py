# dataset_augmente_equilibre locations
dataset_DIR = "dataset_augmente_equilibre"
OUTPUT_DIR = "segmentation"
BALANCED_dataset_DIR = "dataset_augmente_equilibre"
ANALYSIS_DIR = "analysis_result"

# Model defaults
DEFAULT_MODEL_PATH = "driving_cnn.pth"
MULTITASK_MODEL_PATH = "MULTITASK_DRIVING_CNN.pth"
DEFAULT_IMAGE_SIZE = (120, 160)
SAMPLE_PERIOD_MS = 250

# Raw direction labels found in CSV exports.
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
    "other",
]

# Training/inference uses a simplified 3-class decision space.
MODEL_CLASSES = ["left", "forward", "right"]

RAW_CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}
CLASS_TO_IDX = {name: index for index, name in enumerate(MODEL_CLASSES)}
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
