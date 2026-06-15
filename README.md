# Autonomous Driving Car

This repository now groups the full workflow in one place:

1. prepare raw driving logs and images,
2. rebalance the dataset for 3-class direction prediction,
3. train a classifier,
4. run inference on a single image.

## Project layout

- `GUIDE_UTILISATEUR.md`: detailed French guide for the repository architecture and file roles.
- `Traitement/`: dataset preparation and balancing scripts.
- `Entrainement/`: compatibility entrypoints for the training code that used to live there.
- `actions.py`, `dataset.py`, `training.py`, `inference.py`: shared core modules.
- `train_action_model.py`: train the classifier.
- `predict_action.py`: run inference from one image.

## Direction labels

Raw CSV logs still support the fine-grained labels:

- `forward`
- `light_left`
- `light_right`
- `pivot_left`
- `pivot_right`
- `sharp_left`
- `sharp_right`
- `backward`
- `stop`
- `other`

Training and inference collapse them into 3 model classes:

- `left`
- `forward`
- `right`

## Recommended workflow

### 1. Resample raw records and align them with images

```bash
python Traitement/resample_dataset.py
```

This reads raw records from `dataSet/` and creates aligned samples under `segmentation/`.

### 2. Build a balanced dataset

```bash
python Traitement/build_balanced_dataset.py
```

This creates `dataset_augmente_equilibre/` with `train/`, `valid/` and `test/` splits.

### 3. Train the classifier

```bash
python train_action_model.py
```

You can override the default dataset paths:

```bash
python train_action_model.py --csv-file path/to/labels.csv --image-dir path/to/images
```

### 4. Predict an action for one image

```bash
python predict_action.py path/to/image.png
```

## Dependencies

Install the Python packages listed in `requirements.txt`.

## Notes

- Generated datasets, cached files and model weights are ignored through `.gitignore`.
- The repository keeps backward-compatible entrypoints while moving the real logic into reusable modules.
