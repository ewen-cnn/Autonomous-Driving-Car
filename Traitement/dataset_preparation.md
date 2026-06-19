# dataset_augmente_equilibre preparation

The repository now separates raw acquisition, preparation and model training.

## Raw data

Expected raw records live under `dataset_augmente_equilibre/` and each record should contain:

- `labels.csv`
- `Images/`

## Step 1: resampling and image alignment

Run:

```bash
python Traitement/resample_dataset_augmente_equilibre.py
```

This resamples motor commands at a fixed period and associates each row with the latest available image.

## Step 2: balanced dataset_augmente_equilibre generation

Run:

```bash
python Traitement/build_balanced_dataset_augmente_equilibre.py
```

This creates `dataset_augmente_equilibre_augmente_equilibre/` with `train/`, `valid/` and `test/` splits.

## Step 3: distribution inspection

Run:

```bash
python Traitement/class_repartition_analysis.py segmentation
```

This prints per-record and global direction counts.
