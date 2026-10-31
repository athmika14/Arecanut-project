# Harvest Readiness — ML Module

## What this module predicts

This module analyses an image of an **arecanut bunch** and predicts its
**maturity stage** — whether the bunch is ready to harvest, not yet ready,
or at an intermediate stage (if the dataset supports a three-class split).

Getting the harvest timing right reduces crop loss, improves nut quality,
and increases farmer income.

---

## Expected image input

| Property     | Details                                           |
|--------------|---------------------------------------------------|
| Subject      | Arecanut bunch on the tree or freshly cut         |
| Format       | JPG / PNG / WEBP                                  |
| Recommended size | 224 × 224 px (model input size, TBC)          |
| Colour space | RGB                                               |

Images should clearly show the bunch and nut colour. Colour is a key
indicator of maturity, so good lighting matters.

---

## Expected output

Two-class version (if dataset only supports two stages):

```json
{
  "predicted_class": "Ready",
  "confidence": 87.3,
  "all_scores": {
    "Not Ready": 12.7,
    "Ready": 87.3
  }
}
```

Three-class version (if dataset supports an intermediate stage):

```json
{
  "predicted_class": "Intermediate",
  "confidence": 74.1,
  "all_scores": {
    "Not Ready": 10.5,
    "Intermediate": 74.1,
    "Ready": 15.4
  }
}
```

> Note: The number of classes will be finalised once the dataset is
> collected and examined.

---

## Where to place the dataset

Place your dataset inside the `dataset/` folder in this format:

```
ml/harvest/dataset/
└── <suggested structure — update once dataset is obtained>
    ├── NotReady/
    │   ├── image_001.jpg
    │   └── ...
    ├── Intermediate/       ← include only if dataset supports this class
    │   ├── image_001.jpg
    │   └── ...
    └── Ready/
        ├── image_001.jpg
        └── ...
```

Each subfolder name becomes a class label. The training script will read
this structure automatically using `ImageDataGenerator` or a similar
data loader.

**Do not add dataset images to version control.** Add `ml/harvest/dataset/`
to `.gitignore`.

---

## Where the trained model will be saved

After training, the model file will be saved to:

```
ml/harvest/models/harvest_model.h5
```

This file is then referenced by the Flask backend at:

```
backend/models/harvest_model.h5
```

or loaded directly from this path — to be decided when integrating
the model into the API.

---

## Training results

Plots, logs, and evaluation reports generated during training will be
saved to:

```
ml/harvest/results/
```

Expected artifacts:
- `accuracy_loss_plot.png` — training vs validation accuracy/loss curves
- `confusion_matrix.png` — per-class prediction breakdown
- `classification_report.txt` — precision, recall, F1 per class

---

## Status

| Step                        | Status      |
|-----------------------------|-------------|
| Dataset collection          | ⏳ Pending  |
| Dataset preprocessing       | ⏳ Pending  |
| Model architecture design   | ⏳ Pending  |
| Training script             | ⏳ Pending  |
| Model evaluation            | ⏳ Pending  |
| Backend integration         | ⏳ Pending  |
