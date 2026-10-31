# Disease Detection — ML Module

## What this module predicts

This module analyses an image of an arecanut plant or leaf and predicts
whether the plant is **healthy** or affected by a **disease**. It also
returns a **confidence score** (0–100%) for the prediction.

The exact list of disease classes (e.g. Yellow Leaf Disease, Bud Rot,
Koleroga / Mahali, etc.) will be finalised once the dataset is collected.

---

## Expected image input

| Property     | Details                                      |
|--------------|----------------------------------------------|
| Subject      | Arecanut plant, leaf, or close-up of nut     |
| Format       | JPG / PNG / WEBP                             |
| Recommended size | 224 × 224 px (model input size, TBC)   |
| Colour space | RGB                                          |

Images should be clear and well-lit. Blurry or heavily occluded images
will reduce prediction accuracy.

---

## Expected output

```json
{
  "predicted_class": "Yellow Leaf Disease",
  "confidence": 91.4,
  "all_scores": {
    "Healthy": 3.2,
    "Yellow Leaf Disease": 91.4,
    "Bud Rot": 5.4
  }
}
```

> Note: Class names and count will be updated once the dataset structure
> is confirmed.

---

## Where to place the dataset

Place your dataset inside the `dataset/` folder in this format:

```
ml/disease/dataset/
└── <suggested structure — update once dataset is obtained>
    ├── Healthy/
    │   ├── image_001.jpg
    │   └── ...
    ├── DiseaseClass1/
    │   ├── image_001.jpg
    │   └── ...
    └── DiseaseClass2/
        └── ...
```

Each subfolder name becomes a class label. The training script will read
this structure automatically using `ImageDataGenerator` or a similar
data loader.

**Do not add dataset images to version control.** Add `ml/disease/dataset/`
to `.gitignore`.

---

## Where the trained model will be saved

After training, the model file will be saved to:

```
ml/disease/models/disease_model.h5
```

This file is then referenced by the Flask backend at:

```
backend/models/disease_model.h5
```

or loaded directly from this path — to be decided when integrating
the model into the API.

---

## Training results

Plots, logs, and evaluation reports generated during training will be
saved to:

```
ml/disease/results/
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
