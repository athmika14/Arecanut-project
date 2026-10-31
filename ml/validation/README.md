# Arecanut Image Validator

## Purpose

A binary classifier that determines whether an uploaded image is likely to
contain an arecanut plant, leaf, nut, trunk, or plant part (`arecanut`) or
something unrelated (`not_arecanut`).

This validator is used as a **content guard** before the nine-class disease
classifier runs. If the validator rejects an image (e.g., a photo of a baby,
a building, or a car), the user sees "Invalid image" instead of a meaningless
disease prediction.

## Architecture

- **Backbone**: MobileNetV2 (ImageNet pretrained, `include_top=False`)
- **Head**: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(1, sigmoid)
- **Output**: 1 = arecanut, 0 = not_arecanut
- **Loss**: binary crossentropy
- **Training**: Two-phase transfer learning (frozen backbone, then fine-tune top 30 layers)

## Dataset Structure

### Positive class (`arecanut`)

All 12,292 images from the nine-class arecanut disease dataset are used as
positive examples. These images are **not copied** — the training manifests
reference the original files at:

```
ml/disease/dataset/Arecanut_dataset/Arecanut_dataset/{train,test}/<class>/*.jpg
```

Each image's `source_id` (extracted from its filename) is used for
leakage-free source-group splitting. There are 1,226 unique source groups.

### Negative class (`not_arecanut`) — YOU MUST FILL THESE

Create subfolders under `ml/validation/dataset/not_arecanut/` and place JPEG
or PNG images inside each one. Aim for **at least 300 images total** across
the categories below. More is better.

| Subfolder       | What to put inside                              | Minimum |
|-----------------|-------------------------------------------------|---------|
| `people/`       | Photos of people, faces, babies, crowds         | 50      |
| `animals/`      | Dogs, cats, birds, insects, livestock           | 40      |
| `other_plants/` | Non-arecanut plants, leaves, flowers, trees     | 50      |
| `buildings/`    | Houses, walls, interiors, rooms                 | 30      |
| `vehicles/`     | Cars, bikes, buses, trucks                      | 30      |
| `food/`         | Plates, dishes, fruits, cooked food             | 30      |
| `objects/`      | Furniture, electronics, tools, books            | 40      |
| `scenery/`      | Sky, mountains, water, landscapes              | 30      |

**Rules for collecting negative images:**

1. Use real photographs — not illustrations or AI-generated images.
2. Vary the sources: different cameras, angles, lighting, backgrounds.
3. Do not upload near-duplicates (e.g., 50 photos of the same chair from
   slightly different angles). The audit script will detect exact
   duplicates via MD5 hashing.
4. JPG and PNG only. No GIF, BMP, or TIFF.
5. Each image should be at least 100×100 pixels.

## Pipeline Steps

### Step 1: Audit the dataset

```bash
cd backend
python -m ml.validation.training.audit_dataset
```

This scans positive images from the disease dataset manifest and negative
images from the `not_arecanut/` folders, reports per-category counts,
detects exact duplicates, and checks for class imbalance.

### Step 2: Prepare the split

```bash
cd backend
python -m ml.validation.training.prepare_split
```

Creates source-group-stratified train/validation/test manifests (70/15/15)
for positives and hash-deduplicated random splits for negatives. Verifies
no source group crosses partition boundaries.

### Step 3: Train the validator

```bash
cd backend
python -m ml.validation.training.train_validator
```

Trains the MobileNetV2 binary classifier in two phases, saves the best
model to `ml/validation/models/arecanut_validator.keras` and the class
mapping to `ml/validation/models/validator_class_indices.json`.

### Step 4: Evaluate the validator

```bash
cd backend
python -m ml.validation.training.evaluate_validator
```

Loads the best model, evaluates on the test partition, reports accuracy,
precision, recall, F1-score, confusion matrix, and per-class results.

## Integration with the Disease API

After the validator is trained, the content guard in `routes/disease.py`
can be updated to use this model instead of the ImageNet-pretrained
heuristic. This is a separate step — the current disease API continues
to work with the existing heuristic guard until then.

## Limitations

- The validator cannot distinguish arecanut from other palm species or
  similar-looking tropical plants. It only distinguishes "plant-like
  agricultural image" from "clearly unrelated image".
- Performance depends on the diversity and quantity of negative images
  collected. If negatives are too few or too homogeneous, the model
  will overfit to the training distribution.
- The validator should be re-evaluated periodically with new field images.
