# IE7615 Project 1 — Celebrity Detection

**Group:** David Fung (NUID: 003105587)

A discriminative computer-vision pipeline for **celebrity identification and detection**. Across three graded milestones the project (1) curates a small, diverse set of celebrity identities, (2) builds a synthetic **multi-face** detection dataset, and (3) fine-tunes and evaluates a YOLOv8 detector on it. The dataset and the notebook for every step are committed to this repository, so the pipeline is reproducible end-to-end from a fresh clone.

## Project Overview

| Milestone | Folder | Goal |
|:---------:|--------|------|
| **1** | `Milestone_1/` | Select a diverse subset of celebrity identities from CelebA and build a single-face classification baseline. |
| **2** | `Milestone_2/` | Construct a synthetic multi-face detection dataset (2–5 composite faces per image) with YOLO-format annotations and a train/val/test split. |
| **3** | `Milestone_3/` | Fine-tune YOLOv8 on the synthetic dataset, visualise training, and evaluate on the held-out test split. |

The identity-selection rationale is also captured in `DavidFung_Project1_Selection.ipynb` and `Milestone_1/project_1_proposal.pdf`.

## Repository Structure

```
.
├── README.md
├── .gitignore
├── DavidFung_Project1_Selection.ipynb        # Celebrity subset selection
├── Milestone_1/
│   ├── DavidFung_Project1.Milestone1.ipynb   # Selection + single-face classification
│   ├── project_1_proposal.pdf
│   └── Project_1_Milestone_1_Summary.pdf
├── Milestone_2/
│   └── DavidFung_Project1.Milestone2.ipynb   # Synthetic multi-face dataset construction
├── Milestone_3/
│   └── DavidFung_Project1.Milestone3.ipynb   # YOLOv8 fine-tuning + evaluation
├── Milestone_4/                              # (reserved)
├── data/
│   ├── identity_CelebA.txt                   # CelebA identity reference list
│   ├── selected_images/                      # The celebrity subset (one folder per identity)
│   │   ├── 7/    797/    2619/    4428/    7007/
│   └── synthetic_multi_face/                 # Detection dataset (YOLO format)
│       ├── data.yaml
│       ├── images/{train,val,test}/
│       ├── labels/{train,val,test}/
│       └── temp/{images,labels}/             # intermediate, git-ignored (regenerated in M2)
├── models/                                   # (reserved for exported models)
└── weights/
    └── yolo26n.pt
```

## Celebrity Selection and Identities

For the initial phase of the CelebA classification pipeline, a subset of **five distinct identities** was selected. The selection process was governed by three primary criteria: **visual diversity, demographic representation and dataset balance**.

| Class | Identity | Images |
|:-----:|----------|:------:|
| 0 | ID 7 — A. R. Rahman | 24 |
| 1 | ID 797 — Aria Crescendo | 25 |
| 2 | ID 2619 — Dmitry Medvedev | 25 |
| 3 | ID 4428 — Jenna Fischer | 23 |
| 4 | ID 7007 — Monika Brodka | 24 |

**Total: 121 single-face source images** (stored under `data/selected_images/<ID>/`).

The chosen identities represent a wide spectrum of facial features so the model learns robust, generalised characteristics rather than specific demographic biases. By including individuals with varying hair colors (ranging from blonde to black), different skin tones and diverse facial structures (encompassing a range of masculine and feminine features), the network is given a high-contrast training environment. This diversity is crucial for the model to learn "face-specific" features — such as eye shape and bone structure — rather than relying on easy-to-memorise features like hair color or skin tone.

## Dataset

Two datasets live under `data/`:

1. **`data/selected_images/`** — the raw celebrity subset above, one folder per identity ID. This is the source material for the Milestone 1 classifier and the Milestone 2 compositing step.
2. **`data/synthetic_multi_face/`** — the YOLO detection dataset used to train and evaluate the detector (Milestones 2 & 3):
   - Composite **640×640** images, each containing **2–5 celebrity faces** pasted onto a random colour background.
   - A YOLO-format label file (`class cx cy w h`, normalised) alongside every image.
   - **70 / 15 / 15** split into `images/train`, `images/val`, `images/test` (**140 / 30 / 30** images), with matching `labels/` folders.
   - `data.yaml` declares **5 classes** (`nc: 5`) keyed by identity ID (`7, 797, 2619, 4428, 7007`).
   - `temp/{images,labels}/` holds the pre-split composites; it is a regenerable intermediate and is excluded from version control.

## Environment

| Package | Version |
|---------|---------|
| Python | 3.10.21 |
| PyTorch | 2.14.0+cu132 |
| Ultralytics (YOLOv8) | 8.4.163 |
| TensorFlow | 2.13.1 |

A CUDA-capable GPU was used, the code also runs on CPU.

## How to Reproduce the Training Runs

### 1. Clone & install

```bash
git clone https://github.com/ne-fungdavid/Project_1.git
cd Project_1
pip install ultralytics==8.4.163      # pulls in a compatible torch
```

### 2. (Optional) Rebuild the celebrity subset — Milestone 1

Run top-to-bottom:
- `DavidFung_Project1_Selection.ipynb` — reproduces the identity selection.
- `Milestone_1/DavidFung_Project1.Milestone1.ipynb` — single-face classification baseline.

### 3. Build the synthetic multi-face dataset — Milestone 2

Run `Milestone_2/DavidFung_Project1.Milestone2.ipynb` top-to-bottom. It regenerates 200 composite 640×640 images (2–5 distinct faces per frame) with YOLO-format labels, then splits them into train/val/test and writes `data.yaml`.

**Split** — 70 / 15 / 15 (train / val / test) → **140 / 30 / 30** images. Split is seeded (`GLOBAL_SEED = 42`) so it is reproducible, and every identity appears in every partition (identity-balanced, no leakage).

**Augmentation strategy (Milestone 3 training)**

All augmentations are chosen to be *detection-safe* — they transform the frame in a way that maps bounding boxes deterministically, so labels remain correct:

| Augmentation | Parameter | Rationale |
|:------------:|:---------:|-----------|
| Horizontal flip | `hflip=0.5` | Boxes mirror symmetrically |
| Mosaic | `mosaic=1.0` | 4-image composite with box rescaling — matches the multi-face task |
| HSV jitter | `h=0.015, s=0.7, v=0.4` | Mild colour/lighting shift; boxes unaffected |
| Rotation | `±5°` | Small pose variation; boxes rotate with the frame |
| Scale | `0.5` | Zoom in/out; boxes rescale proportionally |
| Mixup | `mixup=0.1` | Low-rate blend avoids blurring box edges |
| Copy-paste | `copypaste=0.1` | Injects extra faces *with* their boxes, raising recall on small faces |

Large rotations or aggressive geometric warps are deliberately avoided because they stretch tight face boxes into shapes inconsistent with the YOLO rectangle format.

**Compositing-time variation (baked in at generation)**

Each face crop is augmented *before* being pasted onto the background: random horizontal flip (50 %), scale to 13–35 % of canvas height, brightness jitter (0.85–1.15), and rotation (±10°). This gives the synthetic set the same visual diversity that the runtime augmentations would produce, without needing a separate augmentation pass at inference time.

### 4. Fine-tune & evaluate — Milestone 3

To be updated.

### 5. Outputs

To be updated.

## Notes

- `yolov8n.pt` is downloaded automatically on the first run if it is not already present.
- Set `EPOCHS` to a small value (8-10) to sanity-check the pipeline before a full 30 to 50-epoch run.
- `data/` is committed in full (except the regenerable `temp/` intermediates) so the pipeline reproduces without re-downloading the source CelebA images.