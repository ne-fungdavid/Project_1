# IE7615 Project 1 — Celebrity Detection

**Group:** David Fung (NUID: 003105587)

A discriminative computer-vision pipeline for **celebrity identification and detection**. Across four graded milestones the project (1) curates a small, diverse set of celebrity identities and builds a single-face **ResNet50** classifier, (2) constructs a synthetic **grid-based multi-face** detection dataset (9 distinct faces per frame), (3) fine-tunes and evaluates a **YOLOv8** detector on it, and (4) integrates the two into an end-to-end pipeline with a final results gallery and report. The dataset, notebooks, trained checkpoints (or retraining scripts), and every figure are committed to this repository, so the pipeline is reproducible end-to-end from a fresh clone.

## Project Overview

| Milestone | Folder | Goal |
|:---------:|--------|------|
| **1** | `Milestone_1/` | Select a diverse subset of celebrity identities from CelebA and build a single-face classification baseline. |
| **2** | `Milestone_2/` | Construct a synthetic 3×3 grid detection dataset (9 distinct celebrity faces per image) with YOLO-format annotations and a train/val/test split. |
| **3** | `Milestone_3/` | Fine-tune YOLOv8 on the synthetic dataset, visualise training, run a hyperparameter sweep, and evaluate on the held-out test split. |
| **4** | `Milestone_4/` | Integrate classification + detection into one pipeline, curate the final results gallery, and write the final report (incl. ethical considerations). |

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
│   └── DavidFung_Project1.Milestone2.ipynb   # Synthetic 3×3 grid dataset construction
├── Milestone_3/
│   ├── DavidFung_Project1.Milestone3.ipynb   # YOLOv8 fine-tuning + evaluation notebook
│   ├── milestone3_demo.py                    # Standalone detection demo script
│   ├── generate_gallery.py                   # Gallery image generator
│   ├── detection_results.md                  # Metrics, sweep summary, interpretation
│   ├── gallery/                              # 11 annotated detection examples + captions
│   ├── demo_output/                          # (generated) annotated test images
│   └── runs/                                 # (generated) training runs, weights, plots
├── Milestone_4/
│   ├── final_report.md                       # Final report (6–8 pp; export to PDF)
│   ├── celebrity_pipeline.py                 # Integrated detection + identification pipeline
│   ├── train_classifier.py                   # Retrain & save ResNet50 classifier
│   ├── generate_gallery.py                   # Final results gallery generator
│   ├── models/
│   │   └── resnet50_classifier.keras         # Saved classifier (~130 MB, local-only / retrainable)
│   ├── gallery/                              # Final results (25 correct + 8 errors + 11 detection)
│   └── pipeline_output/                      # (generated) annotated pipeline outputs
├── data/
│   ├── identity_CelebA.txt                   # CelebA identity reference list
│   ├── selected_images/                      # Milestone 1 celebrity subset (5 identities)
│   ├── class_images/                         # 13 celebrity identities (source for M2 grid)
│   │   ├── 3/  7/  797/  1212/  2336/  2619/  2970/  4422/  4428/  5695/  7007/  8335/  10002/
│   └── synthetic_grid/                       # Detection dataset (YOLO format, 3×3 grid)
│       ├── data.yaml
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── models/                                   # (reserved for exported models)
└── weights/
    └── yolo26n.pt
```

## Celebrity Selection and Identities

For Milestone 1, a subset of **five distinct identities** was selected for the single-face classification baseline. For Milestone 2, an expanded pool of **13 identities** is used to populate the 3×3 grid (9 per image, drawn without replacement). The selection criteria are **visual diversity, demographic representation and dataset balance**.

**Milestone 1 (classification):** 5 identities, 121 face crops (stored under `data/selected_images/<ID>/`).

**Milestone 2 (detection grid):** 13 identities, ≈280 face crops (stored under `data/class_images/<ID>/`). Nine of the 13 are randomly selected per grid image, so every frame shows 9 distinct celebrities.

The chosen identities represent a wide spectrum of facial features so the model learns robust, generalised characteristics rather than specific demographic biases. By including individuals with varying hair colors (ranging from blonde to black), different skin tones and diverse facial structures (encompassing a range of masculine and feminine features), the network is given a high-contrast training environment. This diversity is crucial for the model to learn "face-specific" features — such as eye shape and bone structure — rather than relying on easy-to-memorise features like hair color or skin tone.

## Dataset

Three datasets live under `data/`:

1. **`data/selected_images/`** — the Milestone 1 celebrity subset (5 identities, 121 crops).
2. **`data/class_images/`** — the expanded pool of **13 identities** (≈280 crops) used as source material for the Milestone 2 grid compositing.
3. **`data/synthetic_grid/`** — the YOLO detection dataset used to train and evaluate the detector (Milestones 2 & 3):
   - **640×640** grid images, each a **3×3 layout of 9 distinct celebrity faces** on a varied background.
   - A YOLO-format label file (`class cx cy w h`, normalised) alongside every image — one box per face.
   - **70 / 15 / 15** split into `images/train`, `images/val`, `images/test` (**14 / 3 / 3** images), with matching `labels/` folders.
   - `data.yaml` declares **13 classes** (`nc: 13`) keyed by identity ID.
   - No source image file is ever reused across the dataset (global `used_paths` tracking).

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

### 3. Build the synthetic grid dataset — Milestone 2

Run `Milestone_2/DavidFung_Project1.Milestone2.ipynb` top-to-bottom. It generates 20 composite 640×640 grid images (each a 3×3 layout of 9 distinct celebrity faces) with YOLO-format labels, then splits them into train/val/test and writes `data.yaml`.

**Grid construction:**
- 9 of the 13 available identities are selected per image (without replacement).
- One random face crop is drawn from each identity's pool; a global `used_paths` set ensures no image file is ever reused across the dataset.
- Per-face augmentation: 50 % horizontal flip, brightness jitter (0.85–1.15), scale to 80–95 % of cell size.
- Each face is placed in its grid cell with a ±5 % random offset to break the perfectly-regular pattern.
- Background varies per image: gradient (50 %), blurred noise (30 %), or blurred face (20 %).
- The identity-to-cell assignment is shuffled per image, so the same 9 celebrities appear in different positions across the dataset.

**Split** — 70 / 15 / 15 (train / val / test) → **14 / 3 / 3** images. Seeded (`GLOBAL_SEED = 42`), reproducible, identity-balanced.

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

Each face crop is augmented *before* being pasted into its grid cell: random horizontal flip (50 %), brightness jitter (0.85–1.15), and scale to 80–95 % of cell dimensions. This bakes in visual diversity directly, so the synthetic set already covers a range of orientations and lighting conditions without needing a separate augmentation pass at inference time.

### 4. Fine-tune & evaluate — Milestone 3

The full pipeline is in `Milestone_3/DavidFung_Project1.Milestone3.ipynb`. Run cells top-to-bottom.

**Key hyperparameters** (selected via sweep): `lr0=0.005`, `lrf=0.01`, `imgsz=640`, `batch=16`, `epochs=100`.

**Standalone demo** (no notebook required):

```bash
cd Milestone_3
python milestone3_demo.py                # run on all test images, display in window
python milestone3_demo.py --save         # save annotated images to demo_output/
python milestone3_demo.py --image my.jpg # run on a single image
python milestone3_demo.py --conf 0.05    # lower confidence threshold
```

**Regenerate the gallery:**

```bash
cd Milestone_3
python generate_gallery.py               # writes 11 annotated images to gallery/
```

### 5. Integrate & Demo — Milestone 4

```bash
cd Milestone_4

# Step 1: Train and save the classifier (if not already saved)
python train_classifier.py

# Step 2: Run the integrated pipeline
python celebrity_pipeline.py                          # run on all test images
python celebrity_pipeline.py --image photo.jpg        # run on a specific image
python celebrity_pipeline.py --image photo.jpg --save # save annotated output

# Step 3: Generate the final results gallery
python generate_gallery.py
```

**Pipeline:** Input Image → YOLOv8 (detect faces, 13-class) → ResNet50 (identify each face, 5-class) → Annotated output. Any face the identifier is not confident about falls back to the detector's label, so a box is always returned — important for the live demo on unseen images.

> **Classifier checkpoint.** The 5-class ResNet50 is retrained with `python train_classifier.py`
> (≈ 90 s) and saved to `Milestone_4/models/resnet50_classifier.keras`. That file is ~130 MB
> (over GitHub's 100 MB file limit), so it is kept locally and **not** pushed; the retraining
> script is committed, satisfying the "checkpoint **or** retraining script" requirement. The
> YOLOv8n detector checkpoint (`best.pt`, ~6 MB) **is** committed under `Milestone_3/runs/...`.

### 6. Outputs

| File | Description |
|------|-------------|
| `Milestone_3/runs/.../best.pt` | Trained YOLOv8n model checkpoint (~6.5 MB) |
| `Milestone_3/detection_results.md` | Full results document (metrics, sweep, interpretation) |
| `Milestone_3/gallery/` | 11 side-by-side prediction vs GT annotated images |
| `Milestone_4/models/` | Saved ResNet50 classifier + weights |
| `Milestone_4/gallery/` | Final results (25 correct + 8 errors + 11 detection) |
| `Milestone_4/pipeline_output/` | Annotated pipeline outputs |
| `requirements.txt` | Full dependency list |

## Notes

- `yolov8n.pt` is downloaded automatically on the first run if it is not already present.
- Set `EPOCHS` to a small value (8-10) to sanity-check the pipeline before a full 30 to 50-epoch run.
- `data/` is committed in full so the pipeline reproduces without re-downloading the source CelebA images.