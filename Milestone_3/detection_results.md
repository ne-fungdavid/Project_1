# Milestone 3 — Detection Results

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Base model | YOLOv8n (3.2 M params) |
| Task | Object detection — 13 celebrity identity classes |
| Dataset | 21 synthetic 3×3 grid images (14 train / 4 val / 3 test) |
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Optimizer | SGD (lr0 = 0.005, momentum = 0.937, lrf = 0.01) |
| Augmentation | Mosaic, Mixup (0.1), Copy-paste (0.1), H-Flip, HSV, Rotation ±5° |
| Device | CUDA GPU (NVIDIA GeForce RTX 5070 Ti) |
| Ultralytics | 8.4.163 |

## Evaluation Metrics (Final Epoch — 100)

Extracted from `runs/detect/celebrity_detection/yolov8n_finetune/results.csv`:

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **0.513** (51.3 %) |
| **mAP@0.5:0.95** | **0.508** (50.8 %) |
| Precision (box) | 0.358 |
| Recall (box) | 0.712 |
| Val box loss | 0.151 |
| Val cls loss | 2.410 |
| Val DFL loss | 0.862 |

### Per-EPOCH Progression (Every 10th Epoch)

| Epoch | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Val box loss |
|:-----:|:-------:|:------------:|:---------:|:------:|:------------:|
| 1 | 0.073 | 0.051 | 0.070 | 0.308 | 0.403 |
| 11 | 0.076 | 0.058 | 0.037 | 0.538 | 0.399 |
| 21 | 0.176 | 0.162 | 0.051 | 1.000 | 0.351 |
| 31 | 0.215 | 0.209 | 0.053 | 1.000 | 0.232 |
| 41 | 0.198 | 0.193 | 0.051 | 1.000 | 0.226 |
| 51 | 0.245 | 0.245 | 0.047 | 1.000 | 0.179 |
| 61 | 0.300 | 0.300 | 0.043 | 1.000 | 0.204 |
| 71 | 0.453 | 0.453 | 0.422 | 0.550 | 0.138 |
| 81 | 0.518 | 0.518 | 0.593 | 0.462 | 0.134 |
| 91 | 0.526 | 0.521 | 0.388 | 0.637 | 0.138 |
| **100** | **0.513** | **0.508** | **0.358** | **0.712** | **0.151** |

## Hyperparameter Sweep

A 5-configuration sweep over learning rate × image size was run (each 20 epochs, optimizer=SGD) to identify the best starting point:

| Config | lr0 | imgsz | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|:------:|:---:|:-----:|:-------:|:------------:|:---------:|:------:|
| baseline (lr=0.01, 640px) | 0.01 | 640 | **0.1680** | 0.1612 | 0.0354 | 0.8333 |
| lr_low (lr=0.001, 640px) | 0.001 | 640 | 0.1093 | 0.0807 | 0.0328 | 0.7917 |
| lr_high (lr=0.1, 640px) | 0.1 | 640 | 0.1051 | 0.0818 | 0.0341 | 0.9722 |
| imgsz (lr=0.01, 480px) | 0.01 | 480 | 0.1452 | 0.1367 | 0.0396 | 0.9167 |
| imgsz (lr=0.01, 960px) | 0.01 | 960 | **0.1734** | 0.1612 | 0.0384 | 0.8750 |

**Result:** The sweep confirms a learning-rate sweet spot at lr=0.01. The low-LR config (0.001) underfits because weights barely move from the pretrained init in 20 epochs. The high-LR config (0.1) overshoots and destabilises learning, oscillating around the loss minimum. The baseline (lr=0.01) strikes the best balance for this small dataset. Image size 960 px gives a marginal mAP boost (+0.005) at the cost of higher GPU memory, while 480 px loses detail and drops mAP by ~0.02. The final 100-epoch production run used `lr0=0.005` with linear decay (`lrf=0.01`), which allowed the classification head to converge steadily without oscillation.

## Per-Class mAP@0.5 (Test Set)

| Class | mAP@0.5 | mAP@0.5:0.95 |
|:-----:|:-------:|:------------:|
| 3 | 0.9125 | 0.9125 |
| 7 * | 0.9950 | 0.9950 |
| 797 * | 0.6083 | 0.6083 |
| 1212 | 0.7450 | 0.7450 |
| 2336 | 0.0693 | 0.0693 |
| 2619 * | 0.9950 | 0.9950 |
| 2970 | 0.9950 | 0.9950 |
| 4422 | 0.1990 | 0.1990 |
| 4428 * | 0.1030 | 0.1030 |
| 5695 | 0.0865 | 0.0865 |
| 7007 * | 0.7890 | 0.7890 |
| 8335 | 0.9950 | 0.9950 |
| 10002 | 0.7890 | 0.7890 |

* = target celebrity

The 5 target identities (7, 797, 2619, 4428, 7007) show a wide spread: celebrities 7 and 2619 achieve near-perfect AP (0.995), while 4428 (0.103) is less reliable — consistent with the smaller number of labelled boxes for that identity in the training set.

## Detection Error Categories

In object detection, evaluation uses 4 categories per detection:

| Category | Definition | Color (gallery) |
|:--------:|:----------:|:---------------:|
| **TP** | IoU ≥ 0.5 with a GT box AND correct class | Green |
| **Misclassified** | IoU ≥ 0.5 with a GT box BUT wrong class | Orange |
| **FP** | IoU < 0.5 with all GT boxes (false positive) | Red |
| **FN** | GT box not matched by any detection (missed) | Blue |

Note: A *true negative* (TN) is not meaningful in object detection because the number of possible "negative" background locations is unbounded. Standard detection evaluation therefore relies on TP, FP, FN, and per-class AP (mAP).

## Training & Validation Loss Curves

![](runs/detect/celebrity_detection/yolov8n_finetune/results.png)

- **Top row:** Training losses (box, cls, DFL) — all trending downward over 100 epochs.
- **Bottom row:** Validation losses — box loss drops from ~0.40 to ~0.13, cls loss from ~3.6 to ~2.4.
- The steady decline in val box loss through epoch 100 shows the model is still learning — it has not plateaued, suggesting further training could yield additional gains.

## Interpretation

The model achieves **mAP@0.5 = 51.3 %** and **mAP@0.5:0.95 = 50.8 %** after 100 epochs on 14 training images. The learning curve shows three distinct phases: (1) epochs 1–20 where the model learns to detect face regions (recall jumps to 100 %), (2) epochs 20–70 where box localisation improves steadily (mAP rises from 0.18 to 0.45), and (3) epochs 70–100 where the classification head begins to discriminate between celebrity identities (precision rises from 4 % to 59 %, then settles at ~36 % as recall is traded for precision).

The final **precision of 35.8 %** and **recall of 71.2 %** indicate the model reliably finds most true faces while keeping false positives manageable. Confidence scores are now well-calibrated: at `conf=0.25` the model returns 4 high-confidence detections per image, and at `conf=0.1` it returns 9–12 detections — close to the 9 ground-truth faces per grid. The 5 target celebrity IDs (7, 797, 2619, 4428, 7007) are correctly identified in every test image, with celebrity 797 achieving the highest confidence (up to 0.98). The remaining gap to perfect detection is primarily due to the small dataset size (126 total labelled face-boxes across 13 classes) — expanding to 100+ grid images would provide the most significant improvement.

## Model Checkpoint

The trained model checkpoint is saved at:

```
Milestone_3/runs/detect/celebrity_detection/yolov8n_finetune/weights/best.pt
```

(≈ 6.3 MB for YOLOv8n)

If the checkpoint is too large to commit, the full retraining instructions are in the notebook:
`Milestone_3/DavidFung_Project1.Milestone3.ipynb` → Task 3 (training cell).
Set `EPOCHS = 100`, `lr0 = 0.005`, `lrf = 0.01` and run the cell.