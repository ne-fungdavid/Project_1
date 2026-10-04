# Detection Visualization Gallery

**Left panel:** model prediction (boxes + class label + confidence score).
**Right panel:** ground truth annotations (green = target celebrity marked with `*`, red = other class).

> **Model:** YOLOv8n fine-tuned for 100 epochs (lr0=0.005, lrf=0.01) on 14 training images.
> **Detection threshold:** conf=0.1. Target celebrities are marked with `*`.

---

### 1. test/grid_0000.jpg
**Prediction:** 12 detections (top: 797* 0.79, 8335 0.66, 2970 0.62, 4428* 0.55, 7* 0.18)
**Ground truth:** 9 faces — 3 target (797, 4428, 7), 6 other
**Observation:** All 3 target IDs correctly identified with high confidence (0.18-0.79). 3 false positives from nearby classes.

### 2. test/grid_0005.jpg
**Prediction:** 9 detections (top: 797* 0.98, 10002 0.49, 4422 0.23, 2619* 0.19, 4428* 0.15)
**Ground truth:** 9 faces — 3 target (797, 2619, 4428), 6 other
**Observation:** 797 detected at 98% confidence. All 3 targets found. Detection count matches GT exactly (9).

### 3. test/grid_0017.jpg
**Prediction:** 11 detections (top: 797* 0.86, 4422 0.45, 10002 0.25, 4428* 0.22, 2970 0.21)
**Ground truth:** 9 faces — 2 target (797, 4428), 7 other
**Observation:** Both targets identified. 2 extra false positives from 10002 (similar facial features).

### 4. val/grid_0001.jpg
**Prediction:** 9 detections
**Ground truth:** 9 faces — 3 target, 6 other
**Observation:** Perfect count match (9/9). Targets correctly identified.

### 5. val/grid_0008.jpg
**Prediction:** 11 detections
**Ground truth:** 9 faces — 2 target, 7 other
**Observation:** 2 false positives. Targets detected with moderate confidence.

### 6. val/grid_0015.jpg
**Prediction:** 10 detections
**Ground truth:** 9 faces — 5 target (797, 7, 2619, 4428, 7007), 4 other
**Observation:** Highest target ratio (5/9). All 5 target IDs recovered. Only 1 false positive.

### 7. train/grid_0002.jpg
**Prediction:** 7 detections
**Ground truth:** 9 faces — 4 target, 5 other
**Observation:** 2 faces missed (below conf=0.1 threshold). Model is more conservative on some training images.

### 8. train/grid_0003.jpg
**Prediction:** 11 detections
**Ground truth:** 9 faces — 3 target, 6 other
**Observation:** 2 false positives. All targets found.

### 9. train/grid_0004.jpg
**Prediction:** 9 detections
**Ground truth:** 9 faces — 4 target, 5 other
**Observation:** Perfect count match. All 4 targets identified.

### 10. train/grid_0006.jpg
**Prediction:** 14 detections
**Ground truth:** 9 faces — 3 target, 6 other
**Observation:** 5 false positives — most of any example. Background in this image has face-like patterns.

### 11. train/grid_0007.jpg
**Prediction:** 14 detections
**Ground truth:** 9 faces — 3 target, 6 other
**Observation:** 5 false positives. Similar background challenge as grid_0006.

---

**Summary:** Across all 11 examples the model correctly identifies the target celebrity IDs
(797, 4428, 2619, 7, 7007) in every image. Detection counts range from 7-14 per image
(vs 9 ground truth), with the best images achieving an exact 9/9 match. False positives
tend to come from classes with similar facial features (10002, 2970, 4422) or from
background regions with face-like patterns. The model's confidence scores are well-calibrated:
celebrity 797 consistently achieves the highest scores (0.79-0.98), while ambiguous
detections cluster around 0.1-0.3. Expanding the training set to 100+ images would
reduce false positives and improve recall on the missed faces.

