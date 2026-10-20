# Results Gallery — Validation Split (Baseline 5-class)

**Source:** 3 synthetic grid images

**Color legend:**  
- **Green** = Correct (detected + classified correctly)  
- **Orange** = Misclassified (detected, IoU > 0.5, wrong class)  
- **Red** = False Positive (no GT match, IoU < 0.5)  
- **Blue/Orange border** = False Negative (missed GT)

## grid_0001.jpg

Image: `baseline_grid_0001.jpg`

Detected: 9 faces | Correct: 2 | Incorrect: 7 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 7 | 0.7180 | classifier | 2336 | INCORRECT |
| 2 | 4428 | 0.9979 | yolo | 4428 | CORRECT |
| 3 | 7007 | 0.9971 | yolo | 7007 | CORRECT |
| 4 | 7007 | 0.9060 | classifier | 10002 | INCORRECT |
| 5 | 7 | 0.8789 | classifier | 3 | INCORRECT |
| 6 | 7007 | 0.8752 | classifier | 5695 | INCORRECT |
| 7 | 7 | 0.8845 | classifier | 1212 | INCORRECT |
| 8 | 7007 | 0.7615 | classifier | 2970 | INCORRECT |
| 9 | 7 | 0.5197 | yolo | 2619 | INCORRECT |

## grid_0008.jpg

Image: `baseline_grid_0008.jpg`

Detected: 9 faces | Correct: 3 | Incorrect: 6 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 4428 | 0.9976 | yolo | 4428 | CORRECT |
| 2 | 7007 | 0.9485 | classifier | 4422 | INCORRECT |
| 3 | 1212 | 0.9963 | detector | 1212 | CORRECT |
| 4 | 7 | 0.6955 | classifier | 3 | INCORRECT |
| 5 | 7007 | 0.7890 | classifier | 2970 | INCORRECT |
| 6 | 4428 | 0.5230 | classifier | 5695 | INCORRECT |
| 7 | 7007 | 0.9908 | yolo | 7007 | CORRECT |
| 8 | 7007 | 0.5599 | classifier | 8335 | INCORRECT |
| 9 | 7007 | 0.9376 | classifier | 10002 | INCORRECT |

## grid_0015.jpg

Image: `baseline_grid_0015.jpg`

Detected: 9 faces | Correct: 6 | Incorrect: 3 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 7007 | 0.9988 | yolo | 7007 | CORRECT |
| 2 | 7 | 0.6385 | classifier | 1212 | INCORRECT |
| 3 | 4428 | 0.9980 | yolo | 4428 | CORRECT |
| 4 | 3 | 0.9969 | detector | 3 | CORRECT |
| 5 | 7 | 0.9963 | yolo | 7 | CORRECT |
| 6 | 2619 | 0.9952 | yolo | 2619 | CORRECT |
| 7 | 7007 | 0.9186 | classifier | 10002 | INCORRECT |
| 8 | 4428 | 0.7713 | classifier | 2970 | INCORRECT |
| 9 | 797 | 0.9683 | yolo | 797 | CORRECT |

---

## Summary

| Metric | Value |
|--------|-------|
| Total faces detected | 27 |
| Correctly classified | 11 |
| Incorrectly classified | 16 |
| Missed (FN) | 0 |
| Accuracy | 40.7% |
| Total images | 3 |
| Total GT faces | 27 |
