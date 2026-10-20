# Results Gallery — Validation Split (Improved 13-class)

**Source:** 3 synthetic grid images

**Color legend:**  
- **Green** = Correct (detected + classified correctly)  
- **Orange** = Misclassified (detected, IoU > 0.5, wrong class)  
- **Red** = False Positive (no GT match, IoU < 0.5)  
- **Blue/Orange border** = False Negative (missed GT)

## grid_0001.jpg

Image: `improved_grid_0001.jpg`

Detected: 9 faces | Correct: 6 | Incorrect: 3 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 2336 | 0.9825 | classifier | 2336 | CORRECT |
| 2 | 4428 | 0.9979 | yolo | 4428 | CORRECT |
| 3 | 7007 | 0.9971 | yolo | 7007 | CORRECT |
| 4 | 10002 | 0.8218 | classifier | 10002 | CORRECT |
| 5 | 3 | 0.9983 | classifier | 3 | CORRECT |
| 6 | 7007 | 0.6901 | classifier | 5695 | INCORRECT |
| 7 | 1212 | 0.7547 | classifier | 1212 | CORRECT |
| 8 | 5695 | 0.9520 | detector | 2970 | INCORRECT |
| 9 | 7 | 0.5197 | yolo | 2619 | INCORRECT |

## grid_0008.jpg

Image: `improved_grid_0008.jpg`

Detected: 9 faces | Correct: 8 | Incorrect: 1 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 4428 | 0.9976 | yolo | 4428 | CORRECT |
| 2 | 4422 | 0.9436 | classifier | 4422 | CORRECT |
| 3 | 1212 | 0.9963 | detector | 1212 | CORRECT |
| 4 | 3 | 0.8225 | classifier | 3 | CORRECT |
| 5 | 7007 | 0.6637 | classifier | 2970 | INCORRECT |
| 6 | 5695 | 0.9162 | classifier | 5695 | CORRECT |
| 7 | 7007 | 0.9908 | yolo | 7007 | CORRECT |
| 8 | 8335 | 0.9570 | detector | 8335 | CORRECT |
| 9 | 10002 | 0.7738 | classifier | 10002 | CORRECT |

## grid_0015.jpg

Image: `improved_grid_0015.jpg`

Detected: 9 faces | Correct: 8 | Incorrect: 1 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 7007 | 0.9988 | yolo | 7007 | CORRECT |
| 2 | 1212 | 0.9891 | classifier | 1212 | CORRECT |
| 3 | 4428 | 0.9980 | yolo | 4428 | CORRECT |
| 4 | 3 | 0.9999 | classifier | 3 | CORRECT |
| 5 | 7 | 0.9963 | yolo | 7 | CORRECT |
| 6 | 2619 | 0.9952 | yolo | 2619 | CORRECT |
| 7 | 10002 | 0.8847 | classifier | 10002 | CORRECT |
| 8 | 4422 | 0.5671 | classifier | 2970 | INCORRECT |
| 9 | 797 | 0.9683 | yolo | 797 | CORRECT |

---

## Summary

| Metric | Value |
|--------|-------|
| Total faces detected | 27 |
| Correctly classified | 22 |
| Incorrectly classified | 5 |
| Missed (FN) | 0 |
| Accuracy | 81.5% |
| Total images | 3 |
| Total GT faces | 27 |
