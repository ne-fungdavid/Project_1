# Results Gallery — Test Split (Baseline 5-class)

**Source:** 3 synthetic grid images

**Color legend:**  
- **Green** = Correct (detected + classified correctly)  
- **Orange** = Misclassified (detected, IoU > 0.5, wrong class)  
- **Red** = False Positive (no GT match, IoU < 0.5)  
- **Blue/Orange border** = False Negative (missed GT)

## grid_0000.jpg

Image: `baseline_grid_0000.jpg`

Detected: 9 faces | Correct: 2 | Incorrect: 7 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 2619 | 0.6667 | classifier | 8335 | INCORRECT |
| 2 | 7 | 0.8149 | classifier | 1212 | INCORRECT |
| 3 | 797 | 0.9981 | yolo | 797 | CORRECT |
| 4 | 7007 | 0.5385 | classifier | 2970 | INCORRECT |
| 5 | 7007 | 0.8837 | classifier | 10002 | INCORRECT |
| 6 | 7007 | 0.5951 | classifier | 3 | INCORRECT |
| 7 | 2619 | 0.9775 | yolo | 2619 | CORRECT |
| 8 | 7 | 0.7250 | classifier | 5695 | INCORRECT |
| 9 | 4428 | 0.5216 | classifier | 7007 | INCORRECT |

## grid_0005.jpg

Image: `baseline_grid_0005.jpg`

Detected: 9 faces | Correct: 2 | Incorrect: 7 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 7 | 0.7070 | classifier | 1212 | INCORRECT |
| 2 | 7007 | 0.9944 | classifier | 4422 | INCORRECT |
| 3 | 2619 | 0.9154 | classifier | 8335 | INCORRECT |
| 4 | 797 | 0.9898 | yolo | 797 | CORRECT |
| 5 | 797 | 0.5815 | classifier | 3 | INCORRECT |
| 6 | 7 | 0.9365 | classifier | 2336 | INCORRECT |
| 7 | 2619 | 0.9753 | yolo | 2619 | CORRECT |
| 8 | 7007 | 0.9604 | classifier | 10002 | INCORRECT |
| 9 | 8335 | 0.6854 | detector | 7007 | INCORRECT |

## grid_0017.jpg

Image: `baseline_grid_0017.jpg`

Detected: 9 faces | Correct: 2 | Incorrect: 7 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 7 | 0.7236 | classifier | 1212 | INCORRECT |
| 2 | 7007 | 0.7031 | classifier | 4422 | INCORRECT |
| 3 | 7007 | 0.8341 | classifier | 5695 | INCORRECT |
| 4 | 4428 | 0.9965 | yolo | 4428 | CORRECT |
| 5 | 797 | 0.9944 | yolo | 797 | CORRECT |
| 6 | 7007 | 0.8772 | classifier | 10002 | INCORRECT |
| 7 | 7 | 0.9031 | classifier | 3 | INCORRECT |
| 8 | 2619 | 0.6566 | classifier | 2336 | INCORRECT |
| 9 | 7007 | 0.5737 | classifier | 2970 | INCORRECT |

---

## Summary

| Metric | Value |
|--------|-------|
| Total faces detected | 27 |
| Correctly classified | 6 |
| Incorrectly classified | 21 |
| Missed (FN) | 0 |
| Accuracy | 22.2% |
| Total images | 3 |
| Total GT faces | 27 |
