# Results Gallery — Validation Split (Baseline 5-class)

**Source:** 3 synthetic grid images

**Color legend:**  
- **Green** = Correct (detected + classified correctly)  
- **Orange** = Misclassified (detected, IoU > 0.5, wrong class)  
- **Red** = False Positive (no GT match, IoU < 0.5)  
- **Blue/Orange border** = False Negative (missed GT)

## grid_0001.jpg

Image: `baseline_grid_0001.jpg`

Detected: 3 faces | Correct: 0 | Incorrect: 3 | Missed (FN): 6 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 7007 | 0.5845 | classifier | 4428 | INCORRECT |
| 2 | 7007 | 0.8644 | classifier | 5695 | INCORRECT |
| 3 | 7 | 0.9015 | classifier | 1212 | INCORRECT |

## grid_0008.jpg

Image: `baseline_grid_0008.jpg`

Detected: 6 faces | Correct: 1 | Incorrect: 5 | Missed (FN): 3 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 7007 | 0.9328 | classifier | 4422 | INCORRECT |
| 2 | 7007 | 0.9286 | classifier | 10002 | INCORRECT |
| 3 | 7007 | 0.7615 | classifier | 2970 | INCORRECT |
| 4 | 4428 | 0.9441 | classifier | 4428 | CORRECT |
| 5 | 7 | 0.6628 | classifier | 3 | INCORRECT |
| 6 | 4428 | 0.5103 | classifier | 5695 | INCORRECT |

## grid_0015.jpg

Image: `baseline_grid_0015.jpg`

Detected: 3 faces | Correct: 1 | Incorrect: 2 | Missed (FN): 6 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 7007 | 0.9882 | classifier | 797 | INCORRECT |
| 2 | 4428 | 0.8636 | classifier | 4428 | CORRECT |
| 3 | 4428 | 0.7702 | classifier | 2970 | INCORRECT |

---

## Summary

| Metric | Value |
|--------|-------|
| Total faces detected | 12 |
| Correctly classified | 2 |
| Incorrectly classified | 10 |
| Missed (FN) | 15 |
| Accuracy | 16.7% |
| Total images | 3 |
| Total GT faces | 27 |
