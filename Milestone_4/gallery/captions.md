# Results Gallery — Test Split (Baseline 5-class)

**Source:** 3 synthetic grid images

**Color legend:**  
- **Green** = Correct (detected + classified correctly)  
- **Orange** = Misclassified (detected, IoU > 0.5, wrong class)  
- **Red** = False Positive (no GT match, IoU < 0.5)  
- **Blue/Orange border** = False Negative (missed GT)

## grid_0000.jpg

Image: `baseline_grid_0000.jpg`

Detected: 5 faces | Correct: 1 | Incorrect: 4 | Missed (FN): 4 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 7007 | 0.5620 | classifier | 797 | INCORRECT |
| 2 | 2970 | 0.4830 | detector | 2970 | CORRECT |
| 3 | 2619 | 0.5726 | classifier | 8335 | INCORRECT |
| 4 | 7 | 0.8591 | classifier | 1212 | INCORRECT |
| 5 | 2336 | 0.1180 | detector | 3 | INCORRECT |

## grid_0005.jpg

Image: `baseline_grid_0005.jpg`

Detected: 6 faces | Correct: 1 | Incorrect: 5 | Missed (FN): 3 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 7007 | 0.9929 | classifier | 4422 | INCORRECT |
| 2 | 7 | 0.7070 | classifier | 1212 | INCORRECT |
| 3 | 7007 | 0.9671 | classifier | 10002 | INCORRECT |
| 4 | 797 | 0.3206 | detector | 797 | CORRECT |
| 5 | 2619 | 0.8609 | classifier | 8335 | INCORRECT |
| 6 | 797 | 0.5024 | classifier | 3 | INCORRECT |

## grid_0017.jpg

Image: `baseline_grid_0017.jpg`

Detected: 7 faces | Correct: 2 | Incorrect: 5 | Missed (FN): 2 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 797 | 0.6337 | classifier | 797 | CORRECT |
| 2 | 7007 | 0.5847 | classifier | 2970 | INCORRECT |
| 3 | 7007 | 0.7440 | classifier | 4422 | INCORRECT |
| 4 | 7007 | 0.8104 | classifier | 10002 | INCORRECT |
| 5 | 7 | 0.9323 | classifier | 3 | INCORRECT |
| 6 | 4428 | 0.8523 | classifier | 4428 | CORRECT |
| 7 | 7 | 0.7788 | classifier | 1212 | INCORRECT |

---

## Summary

| Metric | Value |
|--------|-------|
| Total faces detected | 18 |
| Correctly classified | 4 |
| Incorrectly classified | 14 |
| Missed (FN) | 9 |
| Accuracy | 22.2% |
| Total images | 3 |
| Total GT faces | 27 |
