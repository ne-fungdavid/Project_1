# Results Gallery — Test Split (Improved 13-class)

**Source:** 3 synthetic grid images

**Color legend:**  
- **Green** = Correct (detected + classified correctly)  
- **Orange** = Misclassified (detected, IoU > 0.5, wrong class)  
- **Red** = False Positive (no GT match, IoU < 0.5)  
- **Blue/Orange border** = False Negative (missed GT)

## grid_0000.jpg

Image: `improved_grid_0000.jpg`

Detected: 5 faces | Correct: 4 | Incorrect: 1 | Missed (FN): 4 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 797 | 0.9457 | classifier | 797 | CORRECT |
| 2 | 4422 | 0.5229 | classifier | 2970 | INCORRECT |
| 3 | 8335 | 0.9653 | classifier | 8335 | CORRECT |
| 4 | 1212 | 0.9930 | classifier | 1212 | CORRECT |
| 5 | 3 | 0.9988 | classifier | 3 | CORRECT |

## grid_0005.jpg

Image: `improved_grid_0005.jpg`

Detected: 6 faces | Correct: 6 | Incorrect: 0 | Missed (FN): 3 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 4422 | 0.6889 | classifier | 4422 | CORRECT |
| 2 | 1212 | 0.9936 | classifier | 1212 | CORRECT |
| 3 | 10002 | 0.8891 | classifier | 10002 | CORRECT |
| 4 | 797 | 0.6981 | classifier | 797 | CORRECT |
| 5 | 8335 | 0.2959 | detector | 8335 | CORRECT |
| 6 | 3 | 0.8541 | classifier | 3 | CORRECT |

## grid_0017.jpg

Image: `improved_grid_0017.jpg`

Detected: 7 faces | Correct: 6 | Incorrect: 1 | Missed (FN): 2 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 797 | 0.5814 | classifier | 797 | CORRECT |
| 2 | 797 | 0.7415 | detector | 2970 | INCORRECT |
| 3 | 4422 | 0.9308 | classifier | 4422 | CORRECT |
| 4 | 10002 | 0.8248 | classifier | 10002 | CORRECT |
| 5 | 3 | 0.7904 | classifier | 3 | CORRECT |
| 6 | 4428 | 0.5776 | classifier | 4428 | CORRECT |
| 7 | 1212 | 0.9777 | classifier | 1212 | CORRECT |

---

## Summary

| Metric | Value |
|--------|-------|
| Total faces detected | 18 |
| Correctly classified | 16 |
| Incorrectly classified | 2 |
| Missed (FN) | 9 |
| Accuracy | 88.9% |
| Total images | 3 |
| Total GT faces | 27 |
