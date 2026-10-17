# Final Results Gallery (Milestone 4 - Improved 13-class Model)

**Source:** Only the 3 synthetic test images (data/synthetic_grid/images/test/)

**Color legend:**  
- **Green** = Correct (detected + classified correctly)  
- **Orange** = Misclassified (detected, IoU > 0.5, wrong class)  
- **Red** = False Positive (no GT match, IoU < 0.5)  
- **Blue/Orange border** = False Negative (missed GT)

## grid_0000.jpg

Image: `improved_grid_0000.jpg`

Detected: 7 faces | Correct: 5 | Incorrect: 2 | Missed (FN): 4 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 797 | 0.9457 | classifier | 797 | CORRECT |
| 2 | 4422 | 0.5229 | classifier | 2970 | INCORRECT |
| 3 | 8335 | 0.9653 | classifier | 8335 | CORRECT |
| 4 | 1212 | 0.9930 | classifier | 1212 | CORRECT |
| 5 | 4422 | 0.6698 | classifier | 2970 | INCORRECT |
| 6 | 3 | 0.9988 | classifier | 3 | CORRECT |
| 7 | 3 | 0.9991 | classifier | 3 | CORRECT |

## grid_0005.jpg

Image: `improved_grid_0005.jpg`

Detected: 8 faces | Correct: 8 | Incorrect: 0 | Missed (FN): 3 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 4422 | 0.6889 | classifier | 4422 | CORRECT |
| 2 | 1212 | 0.9936 | classifier | 1212 | CORRECT |
| 3 | 10002 | 0.8891 | classifier | 10002 | CORRECT |
| 4 | 797 | 0.6981 | classifier | 797 | CORRECT |
| 5 | 8335 | 0.2959 | detector | 8335 | CORRECT |
| 6 | 3 | 0.8541 | classifier | 3 | CORRECT |
| 7 | 4422 | 0.7468 | classifier | 4422 | CORRECT |
| 8 | 1212 | 0.9980 | classifier | 1212 | CORRECT |

## grid_0017.jpg

Image: `improved_grid_0017.jpg`

Detected: 10 faces | Correct: 9 | Incorrect: 1 | Missed (FN): 2 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 797 | 0.5814 | classifier | 797 | CORRECT |
| 2 | 797 | 0.7415 | detector | 2970 | INCORRECT |
| 3 | 4422 | 0.9308 | classifier | 4422 | CORRECT |
| 4 | 10002 | 0.8248 | classifier | 10002 | CORRECT |
| 5 | 10002 | 0.8026 | classifier | 10002 | CORRECT |
| 6 | 3 | 0.7904 | classifier | 3 | CORRECT |
| 7 | 4422 | 0.9308 | classifier | 4422 | CORRECT |
| 8 | 4422 | 0.8676 | classifier | 4422 | CORRECT |
| 9 | 4428 | 0.5776 | classifier | 4428 | CORRECT |
| 10 | 1212 | 0.9777 | classifier | 1212 | CORRECT |

---

## Summary

| Metric | Value |
|--------|-------|
| Total faces detected | 25 |
| Correctly classified | 22 |
| Incorrectly classified | 3 |
| Missed (FN) | 9 |
| Accuracy | 88.0% |
| Test images | 3 |
| Total GT faces | 27 |
