# Results Gallery — Validation Split (Improved 13-class)

**Source:** 3 synthetic grid images

**Color legend:**  
- **Green** = Correct (detected + classified correctly)  
- **Orange** = Misclassified (detected, IoU > 0.5, wrong class)  
- **Red** = False Positive (no GT match, IoU < 0.5)  
- **Blue/Orange border** = False Negative (missed GT)

## grid_0001.jpg

Image: `improved_grid_0001.jpg`

Detected: 3 faces | Correct: 2 | Incorrect: 1 | Missed (FN): 6 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 4428 | 0.4393 | detector | 4428 | CORRECT |
| 2 | 7007 | 0.6848 | classifier | 5695 | INCORRECT |
| 3 | 1212 | 0.6656 | classifier | 1212 | CORRECT |

## grid_0008.jpg

Image: `improved_grid_0008.jpg`

Detected: 6 faces | Correct: 5 | Incorrect: 1 | Missed (FN): 3 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 4422 | 0.9382 | classifier | 4422 | CORRECT |
| 2 | 10002 | 0.7018 | classifier | 10002 | CORRECT |
| 3 | 7007 | 0.6816 | classifier | 2970 | INCORRECT |
| 4 | 4428 | 0.2838 | detector | 4428 | CORRECT |
| 5 | 3 | 0.8230 | classifier | 3 | CORRECT |
| 6 | 5695 | 0.9012 | classifier | 5695 | CORRECT |

## grid_0015.jpg

Image: `improved_grid_0015.jpg`

Detected: 3 faces | Correct: 0 | Incorrect: 3 | Missed (FN): 6 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 4422 | 0.5739 | classifier | 797 | INCORRECT |
| 2 | 4422 | 0.4613 | detector | 4428 | INCORRECT |
| 3 | 4422 | 0.5117 | classifier | 2970 | INCORRECT |

---

## Summary

| Metric | Value |
|--------|-------|
| Total faces detected | 12 |
| Correctly classified | 7 |
| Incorrectly classified | 5 |
| Missed (FN) | 15 |
| Accuracy | 58.3% |
| Total images | 3 |
| Total GT faces | 27 |
