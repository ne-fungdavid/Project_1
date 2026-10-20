# Results Gallery — Test Split (Improved 13-class)

**Source:** 3 synthetic grid images

**Color legend:**  
- **Green** = Correct (detected + classified correctly)  
- **Orange** = Misclassified (detected, IoU > 0.5, wrong class)  
- **Red** = False Positive (no GT match, IoU < 0.5)  
- **Blue/Orange border** = False Negative (missed GT)

## grid_0000.jpg

Image: `improved_grid_0000.jpg`

Detected: 9 faces | Correct: 8 | Incorrect: 1 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 8335 | 0.9759 | classifier | 8335 | CORRECT |
| 2 | 1212 | 0.9829 | classifier | 1212 | CORRECT |
| 3 | 797 | 0.9981 | yolo | 797 | CORRECT |
| 4 | 2970 | 0.9955 | detector | 2970 | CORRECT |
| 5 | 10002 | 0.9092 | classifier | 10002 | CORRECT |
| 6 | 3 | 0.9997 | classifier | 3 | CORRECT |
| 7 | 2619 | 0.9775 | yolo | 2619 | CORRECT |
| 8 | 4422 | 0.6251 | classifier | 5695 | INCORRECT |
| 9 | 7007 | 0.6234 | classifier | 7007 | CORRECT |

## grid_0005.jpg

Image: `improved_grid_0005.jpg`

Detected: 9 faces | Correct: 6 | Incorrect: 3 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 1212 | 0.9936 | classifier | 1212 | CORRECT |
| 2 | 4422 | 0.7838 | classifier | 4422 | CORRECT |
| 3 | 2619 | 0.6006 | classifier | 8335 | INCORRECT |
| 4 | 797 | 0.9898 | yolo | 797 | CORRECT |
| 5 | 3 | 0.9207 | classifier | 3 | CORRECT |
| 6 | 1212 | 0.8326 | classifier | 2336 | INCORRECT |
| 7 | 2619 | 0.9753 | yolo | 2619 | CORRECT |
| 8 | 10002 | 0.9269 | classifier | 10002 | CORRECT |
| 9 | 2619 | 0.7062 | classifier | 7007 | INCORRECT |

## grid_0017.jpg

Image: `improved_grid_0017.jpg`

Detected: 9 faces | Correct: 6 | Incorrect: 3 | Missed (FN): 0 | GT total: 9

| # | Predicted | Confidence | Source | Ground Truth | Status |
|---|-----------|------------|--------|--------------|--------|
| 1 | 1212 | 0.9684 | classifier | 1212 | CORRECT |
| 2 | 4422 | 0.9022 | classifier | 4422 | CORRECT |
| 3 | 7007 | 0.9167 | classifier | 5695 | INCORRECT |
| 4 | 4428 | 0.9965 | yolo | 4428 | CORRECT |
| 5 | 797 | 0.9944 | yolo | 797 | CORRECT |
| 6 | 10002 | 0.8472 | classifier | 10002 | CORRECT |
| 7 | 3 | 0.9488 | classifier | 3 | CORRECT |
| 8 | 1212 | 0.5834 | classifier | 2336 | INCORRECT |
| 9 | 797 | 0.6955 | classifier | 2970 | INCORRECT |

---

## Summary

| Metric | Value |
|--------|-------|
| Total faces detected | 27 |
| Correctly classified | 20 |
| Incorrectly classified | 7 |
| Missed (FN) | 0 |
| Accuracy | 74.1% |
| Total images | 3 |
| Total GT faces | 27 |
