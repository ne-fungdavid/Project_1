# Final Results Gallery (Milestone 4)

## Single-Face Classification (ResNet50)

**Correct:** 25 examples  **Errors:** 6 examples

### Correctly Classified

1. **7** (conf=0.944) - 002065.jpg
2. **7** (conf=0.684) - 003099.jpg
3. **7** (conf=0.956) - 007335.jpg
4. **7** (conf=0.901) - 009932.jpg
5. **7** (conf=0.979) - 011904.jpg
6. **7** (conf=0.923) - 017587.jpg
7. **7** (conf=0.988) - 019811.jpg
8. **7** (conf=0.990) - 020284.jpg
9. **7** (conf=0.941) - 022384.jpg
10. **7** (conf=0.826) - 038132.jpg
11. **7** (conf=0.947) - 045626.jpg
12. **7** (conf=0.997) - 063094.jpg
13. **7** (conf=0.935) - 090356.jpg
14. **7** (conf=0.959) - 090940.jpg
15. **7** (conf=0.993) - 091155.jpg
16. **7** (conf=0.666) - 095162.jpg
17. **7** (conf=0.960) - 115977.jpg
18. **7** (conf=0.661) - 121574.jpg
19. **7** (conf=0.944) - 130335.jpg
20. **7** (conf=0.816) - 140464.jpg
21. **7** (conf=0.812) - 157436.jpg
22. **7** (conf=0.790) - 157832.jpg
23. **797** (conf=0.840) - 162875.jpg
24. **797** (conf=0.992) - 164872.jpg
25. **797** (conf=0.989) - 165865.jpg

### Misclassified (with analysis)

1. True: **7**  Predicted: **2619**  (confidence=0.538)  — `036895.jpg`
   Analysis: A cross-identity confusion between two rostered celebrities. The softmax assigned the highest (but not decisive) probability to `2619` instead of the true identity `7`. This indicates the two faces share salient features and the model lacked enough separating signal for this specific pair — the usual remedy is more images and harder negatives for both identities, or a similarity/verification stage on top.
2. True: **7**  Predicted: **2619**  (confidence=0.557)  — `131364.jpg`
   Analysis: A cross-identity confusion between two rostered celebrities. The softmax assigned the highest (but not decisive) probability to `2619` instead of the true identity `7`. This indicates the two faces share salient features and the model lacked enough separating signal for this specific pair — the usual remedy is more images and harder negatives for both identities, or a similarity/verification stage on top.
3. True: **797**  Predicted: **7007**  (confidence=0.774)  — `164035.jpg`
   Analysis: A cross-identity confusion between two rostered celebrities. The softmax assigned the highest (but not decisive) probability to `7007` instead of the true identity `797`. This indicates the two faces share salient features and the model lacked enough separating signal for this specific pair — the usual remedy is more images and harder negatives for both identities, or a similarity/verification stage on top.
4. True: **7007**  Predicted: **4428**  (confidence=0.762)  — `010363.jpg`
   Analysis: A cross-identity confusion between two rostered celebrities. The softmax assigned the highest (but not decisive) probability to `4428` instead of the true identity `7007`. This indicates the two faces share salient features and the model lacked enough separating signal for this specific pair — the usual remedy is more images and harder negatives for both identities, or a similarity/verification stage on top.
5. True: **7007**  Predicted: **797**  (confidence=0.446)  — `035390.jpg`
   Analysis: A cross-identity confusion between two rostered celebrities. The softmax assigned the highest (but not decisive) probability to `797` instead of the true identity `7007`. This indicates the two faces share salient features and the model lacked enough separating signal for this specific pair — the usual remedy is more images and harder negatives for both identities, or a similarity/verification stage on top.
6. True: **7007**  Predicted: **7**  (confidence=0.504)  — `095414.jpg`
   Analysis: A cross-identity confusion between two rostered celebrities. The softmax assigned the highest (but not decisive) probability to `7` instead of the true identity `7007`. This indicates the two faces share salient features and the model lacked enough separating signal for this specific pair — the usual remedy is more images and harder negatives for both identities, or a similarity/verification stage on top.

## Multi-Face Detection (YOLOv8)

Reused from Milestone 3 gallery (11 annotated grid images).

Total detection examples: 11
