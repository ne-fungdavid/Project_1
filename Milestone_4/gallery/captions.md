# Final Results Gallery (Milestone 4)

## Single-Face Classification (ResNet50)

**Correct:** 25 examples  **Errors:** 8 examples

### Correctly Classified

1. **7** (conf=0.985) - 002065.jpg
2. **7** (conf=0.863) - 003099.jpg
3. **7** (conf=0.970) - 007335.jpg
4. **7** (conf=0.982) - 009932.jpg
5. **7** (conf=0.979) - 011904.jpg
6. **7** (conf=0.971) - 017587.jpg
7. **7** (conf=0.968) - 019811.jpg
8. **7** (conf=0.992) - 020284.jpg
9. **7** (conf=0.948) - 022384.jpg
10. **7** (conf=0.882) - 036895.jpg
11. **7** (conf=0.980) - 038132.jpg
12. **7** (conf=0.949) - 045626.jpg
13. **7** (conf=0.997) - 063094.jpg
14. **7** (conf=0.937) - 090356.jpg
15. **7** (conf=0.987) - 090940.jpg
16. **7** (conf=0.997) - 091155.jpg
17. **7** (conf=0.768) - 095162.jpg
18. **7** (conf=0.986) - 115977.jpg
19. **7** (conf=0.699) - 121574.jpg
20. **7** (conf=0.978) - 130335.jpg
21. **7** (conf=0.933) - 131364.jpg
22. **7** (conf=0.934) - 140464.jpg
23. **7** (conf=0.965) - 157436.jpg
24. **7** (conf=0.952) - 157832.jpg
25. **797** (conf=0.951) - 162875.jpg

### Misclassified (with analysis)

1. True: **797**  Predicted: **7007**  (conf=0.406)
   Analysis: Model confused similar facial features. Both are target celebrities in the 5-class set.
2. True: **2619**  Predicted: **7**  (conf=0.410)
   Analysis: Model confused similar facial features. Both are target celebrities in the 5-class set.
3. True: **2619**  Predicted: **7**  (conf=0.781)
   Analysis: Model confused similar facial features. Both are target celebrities in the 5-class set.
4. True: **7007**  Predicted: **4428**  (conf=0.615)
   Analysis: Model confused similar facial features. Both are target celebrities in the 5-class set.
5. True: **7007**  Predicted: **7**  (conf=0.548)
   Analysis: Model confused similar facial features. Both are target celebrities in the 5-class set.
6. True: **7007**  Predicted: **797**  (conf=0.467)
   Analysis: Model confused similar facial features. Both are target celebrities in the 5-class set.
7. True: **7007**  Predicted: **7**  (conf=0.914)
   Analysis: Model confused similar facial features. Both are target celebrities in the 5-class set.
8. True: **7007**  Predicted: **7**  (conf=0.507)
   Analysis: Model confused similar facial features. Both are target celebrities in the 5-class set.

## Multi-Face Detection (YOLOv8)

Reused from Milestone 3 gallery (11 annotated grid images).

Total detection examples: 11
