#!/usr/bin/env python
"""
Milestone 3 - Celebrity Detection Demo
=====================================
Loads the fine-tuned YOLO model and runs detection on test images or
a user-supplied image. Displays annotated results and a detection summary.

Usage (from the Milestone_3 directory, or anywhere):
    python milestone3_demo.py                            # run on all held-out test images
    python milestone3_demo.py --image path/to/photo.jpg  # run on a specific image
    python milestone3_demo.py --conf 0.05                # lower confidence threshold
    python milestone3_demo.py --save                     # save annotated outputs
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TARGET_IDS = {"7", "797", "2619", "4428", "7007"}


def find_weights():
    """Locate best.pt, preferring the main training run over sweeps."""
    search_dir = os.path.join(SCRIPT_DIR, "runs", "detect")
    if not os.path.isdir(search_dir):
        return None
    candidates = []
    for root, _dirs, files in os.walk(search_dir):
        if "best.pt" in files:
            candidates.append(os.path.join(root, "best.pt"))
    if not candidates:
        return None
    # Prefer the main run (yolov8n_finetune) over sweep runs
    for c in candidates:
        if "sweep" not in c.lower():
            return c
    # Fall back to most recently modified
    return max(candidates, key=os.path.getmtime)


def test_images_dir():
    return os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "images", "test")


def main():
    parser = argparse.ArgumentParser(description="Celebrity Detection Demo (Milestone 3)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image to run detection on")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Confidence threshold (default 0.1)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default 640)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated images to demo_output/")
    args = parser.parse_args()

    weights = find_weights()
    if weights is None:
        print("ERROR: Could not find best.pt. Run the training notebook first.")
        sys.exit(1)

    print(f"Loading model: {weights}")
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names
    n_cls = len(names)
    target_indices = {i for i, n in names.items() if n in TARGET_IDS}
    print(f"Classes ({n_cls}): {list(names.values())}")
    print(f"Target celebrities: {sorted(TARGET_IDS)}")
    print()

    if args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            sys.exit(1)
        images = [args.image]
    else:
        tdir = test_images_dir()
        if not os.path.isdir(tdir):
            print(f"ERROR: Test images not found: {tdir}")
            sys.exit(1)
        images = sorted(
            os.path.join(tdir, f) for f in os.listdir(tdir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not images:
            print("ERROR: No test images found.")
            sys.exit(1)
        print(f"Running on {len(images)} test image(s)")

    conf = args.conf
    all_results = []
    for img_path in images:
        results = model.predict(img_path, conf=conf, imgsz=args.imgsz, verbose=False)
        detections = []
        dets = results[0].boxes
        for i in range(len(dets)):
            cls_idx = int(dets.cls[i].item())
            detections.append({
                "class_id": names.get(cls_idx, str(cls_idx)),
                "class_idx": cls_idx,
                "confidence": float(dets.conf[i].item()),
                "is_target": cls_idx in target_indices,
                "bbox": [float(x) for x in dets.xyxy[i].tolist()],
            })
        if not detections and conf > 0.001:
            low = 0.001
            print(f"  (0 det at conf={conf}, retrying at conf={low})")
            results = model.predict(img_path, conf=low, imgsz=args.imgsz, verbose=False)
            dets = results[0].boxes
            for i in range(len(dets)):
                cls_idx = int(dets.cls[i].item())
                detections.append({
                    "class_id": names.get(cls_idx, str(cls_idx)),
                    "class_idx": cls_idx,
                    "confidence": float(dets.conf[i].item()),
                    "is_target": cls_idx in target_indices,
                    "bbox": [float(x) for x in dets.xyxy[i].tolist()],
                })
            conf = low
        all_results.append((img_path, results, detections, conf))

    # --- Match detections to ground truth using IoU ---
    def iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-9)

    def load_gt(img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        for split in ("test", "val", "train"):
            lbl = os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "labels", split, base + ".txt")
            if os.path.exists(lbl):
                boxes = []
                with open(lbl) as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) >= 5:
                            boxes.append((int(p[0]), [float(p[1]), float(p[2]), float(p[3]), float(p[4])]))
                return boxes
        return None

    def img_dims(img_path):
        import cv2 as _cv2
        try:
            img = _cv2.imread(img_path)
            if img is not None:
                return img.shape[:2]
        except Exception:
            pass
        return (640, 640)

    for img_path, _res, detections, _conf in all_results:
        gt = load_gt(img_path)
        if gt is None:
            for d in detections:
                d["correct"] = None
            continue
        ih, iw = img_dims(img_path)
        gt_pixel = []
        for cls_idx, (cx, cy, bw, bh) in gt:
            gt_pixel.append((cls_idx, [(cx-bw/2)*iw, (cy-bh/2)*ih, (cx+bw/2)*iw, (cy+bh/2)*ih]))
        matched_gt = set()
        # 4-way classification: TP / Misclassified / FP / FN
        for d in detections:
            best_iou, best_cls = 0, -1
            for gi, (_ci, gbox) in enumerate(gt_pixel):
                v = iou(d["bbox"], gbox)
                if v > best_iou:
                    best_iou, best_cls = v, gi
            if best_iou >= 0.5 and best_cls >= 0:
                gt_cls = gt_pixel[best_cls][0]
                d["status"] = "TP" if d["class_idx"] == gt_cls else "Misclassified"
                d["matched_gt"] = best_cls
                matched_gt.add(best_cls)
            else:
                d["status"] = "FP"
                d["matched_gt"] = -1

    # --- Print summary ---
    # Legend:  ✓ = TP (correct)   ~ = Misclassified   ✗ = FP   — = FN (missed)
    print(f"\n{'=' * 60}")
    print(f"DETECTION RESULTS  (✓ = TP   ~ = Misclassified   ✗ = FP   * = target)")
    print(f"{'=' * 60}")
    totals = {"TP": 0, "Mis": 0, "FP": 0, "FN": 0, "det": 0}
    for img_path, _res, detections, conf_used in all_results:
        img_name = os.path.basename(img_path)
        gt = load_gt(img_path)
        n_tp  = sum(1 for d in detections if d.get("status") == "TP")
        n_mis = sum(1 for d in detections if d.get("status") == "Misclassified")
        n_fp  = sum(1 for d in detections if d.get("status") == "FP")
        n_fn  = 0
        if gt is not None:
            matched = {d.get("matched_gt", -1) for d in detections if d.get("matched_gt", -1) >= 0}
            n_fn = sum(1 for i in range(len(gt)) if i not in matched)
        totals["TP"] += n_tp; totals["Mis"] += n_mis
        totals["FP"] += n_fp; totals["FN"] += n_fn
        totals["det"] += len(detections)
        parts = [f"{n_tp} TP", f"{n_mis} Mis", f"{n_fp} FP"]
        if gt is not None:
            parts.append(f"{n_fn} FN")
        print(f"\n  {img_name}  [conf={conf_used}]  " + ", ".join(parts))
        for d in sorted(detections, key=lambda x: -x["confidence"]):
            mark = {"TP": "  ✓", "Misclassified": "  ~", "FP": "  ✗"}.get(d.get("status", ""), "")
            star = " *" if d["is_target"] else ""
            print(f"    {d['class_id']:>6s}  conf={d['confidence']:.4f}{star}{mark}")
        if gt is not None and n_fn > 0:
            matched = {d.get("matched_gt", -1) for d in detections if d.get("matched_gt", -1) >= 0}
            missed_names = [names.get(gt[i][0], str(gt[i][0])) for i in range(len(gt)) if i not in matched]
            print(f"    MISSED: {', '.join(missed_names)}")

    print(f"\n{'=' * 60}")
    if any(load_gt(p) is not None for p, *_ in all_results):
        print(f"TOTAL: {totals['det']} det  |  {totals['TP']} TP  {totals['Mis']} Mis  {totals['FP']} FP  {totals['FN']} FN")
    else:
        print(f"TOTAL: {totals['det']} detections (no ground truth available)")
    print(f"{'=' * 60}\n")

    # --- Visualize or save (4-color: Green=TP, Orange=Mis, Red=FP, Blue=FN) ---
    import cv2

    def compute_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0

    def get_gt_pixel(img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        for split in ("test", "val", "train"):
            lbl = os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "labels", split, base + ".txt")
            if os.path.exists(lbl):
                img = cv2.imread(img_path)
                if img is None:
                    return []
                ih, iw = img.shape[:2]
                boxes = []
                with open(lbl) as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) >= 5:
                            ci = int(p[0])
                            cx, cy, bw, bh = map(float, p[1:5])
                            boxes.append({"class": ci, "bbox": [
                                int((cx - bw/2) * iw), int((cy - bh/2) * ih),
                                int((cx + bw/2) * iw), int((cy + bh/2) * ih)]})
                return boxes
        return []

    def annotate_4color(img, gt_boxes, dets, class_names):
        annotated = img.copy()
        det_list = []
        for i in range(len(dets)):
            x1, y1, x2, y2 = map(int, dets.xyxy[i].tolist())
            ci = int(dets.cls[i].item())
            conf = float(dets.conf[i].item())
            det_list.append({"bbox": [x1, y1, x2, y2], "class": ci, "conf": conf})
        has_gt = len(gt_boxes) > 0
        gt_matched = [False] * len(gt_boxes)
        label_offset = 0
        for det in det_list:
            if has_gt:
                best_iou, best_gt_idx = 0.0, -1
                for j, gt in enumerate(gt_boxes):
                    v = compute_iou(det["bbox"], gt["bbox"])
                    if v > best_iou:
                        best_iou, best_gt_idx = v, j
                if best_iou >= 0.5 and best_gt_idx >= 0:
                    if det["class"] == gt_boxes[best_gt_idx]["class"]:
                        color = (0, 200, 0)       # Green = TP
                    else:
                        color = (0, 128, 255)     # Orange = Misclassified
                    gt_matched[best_gt_idx] = True
                else:
                    color = (0, 0, 255)           # Red = FP
            else:
                color = (0, 255, 255)             # Yellow = no GT, neutral
            x1, y1, x2, y2 = det["bbox"]
            label = f"{class_names.get(det['class'], str(det['class']))} ({det['conf']:.2f})"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            # Offset label to avoid overlapping labels on stacked boxes
            ly = max(y1 - th - 6 - label_offset, 0)
            cv2.rectangle(annotated, (x1, ly), (x1 + tw, ly + th + 4), color, -1)
            cv2.putText(annotated, label, (x1, ly + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0) if not has_gt else (255, 255, 255), 1)
            label_offset += th + 6
        for j, gt in enumerate(gt_boxes):
            if not gt_matched[j]:
                x1, y1, x2, y2 = gt["bbox"]
                color = (255, 128, 0)         # Blue = FN
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.rectangle(annotated, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), color, 1, cv2.LINE_AA)
                label = f"GT:{class_names.get(gt['class'], str(gt['class']))} (missed)"
                cv2.putText(annotated, label, (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        return annotated

    if args.save:
        out_dir = os.path.join(SCRIPT_DIR, "demo_output")
        os.makedirs(out_dir, exist_ok=True)
        for img_path, results, _d, _c in all_results:
            img = cv2.imread(img_path)
            gt_boxes = get_gt_pixel(img_path)
            annotated = annotate_4color(img, gt_boxes, results[0].boxes, names)
            out_path = os.path.join(out_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, annotated)
            print(f"Saved: {out_path}")
    else:
        try:
            import matplotlib.pyplot as plt
            n = len(all_results)
            n_cols = min(4, n)
            n_rows = (n + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 8 * n_rows))
            fig.suptitle("Green=TP  Orange=Misclassified  Red=FP  Blue=FN  |  Yellow=no GT (manual check)", fontsize=11)
            if n_rows == 1 and n_cols == 1:
                axes = [[axes]]
            elif n_rows == 1:
                axes = [axes]
            for idx, (img_path, results, _d, _c) in enumerate(all_results):
                row, col = idx // n_cols, idx % n_cols
                ax = axes[row][col]
                img = cv2.imread(img_path)
                gt_boxes = get_gt_pixel(img_path)
                annotated = annotate_4color(img, gt_boxes, results[0].boxes, names)
                ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                ax.set_title(os.path.basename(img_path), fontsize=9)
                ax.axis("off")
            for idx in range(n, n_rows * n_cols):
                row, col = idx // n_cols, idx % n_cols
                axes[row][col].set_visible(False)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("(matplotlib not available - use --save to export images)")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()