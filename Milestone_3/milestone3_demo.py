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
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default 0.25)")
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
        for b in results[0].boxes:
            cls_idx = int(b.cls[0])
            detections.append({
                "class_id": names.get(cls_idx, str(cls_idx)),
                "confidence": b.conf[0].item(),
                "is_target": cls_idx in target_indices,
                "bbox": [float(x) for x in b.xyxy[0].tolist()],
            })
        if not detections and conf > 0.001:
            low = 0.001
            print(f"  (0 det at conf={conf}, retrying at conf={low})")
            results = model.predict(img_path, conf=low, imgsz=args.imgsz, verbose=False)
            for b in results[0].boxes:
                cls_idx = int(b.cls[0])
                detections.append({
                    "class_id": names.get(cls_idx, str(cls_idx)),
                    "confidence": b.conf[0].item(),
                    "is_target": cls_idx in target_indices,
                    "bbox": [float(x) for x in b.xyxy[0].tolist()],
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
        for d in detections:
            best_iou, best_cls = 0, -1
            for gi, (_ci, gbox) in enumerate(gt_pixel):
                v = iou(d["bbox"], gbox)
                if v > best_iou:
                    best_iou, best_cls = v, gi
            d["correct"] = best_iou >= 0.5
            d["matched_gt"] = best_cls if best_iou >= 0.5 else -1
            if best_iou >= 0.5:
                matched_gt.add(best_cls)

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print(f"DETECTION RESULTS  (✓ = correct  ✗ = false positive  * = target)")
    print(f"{'=' * 60}")
    n_total = n_correct = n_fp = n_missed_total = 0
    for img_path, _res, detections, conf_used in all_results:
        img_name = os.path.basename(img_path)
        gt = load_gt(img_path)
        n_corr = sum(1 for d in detections if d.get("correct") is True)
        n_fp = sum(1 for d in detections if d.get("correct") is False)
        if gt is not None:
            matched = {d.get("matched_gt", -1) for d in detections if d.get("correct")}
            n_missed = sum(1 for i in range(len(gt)) if i not in matched)
        else:
            n_missed = 0
        n_total += len(detections); n_correct += n_corr; n_fp += n_fp; n_missed_total += n_missed
        status = f"{n_corr} correct, {n_fp} false"
        if gt is not None:
            status += f", {n_missed} missed"
        print(f"\n  {img_name}  [conf={conf_used}]  {status}")
        for d in sorted(detections, key=lambda x: -x["confidence"]):
            if d.get("correct") is True:
                mark = "  ✓"
            elif d.get("correct") is False:
                mark = "  ✗"
            else:
                mark = ""
            star = " *" if d["is_target"] else ""
            print(f"    {d['class_id']:>6s}  conf={d['confidence']:.4f}{star}{mark}")
        if gt is not None and n_missed > 0:
            matched = {d.get("matched_gt", -1) for d in detections if d.get("correct")}
            missed_names = [names.get(gt[i][0], str(gt[i][0])) for i in range(len(gt)) if i not in matched]
            print(f"    MISSED: {', '.join(missed_names)}")

    print(f"\n{'=' * 60}")
    if any(load_gt(p) is not None for p, *_ in all_results):
        print(f"TOTAL: {n_total} detections  |  {n_correct} correct  {n_fp} false  {n_missed_total} missed")
    else:
        print(f"TOTAL: {n_total} detections (no ground truth available)")
    print(f"{'=' * 60}\n")

    if args.save:
        out_dir = os.path.join(SCRIPT_DIR, "demo_output")
        os.makedirs(out_dir, exist_ok=True)
        import cv2
        for img_path, results, _d, _c in all_results:
            annotated = results[0].plot()
            out_path = os.path.join(out_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, annotated)
            print(f"Saved: {out_path}")
    else:
        try:
            import matplotlib.pyplot as plt
            import cv2
            n = len(all_results)
            fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
            if n == 1:
                axes = [axes]
            for idx, (img_path, results, _d, _c) in enumerate(all_results):
                annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                axes[idx].imshow(annotated)
                axes[idx].set_title(os.path.basename(img_path), fontsize=10)
                axes[idx].axis("off")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("(matplotlib not available - use --save to export images)")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()