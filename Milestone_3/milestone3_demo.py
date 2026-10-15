#!/usr/bin/env python
"""
Milestone 3 - Celebrity Detection Demo
=====================================
Loads the fine-tuned YOLO model and reproduces the Milestone 3 notebook's
Detection Gallery: the same image set (all test + all val + first 5 train),
the same 4-color error coding, and the same DETECTION SUMMARY.

  Green  = TP (correct class, IoU >= 0.5)
  Orange = Misclassified (IoU >= 0.5, wrong class)
  Red    = FP (IoU < 0.5, no matching GT)
  Blue   = FN (missed GT box)

Usage (from anywhere):
    python milestone3_demo.py                              # reproduce the notebook gallery
    python milestone3_demo.py --image path/to/photo.jpg    # run on a single image
    python milestone3_demo.py --conf 0.05                  # different confidence threshold
    python milestone3_demo.py --save                       # save annotated images + montage PNG
    python milestone3_demo.py --no-plot                    # skip the interactive plot window
"""

import argparse
import os
import sys
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

NUM_CLASSES = 13
IMG_SIZE = 640
CONF_DEFAULT = 0.1
TARGET_IDS = {"7", "797", "2619", "4428", "7007"}
PROJECT_NAME = "celebrity_detection"
RUN_NAME = "yolov8n_finetune"


def find_weights():
    """Prefer the exact weights the notebook uses; fall back to the newest non-sweep run."""
    canon = os.path.join(SCRIPT_DIR, "runs", "detect", PROJECT_NAME, RUN_NAME, "weights", "best.pt")
    if os.path.exists(canon):
        return canon
    cands = []
    for root, _d, files in os.walk(os.path.join(SCRIPT_DIR, "runs", "detect")):
        if "best.pt" in files and "sweep" not in root.lower():
            cands.append(os.path.join(root, "best.pt"))
    return max(cands, key=os.path.getmtime) if cands else None


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
             + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - inter)
    return inter / union if union > 0 else 0.0


def load_gt_boxes(lbl_path, img_w, img_h, target_indices):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            ci = int(p[0])
            cx, cy, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            boxes.append({"class": ci, "is_target": ci in target_indices,
                          "bbox": [int((cx - bw / 2) * img_w), int((cy - bh / 2) * img_h),
                                  int((cx + bw / 2) * img_w), int((cy + bh / 2) * img_h)]})
    return boxes


def annotate_with_errors(img, gt_boxes, dets, class_names):
    """4-color error coding, identical to the notebook's gallery cell."""
    annotated = img.copy()
    n_tp = n_fp = n_fn = n_conf = 0
    det_list = []
    for i in range(len(dets)):
        x1, y1, x2, y2 = map(int, dets.xyxy[i].tolist())
        det_list.append({"bbox": [x1, y1, x2, y2], "class": int(dets.cls[i].item()),
                         "conf": float(dets.conf[i].item())})
    gt_matched = [False] * len(gt_boxes)
    for det in det_list:
        best_iou, best_gt = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            v = compute_iou(det["bbox"], gt["bbox"])
            if v > best_iou:
                best_iou, best_gt = v, j
        if best_iou > 0.5 and best_gt >= 0:
            if det["class"] == gt_boxes[best_gt]["class"]:
                color = (0, 200, 0)      # Green = TP
                n_tp += 1
            else:
                color = (0, 128, 255)    # Orange = Misclassified
                n_conf += 1
            gt_matched[best_gt] = True
        else:
            color = (0, 0, 255)          # Red = FP
            n_fp += 1
        x1, y1, x2, y2 = det["bbox"]
        label = f"{class_names.get(det['class'], det['class'])} ({det['conf']:.2f})"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(annotated, (x1, max(y1 - th - 6, 0)), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for j, gt in enumerate(gt_boxes):
        if not gt_matched[j]:
            x1, y1, x2, y2 = gt["bbox"]
            color = (255, 128, 0)        # Blue = FN (missed)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.rectangle(annotated, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), color, 1, cv2.LINE_AA)
            cv2.putText(annotated, f"GT:{class_names.get(gt['class'], gt['class'])} (missed)",
                        (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            n_fn += 1
    return annotated, {"TP": n_tp, "FP": n_fp, "FN": n_fn, "Misclassified": n_conf}


def build_items(single_image=None):
    """Image set identical to the notebook: all test, all val, first 5 train."""
    data = os.path.join(PROJECT_ROOT, "data", "synthetic_grid")
    if single_image:
        base = os.path.splitext(os.path.basename(single_image))[0]
        lbl = None
        for split in ("test", "val", "train"):
            p = os.path.join(data, "labels", split, base + ".txt")
            if os.path.exists(p):
                lbl = p
                break
        return [("custom", os.path.basename(single_image), single_image,
                 lbl or os.path.join(data, "labels", "_none.txt"))]
    items = []
    for sname in ("test", "val", "train"):
        idir = os.path.join(data, "images", sname)
        ldir = os.path.join(data, "labels", sname)
        if not os.path.isdir(idir):
            continue
        imgs = sorted(f for f in os.listdir(idir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if sname == "train":
            imgs = imgs[:5]
        for f in imgs:
            items.append((sname, f, os.path.join(idir, f),
                          os.path.join(ldir, f.rsplit(".", 1)[0] + ".txt")))
    return items



def build_montage(annotated_list, results_log):
    import matplotlib.pyplot as plt
    n = len(annotated_list)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 8 * n_rows))
    fig.suptitle("Detection Gallery: Green=TP, Orange=Misclassified, Red=FP, Blue=FN (missed)", fontsize=13)
    if n_rows == 1:
        axes = [axes]
    for idx in range(n):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.imshow(cv2.cvtColor(annotated_list[idx][1], cv2.COLOR_BGR2RGB))
        ax.set_title(results_log[idx]["caption"], fontsize=8)
        ax.axis("off")
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Celebrity Detection Demo (Milestone 3)")
    parser.add_argument("--image", type=str, default=None, help="Run on a single image instead of the gallery set")
    parser.add_argument("--conf", type=float, default=CONF_DEFAULT, help="Confidence threshold (default 0.1)")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Inference image size (default 640)")
    parser.add_argument("--save", action="store_true", help="Save annotated images + montage PNG")
    parser.add_argument("--no-plot", action="store_true", help="Skip the interactive plot window")
    args = parser.parse_args()

    weights = find_weights()
    if weights is None:
        print("ERROR: Could not find best.pt. Run the training notebook first.")
        sys.exit(1)
    print(f"Loading model: {weights}")
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names
    target_indices = {i for i, n in names.items() if n in TARGET_IDS}

    items = build_items(args.image)
    if not items:
        print("ERROR: No images found.")
        sys.exit(1)
    n_imgs = len(items)
    print(f"\n{'='*60}")
    print(f"Detection Gallery - {n_imgs} images, {len(names)} classes")
    print(f"Target celebrity IDs: {sorted(TARGET_IDS)}")
    print(f"{'='*60}")

    results_log = []
    annotated_list = []
    for sname, img_name, img_path, lbl_path in items:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  (skip {img_name}: could not read)")
            continue
        h, w = img.shape[:2]
        gt_boxes = load_gt_boxes(lbl_path, w, h, target_indices)
        preds = model.predict(img_path, conf=args.conf, imgsz=args.imgsz, verbose=False)
        annotated, counts = annotate_with_errors(img, gt_boxes, preds[0].boxes, names)
        n_gt_target = sum(1 for g in gt_boxes if g["is_target"])
        parts = []
        if counts["TP"]:
            parts.append(f"{counts['TP']} correct")
        if counts["Misclassified"]:
            parts.append(f"{counts['Misclassified']} confused")
        if counts["FP"]:
            parts.append(f"{counts['FP']} FP")
        if counts["FN"]:
            parts.append(f"{counts['FN']} missed")
        caption = f"[{sname}] {img_name}: " + (", ".join(parts) if parts else "no detections")
        if n_gt_target:
            caption += f"  (GT targets: {n_gt_target})"
        results_log.append({"image": f"{sname}/{img_name}", "counts": counts, "caption": caption})
        annotated_list.append((img_name, annotated))

    if annotated_list:
        if args.save:
            import matplotlib
            matplotlib.use("Agg")
        try:
            fig = build_montage(annotated_list, results_log)
            if args.save:
                montage_path = os.path.join(SCRIPT_DIR, "runs", "detect", PROJECT_NAME, RUN_NAME,
                                            "detection_gallery_enhanced.png")
                os.makedirs(os.path.dirname(montage_path), exist_ok=True)
                fig.savefig(montage_path, dpi=150, bbox_inches="tight")
                out_dir = os.path.join(SCRIPT_DIR, "demo_output")
                os.makedirs(out_dir, exist_ok=True)
                for img_name, annotated in annotated_list:
                    cv2.imwrite(os.path.join(out_dir, os.path.splitext(img_name)[0] + "_annotated.jpg"), annotated)
                import matplotlib.pyplot as plt
                plt.close(fig)
                print(f"Saved montage: {montage_path}")
                print(f"Saved {len(annotated_list)} annotated images to {out_dir}")
            elif not args.no_plot:
                import matplotlib.pyplot as plt
                plt.show()
        except ImportError:
            print("(matplotlib not available - use --save to export images)")

    print(f"\n{'='*60}")
    print("DETECTION SUMMARY  (Green=TP, Orange=Misclassified, Red=FP, Blue=FN)")
    print(f"{'='*60}")
    for r in results_log:
        c = r["counts"]
        status = "OK" if c["FN"] == 0 and c["Misclassified"] == 0 else "ISSUES"
        print(f"  {r['image']}: TP={c['TP']} FP={c['FP']} FN={c['FN']} Misclassified={c['Misclassified']}  [{status}]")
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
