#!/usr/bin/env python
"""
Generate the detection visualization gallery for Milestone 3.
Creates annotated images with 4-color error coding in gallery/:
  Green  = TP (correct class, IoU >= 0.5)
  Orange = Misclassified (IoU >= 0.5, wrong class)
  Red    = FP (IoU < 0.5, no matching GT)
  Blue   = FN (missed GT box)

Run from anywhere:
    python generate_gallery.py
"""
import os
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TARGET_IDS = {"7", "797", "2619", "4428", "7007"}
GALLERY_DIR = os.path.join(SCRIPT_DIR, "gallery")


def find_weights():
    search_dir = os.path.join(SCRIPT_DIR, "runs", "detect")
    for root, _dirs, files in os.walk(search_dir):
        if "best.pt" in files and "sweep" not in root.lower():
            return os.path.join(root, "best.pt")
    for root, _dirs, files in os.walk(search_dir):
        if "best.pt" in files:
            return os.path.join(root, "best.pt")
    return None


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes(lbl_path, img_w, img_h):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            ci = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            boxes.append({"class": ci, "bbox": [x1, y1, x2, y2]})
    return boxes


def annotate_with_errors(img, gt_boxes, dets, class_names):
    """Annotate image with 4-color coding. Returns (annotated_img, counts_dict)."""
    annotated = img.copy()
    n_tp = n_fp = n_fn = n_mis = 0

    # Build detection list using indexed access
    det_list = []
    for i in range(len(dets)):
        x1, y1, x2, y2 = map(int, dets.xyxy[i].tolist())
        ci = int(dets.cls[i].item())
        conf = float(dets.conf[i].item())
        det_list.append({"bbox": [x1, y1, x2, y2], "class": ci, "conf": conf})

    gt_matched = [False] * len(gt_boxes)

    for det in det_list:
        best_iou, best_gt_idx = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            iou = compute_iou(det["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= 0.5 and best_gt_idx >= 0:
            gt = gt_boxes[best_gt_idx]
            if det["class"] == gt["class"]:
                color = (0, 200, 0)      # Green = TP
                n_tp += 1
            else:
                color = (0, 128, 255)    # Orange = Misclassified
                n_mis += 1
            gt_matched[best_gt_idx] = True
        else:
            color = (0, 0, 255)          # Red = FP
            n_fp += 1

        x1, y1, x2, y2 = det["bbox"]
        label = f"{class_names.get(det['class'], str(det['class']))} ({det['conf']:.2f})"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(annotated, (x1, max(y1 - th - 6, 0)), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw missed GT boxes (Blue = FN)
    for j, gt in enumerate(gt_boxes):
        if not gt_matched[j]:
            x1, y1, x2, y2 = gt["bbox"]
            color = (255, 128, 0)        # Blue in BGR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.rectangle(annotated, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), color, 1, cv2.LINE_AA)
            label = f"GT:{class_names.get(gt['class'], str(gt['class']))} (missed)"
            cv2.putText(annotated, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            n_fn += 1

    return annotated, {"TP": n_tp, "FP": n_fp, "FN": n_fn, "Misclassified": n_mis}


def main():
    from ultralytics import YOLO

    weights = find_weights()
    if weights is None:
        print("ERROR: No best.pt found. Train the model first.")
        return

    print(f"Loading: {weights}")
    model = YOLO(weights)
    names = model.names
    target_indices = {i for i, n in names.items() if n in TARGET_IDS}

    # Pick images: all test + all val + 5 train
    splits = {
        "test":  os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "images", "test"),
        "val":   os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "images", "val"),
        "train": os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "images", "train"),
    }
    lbl_splits = {
        "test":  os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "labels", "test"),
        "val":   os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "labels", "val"),
        "train": os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "labels", "train"),
    }

    selected = []
    for split, take in [("test", None), ("val", None), ("train", 5)]:
        d = splits[split]
        imgs = sorted(f for f in os.listdir(d) if f.endswith((".jpg", ".png")))
        if take:
            imgs = imgs[:take]
        for f in imgs:
            selected.append((split, f))

    print(f"Generating {len(selected)} gallery examples")
    print(f"Color legend: Green=TP  Orange=Misclassified  Red=FP  Blue=FN")
    os.makedirs(GALLERY_DIR, exist_ok=True)

    captions = []
    for split, img_name in selected:
        img_path = os.path.join(splits[split], img_name)
        lbl_path = os.path.join(lbl_splits[split], img_name.rsplit(".", 1)[0] + ".txt")

        # Model prediction
        results = model.predict(img_path, conf=0.1, imgsz=640, verbose=False)
        dets = results[0].boxes

        # Load image and GT
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        gt_boxes = load_gt_boxes(lbl_path, w, h)

        # Annotate with 4-color coding
        annotated, counts = annotate_with_errors(img, gt_boxes, dets, names)

        # Save
        out_name = f"{split}_{img_name}"
        cv2.imwrite(os.path.join(GALLERY_DIR, out_name), annotated)

        # Caption
        n_gt_target = sum(1 for g in gt_boxes if g["class"] in target_indices)
        parts = []
        if counts["TP"] > 0:
            parts.append(f"{counts['TP']} TP")
        if counts["Misclassified"] > 0:
            parts.append(f"{counts['Misclassified']} Misclassified")
        if counts["FP"] > 0:
            parts.append(f"{counts['FP']} FP")
        if counts["FN"] > 0:
            parts.append(f"{counts['FN']} FN (missed)")
        caption = (f"{split}/{img_name} | "
                   f"{"  ".join(parts) if parts else "no detections"} | "
                   f"GT: {len(gt_boxes)} faces ({n_gt_target} target)")
        captions.append(caption)
        print(f"  {out_name}: {counts['TP']} TP, {counts['Misclassified']} Mis, "
              f"{counts['FP']} FP, {counts['FN']} FN")

    # Write captions
    cap_path = os.path.join(GALLERY_DIR, "captions.md")
    with open(cap_path, "w") as f:
        f.write("# Detection Visualization Gallery\n\n")
        f.write("**Color legend:**  \\n")
        f.write("- **Green** = TP (correct class, IoU >= 0.5)  \\n")
        f.write("- **Orange** = Misclassified (IoU >= 0.5, wrong class)  \\n")
        f.write("- **Red** = FP (no GT match, IoU < 0.5)  \\n")
        f.write("- **Blue** = FN (missed GT box)\n\n")
        for i, cap in enumerate(captions):
            fname = cap.split(" | " )[0]
            f.write(f"### {i+1}. {fname}\n")
            f.write(f"{cap.split('|', 2)[-1].strip()}\n\n")
    print(f"\nSaved {len(selected)} images + captions to {GALLERY_DIR}")


if __name__ == "__main__":
    main()
