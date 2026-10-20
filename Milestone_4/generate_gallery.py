#!/usr/bin/env python
"""
Milestone 4 - Generate Final Results Gallery (Baseline 5-class model)
======================================================================
Runs the integrated pipeline (YOLO detection + ResNet50 classification)
on the test, train, and validation synthetic grid images.

Output in gallery/:
  - Annotated test images (root of gallery/) + captions.md
  - train/ subfolder: annotated train images + captions
  - validation/ subfolder: annotated validation images + captions

Run from anywhere:
    python generate_gallery.py
"""

import os
import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.src.utils import python_utils
python_utils.preprocess_input = preprocess_input

from tensorflow.keras.models import load_model
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CLASS_NAMES_5 = ['7', '797', '2619', '4428', '7007']
CLASS_NAMES_13 = ["3", "7", "797", "1212", "2336", "2619", "2970",
                  "4422", "4428", "5695", "7007", "8335", "10002"]
TARGET_IDS = {"7", "797", "2619", "4428", "7007"}

TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "images", "test")
TEST_LBL_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "labels", "test")
TRAIN_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "images", "train")
TRAIN_LBL_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "labels", "train")
VAL_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "images", "val")
VAL_LBL_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "labels", "val")
GALLERY_DIR = os.path.join(SCRIPT_DIR, "gallery")
CLASSIFIER_PATH = os.path.join(SCRIPT_DIR, "models", "resnet50_classifier.keras")
DET_CONF = 0.1
ID_THRESH = 0.5
YOLO_THRESH = 0.3
MIN_KEEP_CONF = 0.05


def find_yolo_weights():
    canon = os.path.join(PROJECT_ROOT, "Milestone_3", "runs", "detect",
                         "celebrity_detection", "yolov8n_finetune", "weights", "best.pt")
    if os.path.exists(canon):
        return canon
    search_dir = os.path.join(PROJECT_ROOT, "Milestone_3", "runs", "detect")
    candidates = []
    for root, _dirs, files in os.walk(search_dir):
        if "best.pt" in files:
            candidates.append(os.path.join(root, "best.pt"))
    for c in candidates:
        if "sweep" not in c.lower():
            return c
    return candidates[0] if candidates else None


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


def nms_class_agnostic(boxes, confs, iou_thresh=0.5):
    """Class-agnostic NMS: suppress overlapping boxes regardless of class label.

    YOLO's built-in NMS is class-aware, so it keeps multiple near-identical boxes
    if they claim different classes. This helper collapses them (keeping the
    highest-confidence box) so a single face is never counted several times.

    Args:
        boxes: list of [x1, y1, x2, y2]
        confs: list of confidence values (same order as boxes)
        iou_thresh: suppress boxes whose IoU with a kept box exceeds this
    Returns:
        list of indices to keep (in descending confidence order)
    """
    if len(boxes) == 0:
        return []
    ba = np.asarray(boxes, dtype=np.float64)
    bx1, by1, bx2, by2 = ba[:, 0], ba[:, 1], ba[:, 2], ba[:, 3]
    areas = np.maximum(0, bx2 - bx1) * np.maximum(0, by2 - by1)
    order = np.argsort(np.asarray(confs))[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(bx1[i], bx1[order[1:]])
        yy1 = np.maximum(by1[i], by1[order[1:]])
        xx2 = np.minimum(bx2[i], bx2[order[1:]])
        yy2 = np.minimum(by2[i], by2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


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
            boxes.append({"class_idx": ci, "class_name": CLASS_NAMES_13[ci],
                          "bbox": [x1, y1, x2, y2]})
    return boxes


def process_split(img_dir, lbl_dir, output_dir, prefix, split_name,
                  classifier, detector, yolo_names, model_desc="Baseline 5-class",
                  captions_name="captions.md"):
    """Process one split: detect, classify, annotate, save images + captions."""
    os.makedirs(output_dir, exist_ok=True)
    img_files = sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not img_files:
        print(f"  [{split_name}] No images found in {img_dir}, skipping.")
        return []
    print(f"\n  [{split_name}] Processing {len(img_files)} images...")
    all_records = []
    for img_path in img_files:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        results = detector.predict(img_path, conf=DET_CONF, imgsz=640, verbose=False)
        dets = results[0].boxes
        gt_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(lbl_dir, gt_name + ".txt")
        gt_boxes = load_gt_boxes(lbl_path, w, h)
        det_list = []
        for i in range(len(dets)):
            x1, y1, x2, y2 = [int(v) for v in dets.xyxy[i].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            ci = int(dets.cls[i].item())
            conf = float(dets.conf[i].item())
            det_list.append({"bbox": [x1, y1, x2, y2], "class_idx": ci,
                             "class_name": yolo_names.get(ci, str(ci)), "conf": conf})
        # Drop very low-confidence boxes before NMS/classification to avoid false
        # positives in empty regions when DET_CONF is set low.
        det_list = [d for d in det_list if d["conf"] >= MIN_KEEP_CONF]
        # Collapse duplicate/overlapping detections of the same face (which YOLO
        # emits with different class labels) using class-agnostic NMS.
        if len(det_list) > 0:
            keep_idx = nms_class_agnostic([d["bbox"] for d in det_list],
                                          [d["conf"] for d in det_list], iou_thresh=0.5)
            det_list = [det_list[i] for i in keep_idx]
        target_indices = {i for i, n in yolo_names.items() if n in TARGET_IDS}
        gt_matched = [False] * len(gt_boxes)
        annotated = img.copy()
        image_faces = []
        for det in det_list:
            best_iou, best_gt_idx = 0.0, -1
            for j, gt in enumerate(gt_boxes):
                iou = compute_iou(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            is_tp = best_iou > 0.5 and best_gt_idx >= 0
            gt_label = gt_boxes[best_gt_idx]["class_name"] if is_tp else None
            if is_tp:
                gt_matched[best_gt_idx] = True
            x1, y1, x2, y2 = det["bbox"]
            # Prefer YOLO's identity when it is confident and in the target set.
            if det["conf"] >= YOLO_THRESH and det["class_idx"] in target_indices:
                pred_label = det["class_name"]
                confidence = det["conf"]
                source = "yolo"
            else:
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (224, 224))
                    crop_arr = crop_resized.astype(np.float32)[np.newaxis, ...]
                    probs = classifier.predict(crop_arr, verbose=0)[0]
                    top_idx = int(np.argmax(probs))
                    top_prob = float(probs[top_idx])
                    if top_prob >= ID_THRESH:
                        pred_label = CLASS_NAMES_5[top_idx]
                        confidence = top_prob
                        source = "classifier"
                    else:
                        pred_label = det["class_name"]
                        confidence = det["conf"]
                        source = "detector"
                else:
                    pred_label = det["class_name"]
                    confidence = det["conf"]
                    source = "detector"
            if is_tp and pred_label == gt_label:
                correct = True; color = (0, 200, 0)
            elif is_tp:
                correct = False; color = (0, 128, 255)
            else:
                correct = False; color = (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{pred_label} ({confidence:.2f})"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if is_tp and pred_label == gt_label:
                tx, ty = x1 + 2, y1 - 2
            elif is_tp:
                tx, ty = x2 - tw - 2, y1 - 2
            else:
                tx, ty = x1 + 2, y2 + 2
            tx = max(0, min(tx, w - tw))
            ty = max(th, min(ty, h))
            cv2.rectangle(annotated, (tx, ty - th), (tx + tw, ty + baseline), color, -1)
            cv2.putText(annotated, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            image_faces.append({"det_label": det["class_name"], "pred_label": pred_label,
                                "confidence": confidence, "source": source,
                                "gt_label": gt_label, "iou": best_iou, "correct": correct})
        for j, gt in enumerate(gt_boxes):
            if not gt_matched[j]:
                gx1, gy1, gx2, gy2 = gt["bbox"]
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (255,128,0), 2)
                fn_text = f"FN: {gt['class_name']}"
                (tw2, th2), bl2 = cv2.getTextSize(fn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                fx = max(0, min(gx2 - tw2 - 2, w - tw2))
                fy = max(th2, min(gy2 + 2, h))
                cv2.rectangle(annotated, (fx, fy - th2), (fx + tw2, fy + bl2), (255,128,0), -1)
                cv2.putText(annotated, fn_text, (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imwrite(os.path.join(output_dir, f"{prefix}{img_name}"), annotated)
        n_correct = sum(1 for f in image_faces if f["correct"])
        n_incorrect = sum(1 for f in image_faces if not f["correct"])
        n_fn = sum(1 for m in gt_matched if not m)
        print(f"    {img_name}: {len(image_faces)} faces -> {n_correct} correct, "
              f"{n_incorrect} incorrect, {n_fn} missed (FN)")
        all_records.append({"image": img_name, "faces": image_faces,
                            "n_correct": n_correct, "n_incorrect": n_incorrect,
                            "n_fn": n_fn, "n_gt": len(gt_boxes)})
    # Write captions
    tc = sum(r["n_correct"] for r in all_records)
    ti = sum(r["n_incorrect"] for r in all_records)
    tfn = sum(r["n_fn"] for r in all_records)
    tfaces = sum(len(r["faces"]) for r in all_records)
    tgt = sum(r["n_gt"] for r in all_records)
    cap_path = os.path.join(output_dir, captions_name)
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(f"# Results Gallery \u2014 {split_name} Split ({model_desc})\n\n")
        f.write(f"**Source:** {len(img_files)} synthetic grid images\n\n")
        f.write("**Color legend:**  \n")
        f.write("- **Green** = Correct (detected + classified correctly)  \n")
        f.write("- **Orange** = Misclassified (detected, IoU > 0.5, wrong class)  \n")
        f.write("- **Red** = False Positive (no GT match, IoU < 0.5)  \n")
        f.write("- **Blue/Orange border** = False Negative (missed GT)\n\n")
        for rec in all_records:
            f.write(f"## {rec['image']}\n\n")
            f.write(f"Image: `{prefix}{rec['image']}`\n\n")
            f.write(f"Detected: {len(rec['faces'])} faces | Correct: {rec['n_correct']} | "
                    f"Incorrect: {rec['n_incorrect']} | Missed (FN): {rec['n_fn']} | "
                    f"GT total: {rec['n_gt']}\n\n")
            f.write("| # | Predicted | Confidence | Source | Ground Truth | Status |\n")
            f.write("|---|-----------|------------|--------|--------------|--------|\n")
            for i, face in enumerate(rec["faces"], 1):
                status = "CORRECT" if face["correct"] else "INCORRECT"
                gt_str = face["gt_label"] if face["gt_label"] else "No match (FP)"
                f.write(f"| {i} | {face['pred_label']} | {face['confidence']:.4f} | "
                        f"{face['source']} | {gt_str} | {status} |\n")
            f.write("\n")
        f.write("---\n\n## Summary\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total faces detected | {tfaces} |\n")
        f.write(f"| Correctly classified | {tc} |\n")
        f.write(f"| Incorrectly classified | {ti} |\n")
        f.write(f"| Missed (FN) | {tfn} |\n")
        if tfaces > 0:
            f.write(f"| Accuracy | {100*tc/tfaces:.1f}% |\n")
        f.write(f"| Total images | {len(img_files)} |\n")
        f.write(f"| Total GT faces | {tgt} |\n")
    print(f"    Saved {len(img_files)} annotated images + captions to: {output_dir}")
    return all_records


def main():
    print("Loading classifier (5-class baseline)...")
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"Classifier not found: {CLASSIFIER_PATH}\nRun: python train_classifier.py")
    classifier = load_model(CLASSIFIER_PATH, compile=False, safe_mode=False)
    from ultralytics import YOLO
    yolo_path = find_yolo_weights()
    if yolo_path is None:
        raise FileNotFoundError("YOLOv8 best.pt not found. Run Milestone 3 first.")
    print(f"Loading YOLOv8: {yolo_path}")
    detector = YOLO(yolo_path)
    yolo_names = detector.names

    # === TEST SET (primary deliverable - output location unchanged) ===
    print("\n" + "="*60)
    print("  TEST SET")
    print("="*60)
    process_split(TEST_IMG_DIR, TEST_LBL_DIR, GALLERY_DIR, "baseline_", "Test",
                  classifier, detector, yolo_names)

    # === TRAIN SET (additional gallery coverage) ===
    print("\n" + "="*60)
    print("  TRAIN SET")
    print("="*60)
    train_out = os.path.join(GALLERY_DIR, "train")
    process_split(TRAIN_IMG_DIR, TRAIN_LBL_DIR, train_out, "baseline_", "Train",
                  classifier, detector, yolo_names)

    # === VALIDATION SET (additional gallery coverage) ===
    print("\n" + "="*60)
    print("  VALIDATION SET")
    print("="*60)
    val_out = os.path.join(GALLERY_DIR, "validation")
    process_split(VAL_IMG_DIR, VAL_LBL_DIR, val_out, "baseline_", "Validation",
                  classifier, detector, yolo_names)

    print(f"\nAll galleries saved to: {GALLERY_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
