#!/usr/bin/env python
"""
Generate the detection visualization gallery for Milestone 3.
Creates annotated images (model predictions + ground truth) in gallery/.

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

    # Pick 12 images: all 3 test + all 4 val + 5 train
    splits = {
        "test": os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "images", "test"),
        "val":  os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "images", "val"),
        "train": os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "images", "train"),
    }
    lbl_splits = {
        "test": os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "labels", "test"),
        "val":  os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "labels", "val"),
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
    os.makedirs(GALLERY_DIR, exist_ok=True)

    captions = []
    for split, img_name in selected:
        img_path = os.path.join(splits[split], img_name)
        lbl_path = os.path.join(lbl_splits[split], img_name.rsplit(".", 1)[0] + ".txt")

        # Model prediction
        results = model.predict(img_path, conf=0.1, imgsz=640, verbose=False)
        annotated = results[0].plot()

        # Ground truth overlay
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        n_gt = 0
        n_target_gt = 0
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_idx = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    color = (0, 200, 0) if cls_idx in target_indices else (200, 100, 100)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = names.get(cls_idx, str(cls_idx))
                    if cls_idx in target_indices:
                        label += " *"
                    cv2.putText(img, label, (x1, max(y1 - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    n_gt += 1
                    if cls_idx in target_indices:
                        n_target_gt += 1

        # Side-by-side: prediction | ground truth
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        combined = np.concatenate([annotated_rgb, img_rgb], axis=1)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        out_name = f"{split}_{img_name}"
        cv2.imwrite(os.path.join(GALLERY_DIR, out_name), combined_bgr)

        # Caption
        n_det = len(results[0].boxes)
        dets = []
        for b in results[0].boxes:
            ci = int(b.cls[0])
            dets.append((names.get(ci, str(ci)), b.conf[0].item(), ci in target_indices))
        dets.sort(key=lambda x: -x[1])
        top = ", ".join(f"{c}{('*' if t else '')}" for c, _, t in dets[:5])
        caption = (f"{split}/{img_name} | "
                   f"Pred: {n_det} det (top: {top}) | "
                   f"GT: {n_gt} faces ({n_target_gt} target)")
        captions.append(caption)
        print(f"  {out_name}: {n_det} det, {n_gt} GT")

    # Write captions
    cap_path = os.path.join(GALLERY_DIR, "captions.md")
    with open(cap_path, "w") as f:
        f.write("# Detection Visualization Gallery\n\n")
        f.write("Left panel: model prediction (boxes + class + confidence).  ")
        f.write("Right panel: ground truth (green = target celebrity `*`, red = other).\n\n")
        for i, cap in enumerate(captions):
            fname = cap.split(" | ")[0]
            f.write(f"### {i+1}. {fname}\n")
            f.write(f"{cap.split('|', 2)[-1].strip()}\n\n")
    print(f"\nSaved {len(selected)} images + captions to {GALLERY_DIR}")


if __name__ == "__main__":
    main()