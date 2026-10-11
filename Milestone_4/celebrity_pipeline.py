#!/usr/bin/env python
"""
Milestone 4 - Integrated Celebrity Detection & Identification Pipeline
======================================================================
Combines M1 ResNet50 classifier + M3 YOLOv8 detector into one system.

Usage:
    python celebrity_pipeline.py --image photo.jpg
    python celebrity_pipeline.py --image photo.jpg --save
    python celebrity_pipeline.py                          # run on test images
"""

import argparse
import os
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CLASSIFIER_PATH = os.path.join(SCRIPT_DIR, "models", "resnet50_classifier.keras")
CLASS_NAMES_5 = ['7', '797', '2619', '4428', '7007']
TARGET_IDS = {"7", "797", "2619", "4428", "7007"}


def find_yolo_weights():
    search_dir = os.path.join(PROJECT_ROOT, "Milestone_3", "runs", "detect")
    candidates = []
    for root, _dirs, files in os.walk(search_dir):
        if "best.pt" in files:
            candidates.append(os.path.join(root, "best.pt"))
    for c in candidates:
        if "sweep" not in c.lower():
            return c
    return candidates[0] if candidates else None


class CelebrityPipeline:
    """Integrated detection + identification pipeline."""

    def __init__(self, conf=0.25, imgsz=640):
        from ultralytics import YOLO
        from tensorflow.keras.models import load_model

        yolo_path = find_yolo_weights()
        if yolo_path is None:
            raise FileNotFoundError("YOLOv8 best.pt not found. Run Milestone 3 first.")
        if not os.path.exists(CLASSIFIER_PATH):
            raise FileNotFoundError(
                f"Classifier not found: {CLASSIFIER_PATH}\n"
                "Run: python train_classifier.py")

        print(f"Loading YOLOv8: {yolo_path}")
        self.detector = YOLO(yolo_path)
        self.yolo_names = self.detector.names

        print(f"Loading ResNet50: {CLASSIFIER_PATH}")
        from tensorflow.keras.applications import ResNet50 as RN50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Input, GlobalAveragePooling2D,
                                             Dense, Dropout, Lambda)
        from tensorflow.keras import regularizers as reg

        NUM_CLS = 5
        l2 = reg.l2(1e-3)
        base = RN50(weights=None, include_top=False, input_shape=(224, 224, 3))
        for layer in base.layers:
            if not layer.name.startswith('conv5_block3'):
                layer.trainable = False

        self.classifier = Sequential([
            Input(shape=(224, 224, 3)),
            Lambda(lambda x: preprocess_input(x)),
            base,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu', kernel_regularizer=l2),
            Dropout(0.30),
            Dense(NUM_CLS, activation='softmax')
        ])
        # Load weights (skip augmentation layers — they have no weights)
        weights_path = os.path.join(SCRIPT_DIR, "models", "resnet50_weights.h5")
        if os.path.exists(weights_path):
            self.classifier.load_weights(weights_path, by_name=True)
        else:
            self.classifier.load_weights(CLASSIFIER_PATH, by_name=True)

        self.conf = conf
        self.imgsz = imgsz
        self.target_indices = {
            i for i, n in self.yolo_names.items() if n in TARGET_IDS
        }
        print(f"Pipeline ready. {len(self.yolo_names)} det classes, "
              f"{len(CLASS_NAMES_5)} cls classes.")

    def detect_and_identify(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]

        results = self.detector.predict(
            image_path, conf=self.conf, imgsz=self.imgsz, verbose=False)
        detections = []
        for b in results[0].boxes:
            cls_idx = int(b.cls[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "yolo_class": self.yolo_names.get(cls_idx, str(cls_idx)),
                "yolo_conf": b.conf[0].item(),
                "is_target": cls_idx in self.target_indices,
            })

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                det.update(identity=det["yolo_class"],
                           confidence=det["yolo_conf"], source="detector")
                continue
            crop_resized = cv2.resize(crop, (224, 224))
            crop_arr = crop_resized.astype(np.float32)[np.newaxis, ...]
            probs = self.classifier.predict(crop_arr, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            if top_prob > 0.5:
                det.update(identity=CLASS_NAMES_5[top_idx],
                           confidence=top_prob, source="classifier")
            else:
                det.update(identity=det["yolo_class"],
                           confidence=det["yolo_conf"], source="detector")

        return detections, img

    def annotate(self, img, detections):
        annotated = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 200, 0) if det["is_target"] else (255, 160, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['identity']} ({det['confidence']:.2f})"
            if det["is_target"]:
                # Classifier (green): label above the box
                cv2.putText(annotated, label, (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            else:
                # Detector (orange): label inside box, bottom-left
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.putText(annotated, label, (x1 + 3, y2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return annotated


def main():
    parser = argparse.ArgumentParser(
        description="Celebrity Detection & Identification Pipeline (Milestone 4)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default 0.25)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default 640)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated images to pipeline_output/")
    args = parser.parse_args()

    pipeline = CelebrityPipeline(conf=args.conf, imgsz=args.imgsz)

    if args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            sys.exit(1)
        images = [args.image]
    else:
        tdir = os.path.join(PROJECT_ROOT, "data", "synthetic_grid", "images", "test")
        if not os.path.isdir(tdir):
            print("ERROR: Test images not found. Use --image to specify a photo.")
            sys.exit(1)
        images = sorted(
            os.path.join(tdir, f) for f in os.listdir(tdir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")))

    print(f"\nRunning on {len(images)} image(s)\n")

    out_dir = os.path.join(SCRIPT_DIR, "pipeline_output")
    if args.save:
        os.makedirs(out_dir, exist_ok=True)

    for img_path in images:
        name = os.path.basename(img_path)
        detections, img = pipeline.detect_and_identify(img_path)
        n_cls = sum(1 for d in detections if d["source"] == "classifier")
        n_det = sum(1 for d in detections if d["source"] == "detector")

        print(f"  {name}: {len(detections)} face(s) "
              f"({n_cls} classified, {n_det} from detector)")
        for d in sorted(detections, key=lambda x: -x["confidence"]):
            star = " *" if d["is_target"] else ""
            src = "cls" if d["source"] == "classifier" else "det"
            print(f"    {d['identity']:>6s}  conf={d['confidence']:.4f}  [{src}]{star}")

        if args.save:
            annotated = pipeline.annotate(img, detections)
            out_path = os.path.join(out_dir, name)
            cv2.imwrite(out_path, annotated)
            print(f"    Saved: {out_path}")

    if not args.save:
        try:
            import matplotlib.pyplot as plt
            n = len(images)
            fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
            if n == 1:
                axes = [axes]
            for idx, img_path in enumerate(images):
                detections, img = pipeline.detect_and_identify(img_path)
                annotated = cv2.cvtColor(
                    pipeline.annotate(img, detections), cv2.COLOR_BGR2RGB)
                axes[idx].imshow(annotated)
                axes[idx].set_title(os.path.basename(img_path), fontsize=10)
                axes[idx].axis("off")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("(matplotlib not available - use --save to export images)")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()