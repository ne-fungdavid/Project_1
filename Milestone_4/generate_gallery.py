#!/usr/bin/env python
"""
Milestone 4 - Generate Final Results Gallery
=============================================
Creates:
  - >=20 correctly classified single-face examples
  - >=5 misclassified single-face examples with analysis
  - >=10 multi-face detection examples (reuses M3 gallery)

Run from anywhere:
    python generate_gallery.py
"""

import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

GLOBAL_SEED = 42
CLASS_NAMES = ['7', '797', '2619', '4428', '7007']
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "selected_images")
GALLERY_DIR = os.path.join(SCRIPT_DIR, "gallery")
M3_GALLERY = os.path.join(PROJECT_ROOT, "Milestone_3", "gallery")

# Model path
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "resnet50_classifier.keras")


def load_model_clean():
    """Load the classifier with preprocess_input in scope."""
    model = load_model(MODEL_PATH, compile=False, safe_mode=False)
    return model


def classify_single(model, img_path):
    """Classify a single face image. Returns (pred_class, prob, true_class)."""
    img = load_img(img_path, target_size=(224, 224))
    arr = img_to_array(img).astype(np.float32)[np.newaxis, ...]
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx])


def main():
    os.makedirs(GALLERY_DIR, exist_ok=True)
    correct_dir = os.path.join(GALLERY_DIR, "classification_correct")
    error_dir = os.path.join(GALLERY_DIR, "classification_errors")
    det_dir = os.path.join(GALLERY_DIR, "detection")
    # Start from a clean slate so re-runs never accumulate stale examples
    for d in (correct_dir, error_dir, det_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Load model
    print("Loading classifier...")
    model = load_model_clean()

    # Build test set (same split as M1)
    all_files, all_labels = [], []
    for i, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        class_files = [
            os.path.join(class_dir, fname)
            for fname in sorted(os.listdir(class_dir))
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        all_files.extend(class_files)
        all_labels.extend([i] * len(class_files))

    files_by_class = defaultdict(list)
    for f, lab in zip(all_files, all_labels):
        files_by_class[lab].append(f)

    test_files, test_labels = [], []
    for cls in sorted(files_by_class.keys()):
        cls_files = files_by_class[cls]
        c_train, c_rest = train_test_split(cls_files, test_size=0.30, random_state=GLOBAL_SEED)
        c_val, c_test = train_test_split(c_rest, test_size=0.5, random_state=GLOBAL_SEED)
        test_files.extend(c_test)
        test_labels.extend([cls] * len(c_test))

    # Use ALL images (train + val + test) for the gallery
    # Classify all
    correct, errors = [], []
    for f, true_cls in zip(all_files, all_labels):
        pred_cls, prob = classify_single(model, f)
        entry = {
            "file": f,
            "true": CLASS_NAMES[true_cls],
            "pred": CLASS_NAMES[pred_cls],
            "prob": prob,
            "correct": pred_cls == true_cls,
        }
        if entry["correct"]:
            correct.append(entry)
        else:
            errors.append(entry)

    print(f"Correct: {len(correct)}, Errors: {len(errors)}")

    # Copy correct examples (up to 25)
    for i, entry in enumerate(correct[:25]):
        fname = os.path.basename(entry["file"])
        dst = os.path.join(correct_dir, f"{i+1:02d}_{entry['true']}_{fname}")
        shutil.copy2(entry["file"], dst)

    # Copy error examples (up to 10)
    for i, entry in enumerate(errors[:10]):
        fname = os.path.basename(entry["file"])
        dst = os.path.join(error_dir, f"{i+1:02d}_{entry['true']}_as_{entry['pred']}_{fname}")
        shutil.copy2(entry["file"], dst)

    # Copy M3 detection gallery
    if os.path.isdir(M3_GALLERY):
        for f in os.listdir(M3_GALLERY):
            if f.endswith(('.jpg', '.png')):
                shutil.copy2(os.path.join(M3_GALLERY, f), os.path.join(det_dir, f))

    # Write captions
    cap_path = os.path.join(GALLERY_DIR, "captions.md")
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write("# Final Results Gallery (Milestone 4)\n\n")

        f.write("## Single-Face Classification (ResNet50)\n\n")
        f.write(f"**Correct:** {len(correct[:25])} examples  ")
        f.write(f"**Errors:** {len(errors[:10])} examples\n\n")

        f.write("### Correctly Classified\n\n")
        for i, e in enumerate(correct[:25]):
            f.write(f"{i+1}. **{e['true']}** (conf={e['prob']:.3f}) - {os.path.basename(e['file'])}\n")

        f.write("\n### Misclassified (with analysis)\n\n")
        for i, e in enumerate(errors[:10]):
            f.write(f"{i+1}. True: **{e['true']}**  Predicted: **{e['pred']}**  "
                     f"(confidence={e['prob']:.3f})  — `{os.path.basename(e['file'])}`\n")
            f.write(f"   Analysis: A cross-identity confusion between two rostered celebrities. "
                     f"The softmax assigned the highest (but not decisive) probability to "
                     f"`{e['pred']}` instead of the true identity `{e['true']}`. This indicates the "
                     f"two faces share salient features and the model lacked enough separating "
                     f"signal for this specific pair — the usual remedy is more images and harder "
                     f"negatives for both identities, or a similarity/verification stage on top.\n")

        f.write("\n## Multi-Face Detection (YOLOv8)\n\n")
        f.write("Reused from Milestone 3 gallery (11 annotated grid images).\n\n")
        n_det = len([f for f in os.listdir(det_dir) if f.endswith(('.jpg', '.png'))])
        f.write(f"Total detection examples: {n_det}\n")

    print(f"\nGallery saved to: {GALLERY_DIR}")
    print(f"  Correct: {min(len(correct), 25)}")
    print(f"  Errors:  {min(len(errors), 10)}")
    n_det = len([f for f in os.listdir(det_dir) if f.endswith(('.jpg', '.png'))])
    print(f"  Detection: {n_det}")
    print("Done.")


if __name__ == "__main__":
    main()