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
                })
            conf = low
        all_results.append((img_path, results, detections, conf))

    print(f"\n{'=' * 60}")
    print(f"DETECTION RESULTS  (* = target celebrity)")
    print(f"{'=' * 60}")
    n_total = n_target = 0
    for img_path, _res, detections, conf_used in all_results:
        img_name = os.path.basename(img_path)
        n_det = len(detections)
        n_targ = sum(1 for d in detections if d["is_target"])
        n_total += n_det
        n_target += n_targ
        print(f"\n  {img_name}  ({n_det} det, {n_targ} target)  [conf={conf_used}]")
        for d in sorted(detections, key=lambda x: -x["confidence"]):
            mark = " *" if d["is_target"] else ""
            print(f"    {d['class_id']:>6s}  conf={d['confidence']:.4f}{mark}")

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {n_total} detections, {n_target} target celebrities")
    if n_total == 0:
        print("NOTE: Model may need more training epochs for confident detections.")
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