#!/usr/bin/env python
"""
Milestone 4 - Retrain and Save the ResNet50 Classifier
========================================================
Extracts the ResNet50 fine-tuning model from the Milestone 1 notebook
and saves it as a .keras file for use in the integrated pipeline.

Run from anywhere:
    python train_classifier.py
"""

import os
import time
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import load_img, img_to_array, to_categorical

# PROJECT CONSTANTS (same as Milestone 1)
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
keras.utils.set_random_seed(GLOBAL_SEED)

BATCH = 16
EPOCHS = 30
IMG_SIZE = (224, 224)
NUM_CLASSES = 5
CLASS_NAMES = ['7', '797', '2619', '4428', '7007']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "selected_images")
SAVE_PATH = os.path.join(SCRIPT_DIR, "models", "resnet50_classifier.keras")

l2_reg = regularizers.l2(1e-3)
LR_FT = 1e-4
FT_UNFREEZE_PREFIXES = ('conv5_block3',)


def set_all_seeds():
    np.random.seed(GLOBAL_SEED)
    keras.utils.set_random_seed(GLOBAL_SEED)


def load_data():
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

    train_files, val_files, test_files = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for cls in sorted(files_by_class.keys()):
        cls_files = files_by_class[cls]
        c_train, c_rest = train_test_split(cls_files, test_size=0.30, random_state=GLOBAL_SEED)
        c_val, c_test = train_test_split(c_rest, test_size=0.5, random_state=GLOBAL_SEED)
        train_files.extend(c_train); val_files.extend(c_val); test_files.extend(c_test)
        train_labels.extend([cls] * len(c_train))
        val_labels.extend([cls] * len(c_val))
        test_labels.extend([cls] * len(c_test))

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    return (train_files, train_labels, val_files, val_labels, test_files, test_labels)


def load_image_arrays(files, labels):
    x = np.zeros((len(files), IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    for i, f in enumerate(files):
        img = load_img(f, target_size=IMG_SIZE)
        x[i] = img_to_array(img).astype(np.float32)
    y = to_categorical(labels, NUM_CLASSES)
    return x, y


def build_model():
    """ResNet50 fine-tuning model (same architecture as M1 Task 4)."""
    set_all_seeds()
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers:
        if not any(layer.name.startswith(p) for p in FT_UNFREEZE_PREFIXES):
            layer.trainable = False

    model = Sequential([
        Input(shape=(224, 224, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1),
        Lambda(lambda x: preprocess_input(x)),
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2_reg),
        Dropout(0.30),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_FT),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    trainable = sum(w.shape.num_elements() for w in model.trainable_weights)
    print(f"Trainable params: {trainable:,}")
    return model


def main():
    print("=" * 60)
    print("ResNet50 Classifier - Retrain & Save")
    print("=" * 60)

    (train_files, train_labels, val_files, val_labels,
     test_files, test_labels) = load_data()

    X_train, y_train = load_image_arrays(train_files, train_labels)
    X_val, y_val = load_image_arrays(val_files, val_labels)
    X_test, y_test = load_image_arrays(test_files, test_labels)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    model = build_model()

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    ckpt_dir = os.path.join(tempfile.gettempdir(), 'm4_classifier_ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, 'best.weights.h5')

    best_cb = ModelCheckpoint(best_ckpt, monitor='val_accuracy', mode='max',
                              save_best_only=True, save_weights_only=True, verbose=0)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)

    print(f"\nTraining for {EPOCHS} epochs...")
    t0 = time.time()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
              validation_data=(X_val, y_val), callbacks=[best_cb, lr_cb], verbose=1)
    train_time = time.time() - t0

    model.load_weights(best_ckpt)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Training Time: {train_time:.1f}s")

    model.save(SAVE_PATH)
    size_mb = os.path.getsize(SAVE_PATH) / 1e6
    print(f"\nModel saved to: {SAVE_PATH} ({size_mb:.1f} MB)")

    import shutil
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()