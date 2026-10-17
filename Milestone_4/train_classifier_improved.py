#!/usr/bin/env python
"""
Milestone 4 - Retrain and Save the ResNet50 Classifier
========================================================
Faithfully reproduces the Milestone 1 two-stage ResNet50 training:
  Stage 1 (feature extraction): frozen ResNet50 + trained classification head.
  Stage 2 (fine-tuning): conv5_block3 unfrozen, warm-started from the best
          feature-extraction weights, then fine-tuned.
The fine-tuned model is saved as a .keras file for the integrated pipeline.

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
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import load_img, img_to_array, to_categorical

# PROJECT CONSTANTS (same as Milestone 1)
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
keras.utils.set_random_seed(GLOBAL_SEED)

BATCH = 8 # Chanaged from 16
EPOCHS = 100 # Changed from 30
IMG_SIZE = (224, 224)
NUM_CLASSES = 13 # Include all 13 classes
CLASS_NAMES = ["3", "7", "797", "1212", "2336", "2619", "2970", "4422", "4428", "5695", "7007", "8335", "10002"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "class_images")
SAVE_PATH = os.path.join(SCRIPT_DIR, "models", "resnet50_classifier_improved.keras")

l2_reg = regularizers.l2(1e-3)
LR_FE = 5e-4 # Changed from 1e-3
LR_FT = 5e-5 # Changed from 1e-4
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


def _augmentation_layers():
    # Changed RandomRotation and RandomZoom
    return [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.52),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ]

def build_model_fe():
    """ResNet50 feature-extraction model (frozen base) — M1 Task 3."""
    set_all_seeds()
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = Sequential([
        Input(shape=(224, 224, 3)),
        *_augmentation_layers(),
        Lambda(lambda x: preprocess_input(x)),
        base,
        # Old Settings
        #GlobalAveragePooling2D(),
        #Dense(256, activation='relu', kernel_regularizer=l2_reg),
        #Dropout(0.50),
        #Dense(NUM_CLASSES, activation='softmax')
        GlobalAveragePooling2D(),
        Dense(512, activation="relu", kernel_regularizer=l2_reg),
        BatchNormalization(),
        Dropout(0.40),
        Dense(256, activation="relu", kernel_regularizer=l2_reg),
        Dropout(0.30),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_FE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    n = sum(w.shape.num_elements() for w in model.trainable_weights)
    print(f"ResNet50 (FE) — trainable params: {n:,}")
    return model


def build_model_ft():
    """ResNet50 fine-tuning model (conv5_block3 unfrozen) — M1 Task 4."""
    set_all_seeds()
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers:
        if not any(layer.name.startswith(p) for p in FT_UNFREEZE_PREFIXES):
            layer.trainable = False
    model = Sequential([
        Input(shape=(224, 224, 3)),
        *_augmentation_layers(),
        Lambda(lambda x: preprocess_input(x)),
        base,
        # Old Settings
        #GlobalAveragePooling2D(),
        #Dense(256, activation='relu', kernel_regularizer=l2_reg),
        #Dropout(0.50),
        #Dense(NUM_CLASSES, activation='softmax')
        GlobalAveragePooling2D(),
        Dense(512, activation="relu", kernel_regularizer=l2_reg),
        BatchNormalization(),
        Dropout(0.40),
        Dense(256, activation="relu", kernel_regularizer=l2_reg),
        Dropout(0.30),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_FT),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    n = sum(w.shape.num_elements() for w in model.trainable_weights)
    print(f"ResNet50 (FT) — trainable params: {n:,}")
    return model


def main():
    print("=" * 60)
    print("ResNet50 Classifier Improved - Retrain & Save (two-stage, M1-faithful)")
    print("=" * 60)

    (train_files, train_labels, val_files, val_labels,
     test_files, test_labels) = load_data()

    X_train, y_train = load_image_arrays(train_files, train_labels)
    X_val, y_val = load_image_arrays(val_files, val_labels)
    X_test, y_test = load_image_arrays(test_files, test_labels)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    ckpt_dir = os.path.join(tempfile.gettempdir(), 'm4_classifier_imp_ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    fe_best = os.path.join(ckpt_dir, 'best_fe_imp.weights.h5')
    ft_best = os.path.join(ckpt_dir, 'best_ft_imp.weights.h5')

    def lr_cb():
        return ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                 patience=3, min_lr=1e-6, verbose=0)

    total_t0 = time.time()

    # ---- Stage 1: feature extraction (frozen base) ----
    print("\n[Stage 1/2] ResNet50 feature extraction (frozen base) ...")
    model_fe = build_model_fe()
    best_fe_cb = ModelCheckpoint(fe_best, monitor='val_accuracy', mode='max',
                                 save_best_only=True, save_weights_only=True, verbose=0)
    model_fe.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
                 validation_data=(X_val, y_val), callbacks=[best_fe_cb, lr_cb()], verbose=1)
    model_fe.load_weights(fe_best)
    _, fe_acc = model_fe.evaluate(X_test, y_test, verbose=0)
    print(f"[Stage 1/2] FE Improved best test accuracy: {fe_acc:.4f}")

    # ---- Stage 2: fine-tuning (conv5_block3 unfrozen), warm-started from FE best ----
    print("\n[Stage 2/2] ResNet50 fine-tuning (warm-start from FE best) ...")
    model_ft = build_model_ft()
    model_ft.load_weights(fe_best)
    print(f"Warm-started fine-tuning model from {os.path.basename(fe_best)}")
    best_ft_cb = ModelCheckpoint(ft_best, monitor='val_accuracy', mode='max',
                                 save_best_only=True, save_weights_only=True, verbose=0)
    model_ft.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
                 validation_data=(X_val, y_val), callbacks=[best_ft_cb, lr_cb()], verbose=1)
    model_ft.load_weights(ft_best)
    _, ft_acc = model_ft.evaluate(X_test, y_test, verbose=0)
    print(f"[Stage 2/2] FT Improved best test accuracy: {ft_acc:.4f}")

    print(f"\nTotal Improved Training Time: {time.time() - total_t0:.1f}s")

    model_ft.save(SAVE_PATH)
    size_mb = os.path.getsize(SAVE_PATH) / 1e6
    print(f"\nModel saved to: {SAVE_PATH} ({size_mb:.1f} MB)")

    import shutil
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()