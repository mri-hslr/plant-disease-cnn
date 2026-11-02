"""
Training pipeline for PlantVillage 38-class classification.

Models: custom_cnn, resnet50, mobilenetv2.
Preprocessing: resize 224x224 -> CLAHE on LAB-L -> Gaussian 3x3 -> per-model norm.
Split: stratified 70 / 15 / 15.
Optimizer: Adam. Loss: categorical cross-entropy. Early stopping: patience 8.

Usage:
    python src/train.py --model custom_cnn
    python src/train.py --model resnet50 --batch 16 --epochs 50
"""

import argparse
import json
import pathlib
import random

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(path, size=(224, 224)):
    """Read, resize, CLAHE on LAB-L, Gaussian blur. Returns uint8 RGB."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, size, method="bicubic")
    img = tf.cast(img, tf.uint8).numpy()

    # CLAHE on the L channel of LAB, not on R/G/B independently.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    # 3x3 Gaussian for acquisition noise
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0)
    return img


def per_model_preprocess(img_uint8, model_name):
    """
    Each pretrained backbone expects its own input scaling.
    Custom CNN: [0, 1].
    ResNet-50 / MobileNetV2: their own `preprocess_input` functions.
    """
    x = img_uint8.astype(np.float32)
    if model_name == "custom_cnn":
        return x / 255.0
    if model_name == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input(x)
    if model_name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)
    raise ValueError(model_name)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def list_dataset(root):
    """
    Walk data/raw/ -> list of (path, class_idx) + class names.
    Assumes the layout the README describes: each subfolder under root is a class.
    """
    root = pathlib.Path(root)
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders under {root}")

    paths, labels = [], []
    for idx, d in enumerate(class_dirs):
        for f in d.rglob("*"):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(f))
                labels.append(idx)
    paths = np.array(paths)
    labels = np.array(labels, dtype=np.int32)
    class_names = [d.name for d in class_dirs]
    return paths, labels, class_names


def stratified_split(paths, labels, seed=42):
    """70 / 15 / 15 stratified."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def make_tf_dataset(paths, labels, model_name, batch=32, training=False, seed=42):
    size = (224, 224)

    def _load(path_t, label_t):
        def _py(path_bytes, label_int):
            img = preprocess_image(path_bytes.decode("utf-8"), size)
            img = per_model_preprocess(img, model_name)
            return img.astype(np.float32), label_int.astype(np.int32)

        img, label = tf.numpy_function(
            _py, [path_t, label_t], [tf.float32, tf.int32]
        )
        img.set_shape((*size, 3))
        label.set_shape(())
        return img, label

    def aug(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        # We removed the broken rotation line for now
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    # 1. Load the image first
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 2. THEN augment it (if training)
    if training:
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds
# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_custom_cnn(num_classes, input_shape=(224, 224, 3)):
    """
    The custom CNN from the synopsis:
    Conv -> ReLU -> MaxPool -> Dropout -> ... -> Dense -> Softmax.
    BatchNorm added for stability (README explains how to remove it).
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def build_transfer(model_name, num_classes, input_shape=(224, 224, 3)):
    if model_name == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif model_name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:
        raise ValueError(model_name)

    base.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=model_name), base


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model_name, data_dir, out_dir, batch, epochs, patience, seed):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    out_dir = pathlib.Path(out_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{model_name}] listing dataset...")
    paths, labels, class_names = list_dataset(data_dir)
    print(f"[{model_name}] {len(paths)} images, {len(class_names)} classes")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = stratified_split(paths, labels, seed)
    print(f"[{model_name}] split: {len(X_tr)}/{len(X_va)}/{len(X_te)}")

    # encode labels as one-hot for categorical_crossentropy
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, len(class_names))
    y_va_oh = tf.keras.utils.to_categorical(y_va, len(class_names))
    y_te_oh = tf.keras.utils.to_categorical(y_te, len(class_names))

    # For the custom CNN, we *could* feed y_tr (integers) with sparse CE;
    # the synopsis says categorical cross-entropy, so one-hot it is.
    train_ds = make_tf_dataset(X_tr, y_tr, model_name, batch=batch, training=True, seed=seed) \
        .map(lambda x, y: (x, tf.one_hot(y, len(class_names))))
    val_ds = make_tf_dataset(X_va, y_va, model_name, batch=batch, training=False) \
        .map(lambda x, y: (x, tf.one_hot(y, len(class_names))))
    test_ds = make_tf_dataset(X_te, y_te, model_name, batch=batch, training=False)

    # Build model
    if model_name == "custom_cnn":
        model = build_custom_cnn(len(class_names))
        base = None
    else:
        model, base = build_transfer(model_name, len(class_names))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt = out_dir / "best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]

    # Phase 1: freeze backbone (or train custom CNN end-to-end)
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=callbacks, verbose=1,
    )

    # Phase 2: fine-tune top 30 layers of the backbone (transfer models only)
    if base is not None:
        print(f"[{model_name}] phase 2: unfreezing top 30 layers, lr=1e-5")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history_ft = model.fit(
            train_ds, validation_data=val_ds,
            epochs=epochs, callbacks=callbacks, verbose=1,
        )
        # append phase-2 history to the CSV
        with open(out_dir / "history.csv", "a") as f:
            for k, v in history_ft.history.items():
                f.write(f"{k}," + ",".join(map(str, v)) + "\n")

    # Evaluate on the untouched test set
    print(f"[{model_name}] evaluating on test set...")
    y_pred_prob = model.predict(test_ds, verbose=1)
    y_pred = y_pred_prob.argmax(axis=1)

    report = classification_report(
        y_te, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_te, y_pred)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    print(report)
    print(f"[{model_name}] artifacts saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["custom_cnn", "resnet50", "mobilenetv2"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train(
        model_name=args.model,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()