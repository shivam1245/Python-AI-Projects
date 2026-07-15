"""
CameraClassifier — Manual Training Script
==========================================
Run this script to train (or retrain) the SVM classifier from the command line
with full control over hyperparameters, data augmentation, and evaluation.

Quick start
-----------
    python train.py                          # train with defaults
    python train.py --kernel rbf             # use RBF kernel
    python train.py --tune                   # grid-search best params
    python train.py --augment                # add flipped/brightened copies
    python train.py --size 80 --kernel rbf   # larger images + RBF

The trained model is saved as  CameraClassifier/trained_model.pkl
and is automatically picked up by the web server and Tkinter GUI.
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
import cv2 as cv
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMG_SIZE  = 50       # pixels (width = height)
DEFAULT_KERNEL    = "linear"
DEFAULT_C         = 1.0
DEFAULT_GAMMA     = "scale"  # only used for rbf / poly
MODEL_OUTPUT_PATH = os.path.join(DEFAULT_BASE_DIR, "trained_model.pkl")


# ── Data loading ───────────────────────────────────────────────────────────────
def load_dataset(base_dir: str, img_size: int, augment: bool = False):
    """
    Scan folders  base_dir/1/  base_dir/2/  … and load every .jpg as a
    flattened grayscale vector.

    Returns
    -------
    X : np.ndarray  shape (N, img_size*img_size)
    y : np.ndarray  shape (N,)  — integer class labels starting at 1
    class_map : dict  {1: folder_name, 2: …}
    """
    X, y, class_map = [], [], {}

    idx = 1
    while True:
        folder = os.path.join(base_dir, str(idx))
        if not os.path.isdir(folder):
            break

        files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
        if not files:
            print(f"  [warn] folder {idx}/ is empty — skipping")
            idx += 1
            continue

        class_map[idx] = os.path.basename(folder)

        for fname in files:
            img = cv.imread(os.path.join(folder, fname), cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv.resize(img, (img_size, img_size))
            X.append(img.flatten())
            y.append(idx)

            if augment:
                # Horizontal flip
                flipped = cv.flip(img, 1)
                X.append(flipped.flatten())
                y.append(idx)

                # Slightly brighter
                bright = cv.convertScaleAbs(img, alpha=1.15, beta=10)
                X.append(bright.flatten())
                y.append(idx)

                # Slightly darker
                dark = cv.convertScaleAbs(img, alpha=0.85, beta=-10)
                X.append(dark.flatten())
                y.append(idx)

                # Small rotation (+5 degrees)
                h, w = img.shape
                M = cv.getRotationMatrix2D((w // 2, h // 2), 5, 1.0)
                rot = cv.warpAffine(img, M, (w, h))
                X.append(rot.flatten())
                y.append(idx)

        count = len([f for f in files]) * (5 if augment else 1)
        print(f"  Class {idx}: {len(files)} raw images"
              + (f" → {count} with augmentation" if augment else ""))
        idx += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=int), class_map


# ── Hyperparameter grid search ─────────────────────────────────────────────────
def tune_hyperparameters(X_train, y_train, kernel: str):
    """Run GridSearchCV and return the best estimator + params."""
    if kernel == "linear":
        param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    elif kernel == "rbf":
        param_grid = {
            "C":     [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01],
        }
    elif kernel == "poly":
        param_grid = {
            "C":      [0.1, 1, 10],
            "degree": [2, 3, 4],
            "gamma":  ["scale", "auto"],
        }
    else:
        param_grid = {"C": [0.1, 1, 10]}

    print(f"\n  Running GridSearchCV ({kernel} kernel) — this may take a minute…")
    clf = svm.SVC(kernel=kernel, probability=True)
    gs  = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=0,
                       scoring="accuracy")
    gs.fit(X_train, y_train)
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV acc : {gs.best_score_:.1%}")
    return gs.best_estimator_, gs.best_params_


# ── Main training routine ──────────────────────────────────────────────────────
def train(args):
    print("\n─── CameraClassifier Manual Trainer ───────────────────────────────")
    print(f"  Base dir  : {args.base_dir}")
    print(f"  Image size: {args.size}×{args.size}")
    print(f"  Kernel    : {args.kernel}")
    print(f"  Augment   : {args.augment}")
    print(f"  Grid tune : {args.tune}")
    print("────────────────────────────────────────────────────────────────────\n")

    # 1. Load data ──────────────────────────────────────────────────────────────
    print("1. Loading dataset…")
    t0 = time.time()
    X, y, class_map = load_dataset(args.base_dir, args.size, args.augment)

    if len(X) == 0:
        print("\n  ERROR: No images found. Capture samples first via the web UI")
        print("         or GUI, then re-run this script.")
        sys.exit(1)

    unique = set(y)
    if len(unique) < 2:
        print("\n  ERROR: Need samples for at least 2 classes.")
        sys.exit(1)

    print(f"\n  Loaded {len(X)} samples across {len(unique)} classes "
          f"in {time.time()-t0:.1f}s")
    for cls_id, name in class_map.items():
        count = int(np.sum(y == cls_id))
        print(f"    Class {cls_id} ({name}): {count} samples")

    # Normalize pixel values to [0, 1]
    X = X / 255.0

    # 2. Train / val split ──────────────────────────────────────────────────────
    print("\n2. Splitting dataset (80 % train / 20 % validation)…")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}")

    # 3. Fit model ──────────────────────────────────────────────────────────────
    print("\n3. Training SVM…")
    t1 = time.time()

    if args.tune:
        clf, best_params = tune_hyperparameters(X_train, y_train, args.kernel)
    else:
        clf_kwargs = {"kernel": args.kernel, "probability": True, "C": args.C}
        if args.kernel in ("rbf", "poly"):
            clf_kwargs["gamma"] = args.gamma
        if args.kernel == "poly":
            clf_kwargs["degree"] = args.degree
        clf = svm.SVC(**clf_kwargs)
        clf.fit(X_train, y_train)

    print(f"  Training done in {time.time()-t1:.1f}s")

    # 4. Evaluate ───────────────────────────────────────────────────────────────
    print("\n4. Evaluation on validation set…")
    y_pred = clf.predict(X_val)
    val_acc = float(np.mean(y_pred == y_val))
    print(f"\n  Validation accuracy: {val_acc:.1%}\n")

    target_names = [f"Class {i} ({class_map.get(i, i)})" for i in sorted(unique)]
    print(classification_report(y_val, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    header = "       " + "  ".join(f"C{i}" for i in sorted(unique))
    print(header)
    for i, row in zip(sorted(unique), cm):
        print(f"  C{i}  |  " + "  ".join(f"{v:2d}" for v in row))

    # 5-fold cross-validation on full dataset
    print("\n  Running 5-fold cross-validation on full dataset…")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  CV scores : {[f'{s:.1%}' for s in cv_scores]}")
    print(f"  CV mean   : {cv_scores.mean():.1%}  ±  {cv_scores.std():.1%}")

    # 5. Save ───────────────────────────────────────────────────────────────────
    out_path = args.output
    payload = {
        "clf":       clf,
        "class_map": class_map,
        "img_size":  args.size,
        "kernel":    args.kernel,
        "val_acc":   val_acc,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n5. Model saved → {out_path}")
    print(f"\n  Done! Start the web server and the new model will be loaded automatically.\n")
    return clf, class_map


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Manual trainer for CameraClassifier SVM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                          # quick train, default settings
  python train.py --kernel rbf --tune      # RBF + grid search (best accuracy)
  python train.py --augment                # data augmentation (4x samples)
  python train.py --size 80 --kernel rbf   # 80x80 images + RBF
  python train.py --C 10 --kernel linear   # stronger regularization
        """,
    )
    p.add_argument("--base-dir",  default=DEFAULT_BASE_DIR, metavar="DIR",
                   help="folder containing class subfolders 1/ 2/ …  (default: CameraClassifier/)")
    p.add_argument("--size",      type=int,   default=DEFAULT_IMG_SIZE,
                   help="resize images to SIZE×SIZE before training (default: 50)")
    p.add_argument("--kernel",    default=DEFAULT_KERNEL,
                   choices=["linear", "rbf", "poly"],
                   help="SVM kernel (default: linear)")
    p.add_argument("--C",         type=float, default=DEFAULT_C,
                   help="SVM regularization parameter (default: 1.0)")
    p.add_argument("--gamma",     default=DEFAULT_GAMMA,
                   help="kernel coefficient for rbf/poly (default: scale)")
    p.add_argument("--degree",    type=int,   default=3,
                   help="polynomial degree for poly kernel (default: 3)")
    p.add_argument("--augment",   action="store_true",
                   help="enable data augmentation (flip, brightness, rotation)")
    p.add_argument("--tune",      action="store_true",
                   help="run GridSearchCV to find best hyperparameters")
    p.add_argument("--output",    default=MODEL_OUTPUT_PATH, metavar="FILE",
                   help="where to save the trained model (default: trained_model.pkl)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
