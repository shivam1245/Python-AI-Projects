# CameraClassifier

A real-time webcam classifier with a Tkinter GUI and full web interface. Capture samples, train a multi-class SVM, and predict live — with face detection, emotion analysis, and object detection built in.

---

## What's New in v2.0

| Feature | Description |
|---|---|
| **Flask Web UI** | Three-tab browser interface — Feature Overview, Live Demo, Object Detection. No GUI needed. |
| **Object Detection** | YOLOv8n detects 80 COCO object classes in real-time with colored bounding boxes. |
| **Face Detection** | Toggle a green bounding box drawn around every detected face using OpenCV's Haar cascade. |
| **Emotion Analysis** | FER-powered 7-emotion detection (happy, sad, angry, fear, surprise, disgust, neutral). |
| **Multi-Class SVM** | Classify any number of things. "Add Class" dynamically adds capture buttons and folders. |
| **Confidence Score** | SVM reports prediction confidence (%) alongside the class label. |

---

## Features

- Live 640×480 webcam preview in browser (MJPEG stream) and Tkinter GUI
- **Three web pages** — Feature Overview, Live Demo, Object Detection
- Capture labeled samples for **any number of classes** via browser or GUI
- Train a **linear SVM** on collected grayscale thumbnails
- Manual and **auto prediction** modes
- **Face Detection** toggle — green bounding box via Haar cascade
- **Emotion Analysis** toggle — 7-class FER model, lazy TensorFlow load
- **Object Detection** — YOLOv8n, 80 COCO classes, confidence threshold slider
- Standalone **CLI training script** with grid search and data augmentation

---

## Requirements

- Python 3.9+
- A functional webcam
- OS: Windows, macOS, Linux

---

## Installation

### Step 1 — Clone and create virtual environment

```bash
git clone https://github.com/shivam1245/Python-AI-Projects.git
cd Python-AI-Projects

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 2 — Install core dependencies

```bash
pip install -r CameraClassifier/requirements.txt
```

### Step 3 — Install web server dependencies

```bash
pip install -r CameraClassifier/website/requirements.txt
```

### Step 4 — Install optional ML features

```bash
# Emotion Analysis (FER + TensorFlow)
pip install fer tensorflow

# Object Detection (YOLOv8 — downloads ~6 MB weights on first use)
pip install ultralytics
```

> **Important — Multiple Python versions:** If you have Python 3.9, 3.10, 3.11, and 3.12 installed, make sure you install packages into the **same Python that runs the server**. The server prints its Python path on startup. Use that exact path to install, e.g.:
> ```bash
> "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe" -m pip install fer tensorflow ultralytics
> ```

---

## How to Run

### Option A — Web Interface (recommended)

```bash
cd CameraClassifier/website
python server.py
```

Then open **http://localhost:5000** in your browser.

The server prints which Python it is using and which features are available:

```
  CameraClassifier Web Server
  ─────────────────────────────
  Python    →  C:\...\python.exe
  Overview  →  http://localhost:5000/
  Live Demo →  http://localhost:5000/demo
  Objects   →  http://localhost:5000/object
  Emotion available: True
  YOLO available:    True
```

| Page | URL | Description |
|---|---|---|
| Feature Overview | `http://localhost:5000/` | Docs, feature cards, install guide |
| Live Demo | `http://localhost:5000/demo` | Capture → Train → Predict in browser |
| Object Detection | `http://localhost:5000/object` | YOLOv8 real-time detection |

### Option B — Tkinter GUI (desktop)

From the repository root:

```bash
python -m CameraClassifier.main
```

Or from inside `CameraClassifier/`:

```bash
cd CameraClassifier
python main.py
```

On startup you will be prompted to name your first two classes (e.g. "Open Hand" / "Closed Fist"). Press Cancel for defaults.

---

## Usage — Web Interface

### Live Demo page (`/demo`)

1. **Add classes** — type a name in the input box and click **+ Add**.
2. **Capture samples** — point your camera at the object/gesture and click **📸 Capture** (aim for 20–50 per class).
3. **Train** — click **🧠 Train Model** and wait for the status bar to show "Trained".
4. **Predict** — click **🎯 Predict Once** for a single result, or enable **🔁 Auto Predict** for continuous classification.
5. **Toggles** — enable **👤 Face Detect** (green boxes), **😊 Emotion Analysis** (orange boxes + emotion label), independently.

### Object Detection page (`/object`)

1. Click **Enable Detection** — YOLOv8n model loads (~2–5 seconds on first run, downloads weights if needed).
2. Adjust the **Min. Confidence** slider to filter low-confidence detections.
3. Detected objects appear in the side panel grouped by class with confidence bars.

---

## Usage — Tkinter GUI

### 1. Collect Samples

- Click **Capture: \<class name\>** to save a grayscale frame for that class.
- Aim for 20–50 samples per class for reliable SVM training.
- Click **+ Add Class** to add a third, fourth, … class at any time.

### 2. Train

Click **Train Model**. The SVM trains on all saved samples across all classes.

### 3. Predict

- **Predict** — classifies the current frame once and shows the class + confidence.
- **Auto Predict: ON** — continuously classifies every frame.

### 4. Face Detection

Click **Face Detect: ON** to overlay a green bounding box on each detected face.

### 5. Emotion Analysis

Click **Emotion: ON** to enable the FER model:
- An orange bounding box appears around each face with the dominant emotion label.
- Re-analyzes every 10 frames (~150 ms) to keep the UI responsive.
- Requires `fer` and `tensorflow`.

### 6. Reset

Click **Reset** to delete all captured images, clear all class buttons, and reset the model.

---

## GUI Layout

```
┌─────────────────────────────────────────────┐
│               Camera Feed (640×480)          │
│   [green box: Face]  [orange box: Emotion]   │
├─────────────────────────────────────────────┤
│  Auto Predict: OFF │ Face Detect: OFF │ Emotion: OFF │
│  Capture: Class 1 │ Capture: Class 2 │ + Add Class  │
│  Train Model       │ Predict          │ Reset        │
│                  CLASS: Open Hand            │
│                  CONFIDENCE: 87%             │
│                  EMOTION: Happy (91%)        │
└─────────────────────────────────────────────┤
```

---

## Project Layout

```
CameraClassifier/
├── main.py                    Entry point (Tkinter GUI)
├── app.py                     Tkinter GUI and application logic
├── camera.py                  OpenCV webcam wrapper
├── model.py                   Multi-class linear SVM (scikit-learn)
├── train.py                   ★ Manual training script (CLI, hyperparameter tuning)
├── emotion.py                 FER emotion detection module
├── test_camera_classifier.py  Headless tests (no webcam/GUI needed)
├── requirements.txt           Python dependencies
├── FEATURES.md                Full feature reference and roadmap
├── trained_model.pkl          Saved model (created by train.py)
├── 1/, 2/, 3/, …              Auto-created sample folders per class
└── website/
    ├── server.py              Flask web server
    ├── detector.py            YOLOv8 object detection wrapper
    ├── templates/             Jinja2 HTML templates
    └── static/                CSS + JavaScript
```

---

## Manual Training Guide

This section covers everything you need to know to train the classifier manually and squeeze maximum accuracy out of the SVM model.

### Which files are involved in training?

| File | Role |
|---|---|
| `model.py` | Core SVM class — `train_model()` and `predict()` |
| `train.py` | **Standalone CLI training script** — hyperparameter tuning, augmentation, metrics |
| `1/`, `2/`, `3/`, … | Training data folders — one folder per class, `.jpg` frames inside |
| `trained_model.pkl` | Saved trained model — loaded automatically by the web server |

---

### Step 1 — Capture Good Training Data

Quality data is the single biggest factor in accuracy. Follow these rules:

**Minimum samples:** 20–50 per class. More is better — aim for 80–100 for real-world use.

**Capture in all conditions you'll predict in:**
- Different lighting (bright, dim, lamp vs. daylight)
- Different distances from camera
- Slightly different angles / orientations
- With and without background clutter

**Avoid class overlap:** Make sure each class looks visually distinct. The SVM draws a linear boundary — the more separable the classes, the higher the accuracy.

**How to capture via the web UI:**
1. Run `cd CameraClassifier/website && python server.py`
2. Open `http://localhost:5000/demo`
3. Add class names → click **📸 Capture** repeatedly (hold or spam-click for fast collection)
4. Aim for consistent posture/object position across captures, then vary it deliberately

**How to capture via the Tkinter GUI:**
```bash
python main.py
```
Click **Capture: \<class name\>** — each click saves one `.jpg` into `CameraClassifier/<class_number>/`.

**Check your data manually:**
```
CameraClassifier/
├── 1/    ← class 1 images (frame1.jpg, frame2.jpg, …)
├── 2/    ← class 2 images
└── 3/    ← class 3 images (if you added a third class)
```

---

### Step 2 — Run the Manual Training Script

`train.py` gives full control over the SVM and prints accuracy metrics.

#### Basic training (same as clicking "Train Model" in the UI)

```bash
cd CameraClassifier
python train.py
```

#### Recommended for best accuracy — RBF kernel + grid search

```bash
python train.py --kernel rbf --tune
```

Grid search tries many combinations of `C` and `gamma` and picks the best using 5-fold cross-validation. Slower but consistently gives 3–8% better accuracy than default linear.

#### Data augmentation (doubles/quadruples your dataset for free)

```bash
python train.py --augment
```

Each captured frame gets four augmented copies (horizontal flip, brighter, darker, slight rotation). Especially useful when you have fewer than 30 samples per class.

#### Combine everything for maximum accuracy

```bash
python train.py --kernel rbf --tune --augment --size 80
```

- `--kernel rbf` — non-linear boundary, better for complex visual patterns
- `--tune` — auto-finds best `C` and `gamma`
- `--augment` — multiplies training data 5×
- `--size 80` — uses 80×80 pixel images instead of 50×50 (more detail, slower)

---

### All `train.py` Options

```
usage: python train.py [OPTIONS]

Options:
  --kernel {linear,rbf,poly}   SVM kernel type           (default: linear)
  --C FLOAT                    Regularization parameter  (default: 1.0)
  --gamma {scale,auto,FLOAT}   Kernel coefficient        (default: scale)
  --degree INT                 Poly kernel degree        (default: 3)
  --size INT                   Image resize dimension    (default: 50)
  --augment                    Enable data augmentation
  --tune                       Run GridSearchCV (finds best params)
  --base-dir DIR               Folder containing 1/ 2/ 3/ subfolders
  --output FILE                Where to save the model   (default: trained_model.pkl)
```

---

### Step 3 — Read the Training Output

A typical run prints:

```
─── CameraClassifier Manual Trainer ───────────────────────────────
  Base dir  : /path/to/CameraClassifier
  Image size: 50×50
  Kernel    : rbf
  Augment   : True
  Grid tune : True
────────────────────────────────────────────────────────────────────

1. Loading dataset…
  Class 1 (OpenHand): 45 raw images → 225 with augmentation
  Class 2 (ClosedFist): 50 raw images → 250 with augmentation

  Loaded 475 samples across 2 classes in 0.3s

2. Splitting dataset (80% train / 20% validation)…
  Train: 380  |  Val: 95

3. Training SVM…
  Running GridSearchCV (rbf kernel)…
  Best params : {'C': 10, 'gamma': 0.001}
  Best CV acc : 97.4%
  Training done in 12.3s

4. Evaluation on validation set…
  Validation accuracy: 96.8%

              precision  recall  f1-score
  Class 1        0.97    0.96      0.97
  Class 2        0.96    0.97      0.97

  CV scores: ['96.8%', '97.1%', '95.4%', '97.8%', '96.2%']
  CV mean  : 96.7%  ±  0.8%

5. Model saved → trained_model.pkl
```

**What to look for:**
- **Validation accuracy > 90%** — good for most use cases
- **CV mean ≈ validation accuracy** — model is not overfitting
- **High precision AND recall** for each class — no class is being ignored
- **Confusion matrix** — off-diagonal entries show which classes get confused

---

### Step 4 — Use the Trained Model

The web server picks up `trained_model.pkl` automatically if present. After running `train.py`, restart the server:

```bash
cd CameraClassifier/website
python server.py
```

Then go to `http://localhost:5000/demo` → click **Predict Once** or enable **Auto Predict**.

> **Note:** The model saved by `train.py` uses the same `model.py` API as the in-UI training, so both paths are fully compatible.

---

### Accuracy Tips

| Problem | Fix |
|---|---|
| Low accuracy (< 75%) | Capture more samples; ensure classes are visually distinct |
| One class always predicted | That class has far more samples — balance your dataset |
| High train acc, low val acc | Overfitting — try `--C 0.1` or `--kernel linear` |
| Low acc on all classes | Try `--kernel rbf --tune` and `--augment` |
| Works in UI but fails live | Lighting/background mismatch — recapture in same conditions as use |
| Slow prediction | Reduce `--size` (try 30) for faster inference |

---

### SVM Kernel Comparison

| Kernel | When to use | Typical accuracy |
|---|---|---|
| `linear` | Visually simple classes, fast training | 80–92% |
| `rbf` | Complex boundaries, recommended default | 88–97% |
| `poly` | Structured patterns (shapes, symbols) | 85–95% |

---

## Testing (Headless)

No webcam or GUI required:

```bash
cd Python-AI-Projects
python -m CameraClassifier.test_camera_classifier
```

Expected output: `All tests passed.`

The tests cover:
- SVM training and prediction on synthetic bright/dark images
- Correct `(class_label, confidence)` return signature from `model.predict()`
- Untrained model returning `(0, 0.0)` safely
- App module import (skipped if Tkinter unavailable)

---

## Troubleshooting

### Web Server

**Flask server won't start**
- Make sure you installed web requirements: `pip install -r CameraClassifier/website/requirements.txt`
- Run from the `website/` folder: `cd CameraClassifier/website && python server.py`

**`fer` / `ultralytics` not found even after pip install**
- You have multiple Python versions installed. The server prints which Python it uses at startup:
  ```
  Python  →  C:\Users\you\AppData\Local\Programs\Python\Python312\python.exe
  ```
- Install into that exact Python:
  ```bash
  "C:\...\python.exe" -m pip install fer tensorflow ultralytics
  ```

**Emotion toggle says "not installed" on the web UI**
- Restart the server after installing `fer` — the availability check re-runs automatically on each toggle click now, so you may not even need to restart.

**Object Detection toggle — model loads slowly**
- Expected on first run: YOLOv8n downloads ~6 MB of weights from the internet. Subsequent loads use the local cache (`~/.cache/ultralytics/`).

**Camera feed shows "Camera starting…" black screen**
- The camera takes 1–3 seconds to warm up. If it stays black, another app may be holding the camera.
- Windows: check Task Manager for other camera apps.
- macOS: grant Python camera permissions in System Settings → Privacy → Camera.

### GUI (Tkinter)

**Webcam not opening / black screen**
- Another app may be using the camera. Close it and retry.
- Linux/Wayland: check `/dev/video*` access; try `index=1` in `camera.py`.

**Tkinter import errors**
- Linux: `sudo apt-get install python3-tk`
- macOS (Homebrew Python): `brew install python-tk@3.13` then recreate venv.
- Conda: `conda install tk`

**"Not enough data to train"**
- Capture at least a few samples for **each** class before clicking Train Model.

**Emotion toggle freezes the GUI briefly**
- Expected — TensorFlow loads the FER model on first activation (5–15 seconds). Subsequent toggles are instant.

**Pillow ANTIALIAS warning**
- The code targets Pillow 10.x (`Resampling.LANCZOS`) with a fallback for older versions.

---

## Python 3.13 / Apple Silicon Notes

- NumPy 2.x, SciPy 1.14+, scikit-learn 1.5.x are required on Python 3.13 arm64.
- Install: `pip install -r CameraClassifier/requirements.txt`
- Upgrade pip first if you see binary compatibility errors: `python -m pip install --upgrade pip`

---

## Screenshot

![Prediction](image.png)

---

## Attribution

Inspired by NeuralNine's "Camera Classifier v0.1 Alpha" concept. Extended with face detection, emotion analysis, multi-class support, and confidence scoring.
