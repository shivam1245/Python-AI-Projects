# CameraClassifier — Feature Reference

## Current Features (v1.0 — Baseline)

| Feature | Description | Status |
|---|---|---|
| Live webcam preview | 640×480 OpenCV feed rendered in Tkinter canvas | Implemented |
| Two-class capture | Labeled buttons save grayscale 150×150 thumbnails to `1/` and `2/` | Implemented |
| Linear SVM training | scikit-learn `SVC(kernel='linear')` trained on flattened 50×50 images | Implemented |
| Manual prediction | Classify current frame on demand via Predict button | Implemented |
| Auto prediction | Continuously classify frames on a 15 ms loop | Implemented |
| Custom class names | Dialog prompts at startup for user-defined class labels | Implemented |
| Reset | Clears all captured samples and resets model state | Implemented |
| Headless tests | Synthetic image tests; no webcam or GUI required | Implemented |

---

## Tier 1 Features — High Impact, Moderate Effort (v2.0)

### 1. Face Detection with Bounding Box
- **What:** Detects faces in real-time using OpenCV's Haar cascade and draws a green bounding box.
- **Why:** Visual confirmation that the camera is finding faces; prerequisite context for emotion analysis.
- **Toggle:** "Face Detect" button (ON/OFF).
- **Dependencies:** None — uses `cv2.CascadeClassifier` (bundled with OpenCV).
- **Status:** Implemented

### 2. Emotion Analysis
- **What:** Detects facial emotions per frame using the `fer` (Facial Expression Recognition) library.
- **Emotions:** `happy`, `sad`, `angry`, `fear`, `surprise`, `disgust`, `neutral`
- **Display:** Orange bounding box per face + dominant emotion label + confidence % overlaid on the camera feed. Separate EMOTION label in the GUI.
- **Throttle:** Re-analyzes every 10 frames (~150 ms) to keep the UI responsive.
- **Toggle:** "Emotion" button (ON/OFF); model is lazy-loaded on first activation.
- **Dependencies:** `fer>=22.5.1`, `tensorflow>=2.0.0`
- **Status:** Implemented

### 3. Multi-Class Classification
- **What:** Extends classification from 2 fixed classes to any number of user-defined classes.
- **How:** "Add Class" button prompts for a name, creates a new numbered folder, and adds a capture button dynamically.
- **Confidence Score:** SVM now uses `probability=True`; confidence % shown alongside the predicted class label.
- **Dependencies:** None — scikit-learn SVM supports multi-class natively.
- **Status:** Implemented

---

## Tier 2 Features — Advanced, High Value (Planned)

### 4. Hand Gesture Recognition
- **What:** Detect 21 hand landmarks and classify gestures (thumbs up, peace, fist, open palm, etc.).
- **Library:** `mediapipe`
- **Use case:** Touchless control, sign language demos, gesture-based UX.

### 5. Head Pose Estimation
- **What:** Detect the direction the user is facing — left, right, up, down, forward.
- **Library:** MediaPipe Face Mesh + OpenCV PnP solver.
- **Use case:** Attention tracking, gaze monitoring, driver safety demos.

### 6. Eye Blink / Drowsiness Detection
- **What:** Compute the Eye Aspect Ratio (EAR) from face landmarks; trigger alert if eyes close too long.
- **Library:** MediaPipe Face Mesh or `dlib`.
- **Use case:** Fatigue detection, focus monitoring.

### 7. Age & Gender Estimation
- **What:** Estimate age range (e.g., 20–30) and binary gender from a face crop.
- **Library:** `deepface` or OpenCV DNN with pre-trained weights.
- **Use case:** Demographics analysis, audience profiling demos.

---

## Tier 3 Features — UX & Quality-of-Life (Planned)

### 8. Real-Time FPS Counter
- **What:** Overlay the actual frames-per-second on the camera canvas.
- **How:** Track time between `update()` calls; display with `cv.putText`.
- **Dependencies:** None.

### 9. Session Recording
- **What:** Save the annotated camera session as a `.mp4` video clip.
- **How:** `cv2.VideoWriter` wrapped in a record toggle.
- **Dependencies:** None (OpenCV built-in).

### 10. Data Augmentation on Capture
- **What:** Automatically save flipped and slightly rotated variants alongside each captured sample.
- **Why:** Multiplies training data 3–4× with no extra user effort; improves SVM accuracy on small datasets.
- **Dependencies:** None.

### 11. Background Blur / Segmentation
- **What:** Blur or replace the background in real-time using body segmentation.
- **Library:** MediaPipe Selfie Segmentation.
- **Use case:** Privacy-preserving demos, virtual backgrounds.

### 12. Snapshot with Annotations
- **What:** Save the current annotated frame (with face boxes, emotion labels, class label) as a `.png`.
- **How:** Single button; uses `PIL.Image.save`.
- **Dependencies:** None.

---

## Feature Priority Matrix

| # | Feature | Effort | Impact | New Dependency | Version |
|---|---|---|---|---|---|
| 1 | Face Detection | Low | High | None | v2.0 |
| 2 | Emotion Analysis | Medium | High | `fer`, `tensorflow` | v2.0 |
| 3 | Multi-Class + Confidence | Medium | High | None | v2.0 |
| 4 | Hand Gesture | Medium | High | `mediapipe` | Planned |
| 5 | Head Pose | Medium | Medium | `mediapipe` | Planned |
| 6 | Eye Blink / Drowsiness | Medium | Medium | `mediapipe`/`dlib` | Planned |
| 7 | Age & Gender | Low | Medium | `deepface` | Planned |
| 8 | FPS Counter | Low | Low | None | Planned |
| 9 | Session Recording | Low | Medium | None | Planned |
| 10 | Data Augmentation | Low | High | None | Planned |
| 11 | Background Blur | Medium | Low | `mediapipe` | Planned |
| 12 | Snapshot | Low | Medium | None | Planned |
