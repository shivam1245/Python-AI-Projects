"""
CameraClassifier Web Server
Flask backend that streams the webcam, handles face detection / emotion analysis,
and exposes a REST API for training and prediction.

Run:
    cd CameraClassifier/website
    python server.py

Open:  http://localhost:5000
"""

import sys
import os
import threading
import time

# ── Make CameraClassifier importable ───────────────────────────────────
_CC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CC_ROOT)

import cv2 as cv
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

from camera import Camera as _Camera
from model import Model as _Model

try:
    from emotion import EmotionDetector, is_available as _emotion_available
    EMOTION_AVAILABLE = _emotion_available()
except Exception:
    EMOTION_AVAILABLE = False

try:
    from detector import ObjectDetector, class_color, is_available as _obj_available
    OBJ_AVAILABLE = _obj_available()
except Exception:
    OBJ_AVAILABLE = False

# ── Flask app ───────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Data directory (where class folders 1/ 2/ 3/ … live) ───────────────
DATA_DIR = _CC_ROOT

# ── OpenCV Haar cascade ─────────────────────────────────────────────────
_face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Shared state ────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_state = {
    "classes":        [],       # list of class name strings
    "counters":       [],       # sample count per class (int)
    "face_detect":    False,
    "emotion":        False,
    "auto_predict":   False,
    "object_detect":  False,
    "trained":        False,
    "training":       False,    # True while train_model() is running
    "prediction":     {"class_name": None, "confidence": 0.0},
    "emotion_result": {"emotion": None,    "confidence": 0.0},
    "obj_detections": [],       # [{label, confidence}]
}

_model = _Model()

_emotion_detector      = None
_emotion_detector_lock = threading.Lock()

_obj_detector      = None
_obj_detector_lock = threading.Lock()

# ── Frame buffers ───────────────────────────────────────────────────────
_frame_lock          = threading.Lock()
_raw_frame           = None   # RGB, no overlays — used for capture & predict
_display_frame       = None   # RGB, with overlays — streamed to /video_feed
_obj_display_frame   = None   # RGB, with YOLO boxes — streamed to /object_feed

_emotion_cache: list = []
_emotion_tick        = 0
_obj_cache:    list  = []
_obj_tick            = 0

# ── Camera ──────────────────────────────────────────────────────────────
try:
    _cam = _Camera()
    _cam_ok = True
except Exception:
    _cam = None
    _cam_ok = False


# ── Background frame-processing thread ─────────────────────────────────
def _frame_worker():
    global _raw_frame, _display_frame, _obj_display_frame
    global _emotion_tick, _emotion_cache, _obj_tick, _obj_cache

    while True:
        if not _cam_ok:
            time.sleep(0.5)
            continue

        ret, frame = _cam.get_frame()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        with _state_lock:
            face_on   = _state["face_detect"]
            emo_on    = _state["emotion"]
            auto_on   = _state["auto_predict"]
            obj_on    = _state["object_detect"]
            classes   = list(_state["classes"])

        display     = frame.copy()
        obj_display = frame.copy()

        # ── Emotion analysis (throttled every 10 frames) ──────────────
        if emo_on:
            with _emotion_detector_lock:
                det = _emotion_detector
            if det is not None:
                _emotion_tick += 1
                if _emotion_tick % 10 == 0:
                    try:
                        _emotion_cache = det.analyze(frame)
                        if _emotion_cache:
                            top = _emotion_cache[0]
                            with _state_lock:
                                _state["emotion_result"] = {
                                    "emotion":    top["emotion"],
                                    "confidence": round(top["confidence"], 3),
                                }
                        else:
                            with _state_lock:
                                _state["emotion_result"] = {"emotion": "No face", "confidence": 0.0}
                    except Exception:
                        pass

        # ── Draw demo overlays ────────────────────────────────────────
        if emo_on and _emotion_cache:
            for r in _emotion_cache:
                x, y, w, h = r["box"]
                label = f"{r['emotion'].capitalize()} {r['confidence']:.0%}"
                cv.rectangle(display, (x, y), (x + w, y + h), (255, 165, 0), 2)
                cv.putText(display, label, (x, max(y - 8, 14)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
        elif face_on:
            gray  = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            for (x, y, w, h) in faces:
                cv.rectangle(display, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv.putText(display, "Face", (x, max(y - 8, 14)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # ── Auto-predict ──────────────────────────────────────────────
        if auto_on and _model.trained and classes:
            pred, conf = _model.predict(frame)
            if 1 <= pred <= len(classes):
                with _state_lock:
                    _state["prediction"] = {
                        "class_name": classes[pred - 1],
                        "confidence": round(conf, 3),
                    }

        # ── Object detection (throttled every 3 frames) ───────────────
        if obj_on:
            with _obj_detector_lock:
                odet = _obj_detector
            if odet is not None:
                _obj_tick += 1
                if _obj_tick % 3 == 0:
                    try:
                        _obj_cache = odet.detect(frame)
                        with _state_lock:
                            _state["obj_detections"] = [
                                {"label": d["label"], "confidence": d["confidence"]}
                                for d in _obj_cache
                            ]
                    except Exception:
                        pass
                for d in _obj_cache:
                    x1, y1, x2, y2 = d["box"]
                    color = class_color(d["cls_id"])
                    lbl   = f"{d['label']} {d['confidence']:.0%}"
                    cv.rectangle(obj_display, (x1, y1), (x2, y2), color, 2)
                    cv.putText(obj_display, lbl, (x1, max(y1 - 8, 14)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            else:
                cv.putText(obj_display, "Loading model...", (20, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)

        with _frame_lock:
            _raw_frame         = frame
            _display_frame     = display
            _obj_display_frame = obj_display

        time.sleep(0.033)  # ~30 fps


threading.Thread(target=_frame_worker, daemon=True).start()


# ── MJPEG stream helpers ─────────────────────────────────────────────────
def _frame_to_jpeg(frame_rgb) -> bytes:
    bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
    _, buf = cv.imencode(".jpg", bgr, [cv.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes()


def _blank_frame(msg: str = "Camera starting...") -> "np.ndarray":
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv.putText(blank, msg, (160, 240),
               cv.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    return blank


def _generate_mjpeg():
    while True:
        with _frame_lock:
            frame = _display_frame
        if frame is None:
            frame = _blank_frame()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + _frame_to_jpeg(frame) + b"\r\n")
        time.sleep(0.033)


def _generate_obj_mjpeg():
    while True:
        with _frame_lock:
            frame = _obj_display_frame
        if frame is None:
            frame = _blank_frame()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + _frame_to_jpeg(frame) + b"\r\n")
        time.sleep(0.033)


# ══════════════════════════════════════════════════════════════════════
# Routes — Pages
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def page_overview():
    return render_template("index.html", active="overview")


@app.route("/demo")
def page_demo():
    with _state_lock:
        snap = {k: v for k, v in _state.items()}
    return render_template("demo.html", active="demo",
                           state=snap, emotion_available=EMOTION_AVAILABLE)


@app.route("/object")
def page_object():
    with _obj_detector_lock:
        det_loaded = _obj_detector is not None
    with _state_lock:
        enabled = _state["object_detect"]
    return render_template("object.html", active="object",
                           obj_available=OBJ_AVAILABLE,
                           det_loaded=det_loaded,
                           obj_enabled=enabled)


@app.route("/video_feed")
def video_feed():
    return Response(_generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/object_feed")
def object_feed():
    return Response(_generate_obj_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ══════════════════════════════════════════════════════════════════════
# REST API
# ══════════════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    with _state_lock:
        snap = {k: v for k, v in _state.items()}
    snap["trained"]           = _model.trained
    snap["emotion_available"] = EMOTION_AVAILABLE
    return jsonify(snap)


@app.route("/api/add_class", methods=["POST"])
def api_add_class():
    data = request.get_json(force=True, silent=True) or {}
    with _state_lock:
        idx  = len(_state["classes"]) + 1
        name = data.get("name", "").strip() or f"Class {idx}"
        _state["classes"].append(name)
        _state["counters"].append(0)

    os.makedirs(os.path.join(DATA_DIR, str(idx)), exist_ok=True)
    return jsonify({"success": True, "class_idx": idx, "name": name})


@app.route("/api/capture/<int:class_num>", methods=["POST"])
def api_capture(class_num):
    with _state_lock:
        n = len(_state["classes"])
    if class_num < 1 or class_num > n:
        return jsonify({"error": "Invalid class number"}), 400

    with _frame_lock:
        frame = _raw_frame
    if frame is None:
        return jsonify({"error": "No camera frame available"}), 503

    folder = os.path.join(DATA_DIR, str(class_num))
    os.makedirs(folder, exist_ok=True)

    with _state_lock:
        count = _state["counters"][class_num - 1]
        _state["counters"][class_num - 1] += 1

    path = os.path.join(folder, f"frame{count + 1}.jpg")
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    gray = cv.resize(gray, (150, 150))
    cv.imwrite(path, gray)

    return jsonify({"success": True, "count": count + 1})


def _do_train(num_classes):
    global _model
    with _state_lock:
        _state["training"] = True
    try:
        _model.train_model(num_classes, base_dir=DATA_DIR)
        with _state_lock:
            _state["trained"]  = _model.trained
            _state["training"] = False
    except Exception as exc:
        with _state_lock:
            _state["training"] = False
        print(f"[train] error: {exc}")


@app.route("/api/train", methods=["POST"])
def api_train():
    with _state_lock:
        if _state["training"]:
            return jsonify({"error": "Training already in progress"}), 409
        num = len(_state["classes"])
    if num < 2:
        return jsonify({"error": "Need at least 2 classes"}), 400

    threading.Thread(target=_do_train, args=(num,), daemon=True).start()
    return jsonify({"success": True, "message": "Training started"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    with _frame_lock:
        frame = _raw_frame
    if frame is None:
        return jsonify({"error": "No frame"}), 503

    with _state_lock:
        classes = list(_state["classes"])

    if not _model.trained:
        return jsonify({"error": "Model not trained"}), 400

    pred, conf = _model.predict(frame)
    if 1 <= pred <= len(classes):
        result = {"class_name": classes[pred - 1], "confidence": round(conf, 3)}
        with _state_lock:
            _state["prediction"] = result
        return jsonify(result)
    return jsonify({"class_name": None, "confidence": 0.0})


@app.route("/api/toggle/<feature>", methods=["POST"])
def api_toggle(feature):
    global _emotion_detector, _obj_detector
    allowed = {"face_detect", "emotion", "auto_predict", "object_detect"}
    if feature not in allowed:
        return jsonify({"error": "Unknown feature"}), 400
    if feature == "emotion":
        from emotion import is_available as _chk_emo
        if not _chk_emo():
            import sys
            return jsonify({
                "error": f"fer not found in this Python ({sys.executable}). "
                         f"Run: \"{sys.executable}\" -m pip install fer tensorflow"
            }), 400
    if feature == "object_detect" and not OBJ_AVAILABLE:
        return jsonify({"error": "ultralytics not installed. Run: pip install ultralytics"}), 400

    with _state_lock:
        _state[feature] = not _state[feature]
        new_val = _state[feature]

    # Lazy-load FER model in background to avoid blocking the response
    if feature == "emotion" and new_val:
        def _load_emotion():
            global _emotion_detector
            with _emotion_detector_lock:
                if _emotion_detector is None:
                    try:
                        _emotion_detector = EmotionDetector()
                    except Exception as exc:
                        print(f"[emotion] load error: {exc}")
                        with _state_lock:
                            _state["emotion"] = False
        threading.Thread(target=_load_emotion, daemon=True).start()
    elif feature == "emotion" and not new_val:
        global _emotion_cache
        _emotion_cache = []
        with _state_lock:
            _state["emotion_result"] = {"emotion": None, "confidence": 0.0}

    # Lazy-load YOLO model in background
    if feature == "object_detect" and new_val:
        def _load_obj():
            global _obj_detector
            with _obj_detector_lock:
                if _obj_detector is None:
                    try:
                        _obj_detector = ObjectDetector()
                    except Exception as exc:
                        print(f"[object] load error: {exc}")
                        with _state_lock:
                            _state["object_detect"] = False
        threading.Thread(target=_load_obj, daemon=True).start()
    elif feature == "object_detect" and not new_val:
        global _obj_cache
        _obj_cache = []
        with _state_lock:
            _state["obj_detections"] = []

    return jsonify({"feature": feature, "enabled": new_val})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global _model, _emotion_cache
    with _state_lock:
        num = len(_state["classes"])

    for i in range(1, num + 1):
        folder = os.path.join(DATA_DIR, str(i))
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                fp = os.path.join(folder, fname)
                if os.path.isfile(fp):
                    try:
                        os.unlink(fp)
                    except OSError:
                        pass

    _model = _Model()
    _emotion_cache = []

    with _state_lock:
        _state.update({
            "classes":        [],
            "counters":       [],
            "trained":        False,
            "training":       False,
            "prediction":     {"class_name": None, "confidence": 0.0},
            "emotion_result": {"emotion": None,    "confidence": 0.0},
        })

    return jsonify({"success": True})


@app.route("/api/object/status")
def api_object_status():
    with _state_lock:
        enabled = _state["object_detect"]
        dets    = list(_state["obj_detections"])
    with _obj_detector_lock:
        loaded = _obj_detector is not None
    return jsonify({
        "enabled":    enabled,
        "loaded":     loaded,
        "detections": dets,
        "available":  OBJ_AVAILABLE,
    })


@app.route("/api/object/threshold", methods=["POST"])
def api_object_threshold():
    data      = request.get_json(force=True, silent=True) or {}
    threshold = float(data.get("threshold", 0.40))
    threshold = max(0.10, min(0.95, threshold))
    with _obj_detector_lock:
        if _obj_detector is not None:
            _obj_detector.conf = threshold
    return jsonify({"threshold": round(threshold, 2)})


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys as _sys
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  CameraClassifier Web Server")
    print(f"  ─────────────────────────────")
    print(f"  Python    →  {_sys.executable}")
    print(f"  Overview  →  http://localhost:{port}/")
    print(f"  Live Demo →  http://localhost:{port}/demo")
    print(f"  Objects   →  http://localhost:{port}/object")
    print(f"  Emotion available: {EMOTION_AVAILABLE}")
    print(f"  YOLO available:    {OBJ_AVAILABLE}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
