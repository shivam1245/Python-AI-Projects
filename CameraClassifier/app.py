"""
Camera Classifier — v2.0
Tier 1 features: face detection, emotion analysis, multi-class SVM classification.
"""

import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2 as cv
import os
import PIL.Image
import PIL.ImageTk
import model
import camera
import emotion as emotion_module


class App:
    # How many update() frames to skip between full emotion analyses (keeps UI fluid)
    _EMOTION_INTERVAL = 10

    def __init__(self, window: tk.Tk | None = None, window_title: str = "Camera Classifier"):
        self.window = window or tk.Tk()
        self.window.title(window_title)

        self.model = model.Model()
        self.camera = camera.Camera()

        # Multi-class state
        self.classes: list[str] = []
        self.counters: list[int] = []
        self.class_buttons: list[tk.Button] = []

        # Feature toggles
        self.auto_predict = False
        self.face_detection_enabled = False
        self.emotion_enabled = False

        # Emotion state — lazy init to avoid blocking startup with TF loading
        self._emotion_detector: emotion_module.EmotionDetector | None = None
        self._current_emotions: list[dict] = []
        self._emotion_tick = 0

        # Haar cascade for face-only detection (no emotion label)
        _cascade = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._face_cascade = cv.CascadeClassifier(_cascade)

        self._init_gui()
        self._prompt_initial_classes()

        self.delay = 15
        self._update()

        try:
            self.window.attributes("-topmost", True)
        except Exception:
            pass
        self.window.mainloop()

    # ------------------------------------------------------------------
    # GUI setup
    # ------------------------------------------------------------------

    def _init_gui(self):
        # Camera canvas
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        # Toggle row
        toggle_frame = tk.Frame(self.window)
        toggle_frame.pack(fill=tk.X, padx=4, pady=(4, 0))

        self._btn_auto = tk.Button(toggle_frame, text="Auto Predict: OFF", width=22,
                                   command=self._toggle_auto_predict)
        self._btn_auto.pack(side=tk.LEFT, padx=2)

        self._btn_face = tk.Button(toggle_frame, text="Face Detect: OFF", width=22,
                                   command=self._toggle_face_detect)
        self._btn_face.pack(side=tk.LEFT, padx=2)

        self._btn_emotion = tk.Button(toggle_frame, text="Emotion: OFF", width=22,
                                      command=self._toggle_emotion)
        self._btn_emotion.pack(side=tk.LEFT, padx=2)

        # Class capture row (buttons added dynamically)
        self._class_frame = tk.Frame(self.window)
        self._class_frame.pack(fill=tk.X, padx=4, pady=(4, 0))

        self._btn_add_class = tk.Button(self._class_frame, text="+ Add Class", width=14,
                                        command=self._add_class)
        self._btn_add_class.pack(side=tk.RIGHT, padx=2)

        # Action row
        action_frame = tk.Frame(self.window)
        action_frame.pack(fill=tk.X, padx=4, pady=(4, 0))

        tk.Button(action_frame, text="Train Model", width=20,
                  command=self._train).pack(side=tk.LEFT, padx=2)
        tk.Button(action_frame, text="Predict", width=20,
                  command=self._predict).pack(side=tk.LEFT, padx=2)
        tk.Button(action_frame, text="Reset", width=20,
                  command=self._reset).pack(side=tk.LEFT, padx=2)

        # Info labels
        self._lbl_class = tk.Label(self.window, text="CLASS: —", font=("Arial", 16))
        self._lbl_class.pack(anchor=tk.CENTER, pady=(6, 0))

        self._lbl_confidence = tk.Label(self.window, text="CONFIDENCE: —", font=("Arial", 12))
        self._lbl_confidence.pack(anchor=tk.CENTER)

        self._lbl_emotion = tk.Label(self.window, text="EMOTION: —", font=("Arial", 14))
        self._lbl_emotion.pack(anchor=tk.CENTER, pady=(4, 6))

    # ------------------------------------------------------------------
    # Class management
    # ------------------------------------------------------------------

    def _prompt_initial_classes(self):
        for i in range(1, 3):
            name = simpledialog.askstring(
                f"Class {i}", f"Enter name for class {i}:", parent=self.window
            ) or f"Class {i}"
            self._register_class(name)

    def _register_class(self, name: str):
        idx = len(self.classes) + 1
        self.classes.append(name)
        self.counters.append(1)
        os.makedirs(str(idx), exist_ok=True)
        btn = tk.Button(self._class_frame, text=f"Capture: {name}", width=22,
                        command=lambda i=idx: self._save_for_class(i))
        btn.pack(side=tk.LEFT, padx=2, pady=2)
        self.class_buttons.append(btn)

    def _add_class(self):
        default = f"Class {len(self.classes) + 1}"
        name = simpledialog.askstring("Add Class", "Enter new class name:",
                                      parent=self.window) or default
        self._register_class(name)

    # ------------------------------------------------------------------
    # Toggle handlers
    # ------------------------------------------------------------------

    def _toggle_auto_predict(self):
        self.auto_predict = not self.auto_predict
        self._btn_auto.config(text=f"Auto Predict: {'ON' if self.auto_predict else 'OFF'}")

    def _toggle_face_detect(self):
        self.face_detection_enabled = not self.face_detection_enabled
        self._btn_face.config(text=f"Face Detect: {'ON' if self.face_detection_enabled else 'OFF'}")

    def _toggle_emotion(self):
        if not emotion_module.is_available():
            messagebox.showerror(
                "Not Available",
                "The 'fer' library is not installed.\nRun: pip install fer tensorflow"
            )
            return

        self.emotion_enabled = not self.emotion_enabled
        self._btn_emotion.config(text=f"Emotion: {'ON' if self.emotion_enabled else 'OFF'}")

        if self.emotion_enabled and self._emotion_detector is None:
            self._lbl_emotion.config(text="EMOTION: Loading model…")
            self.window.update_idletasks()
            try:
                self._emotion_detector = emotion_module.EmotionDetector()
                self._lbl_emotion.config(text="EMOTION: —")
            except Exception as exc:
                self.emotion_enabled = False
                self._btn_emotion.config(text="Emotion: OFF")
                self._lbl_emotion.config(text="EMOTION: —")
                messagebox.showerror("Emotion Error", str(exc))

        if not self.emotion_enabled:
            self._current_emotions = []
            self._lbl_emotion.config(text="EMOTION: —")

    # ------------------------------------------------------------------
    # Data capture
    # ------------------------------------------------------------------

    def _save_for_class(self, class_num: int):
        ret, frame = self.camera.get_frame()
        if not ret:
            return
        folder = str(class_num)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"frame{self.counters[class_num - 1]}.jpg")
        cv.imwrite(path, cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        try:
            img = PIL.Image.open(path)
            img.thumbnail((150, 150), PIL.Image.Resampling.LANCZOS)
            img.save(path)
        except AttributeError:
            try:
                img = PIL.Image.open(path)
                img.thumbnail((150, 150), PIL.Image.ANTIALIAS)  # type: ignore[attr-defined]
                img.save(path)
            except Exception:
                pass
        except Exception:
            pass
        self.counters[class_num - 1] += 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self):
        self.model.train_model(len(self.classes))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict(self, frame=None) -> str | None:
        if frame is None:
            ret, frame = self.camera.get_frame()
            if not ret:
                return None
        pred, confidence = self.model.predict(frame)
        if 1 <= pred <= len(self.classes):
            name = self.classes[pred - 1]
            self._lbl_class.config(text=f"CLASS: {name}")
            self._lbl_confidence.config(text=f"CONFIDENCE: {confidence:.0%}")
            return name
        return None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset(self):
        for i in range(1, len(self.classes) + 1):
            folder = str(i)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                fp = os.path.join(folder, fname)
                if os.path.isfile(fp):
                    try:
                        os.unlink(fp)
                    except Exception:
                        pass

        for btn in self.class_buttons:
            btn.destroy()
        self.class_buttons.clear()
        self.classes.clear()
        self.counters.clear()
        self._current_emotions.clear()

        self.model = model.Model()
        self._lbl_class.config(text="CLASS: —")
        self._lbl_confidence.config(text="CONFIDENCE: —")
        self._lbl_emotion.config(text="EMOTION: —")
        self._prompt_initial_classes()

    # ------------------------------------------------------------------
    # Frame annotation helpers
    # ------------------------------------------------------------------

    def _draw_face_boxes(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv.putText(frame, "Face", (x, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        return frame

    def _draw_emotion_overlays(self, frame):
        for result in self._current_emotions:
            x, y, w, h = result["box"]
            label = emotion_module.EmotionDetector.format_label(
                result["emotion"], result["confidence"]
            )
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
            cv.putText(frame, label, (x, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
        return frame

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------

    def _update(self):
        ret, frame = self.camera.get_frame()

        if ret:
            display = frame.copy()

            # Emotion analysis — run every _EMOTION_INTERVAL frames
            if self.emotion_enabled and self._emotion_detector is not None:
                self._emotion_tick += 1
                if self._emotion_tick % self._EMOTION_INTERVAL == 0:
                    try:
                        self._current_emotions = self._emotion_detector.analyze(frame)
                        if self._current_emotions:
                            top = self._current_emotions[0]
                            self._lbl_emotion.config(
                                text=f"EMOTION: {emotion_module.EmotionDetector.format_label(top['emotion'], top['confidence'])}"
                            )
                        else:
                            self._lbl_emotion.config(text="EMOTION: No face detected")
                    except Exception:
                        pass
            elif not self.emotion_enabled:
                self._current_emotions = []

            # Draw overlays — emotion takes priority over plain face boxes
            if self.emotion_enabled and self._current_emotions:
                display = self._draw_emotion_overlays(display)
            elif self.face_detection_enabled:
                display = self._draw_face_boxes(display)

            # Auto-predict
            if self.auto_predict:
                self._predict(frame=frame)

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(display))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self._update)


if __name__ == "__main__":
    App()
