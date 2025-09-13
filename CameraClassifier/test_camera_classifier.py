"""
Headless verification tests for CameraClassifier components.
Run: python -m CameraClassifier.test_camera_classifier
"""
import os
import shutil
import numpy as np
import cv2 as cv
import importlib

from . import model as model_module

# Detect Tk availability to make tests robust in environments without Tk support
_HAS_TK = True
try:
    import tkinter  # noqa: F401
except Exception:
    _HAS_TK = False


def setup_synthetic_data():
    os.makedirs('1', exist_ok=True)
    os.makedirs('2', exist_ok=True)
    for i in range(8):
        img1 = np.full((150, 150), 220, dtype=np.uint8)
        img2 = np.full((150, 150), 30, dtype=np.uint8)
        cv.imwrite(f'1/frame{i+1}.jpg', img1)
        cv.imwrite(f'2/frame{i+1}.jpg', img2)


def cleanup_synthetic_data():
    shutil.rmtree('1', ignore_errors=True)
    shutil.rmtree('2', ignore_errors=True)


def test_training_and_prediction():
    setup_synthetic_data()
    try:
        m = model_module.Model()
        m.train_model([9, 9])
        assert m.trained, "Model did not set trained=True after training"
        bright_rgb = cv.cvtColor(np.full((150, 150), 220, dtype=np.uint8), cv.COLOR_GRAY2RGB)
        dark_rgb = cv.cvtColor(np.full((150, 150), 30, dtype=np.uint8), cv.COLOR_GRAY2RGB)
        p1 = m.predict(bright_rgb)
        p2 = m.predict(dark_rgb)
        assert p1 in (1, 2) and p2 in (1, 2), "Predictions should be class labels 1 or 2"
        assert p1 != p2, "Synthetic bright/dark samples should map to different classes"
    finally:
        cleanup_synthetic_data()


def test_app_import():
    # Ensure app module imports without executing mainloop
    if not _HAS_TK:
        print('[INFO] Skipping GUI import test because Tkinter is not available in this environment.')
        return
    importlib.import_module('CameraClassifier.app')


if __name__ == '__main__':
    # Simple runner
    test_training_and_prediction()
    test_app_import()
    print('All tests passed.')
