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
        # train_model now takes num_classes (int), not a counters list
        m.train_model(2)
        assert m.trained, "Model did not set trained=True after training"

        bright_rgb = cv.cvtColor(np.full((150, 150), 220, dtype=np.uint8), cv.COLOR_GRAY2RGB)
        dark_rgb = cv.cvtColor(np.full((150, 150), 30, dtype=np.uint8), cv.COLOR_GRAY2RGB)

        # predict() now returns (class_label, confidence)
        p1, c1 = m.predict(bright_rgb)
        p2, c2 = m.predict(dark_rgb)

        assert p1 in (1, 2), f"Expected class label 1 or 2, got {p1}"
        assert p2 in (1, 2), f"Expected class label 1 or 2, got {p2}"
        assert p1 != p2, "Synthetic bright/dark samples should map to different classes"
        assert 0.0 <= c1 <= 1.0, f"Confidence out of range: {c1}"
        assert 0.0 <= c2 <= 1.0, f"Confidence out of range: {c2}"
    finally:
        cleanup_synthetic_data()


def test_untrained_model_returns_zero():
    m = model_module.Model()
    frame = cv.cvtColor(np.zeros((100, 100), dtype=np.uint8), cv.COLOR_GRAY2RGB)
    pred, conf = m.predict(frame)
    assert pred == 0 and conf == 0.0, "Untrained model should return (0, 0.0)"


def test_app_import():
    if not _HAS_TK:
        print('[INFO] Skipping GUI import test — Tkinter not available.')
        return
    importlib.import_module('CameraClassifier.app')


if __name__ == '__main__':
    test_training_and_prediction()
    test_untrained_model_returns_zero()
    test_app_import()
    print('All tests passed.')
