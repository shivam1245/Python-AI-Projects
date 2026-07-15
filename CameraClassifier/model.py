import os
import cv2 as cv
import numpy as np
from sklearn import svm


class Model:
    def __init__(self):
        # probability=True enables predict_proba() for confidence scores
        self.clf = svm.SVC(kernel='linear', probability=True)
        self.trained = False

    def train_model(self, num_classes: int, base_dir: str = '.'):
        """
        Load images from folders 1/ through num_classes/ under base_dir, train a linear SVM.
        base_dir defaults to '.' so existing Tkinter app behaviour is unchanged.
        """
        X, y = [], []
        for label in range(1, num_classes + 1):
            folder = os.path.join(base_dir, str(label))
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.lower().endswith('.jpg'):
                    continue
                path = os.path.join(folder, fname)
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv.resize(img, (50, 50))
                X.append(img.flatten())
                y.append(label)

        unique_classes = set(y)
        if len(X) >= 2 and len(unique_classes) >= 2:
            self.clf.fit(np.array(X), np.array(y))
            self.trained = True
            print(f"Model trained on {len(X)} samples across {len(unique_classes)} classes.")
        else:
            print("Not enough data to train. Capture samples for at least 2 classes.")

    def predict(self, frame) -> tuple[int, float]:
        """
        Classify an RGB frame.
        Returns (class_label, confidence) where class_label is 0 if untrained.
        """
        frame = frame[1] if isinstance(frame, tuple) else frame
        if frame is None or not self.trained:
            return 0, 0.0
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        gray = cv.resize(gray, (50, 50)).flatten().reshape(1, -1)
        pred = int(self.clf.predict(gray)[0])
        confidence = float(max(self.clf.predict_proba(gray)[0]))
        return pred, confidence
