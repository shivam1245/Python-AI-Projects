import os
import cv2 as cv
import numpy as np
from sklearn import svm


class Model:
    def __init__(self):
        self.clf = svm.SVC(kernel='linear')
        self.trained = False

    def train_model(self, counters):
        X = []
        y = []
        for label in [1, 2]:
            folder = str(label)
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
        if len(X) >= 2 and len(set(y)) >= 2:
            self.clf.fit(np.array(X), np.array(y))
            self.trained = True
        else:
            print("Not enough data to train. Capture images for both classes.")

    def predict(self, frame_tuple):
        # frame_tuple can be (ret, frame) or just frame, handle both
        frame = frame_tuple[1] if isinstance(frame_tuple, tuple) else frame_tuple
        if frame is None or not self.trained:
            return 0
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        gray = cv.resize(gray, (50, 50)).flatten().reshape(1, -1)
        return int(self.clf.predict(gray)[0])
