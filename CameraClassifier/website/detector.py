"""
Object detection wrapper using YOLOv8 (ultralytics).
Weights (yolov8n.pt, ~6 MB) are downloaded automatically from the
Ultralytics CDN on first use — pre-trained on the public COCO dataset
which covers 80 common object categories.
"""

import cv2 as cv

try:
    from ultralytics import YOLO as _YOLO
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def is_available() -> bool:
    return _AVAILABLE


# Distinct BGR colors, one per class slot (cycles for 80 COCO classes)
_PALETTE = [
    (86,  180, 233),   # sky-blue
    (230, 159,   0),   # orange
    (0,   158, 115),   # teal
    (213,  94,   0),   # vermilion
    (0,   114, 178),   # blue
    (204, 121, 167),   # pink
    (240, 228,  66),   # yellow
    (0,   202, 148),   # mint
    (189,  64,  35),   # brick
    (117, 112, 179),   # lavender
]


def class_color(cls_id: int) -> tuple:
    """Return a consistent BGR color for a COCO class ID."""
    return _PALETTE[cls_id % len(_PALETTE)]


class ObjectDetector:
    """Real-time object detector using YOLOv8n (COCO, 80 classes)."""

    def __init__(self, conf: float = 0.40):
        if not _AVAILABLE:
            raise RuntimeError(
                "ultralytics not installed.\n"
                "Run: pip install ultralytics"
            )
        # yolov8n.pt is downloaded from the official Ultralytics release
        # page on first call (~6 MB, cached in ~/.cache/ultralytics/)
        self._model = _YOLO("yolov8n.pt")
        self.conf   = conf

    def detect(self, frame_rgb) -> list:
        """
        Detect objects in an RGB frame.

        Returns list of dicts:
            {
                'label':      str,
                'confidence': float,     # 0.0 – 1.0
                'box':        (x1,y1,x2,y2),
                'cls_id':     int,
            }
        """
        bgr     = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        results = self._model(bgr, conf=self.conf, verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label  = self._model.names.get(cls_id, f"class_{cls_id}")
                out.append({
                    "label":      label,
                    "confidence": round(conf, 3),
                    "box":        (x1, y1, x2, y2),
                    "cls_id":     cls_id,
                })
        return out
