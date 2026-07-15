"""
Emotion detection module using the FER (Facial Expression Recognition) library.
Lazy-loads the TensorFlow model on first use to avoid blocking app startup.
"""

import cv2 as cv

try:
    from fer import FER as _FER
    _FER_AVAILABLE = True
except ImportError:
    _FER_AVAILABLE = False


def is_available() -> bool:
    """Re-check at call time so a restart isn't needed after pip install."""
    if _FER_AVAILABLE:
        return True
    try:
        import importlib
        importlib.import_module("fer")
        return True
    except ImportError:
        return False


class EmotionDetector:
    """
    Wraps the FER library to detect emotions from RGB frames.
    Uses OpenCV Haar cascade internally (mtcnn=False) for faster face detection.
    """

    EMOJI = {
        "happy": "😄",
        "sad": "😢",
        "angry": "😠",
        "fear": "😨",
        "surprise": "😲",
        "disgust": "🤢",
        "neutral": "😐",
    }

    def __init__(self):
        if not _FER_AVAILABLE:
            raise RuntimeError(
                "The 'fer' library is not installed.\n"
                "Run: pip install fer tensorflow"
            )
        self._detector = _FER(mtcnn=False)

    def analyze(self, frame_rgb: "np.ndarray") -> list:
        """
        Detect emotions in an RGB frame.

        Returns a list of dicts for each detected face:
            {
                'box': (x, y, w, h),
                'emotion': str,       # dominant emotion label
                'confidence': float,  # 0.0 – 1.0
                'all': dict,          # full scores for all emotions
            }
        """
        bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        results = self._detector.detect_emotions(bgr)

        output = []
        for face in results:
            emotions = face.get("emotions", {})
            if not emotions:
                continue
            dominant = max(emotions, key=emotions.get)
            output.append({
                "box": face["box"],
                "emotion": dominant,
                "confidence": emotions[dominant],
                "all": emotions,
            })
        return output

    @staticmethod
    def format_label(emotion: str, confidence: float) -> str:
        return f"{emotion.capitalize()} ({confidence:.0%})"
