import cv2 as cv

class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.vid = cv.VideoCapture(index)
        self.width = int(width)
        self.height = int(height)
        self.vid.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        ret, frame = self.vid.read()
        if not ret:
            return False, None
        # convert to RGB for Tkinter compatibility
        return True, cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def __del__(self):
        if hasattr(self, 'vid') and self.vid.isOpened():
            self.vid.release()
