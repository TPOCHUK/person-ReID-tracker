import sys

import cv2


class Camera:
    def __init__(self, source="0", width=640, height=480):
        # Numeric sources are camera indices.
        src = int(source) if str(source).isdigit() else source

        # Use the Linux camera backend for local devices.
        if isinstance(src, int) and sys.platform.startswith("linux"):
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Failed to open camera source: {source}")

        # Request the configured camera size.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Request a short buffer; some backends ignore this.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
