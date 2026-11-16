import cv2


class FrameSource:
    """Read a camera or loop a video file for timed benchmarks."""

    def __init__(self, source: str, width: int = 640, height: int = 480):
        self.source = str(source)
        self.is_file = not self.source.isdigit()

        target = self.source if self.is_file else int(self.source)
        self.cap = cv2.VideoCapture(target)

        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Failed to open benchmark source: {source}")

        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ok, frame = self.cap.read()

        if not ok and self.is_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()

        return ok, frame

    def describe(self) -> str:
        if self.is_file:
            return f"file: {self.source}"
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return f"camera {self.source} ({w}x{h})"

    def release(self):
        self.cap.release()
