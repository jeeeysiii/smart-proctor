import cv2
import time


class OpenCVCameraSource:
    def __init__(self, device="/dev/video0", width=640, height=360, fps=20):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Some camera stacks report "opened" before frames are actually ready.
        # Probe a few reads so create_camera_source can fall back to Picamera2 when
        # OpenCV cannot deliver frames.
        for _ in range(8):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return True
            time.sleep(0.05)

        self.release()
        return False

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class Picamera2Source:
    def __init__(self, width=640, height=360, fps=20):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.picam2 = None

    def open(self):
        try:
            from picamera2 import Picamera2
        except ImportError:
            return False

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps},
        )
        self.picam2.configure(config)
        self.picam2.start()
        return True

    def read(self):
        if self.picam2 is None:
            return False, None
        try:
            frame_rgb = self.picam2.capture_array("main")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return False, None
        return True, frame_bgr

    def release(self):
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None


def create_camera_source(width=640, height=360, fps=20, device="/dev/video0"):
    opencv_source = OpenCVCameraSource(device=device, width=width, height=height, fps=fps)
    if opencv_source.open():
        return opencv_source, "opencv"

    picamera_source = Picamera2Source(width=width, height=height, fps=fps)
    if picamera_source.open():
        return picamera_source, "picamera2"

    raise RuntimeError(
        "Unable to open camera via OpenCV (/dev/video0) and Picamera2 fallback. "
        "Check camera connection and permissions."
    )
