import cv2
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CameraInitReport:
    opencv_indices_tried: List[int] = field(default_factory=list)
    opencv_errors: Dict[int, str] = field(default_factory=dict)
    picamera2_error: Optional[str] = None


class OpenCVCameraSource:
    def __init__(self, device=0, width=640, height=360, fps=20, warmup_frames=5):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.warmup_frames = int(warmup_frames)
        self.cap = None
        self.last_error = None

    def open(self):
        self.last_error = None
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            self.last_error = "cannot open capture device"
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        success_reads = 0
        last_shape = None
        for _ in range(max(3, self.warmup_frames)):
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.size > 0:
                success_reads += 1
                last_shape = frame.shape
            else:
                time.sleep(0.05)

        if success_reads == 0:
            self.last_error = "opened but failed to read any frames"
            self.release()
            return False

        if last_shape is None:
            self.last_error = "opened but returned empty frames"
            self.release()
            return False

        return True

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class Picamera2Source:
    def __init__(self, width=640, height=360, fps=20, warmup_frames=5):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.warmup_frames = int(warmup_frames)
        self.picam2 = None
        self.last_error = None

    def _to_bgr(self, frame):
        if frame is None or getattr(frame, "size", 0) == 0:
            raise RuntimeError("captured empty frame")

        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if len(frame.shape) != 3:
            raise RuntimeError(f"unexpected frame shape: {frame.shape}")

        channels = frame.shape[2]
        if channels == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if channels == 4:
            rgb = frame[:, :, :3]
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        raise RuntimeError(f"unsupported channel count: {channels}")

    def open(self):
        self.last_error = None
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            self.last_error = f"import failed: {exc}"
            return False

        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height)},
                controls={"FrameRate": self.fps},
            )
            self.picam2.configure(config)
            self.picam2.start()

            success_reads = 0
            for _ in range(max(5, self.warmup_frames)):
                frame = self.picam2.capture_array("main")
                frame_bgr = self._to_bgr(frame)
                if frame_bgr is not None and frame_bgr.size > 0:
                    success_reads += 1
                else:
                    time.sleep(0.05)

            if success_reads == 0:
                raise RuntimeError("started but failed to capture warmup frames")

            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.release()
            return False

    def read(self):
        if self.picam2 is None:
            return False, None
        try:
            frame = self.picam2.capture_array("main")
            frame_bgr = self._to_bgr(frame)
        except Exception as exc:
            self.last_error = str(exc)
            return False, None
        return True, frame_bgr

    def release(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None


def create_camera_source(width=640, height=360, fps=20, device=None, opencv_index=None):
    report = CameraInitReport()

    indices_to_try = []
    if opencv_index is not None:
        indices_to_try = [int(opencv_index)]
    elif device is not None:
        try:
            indices_to_try = [int(device)]
        except (TypeError, ValueError):
            indices_to_try = [device]
    else:
        indices_to_try = list(range(5))

    for idx in indices_to_try:
        report.opencv_indices_tried.append(idx)
        source = OpenCVCameraSource(device=idx, width=width, height=height, fps=fps)
        if source.open():
            print(f"[INFO] OpenCV camera index {idx} opened successfully.", flush=True)
            return source, "opencv", report

        reason = source.last_error or "unknown error"
        report.opencv_errors[idx] = reason
        print(f"[WARN] OpenCV camera index {idx} failed: {reason}", flush=True)

    picamera_source = Picamera2Source(width=width, height=height, fps=fps)
    if picamera_source.open():
        print("[INFO] Picamera2 opened successfully.", flush=True)
        return picamera_source, "picamera2", report

    report.picamera2_error = picamera_source.last_error or "unknown error"
    print(f"[WARN] Picamera2 failed: {report.picamera2_error}", flush=True)

    opencv_diag = ", ".join(
        f"{idx}: {report.opencv_errors.get(idx, 'unknown error')}" for idx in report.opencv_indices_tried
    ) or "none"

    raise RuntimeError(
        "Unable to open camera via OpenCV and Picamera2 fallback.\n"
        f"OpenCV indices tried: {report.opencv_indices_tried}\n"
        f"OpenCV failures: {opencv_diag}\n"
        f"Picamera2 failure: {report.picamera2_error}\n"
        "Suggestions:\n"
        "- Enable camera stack in raspi-config and reboot.\n"
        "- Check CSI cable orientation and seating on both ends.\n"
        "- Confirm user is in the 'video' group (or use sudo only for debugging)."
    )
