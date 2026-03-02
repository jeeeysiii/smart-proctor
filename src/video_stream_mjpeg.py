import argparse
import os
import threading
import time
from typing import Optional

import cv2
from flask import Flask, Response, abort, request, send_from_directory
from werkzeug.utils import secure_filename

from .camera_source import create_camera_source


class FrameBroadcaster:
    def __init__(self, camera_source, max_fps: float, jpeg_quality: int, flip_mode: str):
        self.camera_source = camera_source
        self.frame_interval = 1.0 / max(0.1, float(max_fps))
        self.jpeg_quality = int(max(1, min(100, jpeg_quality)))
        self.flip_mode = flip_mode

        self._condition = threading.Condition()
        self._latest_jpeg = None
        self._frame_id = 0
        self._running = False
        self._thread = None

    def _apply_flip(self, frame):
        if self.flip_mode == "h":
            return cv2.flip(frame, 1)
        if self.flip_mode == "v":
            return cv2.flip(frame, 0)
        if self.flip_mode == "hv":
            return cv2.flip(frame, -1)
        return frame

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="mjpeg-capture", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        next_frame_at = time.monotonic()
        while self._running:
            ok, frame = self.camera_source.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.05)
                continue

            frame = self._apply_flip(frame)
            encoded, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if not encoded:
                continue

            with self._condition:
                self._latest_jpeg = buffer.tobytes()
                self._frame_id += 1
                self._condition.notify_all()

            next_frame_at += self.frame_interval
            sleep_for = next_frame_at - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # If camera/encode is slower than target, avoid unbounded drift.
                next_frame_at = time.monotonic()

    def wait_next(self, last_frame_id: int, timeout: float = 2.0):
        with self._condition:
            if self._frame_id <= last_frame_id and self._running:
                self._condition.wait(timeout=timeout)
            return self._frame_id, self._latest_jpeg


def parse_args():
    parser = argparse.ArgumentParser(description="Serve Raspberry Pi camera as MJPEG stream")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=360, help="Camera height")
    parser.add_argument("--fps", type=int, default=20, help="Camera capture target FPS")
    parser.add_argument("--max-fps", type=float, default=15.0, help="Maximum MJPEG stream FPS")
    parser.add_argument("--camera-device", default=None, help="Optional OpenCV camera device/index override")
    parser.add_argument("--opencv-index", type=int, default=None, help="Explicit OpenCV camera index to try first")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1..100")
    parser.add_argument("--flip", choices=["none", "h", "v", "hv"], default="none", help="Optional frame flip")
    parser.add_argument("--token", default=None, help="Optional token required by /mjpeg and /")
    return parser.parse_args()


def token_required(expected_token: Optional[str]):
    if not expected_token:
        return
    if request.args.get("token") != expected_token:
        abort(401)


def build_app(broadcaster: FrameBroadcaster, args):
    app = Flask(__name__)
    evidence_dir = getattr(args, "evidence_dir", "evidence")
    evidence_dir = os.path.abspath(evidence_dir)

    @app.route("/health")
    def health():
        return {"status": "ok"}, 200

    @app.route("/")
    def index():
        token_required(args.token)
        token_qs = f"?token={args.token}" if args.token else ""
        return (
            "<html><body><h3>smart-proctor MJPEG stream</h3>"
            f"<img src='/mjpeg{token_qs}' style='max-width:100%;height:auto;' />"
            "</body></html>"
        )

    @app.route("/mjpeg")
    def mjpeg():
        token_required(args.token)

        def generate():
            last_frame_id = -1
            while True:
                frame_id, jpg = broadcaster.wait_next(last_frame_id)
                if frame_id == last_frame_id or jpg is None:
                    continue
                last_frame_id = frame_id
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Access-Control-Allow-Origin": "https://smartproctoring.online",
                "Access-Control-Allow-Methods": "GET",
                "Access-Control-Allow-Headers": "*",
            }
        )

    @app.route("/clip/<path:filename>")
    def clip(filename):
        token_required(args.token)

        if (
            "/" in filename
            or "\\" in filename
            or ".." in filename
            or not filename.lower().endswith(".mp4")
            or filename != secure_filename(filename)
        ):
            abort(400)

        if not os.path.isdir(evidence_dir):
            abort(404)

        response = send_from_directory(
            evidence_dir,
            filename,
            as_attachment=True,
            download_name=filename,
        )
        response.headers["Access-Control-Allow-Origin"] = "https://smartproctoring.online"
        response.headers["Access-Control-Allow-Methods"] = "GET"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

    return app


def main():
    args = parse_args()
    print(
        f"[INFO] Starting MJPEG server on {args.host}:{args.port} "
        f"(resolution={args.width}x{args.height}, capture_fps={args.fps}, max_fps={args.max_fps}, "
        f"jpeg_quality={args.jpeg_quality}, flip={args.flip})",
        flush=True,
    )

    try:
        camera_source, backend, _report = create_camera_source(
            width=args.width,
            height=args.height,
            fps=args.fps,
            device=args.camera_device,
            opencv_index=args.opencv_index,
        )
    except Exception as exc:
        print("[ERROR] Camera initialization failed.", flush=True)
        print(f"[ERROR] {exc}", flush=True)
        raise

    if backend == "opencv" and getattr(camera_source, "cap", None) is not None:
        backend_id = int(camera_source.cap.get(cv2.CAP_PROP_BACKEND))
        backend_name = cv2.videoio_registry.getBackendName(backend_id) if backend_id > 0 else "unknown"
        print(f"[INFO] Camera backend: opencv ({backend_name}, id={backend_id})", flush=True)
    else:
        print(f"[INFO] Camera backend: {backend}", flush=True)

    broadcaster = FrameBroadcaster(
        camera_source=camera_source,
        max_fps=args.max_fps,
        jpeg_quality=args.jpeg_quality,
        flip_mode=args.flip,
    )
    broadcaster.start()

    app = build_app(broadcaster, args)
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        print("[INFO] Stopping broadcaster and releasing camera source.", flush=True)
        broadcaster.stop()
        camera_source.release()


if __name__ == "__main__":
    main()
