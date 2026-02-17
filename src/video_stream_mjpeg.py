import argparse
import time
from typing import Optional

import cv2
from flask import Flask, Response, abort, request

from .camera_source import create_camera_source


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


def apply_flip(frame, flip_mode: str):
    if flip_mode == "h":
        return cv2.flip(frame, 1)
    if flip_mode == "v":
        return cv2.flip(frame, 0)
    if flip_mode == "hv":
        return cv2.flip(frame, -1)
    return frame


def token_required(expected_token: Optional[str]):
    if not expected_token:
        return
    if request.args.get("token") != expected_token:
        abort(401)


def build_app(camera_source, args):
    app = Flask(__name__)

    quality = int(max(1, min(100, args.jpeg_quality)))
    frame_interval = 1.0 / max(0.1, float(args.max_fps))

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
            last_sent = 0.0
            while True:
                ok, frame = camera_source.read()
                if not ok or frame is None or frame.size == 0:
                    time.sleep(0.05)
                    continue

                now = time.monotonic()
                since = now - last_sent
                if since < frame_interval:
                    time.sleep(frame_interval - since)

                frame = apply_flip(frame, args.flip)
                encoded, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if not encoded:
                    continue

                jpg = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
                last_sent = time.monotonic()

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

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

    app = build_app(camera_source, args)
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        print("[INFO] Releasing camera source.", flush=True)
        camera_source.release()


if __name__ == "__main__":
    main()
