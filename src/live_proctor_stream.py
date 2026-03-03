import argparse
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import cv2
import mediapipe as mp
import requests
from werkzeug.serving import make_server

from . import live_proctor as lp
from .camera_source import create_camera_source
from .utils_rois import crop, load_rois
from .video_stream_mjpeg import build_app

CONTROL_URL = "https://smartproctoring.online/get_status.php"
CONTROL_POLL_SEC = 2.0
DEVICE_ID = "2026"


class ControlPoller:
    def __init__(self, device_id, interval_sec):
        self.device_id = device_id
        self.interval_sec = interval_sec
        self._running = False
        self._thread = None
        self.current_state = 0
        self._lock = threading.Lock()
        self._last_warn = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_state(self):
        with self._lock:
            return self.current_state

    def _loop(self):
        session = requests.Session()
        while self._running:
            try:
                r = session.get(
                    CONTROL_URL,
                    params={"device_id": self.device_id},
                    timeout=3.0,
                )
                if r.status_code == 200:
                    data = r.json()
                    if "start" in data:
                        with self._lock:
                            self.current_state = int(data["start"])
                else:
                    self._warn()
            except Exception:
                self._warn()
            time.sleep(self.interval_sec)

    def _warn(self):
        now = time.time()
        if now - self._last_warn > 10:
            print("[WARN] Control poll failed (keeping last state)", flush=True)
            self._last_warn = now


class FrameHub:
    def __init__(self, camera_source):
        self.camera_source = camera_source
        self.latest_frame = None
        self.latest_timestamp = None
        self.frame_id = 0
        self._running = False
        self._thread = None
        self._cond = threading.Condition()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="frame-hub-capture", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        while self._running:
            ok, frame = self.camera_source.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.01)
                continue
            ts = time.time()
            with self._cond:
                self.latest_frame = frame.copy()
                self.latest_timestamp = ts
                self.frame_id += 1
                self._cond.notify_all()

    def wait_next(self, last_frame_id, timeout=2.0):
        with self._cond:
            if self.frame_id <= last_frame_id and self._running:
                self._cond.wait(timeout=timeout)
            return self.latest_frame, self.latest_timestamp, self.frame_id

    def get_latest(self):
        with self._cond:
            return self.latest_frame, self.latest_timestamp, self.frame_id


class HubBroadcaster:
    def __init__(self, hub, max_fps, jpeg_quality, flip_mode):
        self.hub = hub
        self.frame_interval = 1.0 / max(0.1, float(max_fps))
        self.jpeg_quality = int(max(1, min(100, jpeg_quality)))
        self.flip_mode = flip_mode
        self._last_emit = 0.0
        self._overlay_frame = None
        self._overlay_frame_id = 0
        self._lock = threading.Lock()

    def _apply_flip(self, frame):
        if self.flip_mode == "h":
            return cv2.flip(frame, 1)
        if self.flip_mode == "v":
            return cv2.flip(frame, 0)
        if self.flip_mode == "hv":
            return cv2.flip(frame, -1)
        return frame

    def set_overlay_frame(self, frame, frame_id):
        if frame is None:
            return
        with self._lock:
            self._overlay_frame = frame.copy()
            self._overlay_frame_id = int(frame_id)

    def start(self):
        return

    def stop(self):
        return

    def wait_next(self, last_frame_id, timeout=2.0):
        with self._lock:
            overlay_frame = None if self._overlay_frame is None else self._overlay_frame.copy()
            overlay_frame_id = self._overlay_frame_id

        if overlay_frame is not None and overlay_frame_id > last_frame_id:
            frame = overlay_frame
            frame_id = overlay_frame_id
        else:
            frame, _ts, frame_id = self.hub.wait_next(last_frame_id, timeout=timeout)
            if frame is None:
                return frame_id, None
            if frame_id == last_frame_id:
                time.sleep(0.01)
                return frame_id, None

        now = time.monotonic()
        sleep_for = (self._last_emit + self.frame_interval) - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_emit = time.monotonic()

        stream_frame = self._apply_flip(frame)
        encoded, buffer = cv2.imencode(".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not encoded:
            return frame_id, None
        return frame_id, buffer.tobytes()


def parse_args():
    parser = argparse.ArgumentParser(description="Live smart proctor + MJPEG streaming")
    parser.add_argument("--rois", default="configs/rois_live.json", help="Path to ROI JSON")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=360, help="Camera height")
    parser.add_argument("--fps", type=int, default=20, help="Camera FPS target")
    parser.add_argument("--headless", action="store_true", help="Disable GUI preview")
    parser.add_argument("--model-complexity", type=int, default=0, choices=[0, 1, 2], help="MediaPipe Pose model complexity")
    parser.add_argument("--save-evidence", action="store_true", help="Reserved for future evidence saving (currently no-op)")
    parser.add_argument("--camera-device", default=None, help="Optional OpenCV camera device/index override")
    parser.add_argument("--opencv-index", type=int, default=None, help="Explicit OpenCV camera index to try first")
    parser.add_argument(
        "--enabled-rois",
        default="all",
        help='ROI ids to process: "all" (default) or comma-separated ids (e.g. S2 or S1,S3)',
    )
    parser.add_argument("--show-disabled", action="store_true", help="Include disabled ROIs in periodic console summary")

    parser.add_argument("--enable-stream", action="store_true", help="Enable MJPEG streaming server")
    parser.add_argument("--stream-host", default="0.0.0.0", help="MJPEG bind host")
    parser.add_argument("--stream-port", type=int, default=8000, help="MJPEG bind port")
    parser.add_argument("--stream-max-fps", type=float, default=15.0, help="Maximum MJPEG stream FPS")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1..100")
    parser.add_argument("--flip", choices=["none", "h", "v", "hv"], default="none", help="Optional stream frame flip")
    parser.add_argument("--token", default=None, help="Optional token required by /mjpeg and /")
    return parser.parse_args()


def main():
    args = parse_args()
    session_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    print(f"[INFO] Local now: {datetime.now().astimezone().isoformat()}", flush=True)

    lp.os.makedirs(lp.LOG_DIR, exist_ok=True)
    log_file = lp.os.path.join(lp.LOG_DIR, f"session_{session_id}.jsonl")

    post_url = lp.os.environ.get("SP_LOG_POST_URL", "").strip()
    timeout_sec = float(lp.os.environ.get("SP_LOG_TIMEOUT_SEC", "6.0"))
    max_replay = int(lp.os.environ.get("SP_LOG_MAX_PENDING_REPLAY", "5000"))
    if post_url:
        uploader = lp.AsyncEventUploader(post_url, timeout_sec=timeout_sec, pending_file=lp.PENDING_FILE, max_replay=max_replay)
        uploader.start()
        print(f"[INFO] Event log POST enabled: {post_url}", flush=True)
    else:
        uploader = None
        print("[WARN] SP_LOG_POST_URL not set; uploading disabled (session JSONL only).", flush=True)

    _, rois = load_rois(args.rois)
    lp.validate_live_rois(rois)
    enabled_roi_ids = lp.resolve_enabled_roi_ids(rois, args.enabled_rois)
    enabled_rois = [roi for roi in rois if roi["id"] in enabled_roi_ids]

    headless = args.headless
    if not headless and not lp.display_available():
        print("[WARN] DISPLAY not available. Falling back to headless mode.", flush=True)
        headless = True

    camera, backend, _ = create_camera_source(
        width=args.width,
        height=args.height,
        fps=args.fps,
        device=args.camera_device,
        opencv_index=args.opencv_index,
    )
    print(f"[INFO] Camera backend: {backend}", flush=True)

    hub = FrameHub(camera)
    hub.start()

    stream_thread = None
    stream_broadcaster = None
    stream_server = None
    if args.enable_stream:
        stream_broadcaster = HubBroadcaster(
            hub=hub,
            max_fps=args.stream_max_fps,
            jpeg_quality=args.jpeg_quality,
            flip_mode=args.flip,
        )
        stream_args = SimpleNamespace(token=args.token)
        app = build_app(stream_broadcaster, stream_args)

        try:
            stream_server = make_server(
                args.stream_host,
                args.stream_port,
                app,
                threaded=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to start MJPEG server on {args.stream_host}:{args.stream_port}"
            ) from exc

        def run_stream():
            stream_server.serve_forever()

        stream_thread = threading.Thread(target=run_stream, name="mjpeg-server", daemon=True)
        stream_thread.start()
        print(
            f"[INFO] MJPEG server started on {args.stream_host}:{args.stream_port} "
            f"(max_fps={args.stream_max_fps}, jpeg_quality={args.jpeg_quality}, flip={args.flip})",
            flush=True,
        )

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    states = {roi["id"]: lp.StudentState(roi["id"]) for roi in rois}
    evidence = lp.EvidenceManager(
        student_ids=list(states.keys()),
        fps=args.fps,
        session_id=session_id,
        log_file=log_file,
        uploader=uploader,
        camera_fps=args.fps,
        evidence_fps=lp.EVIDENCE_FPS,
    )

    poller = ControlPoller(DEVICE_ID, CONTROL_POLL_SEC)
    poller.start()
    detection_active = False

    roi_index = 0
    last_print_ts = time.time()
    last_processed_frame_id = 0

    try:
        while True:
            remote_state = poller.get_state()
            if not detection_active and remote_state == 1:
                detection_active = True
                session_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
                states = {roi["id"]: lp.StudentState(roi["id"]) for roi in rois}
                evidence.close_all(time.time())
                evidence = lp.EvidenceManager(
                    student_ids=list(states.keys()),
                    fps=args.fps,
                    session_id=session_id,
                    log_file=lp.os.path.join(lp.LOG_DIR, f"session_{session_id}.jsonl"),
                    uploader=uploader,
                    camera_fps=args.fps,
                    evidence_fps=lp.EVIDENCE_FPS,
                )
                print(f"[INFO] Detection STARTED (session={session_id})", flush=True)

            if detection_active and remote_state == 0:
                detection_active = False
                evidence.close_all(time.time())
                print("[INFO] Detection STOPPED", flush=True)

            frame, now_ts, frame_id = hub.get_latest()
            if frame is None:
                time.sleep(0.01)
                continue
            if frame_id == last_processed_frame_id:
                frame, now_ts, frame_id = hub.wait_next(last_processed_frame_id, timeout=2.0)
                if frame is None or frame_id == last_processed_frame_id:
                    continue
            last_processed_frame_id = frame_id

            if detection_active:
                evidence.update_frame(frame, now_ts)

                roi = enabled_rois[roi_index]
                sid = roi["id"]
                student = states[sid]
                current_index = next(i for i, item in enumerate(rois) if item["id"] == sid)
                neighbor_roi = None
                if len(rois) > 1:
                    candidate_neighbor = rois[(current_index + 1) % len(rois)]
                    if candidate_neighbor["id"] != sid and candidate_neighbor["id"] in enabled_roi_ids:
                        neighbor_roi = candidate_neighbor
                roi_index = (roi_index + 1) % len(enabled_rois)

                roi_crop = crop(frame, roi)
                if roi_crop.size > 0:
                    rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)
                    if result.pose_landmarks:
                        signals, metrics, reliable_pose = lp.compute_signals(
                            result.pose_landmarks.landmark,
                            roi,
                            neighbor_roi,
                            student.baseline,
                        )
                        if reliable_pose:
                            student.add_baseline(metrics, signals)
                            student.update_with_signals(signals, metrics)
                            student.adapt_baseline(metrics, signals)
                        else:
                            signals["EMPTY"] = True
                            student.update_with_signals(signals, metrics)
                    else:
                        student.update_no_pose()
                else:
                    student.update_no_pose()

                evidence.update_student(sid, student.window[-1]["signals"], now_ts)
                evidence.write_active_events(now_ts)

            out = frame.copy()
            lp.draw_proctor_overlay(out, rois, states, enabled_roi_ids)
            if stream_broadcaster is not None:
                stream_broadcaster.set_overlay_frame(out, frame_id)

            if not headless:
                cv2.imshow("Smart Proctor Live", out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    for s in states.values():
                        s.reset_baseline()
                if key == 27:
                    break

            now = time.time()
            if now - last_print_ts >= 1.0:
                lp.print_periodic_summary(rois, states, enabled_roi_ids, show_disabled=args.show_disabled)
                last_print_ts = now

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Shutting down.", flush=True)
    finally:
        poller.stop()
        hub.stop()
        evidence.close_all(time.time())
        if uploader is not None:
            uploader.stop()
            uploader.join(3.0)
        if stream_broadcaster is not None:
            stream_broadcaster.stop()
        if stream_server is not None:
            stream_server.shutdown()
        camera.release()
        pose.close()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
