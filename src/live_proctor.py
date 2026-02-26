import argparse
import json
import math
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timezone

import cv2
import mediapipe as mp
import numpy as np

from .camera_source import create_camera_source
from .utils_rois import crop, load_rois

SIGNAL_NAMES = ["ROT", "BOUND", "REACH", "LEAN", "STAND", "EMPTY"]
POINTS = {
    "ROT": 1,
    "BOUND": 2,
    "REACH": 3,
    "LEAN": 2,
    "STAND": 3,
    "EMPTY": 3,
}

VIS_THRESH = 0.5
SHOULDER_VIS_THRESH = 0.50
HEAD_VIS_THRESH = 0.30
BASELINE_SAMPLES = 30
BASELINE_ALPHA = 0.02

LEAN_X_THRESH = 0.40
STAND_Y_THRESH = 0.12
ROT_ROLL_THRESH = 7.0

LEAN_WARN_N = 3
LEAN_WARN_COUNT = 2
LEAN_FLAG_N = 4
LEAN_FLAG_COUNT = 3

ROT_WARN_N = 4
ROT_WARN_COUNT = 3
ROT_FLAG_N = 5
ROT_FLAG_COUNT = 4

DEBUG_OVERLAY = False

ROLLING_N = 10
WARN_POINTS = 3
FLAG_SUM = 12
FLAG_K = 3
# Clear slightly faster once suspicious signals stop.
CLEAR_SUM = 3
EMPTY_WARN_COUNT = 3
EMPTY_WARN_N = 6
EMPTY_FLAG_COUNT = 6
EMPTY_FLAG_N = 10
STAND_FLAG_K = 2
STRONG_SIGNALS = {"BOUND", "REACH", "STAND", "EMPTY"}

PRE_EVENT_SEC = 3
POST_SILENCE_SEC = 3
EVIDENCE_FPS = 12
EVIDENCE_DIR = "evidence"
LOG_DIR = "evidence/logs"


class StudentState:
    def __init__(self, student_id):
        self.student_id = student_id
        self.state = "NO_POSE"
        self.baseline_samples = []
        self.baseline = None
        self.window = deque(maxlen=ROLLING_N)
        self.active_signals = []
        self.last_points = 0
        self.last_reliable = False
        self.last_metrics = {}

    def reset_baseline(self):
        self.baseline_samples = []
        self.baseline = None

    def add_baseline(self, metrics, signals):
        if self.baseline is not None:
            return
        required = [
            "shoulder_angle_deg",
            "shoulder_mid_x",
            "shoulder_mid_y",
            "shoulder_width",
        ]
        if any(metrics.get(k) is None for k in required):
            return
        if any(signals.get(name, False) for name in STRONG_SIGNALS):
            return
        if signals.get("EMPTY", False):
            return

        self.baseline_samples.append({k: float(metrics[k]) for k in required})
        if len(self.baseline_samples) >= BASELINE_SAMPLES:
            self.baseline = {k: float(np.median([s[k] for s in self.baseline_samples])) for k in required}

    def adapt_baseline(self, metrics, signals):
        if self.baseline is None:
            return
        if self.state != "OK":
            return
        if any(signals.get(name, False) for name in STRONG_SIGNALS):
            return
        if signals.get("EMPTY", False):
            return
        if signals.get("ROT", False):
            return
        for key in self.baseline.keys():
            cur = metrics.get(key)
            if cur is None:
                return
        for key in self.baseline.keys():
            self.baseline[key] = (1.0 - BASELINE_ALPHA) * self.baseline[key] + BASELINE_ALPHA * float(metrics[key])

    def update_with_signals(self, signals, metrics):
        points = sum(POINTS[name] for name, on in signals.items() if on)
        self.window.append({"signals": signals.copy(), "points": points})
        self.last_points = points
        self.active_signals = [name for name, on in signals.items() if on]
        self.last_reliable = bool(metrics.get("reliable_pose", False))
        self.last_metrics = metrics
        self._update_state()

    def update_no_pose(self):
        signals = {name: False for name in SIGNAL_NAMES}
        signals["EMPTY"] = True
        metrics = {"reliable_pose": False}
        self.last_reliable = False
        self.update_with_signals(signals, metrics)

    def rolling_sum(self):
        return int(sum(item["points"] for item in self.window))

    def rolling_count(self, signal_name, n=None):
        items = self.window if n is None else list(self.window)[-n:]
        return int(sum(1 for item in items if item["signals"].get(signal_name, False)))

    def _update_state(self):
        cur_points = self.last_points
        strong_now = any(name in STRONG_SIGNALS for name in self.active_signals)
        roll_sum = self.rolling_sum()
        lean_warn = self.rolling_count("LEAN", n=LEAN_WARN_N) >= LEAN_WARN_COUNT
        lean_flag = self.rolling_count("LEAN", n=LEAN_FLAG_N) >= LEAN_FLAG_COUNT
        rot_warn = self.rolling_count("ROT", n=ROT_WARN_N) >= ROT_WARN_COUNT
        rot_flag = self.rolling_count("ROT", n=ROT_FLAG_N) >= ROT_FLAG_COUNT
        reach_count = self.rolling_count("REACH")
        bound_count = self.rolling_count("BOUND")
        stand_count = self.rolling_count("STAND")
        empty_warn_count = self.rolling_count("EMPTY", EMPTY_WARN_N)
        empty_flag_count = self.rolling_count("EMPTY", EMPTY_FLAG_N)
        roll_sum_for_flag = roll_sum

        recent = list(self.window)
        stand_pattern_count = 0
        for i in range(1, len(recent)):
            if recent[i]["signals"].get("EMPTY", False) and recent[i - 1]["signals"].get("BOUND", False):
                stand_pattern_count += 1

        warn = cur_points >= WARN_POINTS or strong_now
        flag = (
            roll_sum_for_flag >= FLAG_SUM
            or reach_count >= FLAG_K
            or bound_count >= FLAG_K
            or stand_count >= STAND_FLAG_K
            or stand_pattern_count >= STAND_FLAG_K
            or empty_flag_count >= EMPTY_FLAG_COUNT
            or lean_flag
            or rot_flag
        )
        warn = warn or empty_warn_count >= EMPTY_WARN_COUNT or stand_count >= 1 or stand_pattern_count >= 1
        warn = warn or lean_warn or rot_warn

        if self.state == "FLAG":
            recent_strong = reach_count > 0 or bound_count > 0 or stand_count > 0
            recent_empty = empty_warn_count > 0
            if roll_sum_for_flag < CLEAR_SUM and not recent_strong and not recent_empty:
                self.state = "OK"
            return

        if flag:
            self.state = "FLAG"
        elif warn:
            self.state = "WARN"
        else:
            self.state = "OK"


class EvidenceManager:
    def __init__(self, student_ids, fps, session_id, log_file, api_client, camera_fps=None, evidence_fps=EVIDENCE_FPS):
        self.fps = int(evidence_fps)
        self.camera_fps = int(fps)
        self.session_id = session_id
        self.log_file = log_file
        self.api_client = api_client
        self.frame_interval = 1.0 / float(max(1, self.fps))
        self.frame_buffer = deque(maxlen=max(1, int(PRE_EVENT_SEC * max(1, camera_fps or self.fps))))
        self.latest_frame = None
        self.events = {}
        for student_id in student_ids:
            self.events[student_id] = {
                "active": False,
                "start_time": None,
                "last_motion_time": None,
                "suspicious": False,
                "writer": None,
                "clip_file": None,
                "signals": set(),
                "frame_count": 0,
                "last_write_time": None,
                "next_write_time": None,
            }
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def update_frame(self, frame, timestamp):
        frame_copy = frame.copy()
        self.latest_frame = frame_copy
        self.frame_buffer.append((timestamp, frame_copy))

    def _should_write(self, event, now):
        if event["next_write_time"] is None:
            return True
        return now >= event["next_write_time"]

    def _write_frame(self, event, frame, write_time):
        event["writer"].write(frame)
        event["frame_count"] += 1
        event["last_write_time"] = write_time
        if event["next_write_time"] is None:
            event["next_write_time"] = write_time + self.frame_interval
        else:
            event["next_write_time"] += self.frame_interval

    def _write_buffered_pre_event(self, event, timestamp):
        cutoff = timestamp - PRE_EVENT_SEC
        buffered = [(ts, frame) for ts, frame in self.frame_buffer if ts >= cutoff and ts <= timestamp]
        if not buffered:
            self._write_frame(event, self.latest_frame, timestamp)
            return

        buffered.sort(key=lambda item: item[0])
        slot_time = buffered[0][0]
        buffer_idx = 0
        frame_for_slot = buffered[0][1]

        while slot_time <= timestamp:
            while buffer_idx + 1 < len(buffered) and buffered[buffer_idx + 1][0] <= slot_time:
                buffer_idx += 1
                frame_for_slot = buffered[buffer_idx][1]
            self._write_frame(event, frame_for_slot, slot_time)
            slot_time += self.frame_interval

        if event["last_write_time"] is None or event["last_write_time"] < timestamp:
            self._write_frame(event, self.latest_frame, timestamp)

    def update_student(self, student_id, signals, timestamp):
        suspicious = any(signals.get(s, False) for s in ["LEAN", "ROT", "REACH", "BOUND", "STAND"])
        event = self.events[student_id]
        event["suspicious"] = suspicious

        if not event["active"] and suspicious:
            timestamp_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            clip_file = f"{student_id}_{timestamp_str}.mp4"
            filepath = os.path.join(EVIDENCE_DIR, clip_file)
            if self.latest_frame is None:
                return
            frame_height, frame_width = self.latest_frame.shape[:2]
            if frame_height <= 0 or frame_width <= 0:
                return
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(filepath, fourcc, self.fps, (frame_width, frame_height))

            event["active"] = True
            event["start_time"] = timestamp
            event["last_motion_time"] = timestamp
            event["writer"] = writer
            event["clip_file"] = clip_file
            event["signals"] = {s for s, enabled in signals.items() if enabled}
            self._write_buffered_pre_event(event, timestamp)

        elif event["active"]:
            if suspicious:
                event["last_motion_time"] = timestamp
                event["signals"].update({s for s, enabled in signals.items() if enabled})

    def write_active_events(self, timestamp):
        if self.latest_frame is None:
            return

        for student_id, event in self.events.items():
            if not event["active"]:
                continue

            if self._should_write(event, timestamp):
                while self._should_write(event, timestamp):
                    write_time = event["next_write_time"] if event["next_write_time"] is not None else timestamp
                    self._write_frame(event, self.latest_frame, write_time)

            if not event["suspicious"] and (timestamp - event["last_motion_time"] > POST_SILENCE_SEC):
                self._close_event(student_id, timestamp)

    def _close_event(self, student_id, timestamp):
        event = self.events[student_id]
        if not event["active"]:
            return

        event["writer"].release()
        duration = round(event["frame_count"] / float(self.fps), 2)
        entry = {
            "session_id": self.session_id,
            "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
            "student_id": student_id,
            "signals": sorted(event["signals"]),
            "clip_file": event["clip_file"],
            "duration_sec": duration,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if self.api_client is not None:
            ok = self.api_client.send_event(entry)
            if not ok:
                print("[WARN] Failed to POST event log; kept in JSONL.", flush=True)

        event["active"] = False
        event["start_time"] = None
        event["last_motion_time"] = None
        event["suspicious"] = False
        event["writer"] = None
        event["clip_file"] = None
        event["signals"] = set()
        event["frame_count"] = 0
        event["last_write_time"] = None
        event["next_write_time"] = None

    def close_all(self, timestamp):
        for student_id in self.events.keys():
            self._close_event(student_id, timestamp)


class EventAPIClient:
    def __init__(self, url, mode="json", timeout=2.0, max_queue_size=1000):
        self.url = url
        self.mode = mode if mode in {"json", "form"} else "json"
        self.timeout = float(timeout)
        self.queue = queue.Queue(maxsize=max_queue_size)
        self._stop_sentinel = object()
        self._worker = threading.Thread(target=self._upload_worker, daemon=True)
        self._worker.start()

    def send_event(self, entry: dict) -> bool:
        try:
            self.queue.put_nowait(entry)
            return True
        except queue.Full:
            print("[WARN] Event upload queue is full; dropping event upload.", flush=True)
            return False

    def _upload_worker(self):
        while True:
            item = self.queue.get()
            try:
                if item is self._stop_sentinel:
                    return
                self._send_event_sync(item)
            finally:
                self.queue.task_done()

    def _send_event_sync(self, entry: dict) -> bool:
        try:
            if self.mode == "form":
                body = urllib.parse.urlencode({"event": json.dumps(entry)}).encode("utf-8")
                headers = {"Content-Type": "application/x-www-form-urlencoded"}
            else:
                body = json.dumps(entry).encode("utf-8")
                headers = {"Content-Type": "application/json"}

            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.status == 200
        except Exception as exc:
            print(f"[WARN] Event log POST failed: {exc}", flush=True)
            return False

    def close(self):
        self.queue.put(self._stop_sentinel)
        self._worker.join(timeout=self.timeout)
        if self._worker.is_alive():
            print("[WARN] Event upload worker did not stop cleanly.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Live smart proctor with MediaPipe Pose on Raspberry Pi")
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
    return parser.parse_args()


def resolve_enabled_roi_ids(rois, enabled_rois_arg):
    roi_ids = [roi["id"] for roi in rois]
    if enabled_rois_arg.strip().lower() == "all":
        return set(roi_ids)

    requested_ids = [item.strip() for item in enabled_rois_arg.split(",") if item.strip()]
    requested_set = set(requested_ids)
    invalid_ids = sorted([rid for rid in requested_set if rid not in roi_ids])
    if invalid_ids:
        valid = ", ".join(roi_ids)
        invalid = ", ".join(invalid_ids)
        raise ValueError(f"Unknown ROI id(s): {invalid}. Valid ROI ids: {valid}")
    if not requested_set:
        raise ValueError("Enabled ROI list is empty. Use --enabled-rois all or provide at least one ROI id.")

    enabled_in_order = [rid for rid in roi_ids if rid in requested_set]
    if not enabled_in_order:
        raise ValueError("Enabled ROI list resolved to empty after validation.")
    return set(enabled_in_order)

def compute_signals(landmarks, roi, neighbor_roi, baseline):
    lm = mp.solutions.pose.PoseLandmark
    nose = landmarks[lm.NOSE]
    l_sh = landmarks[lm.LEFT_SHOULDER]
    r_sh = landmarks[lm.RIGHT_SHOULDER]
    l_wr = landmarks[lm.LEFT_WRIST]
    r_wr = landmarks[lm.RIGHT_WRIST]
    l_ear = landmarks[lm.LEFT_EAR]
    r_ear = landmarks[lm.RIGHT_EAR]

    shoulders_reliable = l_sh.visibility >= SHOULDER_VIS_THRESH and r_sh.visibility >= SHOULDER_VIS_THRESH
    nose_reliable = nose.visibility >= HEAD_VIS_THRESH
    ears_both_reliable = l_ear.visibility >= HEAD_VIS_THRESH and r_ear.visibility >= HEAD_VIS_THRESH
    ear_one_reliable = (l_ear.visibility >= HEAD_VIS_THRESH) ^ (r_ear.visibility >= HEAD_VIS_THRESH)
    wrist_reliable = l_wr.visibility >= VIS_THRESH or r_wr.visibility >= VIS_THRESH
    head_anchor_reliable = nose_reliable or ears_both_reliable or ear_one_reliable
    reliable_pose = shoulders_reliable and head_anchor_reliable

    signals = {name: False for name in SIGNAL_NAMES}
    metrics = {
        "shoulder_angle_deg": None,
        "shoulder_roll": None,
        "lean_x": None,
        "shoulder_mid_x": None,
        "shoulder_mid_y": None,
        "shoulder_mid_y_delta": None,
        "shoulder_width": None,
        "reliable_pose": reliable_pose,
    }

    roi_w = float(roi["w"])
    roi_h = float(roi["h"])

    if reliable_pose:
        shoulder_mid_x = (l_sh.x + r_sh.x) / 2.0
        shoulder_mid_y = (l_sh.y + r_sh.y) / 2.0
        shoulder_width = abs(l_sh.x - r_sh.x)
        metrics["shoulder_mid_x"] = float(shoulder_mid_x)
        metrics["shoulder_mid_y"] = float(shoulder_mid_y)
        metrics["shoulder_width"] = float(shoulder_width)

        head_anchor_xy = None
        if shoulder_width > 1e-4:
            if nose_reliable:
                head_anchor_xy = (nose.x, nose.y)
            elif ears_both_reliable:
                head_anchor_xy = ((l_ear.x + r_ear.x) / 2.0, (l_ear.y + r_ear.y) / 2.0)
            elif ear_one_reliable:
                if l_ear.visibility >= HEAD_VIS_THRESH:
                    head_anchor_xy = (l_ear.x, l_ear.y)
                else:
                    head_anchor_xy = (r_ear.x, r_ear.y)

            if baseline is not None:
                lean_x = (shoulder_mid_x - baseline["shoulder_mid_x"]) / shoulder_width
                stand_y_delta = baseline["shoulder_mid_y"] - shoulder_mid_y
                metrics["lean_x"] = float(lean_x)
                metrics["shoulder_mid_y_delta"] = float(stand_y_delta)
                signals["LEAN"] = abs(lean_x) > LEAN_X_THRESH
                signals["STAND"] = stand_y_delta > STAND_Y_THRESH

        dx = r_sh.x - l_sh.x
        dy = r_sh.y - l_sh.y
        shoulder_angle_deg = math.degrees(math.atan2(dy, dx))

        # Roll relative to horizontal (0°), normalized modulo 180° so that a
        # level torso is always near 0° regardless of shoulder ordering.
        shoulder_roll = ((shoulder_angle_deg + 90.0) % 180.0) - 90.0

        metrics["shoulder_angle_deg"] = float(shoulder_angle_deg)
        metrics["shoulder_roll"] = float(shoulder_roll)

        # Symmetric roll detection (left or right)
        signals["ROT"] = abs(shoulder_roll) > ROT_ROLL_THRESH

        margin_x = 0.1 * roi_w
        margin_y = 0.1 * roi_h
        head_anchor_px = None
        if head_anchor_xy is not None:
            head_anchor_px = (head_anchor_xy[0] * roi_w, head_anchor_xy[1] * roi_h)
        shoulder_mid_px = (shoulder_mid_x * roi_w, shoulder_mid_y * roi_h)

        def outside_safe(pt):
            return (
                pt[0] < margin_x
                or pt[0] > roi_w - margin_x
                or pt[1] < margin_y
                or pt[1] > roi_h - margin_y
            )

        head_boundary_hit = head_anchor_px is not None and outside_safe(head_anchor_px)
        signals["BOUND"] = head_boundary_hit or outside_safe(shoulder_mid_px)

    if wrist_reliable:
        margin_x = 0.1 * roi_w
        margin_y = 0.1 * roi_h

        wrists = []
        if l_wr.visibility >= VIS_THRESH:
            wrists.append((l_wr.x * roi_w, l_wr.y * roi_h))
        if r_wr.visibility >= VIS_THRESH:
            wrists.append((r_wr.x * roi_w, r_wr.y * roi_h))

        near_boundary = any(
            wx < margin_x or wx > roi_w - margin_x or wy < margin_y or wy > roi_h - margin_y
            for wx, wy in wrists
        )

        neighbor_hit = False
        if neighbor_roi is not None:
            nx = float(neighbor_roi["x"])
            ny = float(neighbor_roi["y"])
            nw = float(neighbor_roi["w"])
            nh = float(neighbor_roi["h"])
            ox = float(roi["x"])
            oy = float(roi["y"])
            for wx, wy in wrists:
                gx = ox + wx
                gy = oy + wy
                if nx <= gx <= nx + nw and ny <= gy <= ny + nh:
                    neighbor_hit = True
                    break

        signals["REACH"] = near_boundary or neighbor_hit


    return signals, metrics, reliable_pose


def draw_overlay(frame, rois, states, enabled_roi_ids, debug_overlay=False):
    colors = {
        "OK": (255, 255, 255),
        "WARN": (0, 255, 255),
        "FLAG": (0, 0, 255),
        "NO_POSE": (128, 128, 128),
    }
    for roi in rois:
        sid = roi["id"]
        st = states[sid]
        x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
        is_enabled = sid in enabled_roi_ids
        if not is_enabled:
            disabled_color = (80, 80, 80)
            cv2.rectangle(frame, (x, y), (x + w, y + h), disabled_color, 2)
            cv2.putText(
                frame,
                f"{sid} DISABLED",
                (x + 4, max(12, y + 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                disabled_color,
                1,
                cv2.LINE_AA,
            )
            continue

        color = colors.get(st.state, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        short = {
            "LEAN": "L",
            "REACH": "R",
            "BOUND": "B",
            "STAND": "S",
            "EMPTY": "E",
            "ROT": "O",
        }
        ordered = [name for name in SIGNAL_NAMES if name in short]
        sig_items = [short[name] for name in ordered if name in st.active_signals]
        sig_text = " ".join(sig_items) if sig_items else "-"
        line1 = f"{sid} {st.state} [{sig_text}] {st.rolling_sum()}/{st.last_points}"
        cv2.putText(frame, line1, (x + 4, max(12, y + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if debug_overlay:
            m = st.last_metrics

            def fmt(key):
                value = m.get(key)
                if value is None:
                    return None
                return f"{float(value):+.2f}"

            line2 = None
            lean = fmt("lean_x")
            if lean is not None:
                line2 = f"lean={lean}"

            roll = fmt("shoulder_roll")
            if roll is not None:
                line2 = f"{line2} roll={roll}" if line2 else f"roll={roll}"

            if line2:
                y2 = min(y + h - 4, max(12, y + 16))
                cv2.putText(frame, line2, (x + 4, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def print_periodic_summary(rois, states, enabled_roi_ids, show_disabled=False):
    parts = []
    for roi in rois:
        sid = roi["id"]
        if sid not in enabled_roi_ids and not show_disabled:
            continue
        st = states[sid]
        if sid not in enabled_roi_ids:
            parts.append(f"{sid}:DISABLED")
        else:
            sig = ",".join(st.active_signals) if st.active_signals else "-"
            baseline_count = len(st.baseline_samples)
            if st.baseline is not None:
                bl_status = f"BL=READY n={baseline_count}"
            else:
                bl_status = f"BL={baseline_count}/{BASELINE_SAMPLES}"
            parts.append(f"{sid}:{st.state} sum={st.rolling_sum()} pts={st.last_points} signals={sig} {bl_status}")
    print(" | ".join(parts), flush=True)


def display_available():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def validate_live_rois(rois):
    if len(rois) == 0:
        raise ValueError("Expected at least 1 ROI for live mode, got 0")
    if len(rois) != 2:
        print(f"[WARN] Live mode often uses 2 ROIs; got {len(rois)}", flush=True)


def main():
    args = parse_args()
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"session_{session_id}.jsonl")

    post_url = os.environ.get("SP_LOG_POST_URL")
    mode = os.environ.get("SP_LOG_POST_MODE", "json").lower()
    timeout = float(os.environ.get("SP_LOG_TIMEOUT_SEC", "2.0"))
    if post_url:
        api_client = EventAPIClient(post_url, mode=mode, timeout=timeout)
    else:
        api_client = None
        print("[WARN] SP_LOG_POST_URL is not set; API upload is disabled.", flush=True)

    _, rois = load_rois(args.rois)
    validate_live_rois(rois)
    enabled_roi_ids = resolve_enabled_roi_ids(rois, args.enabled_rois)
    enabled_rois = [roi for roi in rois if roi["id"] in enabled_roi_ids]

    headless = args.headless
    if not headless and not display_available():
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

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    states = {roi["id"]: StudentState(roi["id"]) for roi in rois}
    evidence = EvidenceManager(
        student_ids=list(states.keys()),
        fps=args.fps,
        session_id=session_id,
        log_file=log_file,
        api_client=api_client,
        camera_fps=args.fps,
        evidence_fps=EVIDENCE_FPS,
    )
    roi_index = 0
    last_print_ts = time.time()
    debug_overlay = DEBUG_OVERLAY

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("[WARN] Camera frame read failed.", flush=True)
                continue

            now_ts = time.time()
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
                    signals, metrics, reliable_pose = compute_signals(
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

            if not headless:
                out = frame.copy()
                draw_overlay(out, rois, states, enabled_roi_ids, debug_overlay=debug_overlay)
                cv2.imshow("Smart Proctor Live", out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    for s in states.values():
                        s.reset_baseline()
                if key == ord("d"):
                    debug_overlay = not debug_overlay
                if key == 27:
                    break

            now = time.time()
            if now - last_print_ts >= 1.0:
                print_periodic_summary(rois, states, enabled_roi_ids, show_disabled=args.show_disabled)
                last_print_ts = now

    finally:
        evidence.close_all(time.time())
        if api_client is not None:
            api_client.close()
        camera.release()
        pose.close()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
