import argparse
import math
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from .camera_source import create_camera_source
from .utils_rois import crop, load_rois

SIGNAL_NAMES = ["TURN", "ROT", "BOUND", "REACH"]
POINTS = {"TURN": 1, "ROT": 1, "BOUND": 2, "REACH": 3}

VIS_THRESH = 0.5
BASELINE_SAMPLES = 20
ROLLING_N = 10
WARN_POINTS = 2
FLAG_SUM = 10
FLAG_K = 3
CLEAR_SUM = 4
STRONG_SIGNALS = {"BOUND", "REACH"}


class StudentState:
    def __init__(self, student_id):
        self.student_id = student_id
        self.state = "NO_POSE"
        self.baseline_samples = []
        self.baseline_angle = None
        self.window = deque(maxlen=ROLLING_N)
        self.active_signals = []
        self.last_points = 0
        self.last_reliable = False
        self.last_metrics = {}

    def add_baseline(self, angle_deg, signals):
        if self.baseline_angle is not None:
            return
        if angle_deg is None:
            return
        if any(signals.values()):
            return
        self.baseline_samples.append(angle_deg)
        if len(self.baseline_samples) >= BASELINE_SAMPLES:
            self.baseline_angle = float(np.median(self.baseline_samples))

    def update_with_signals(self, signals, metrics):
        points = sum(POINTS[name] for name, on in signals.items() if on)
        self.window.append({"signals": signals.copy(), "points": points})
        self.last_points = points
        self.active_signals = [name for name, on in signals.items() if on]
        self.last_reliable = True
        self.last_metrics = metrics
        self._update_state()

    def update_no_pose(self):
        self.last_reliable = False
        self.last_points = 0
        self.active_signals = []
        self.last_metrics = {}
        if self.state != "FLAG":
            self.state = "NO_POSE"

    def rolling_sum(self):
        return int(sum(item["points"] for item in self.window))

    def rolling_count(self, signal_name):
        return int(sum(1 for item in self.window if item["signals"].get(signal_name, False)))

    def _update_state(self):
        cur_points = self.last_points
        strong_now = any(name in STRONG_SIGNALS for name in self.active_signals)
        roll_sum = self.rolling_sum()
        reach_count = self.rolling_count("REACH")
        bound_count = self.rolling_count("BOUND")

        warn = cur_points >= WARN_POINTS or strong_now
        flag = (
            roll_sum >= FLAG_SUM
            or reach_count >= FLAG_K
            or bound_count >= FLAG_K
        )

        if self.state == "FLAG":
            recent_strong = reach_count > 0 or bound_count > 0
            if roll_sum < CLEAR_SUM and not recent_strong:
                self.state = "OK"
            return

        if flag:
            self.state = "FLAG"
        elif warn:
            self.state = "WARN"
        else:
            self.state = "OK"


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
    return parser.parse_args()


def normalize_angle_diff(current, baseline):
    if baseline is None:
        return 0.0
    diff = current - baseline
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def compute_signals(landmarks, roi, neighbor_roi, baseline_angle):
    lm = mp.solutions.pose.PoseLandmark
    nose = landmarks[lm.NOSE]
    l_sh = landmarks[lm.LEFT_SHOULDER]
    r_sh = landmarks[lm.RIGHT_SHOULDER]
    l_wr = landmarks[lm.LEFT_WRIST]
    r_wr = landmarks[lm.RIGHT_WRIST]

    shoulders_reliable = l_sh.visibility >= VIS_THRESH and r_sh.visibility >= VIS_THRESH
    nose_reliable = nose.visibility >= VIS_THRESH
    wrist_reliable = l_wr.visibility >= VIS_THRESH or r_wr.visibility >= VIS_THRESH
    reliable_pose = shoulders_reliable and nose_reliable

    signals = {name: False for name in SIGNAL_NAMES}
    metrics = {
        "head_offset": None,
        "shoulder_angle_deg": None,
        "shoulder_angle_delta": None,
        "reliable_pose": reliable_pose,
    }

    roi_w = float(roi["w"])
    roi_h = float(roi["h"])

    if reliable_pose:
        shoulder_mid_x = (l_sh.x + r_sh.x) / 2.0
        shoulder_mid_y = (l_sh.y + r_sh.y) / 2.0
        shoulder_width = abs(l_sh.x - r_sh.x)

        if shoulder_width > 1e-4:
            head_offset = (nose.x - shoulder_mid_x) / shoulder_width
            metrics["head_offset"] = float(head_offset)
            signals["TURN"] = abs(head_offset) > 0.35

        dx = r_sh.x - l_sh.x
        dy = r_sh.y - l_sh.y
        shoulder_angle_deg = math.degrees(math.atan2(dy, dx))
        angle_delta = normalize_angle_diff(shoulder_angle_deg, baseline_angle)
        metrics["shoulder_angle_deg"] = float(shoulder_angle_deg)
        metrics["shoulder_angle_delta"] = float(angle_delta)
        signals["ROT"] = baseline_angle is not None and abs(angle_delta) > 15.0

        margin_x = 0.1 * roi_w
        margin_y = 0.1 * roi_h
        nose_px = (nose.x * roi_w, nose.y * roi_h)
        shoulder_mid_px = (shoulder_mid_x * roi_w, shoulder_mid_y * roi_h)

        def outside_safe(pt):
            return (
                pt[0] < margin_x
                or pt[0] > roi_w - margin_x
                or pt[1] < margin_y
                or pt[1] > roi_h - margin_y
            )

        signals["BOUND"] = outside_safe(nose_px) or outside_safe(shoulder_mid_px)

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


def draw_overlay(frame, rois, states):
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
        color = colors.get(st.state, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        signals_txt = " ".join(st.active_signals) if st.active_signals else "-"
        label = f"{sid} {st.state} [{signals_txt}] sum={st.rolling_sum()} pts={st.last_points}"
        cv2.putText(frame, label, (x + 4, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def print_periodic_summary(rois, states):
    parts = []
    for roi in rois:
        sid = roi["id"]
        st = states[sid]
        sig = ",".join(st.active_signals) if st.active_signals else "-"
        parts.append(f"{sid}:{st.state} sum={st.rolling_sum()} pts={st.last_points} signals={sig}")
    print(" | ".join(parts), flush=True)


def display_available():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def validate_live_rois(rois):
    if len(rois) != 2:
        raise ValueError(f"Expected exactly 2 ROIs for live mode, got {len(rois)}")


def main():
    args = parse_args()
    _, rois = load_rois(args.rois)
    validate_live_rois(rois)

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
    roi_index = 0
    last_print_ts = time.time()

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("[WARN] Camera frame read failed.", flush=True)
                continue

            roi = rois[roi_index]
            sid = roi["id"]
            student = states[sid]
            neighbor_roi = rois[1 - roi_index]
            roi_index = 1 - roi_index

            roi_crop = crop(frame, roi)
            if roi_crop.size > 0:
                rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    signals, metrics, reliable_pose = compute_signals(
                        result.pose_landmarks.landmark,
                        roi,
                        neighbor_roi,
                        student.baseline_angle,
                    )
                    if reliable_pose:
                        student.add_baseline(metrics.get("shoulder_angle_deg"), signals)
                        student.update_with_signals(signals, metrics)
                    else:
                        student.update_no_pose()
                else:
                    student.update_no_pose()
            else:
                student.update_no_pose()

            if not headless:
                out = frame.copy()
                draw_overlay(out, rois, states)
                cv2.imshow("Smart Proctor Live", out)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            now = time.time()
            if now - last_print_ts >= 1.0:
                print_periodic_summary(rois, states)
                last_print_ts = now

    finally:
        camera.release()
        pose.close()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
