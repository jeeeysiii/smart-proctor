import argparse
import json
import math
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from .utils_rois import crop, load_rois


"""
Porting to Raspberry Pi notes:
- MediaPipe Pose can be heavy; use model_complexity=0 and lower input resolution to keep FPS stable.
- Consider processing only on motion or reducing the per-student update rate beyond 1/3 frames.
- MP4 codec support on Pi can vary; prefer H.264 Baseline or re-encode inputs if VideoCapture fails.
- GUI display (cv2.imshow/selectROIs) may not be available in headless setups; use display flag sparingly.
"""


SIGNAL_NAMES = ["TURN", "LEAN", "BOUND", "REACH"]


class StudentState:
    def __init__(self, student_id):
        self.student_id = student_id
        self.state = "OK"
        self.last_update_frame = None
        self.warning_timer = 0.0
        self.boundary_timer = 0.0
        self.flag_timer = 0.0
        self.evidence_saved = False
        self.history = deque()
        self.baseline_angles = []
        self.baseline_angle = None
        self.last_metrics = {}
        self.active_signals = []

    def update_baseline(self, angle_deg, stable):
        if self.baseline_angle is not None:
            return
        if stable:
            self.baseline_angles.append(angle_deg)
            if len(self.baseline_angles) >= 15:
                self.baseline_angle = float(np.median(self.baseline_angles))


def parse_args():
    parser = argparse.ArgumentParser(description="Run MediaPipe Pose Layer-2 signals on MP4.")
    parser.add_argument("--video", required=True, help="Path to MP4 video")
    parser.add_argument("--rois", required=True, help="Path to ROI JSON")
    parser.add_argument("--outdir", required=True, help="Directory for evidence output")
    parser.add_argument("--display", action="store_true", help="Display annotated video")
    return parser.parse_args()


def compute_signals(landmarks, roi_w, roi_h, baseline_angle):
    lm = mp.solutions.pose.PoseLandmark
    nose = landmarks[lm.NOSE]
    l_sh = landmarks[lm.LEFT_SHOULDER]
    r_sh = landmarks[lm.RIGHT_SHOULDER]
    l_wr = landmarks[lm.LEFT_WRIST]
    r_wr = landmarks[lm.RIGHT_WRIST]

    shoulder_mid_x = (l_sh.x + r_sh.x) / 2.0
    shoulder_mid_y = (l_sh.y + r_sh.y) / 2.0
    shoulder_width = abs(l_sh.x - r_sh.x)
    head_offset = 0.0
    if shoulder_width > 1e-4:
        head_offset = (nose.x - shoulder_mid_x) / shoulder_width

    dx = r_sh.x - l_sh.x
    dy = r_sh.y - l_sh.y
    angle_deg = math.degrees(math.atan2(dy, dx))
    angle_delta = 0.0
    if baseline_angle is not None:
        angle_delta = angle_deg - baseline_angle

    margin_x = 0.1 * roi_w
    margin_y = 0.1 * roi_h

    nose_px = (nose.x * roi_w, nose.y * roi_h)
    shoulder_mid_px = (shoulder_mid_x * roi_w, shoulder_mid_y * roi_h)

    def outside_inner_box(pt):
        return pt[0] < margin_x or pt[0] > roi_w - margin_x or pt[1] < margin_y or pt[1] > roi_h - margin_y

    boundary = outside_inner_box(nose_px) or outside_inner_box(shoulder_mid_px)

    left_wrist_px = (l_wr.x * roi_w, l_wr.y * roi_h)
    right_wrist_px = (r_wr.x * roi_w, r_wr.y * roi_h)
    reach = left_wrist_px[0] < margin_x or right_wrist_px[0] < margin_x or left_wrist_px[0] > roi_w - margin_x or right_wrist_px[0] > roi_w - margin_x

    turn = abs(head_offset) > 0.35
    lean = baseline_angle is not None and abs(angle_delta) > 15.0

    metrics = {
        "head_offset": head_offset,
        "shoulder_angle_deg": angle_deg,
        "shoulder_angle_delta": angle_delta,
    }
    signals = {
        "TURN": turn,
        "LEAN": lean,
        "BOUND": boundary,
        "REACH": reach,
    }
    return signals, metrics


def update_state(student, signals, metrics, now_s, dt_s):
    any_signal = any(signals.values())
    student.warning_timer = student.warning_timer + dt_s if any_signal else 0.0

    boundary_or_reach = signals["BOUND"] or signals["REACH"]
    student.boundary_timer = student.boundary_timer + dt_s if boundary_or_reach else 0.0

    student.history.append((now_s, signals))
    while student.history and now_s - student.history[0][0] > 2.0:
        student.history.popleft()

    signals_in_window = set()
    for _, sigs in student.history:
        for name, active in sigs.items():
            if active:
                signals_in_window.add(name)

    warn_condition = student.warning_timer >= 0.7
    flag_condition = len(signals_in_window) >= 2 or student.boundary_timer >= 1.5

    if student.state == "OK":
        if warn_condition:
            student.state = "WARN"
    elif student.state == "WARN":
        if flag_condition:
            student.state = "FLAG"
            student.flag_timer = 0.0
            student.evidence_saved = False
        elif not warn_condition:
            student.state = "OK"
    elif student.state == "FLAG":
        if not flag_condition and not warn_condition:
            student.state = "OK"
            student.flag_timer = 0.0
            student.evidence_saved = False

    if student.state == "FLAG":
        student.flag_timer += dt_s
    else:
        student.flag_timer = 0.0

    student.last_metrics = metrics
    student.active_signals = [name for name, active in signals.items() if active]


def save_evidence(outdir, frame_bgr, frame_idx, student_id, state, signals, metrics):
    os.makedirs(outdir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"{timestamp}_S{student_id}_FLAG"
    img_path = os.path.join(outdir, f"{base}.jpg")
    json_path = os.path.join(outdir, f"{base}.json")

    cv2.imwrite(img_path, frame_bgr)
    payload = {
        "timestamp": timestamp,
        "frame_idx": frame_idx,
        "student_id": student_id,
        "state": state,
        "active_signals": signals,
        "metrics": metrics,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    for path in ["configs", "data/events", "data/videos"]:
        os.makedirs(path, exist_ok=True)

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    frame_size, rois = load_rois(args.rois)
    if len(rois) != 3:
        raise ValueError(f"Expected 3 ROIs, found {len(rois)}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    dt_per_frame = 1.0 / fps

    states = {roi["id"]: StudentState(roi["id"]) for roi in rois}

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        process_roi = rois[frame_idx % len(rois)]
        student = states[process_roi["id"]]
        now_s = frame_idx / fps
        if student.last_update_frame is None:
            dt_s = dt_per_frame
        else:
            dt_s = (frame_idx - student.last_update_frame) / fps
        student.last_update_frame = frame_idx

        roi_crop = crop(frame, process_roi)
        roi_h, roi_w = roi_crop.shape[:2]

        signals = {name: False for name in SIGNAL_NAMES}
        metrics = {"head_offset": None, "shoulder_angle_deg": None, "shoulder_angle_delta": None}

        if roi_crop.size > 0:
            roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
            result = pose.process(roi_rgb)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                signals, metrics = compute_signals(landmarks, roi_w, roi_h, student.baseline_angle)

                stable_for_baseline = not any(signals.values())
                if metrics["shoulder_angle_deg"] is not None:
                    student.update_baseline(metrics["shoulder_angle_deg"], stable_for_baseline)

        update_state(student, signals, metrics, now_s, dt_s)

        if student.state == "FLAG" and student.flag_timer >= 2.0 and not student.evidence_saved:
            annotated = frame.copy()
            overlay_annotations(annotated, rois, states)
            save_evidence(args.outdir, annotated, frame_idx, student.student_id, student.state, student.active_signals, student.last_metrics)
            student.evidence_saved = True

        overlay_annotations(frame, rois, states)

        if args.display:
            cv2.imshow("Pose Layer2", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    cap.release()
    if args.display:
        cv2.destroyAllWindows()


def overlay_annotations(frame, rois, states):
    for roi in rois:
        x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
        student = states[roi["id"]]
        color = (0, 255, 0) if student.state == "OK" else (0, 255, 255) if student.state == "WARN" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"S{roi['id']} {student.state}"
        signals_text = ",".join(student.active_signals) if student.active_signals else ""
        cv2.putText(frame, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if signals_text:
            cv2.putText(frame, signals_text, (x + 5, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


if __name__ == "__main__":
    main()
