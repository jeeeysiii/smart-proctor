import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


EVENT_NOT_IN = "NOT_IN_EVENT"
EVENT_IN = "IN_EVENT"
EVENT_COOLDOWN = "COOLDOWN"


class RingBuffer:
    def __init__(self, seconds: float, fps: float, max_frames: int | None = None):
        derived_max = int(max(1, seconds * max(1.0, fps) * 1.5))
        self.max_frames = max_frames or derived_max
        self.seconds = float(seconds)
        self._frames = deque(maxlen=self.max_frames)

    def append(self, ts_unix: float, frame_bgr):
        self._frames.append((float(ts_unix), frame_bgr.copy()))
        self._trim_by_time(ts_unix)

    def _trim_by_time(self, now_ts: float):
        cutoff = now_ts - self.seconds
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def frames_since(self, start_ts: float):
        return [(ts, frm.copy()) for ts, frm in self._frames if ts >= start_ts]


class EvidenceWorker(threading.Thread):
    def __init__(self, run_dir: Path, run_jsonl_path: Path, events_path: Path, clip_fps: int):
        super().__init__(daemon=True)
        self.run_dir = run_dir
        self.run_jsonl_path = run_jsonl_path
        self.events_path = events_path
        self.clip_fps = int(clip_fps)
        self.tasks: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=512)
        self._events: list[dict[str, Any]] = []

    def submit(self, task: dict[str, Any]):
        try:
            self.tasks.put_nowait(task)
        except queue.Full:
            # Non-blocking by design: drop task if writer is saturated.
            pass

    def run(self):
        while True:
            task = self.tasks.get()
            if task is None:
                self._flush_events()
                return
            ttype = task.get("type")
            if ttype == "jsonl":
                self._append_jsonl(task["payload"])
            elif ttype == "event":
                event_payload = task["payload"]
                clip_path = self._write_clip(event_payload)
                snap_path = self._write_snapshot(event_payload)
                event_payload["clip_path"] = str(clip_path) if clip_path else None
                event_payload["snapshot_path"] = str(snap_path) if snap_path else None
                event_payload.pop("frames", None)
                event_payload.pop("snapshot", None)
                self._events.append(event_payload)
                self._flush_events()
            elif ttype == "mark":
                self._events.append(task["payload"])
                self._flush_events()
            elif ttype == "flush":
                self._flush_events()

    def shutdown(self):
        self.tasks.put(None)
        self.join(timeout=10)

    def _append_jsonl(self, payload: dict[str, Any]):
        with self.run_jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_clip(self, event_payload: dict[str, Any]):
        frames = event_payload.get("frames") or []
        if not frames:
            return None
        clips_dir = self.run_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_path = clips_dir / f"{event_payload['event_id']}.mp4"

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, float(self.clip_fps), (w, h))

        if not writer.isOpened():
            clip_path = clip_path.with_suffix(".avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(clip_path), fourcc, float(self.clip_fps), (w, h))

        if not writer.isOpened():
            return None

        for frame in frames:
            writer.write(frame)
        writer.release()
        return clip_path

    def _write_snapshot(self, event_payload: dict[str, Any]):
        frame = event_payload.get("snapshot")
        if frame is None:
            return None
        snaps_dir = self.run_dir / "snapshots"
        snaps_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snaps_dir / f"{event_payload['event_id']}.jpg"
        cv2.imwrite(str(snap_path), frame)
        return snap_path

    def _flush_events(self):
        with self.events_path.open("w", encoding="utf-8") as fh:
            json.dump(self._events, fh, indent=2, ensure_ascii=False)


@dataclass
class EventRuntime:
    event_id: str
    roi_id: str
    start_ts: float
    trigger_ts: float
    trigger_reason: str
    post_until_ts: float
    summary: dict[str, list[float]] = field(default_factory=dict)
    frames: list[Any] = field(default_factory=list)
    snapshot: Any = None
    last_frame_ts: float | None = None
    clear_stable_count: int = 0
    close_ts: float | None = None


class EvidenceManager:
    def __init__(
        self,
        run_id: str,
        log_dir: str,
        rois: list[dict[str, Any]],
        run_config_path: str,
        git_commit: str,
        pre_seconds: float = 5.0,
        post_seconds: float = 5.0,
        clip_fps: int = 12,
        reliable_window: int = 7,
        reliable_required: int = 5,
        clear_stable_frames: int = 8,
        cooldown_seconds: float = 8.0,
    ):
        self.run_id = run_id
        self.pre_seconds = float(pre_seconds)
        self.post_seconds = float(post_seconds)
        self.clip_fps = int(clip_fps)
        self.clear_stable_frames = int(clear_stable_frames)
        self.cooldown_seconds = float(cooldown_seconds)
        self.reliable_required = int(reliable_required)
        self.strong_signals = {"BOUND", "REACH"}

        self.run_dir = Path(log_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_jsonl_path = self.run_dir / f"run_{run_id}.jsonl"
        self.events_path = self.run_dir / f"events_{run_id}.json"
        self.config_path = run_config_path
        self.git_commit = git_commit

        self.ring = RingBuffer(seconds=max(pre_seconds + 1.0, 6.0), fps=max(clip_fps, 12))
        self.worker = EvidenceWorker(self.run_dir, self.run_jsonl_path, self.events_path, clip_fps=self.clip_fps)
        self.worker.start()

        self.roi_event_state = {roi["id"]: EVENT_NOT_IN for roi in rois}
        self.roi_prev_state = {roi["id"]: "NO_POSE" for roi in rois}
        self.roi_cooldown_until = {roi["id"]: 0.0 for roi in rois}
        self.roi_reliable = {roi["id"]: deque(maxlen=reliable_window) for roi in rois}
        self.active_events: dict[str, EventRuntime] = {}
        self._event_seq = 0

    @staticmethod
    def _iso(ts_unix: float):
        return datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()

    def shutdown(self):
        now_ts = time.time()
        for runtime in list(self.active_events.values()):
            if runtime.close_ts is None:
                runtime.close_ts = now_ts
            self._finish_event(runtime)

        self.worker.submit({"type": "flush"})
        self.worker.shutdown()

    def add_frame(self, ts_unix: float, frame_bgr):
        self.ring.append(ts_unix, frame_bgr)
        min_interval = 1.0 / max(1, self.clip_fps)

        for runtime in list(self.active_events.values()):
            if runtime.last_frame_ts is None or (ts_unix - runtime.last_frame_ts) >= min_interval:
                runtime.frames.append(frame_bgr.copy())
                runtime.last_frame_ts = ts_unix

            if ts_unix >= runtime.post_until_ts:
                if runtime.close_ts is None:
                    runtime.close_ts = ts_unix
                self._finish_event(runtime)

    def log_roi_update(self, payload: dict[str, Any]):
        self.worker.submit({"type": "jsonl", "payload": payload})

    def mark(self, ts_unix: float):
        payload = {
            "type": "MARK",
            "ts_wall": self._iso(ts_unix),
            "ts_unix": ts_unix,
            "event_id": f"mark_{self.run_id}_{int(ts_unix * 1000)}",
        }
        self.worker.submit({"type": "jsonl", "payload": payload})
        self.worker.submit({"type": "mark", "payload": payload.copy()})

    def handle_roi_state(
        self,
        ts_unix: float,
        frame_idx: int,
        roi_id: str,
        state: str,
        active_signals: list[str],
        rolling_sum: int,
        metrics: dict[str, Any],
        baseline_ready: bool,
    ):
        rel_hist = self.roi_reliable[roi_id]
        rel_hist.append(bool(metrics.get("reliable_pose", False)))
        quality_ready = sum(rel_hist) >= self.reliable_required and len(rel_hist) >= self.reliable_required
        eligible = baseline_ready and quality_ready

        event_state = self.roi_event_state[roi_id]
        prev = self.roi_prev_state[roi_id]
        strong_now = any(sig in self.strong_signals for sig in active_signals)

        if event_state == EVENT_COOLDOWN and ts_unix >= self.roi_cooldown_until[roi_id]:
            self.roi_event_state[roi_id] = EVENT_NOT_IN
            event_state = EVENT_NOT_IN

        if event_state == EVENT_NOT_IN:
            if prev != "FLAG" and state == "FLAG" and eligible and ts_unix >= self.roi_cooldown_until[roi_id]:
                self._start_event(ts_unix, frame_idx, roi_id, active_signals)
                self.roi_event_state[roi_id] = EVENT_IN

        elif event_state == EVENT_IN:
            runtime = self.active_events.get(roi_id)
            if runtime is not None:
                self._update_summary(runtime, metrics)
                if state in {"OK", "WARN"} and not strong_now:
                    runtime.clear_stable_count += 1
                else:
                    runtime.clear_stable_count = 0

                if runtime.clear_stable_count >= self.clear_stable_frames and runtime.close_ts is None:
                    runtime.close_ts = ts_unix
                    self.roi_event_state[roi_id] = EVENT_COOLDOWN
                    self.roi_cooldown_until[roi_id] = ts_unix + self.cooldown_seconds

        self.roi_prev_state[roi_id] = state

    def _start_event(self, ts_unix: float, frame_idx: int, roi_id: str, active_signals: list[str]):
        self._event_seq += 1
        event_id = f"evt_{self.run_id}_{self._event_seq:04d}_{roi_id}"
        start_ts = ts_unix - self.pre_seconds

        runtime = EventRuntime(
            event_id=event_id,
            roi_id=roi_id,
            start_ts=start_ts,
            trigger_ts=ts_unix,
            trigger_reason=",".join(active_signals) if active_signals else "FLAG_RISE",
            post_until_ts=ts_unix + self.post_seconds,
        )
        runtime.frames = [frm for _, frm in self.ring.frames_since(start_ts)]
        runtime.snapshot = runtime.frames[-1].copy() if runtime.frames else None
        runtime.last_frame_ts = ts_unix
        self.active_events[roi_id] = runtime

        self.worker.submit(
            {
                "type": "jsonl",
                "payload": {
                    "type": "EVENT_START",
                    "event_id": event_id,
                    "roi_id": roi_id,
                    "frame_idx": frame_idx,
                    "ts_wall": self._iso(ts_unix),
                    "ts_unix": ts_unix,
                },
            }
        )

    def _update_summary(self, runtime: EventRuntime, metrics: dict[str, Any]):
        keys = [
            "head_offset_delta",
            "asym_delta",
            "lean_x",
            "head_drop_delta",
            "head_offset",
            "shoulder_angle_delta",
        ]
        for key in keys:
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                runtime.summary.setdefault(key, []).append(float(val))

    def _finish_event(self, runtime: EventRuntime):
        summary_stats = {}
        for key, values in runtime.summary.items():
            if not values:
                continue
            summary_stats[key] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

        payload = {
            "type": "EVENT",
            "event_id": runtime.event_id,
            "roi_id": runtime.roi_id,
            "start_ts": self._iso(runtime.start_ts),
            "end_ts": self._iso(runtime.close_ts or runtime.post_until_ts),
            "trigger_ts": self._iso(runtime.trigger_ts),
            "trigger_reason": runtime.trigger_reason,
            "frames": runtime.frames,
            "snapshot": runtime.snapshot,
            "config_path": self.config_path,
            "git_commit": self.git_commit,
            "summary_stats": summary_stats,
        }
        self.worker.submit({"type": "event", "payload": payload})
        self.worker.submit(
            {
                "type": "jsonl",
                "payload": {
                    "type": "EVENT_END",
                    "event_id": runtime.event_id,
                    "roi_id": runtime.roi_id,
                    "ts_wall": self._iso(runtime.close_ts or runtime.post_until_ts),
                    "ts_unix": runtime.close_ts or runtime.post_until_ts,
                },
            }
        )

        self.active_events.pop(runtime.roi_id, None)
        if self.roi_event_state.get(runtime.roi_id) != EVENT_COOLDOWN:
            end_ts = runtime.close_ts or runtime.post_until_ts
            self.roi_event_state[runtime.roi_id] = EVENT_COOLDOWN
            self.roi_cooldown_until[runtime.roi_id] = end_ts + self.cooldown_seconds
