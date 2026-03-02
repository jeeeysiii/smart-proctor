# smart-proctor

Smart Proctor is a Raspberry Pi system for classroom behavior monitoring.

This repository currently has **one main runtime pipeline**:
- `src/live_proctor_stream.py` (recommended): live detection + optional MJPEG streaming + remote start/stop polling.

It also includes helper and legacy scripts used during development/testing (`src/live_proctor.py`, `src/video_stream_mjpeg.py`, `src/camera_diagnose.py`, `src/pose_layer2_mp4.py`, `src/roi_pick_mp4.py`).

---

## 1) Current architecture (what actually runs now)

### Main flow (`python -m src.live_proctor_stream`)

1. **Camera input layer**
   - Uses `src/camera_source.py`.
   - Tries Picamera2 first on Raspberry Pi (when camera is not explicitly forced), then OpenCV devices, then Picamera2 fallback.

2. **Frame hub layer**
   - `FrameHub` captures frames continuously in a background thread.
   - Keeps latest frame in memory so both detection and stream can reuse the same source.

3. **Remote control layer**
   - `ControlPoller` checks `https://smartproctoring.online/get_status.php?device_id=2026` every 2 seconds.
   - `start=1` -> detection ON.
   - `start=0` -> detection OFF.
   - Poll failures do not stop pipeline; it keeps last known state.

4. **Detection layer (MediaPipe Pose + signal logic)**
   - Reuses logic from `src/live_proctor.py`.
   - Loads ROIs from JSON (`configs/rois_live.json` by default).
   - Alternates one enabled ROI per frame (round-robin).
   - For each ROI:
     - crop ROI,
     - run MediaPipe Pose,
     - compute signals,
     - update student state (OK/WARN/FLAG/NO_POSE).

5. **Evidence layer**
   - `EvidenceManager` records suspicious periods as MP4 clips in `evidence/`.
   - Writes event metadata to `evidence/logs/session_<timestamp>.jsonl`.
   - Optional async upload if `SP_LOG_POST_URL` is set.

6. **Output layer**
   - Optional local preview window (`cv2.imshow`) unless `--headless`.
   - Console summary every ~1 second.
   - Optional MJPEG server (`--enable-stream`) sharing the same camera frames.

---

## 2) Core detection behavior

Signals computed per student ROI:
- `ROT`: shoulder roll above threshold.
- `BOUND`: head/shoulder center outside ROI inner safe zone.
- `REACH`: wrist near ROI border or entering neighbor ROI.
- `LEAN`: shoulder midpoint shift from baseline.
- `STAND`: vertical shoulder shift (standing-like movement).
- `EMPTY`: no reliable pose.

State logic:
- Each signal has points (`ROT=1, BOUND=2, REACH=3, LEAN=2, STAND=3, EMPTY=3`).
- Rolling window + rule triggers set:
  - `OK`
  - `WARN`
  - `FLAG`
  - `NO_POSE` (display color/state when pose is unavailable)
- Hysteresis is used so flags do not flicker.

Baseline logic:
- Per student baseline is learned from stable frames.
- Baseline adapts slowly while student stays in `OK` and no strong signals are active.

---

## 3) Repository structure

- `src/live_proctor_stream.py` -> **main production script** (detection + stream + remote start/stop).
- `src/live_proctor.py` -> core detection classes/functions (student state, evidence manager, uploader, overlay).
- `src/camera_source.py` -> camera backend selection and adapters (OpenCV/Picamera2).
- `src/video_stream_mjpeg.py` -> Flask MJPEG app/broadcaster (also reusable by main script).
- `src/utils_rois.py` -> ROI load/save/crop helpers.
- `src/camera_diagnose.py` -> diagnostic utility for camera backend troubleshooting.
- `src/pose_layer2_mp4.py` -> offline MP4 demo/prototype pipeline (3 ROIs).
- `src/roi_pick_mp4.py` -> GUI helper to pick 3 ROIs from MP4 first frame.
- `configs/` -> ROI config files.
- `requirements/` -> dependency sets for Pi vs dev.
- `evidence/` (generated) -> event clips and logs.

---

## 4) Setup (Raspberry Pi, Bookworm, Python 3.11)

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
sudo apt update
sudo apt install -y python3-opencv python3-picamera2
pip install -r requirements/requirements-pi.txt
```

For local development (non-Pi), use:

```bash
pip install -r requirements/requirements-dev.txt
```

---

## 5) Run commands

### A) Inference + Evidence

```bash
SP_LOG_POST_URL="https://smartproctoring.online/json_data.php" \
SP_LOG_TIMEOUT_SEC="6.0" \
python -m src.live_proctor_stream \
  --rois configs/rois_4_default.json \
  --width 640 --height 360 --fps 12 \
  --model-complexity 0 \
  --enabled-rois S1,S2,S3,S4
```

### B) Inference + Evidence + MJPEG Stream

```bash
SP_LOG_POST_URL="https://smartproctoring.online/json_data.php" \
SP_LOG_TIMEOUT_SEC="6.0" \
python -m src.live_proctor_stream \
  --rois configs/rois_4_default.json \
  --width 640 --height 360 --fps 20 \
  --model-complexity 0 \
  --enabled-rois S1,S2,S3,S4 \
  --enable-stream \
  --stream-host 0.0.0.0 \
  --stream-port 8000 \
  --stream-max-fps 15 \
  --jpeg-quality 80 \
  --flip none \
  --token sp2026
```

Endpoints:
- `GET /health`
- `GET /` (optional token)
- `GET /mjpeg` (optional token)
- `GET /clip/<filename.mp4>` (optional token)

### C) Stream-only mode (no detection)

```bash
python -m src.video_stream_mjpeg --host 0.0.0.0 --port 8000 --width 640 --height 360 --fps 20 --max-fps 15 --jpeg-quality 80 --token YOURTOKEN
```

### D) Camera diagnostics

```bash
python -m src.camera_diagnose
```

---

## 6) Main script parameters (`src.live_proctor_stream`)

### Detection / camera
- `--rois` (default: `configs/rois_live.json`)
- `--width` (default: `640`)
- `--height` (default: `360`)
- `--fps` (default: `20`)
- `--headless` (flag)
- `--model-complexity` (default: `0`, choices: `0|1|2`)
- `--save-evidence` (flag, currently reserved/no-op)
- `--camera-device` (default: `None`, explicit device override)
- `--opencv-index` (default: `None`, explicit OpenCV index)
- `--enabled-rois` (default: `all`, ex: `S2` or `S1,S2`)
- `--show-disabled` (flag)

### Streaming
- `--enable-stream` (flag)
- `--stream-host` (default: `0.0.0.0`)
- `--stream-port` (default: `8000`)
- `--stream-max-fps` (default: `15.0`)
- `--jpeg-quality` (default: `80`, range 1..100)
- `--flip` (default: `none`, choices: `none|h|v|hv`)
- `--token` (default: `None`)

### Runtime keyboard controls (preview mode only)
- `r` -> reset baselines
- `d` -> toggle debug overlay
- `Esc` -> exit

---

## 7) Environment variables

Used by uploader in `live_proctor_stream` / `live_proctor`:
- `SP_LOG_POST_URL`: if set, JSON events are posted asynchronously.
- `SP_LOG_TIMEOUT_SEC` (default `6.0`): HTTP timeout.
- `SP_LOG_MAX_PENDING_REPLAY` (default `5000`): max pending events replayed per cycle.

If upload fails, events are kept in:
- `evidence/logs/pending_uploads.jsonl`

---

## 8) ROI configuration

Example (`configs/rois_live.json`):

```json
{
  "frame_size": [640, 360],
  "rois": [
    {"id": "S1", "x": 40, "y": 40, "w": 240, "h": 280},
    {"id": "S2", "x": 360, "y": 40, "w": 240, "h": 280}
  ]
}
```

Notes:
- `frame_size` should match capture resolution.
- Keep ROI IDs stable (recommended: `S1`, `S2`, ...).
- Current live setup is optimized around 2 seats/ROIs.

---

## 9) Evidence and logs

Generated during runtime:
- `evidence/<student>_<timestamp>.mp4` (event clips)
- `evidence/logs/session_<timestamp>.jsonl` (event metadata)
- `evidence/logs/pending_uploads.jsonl` (retry queue when POST fails)

Event metadata includes:
- `event_id`, `timestamp`, `device_id`, `session_id`, `student_id`, `signals`, `clip_file`, `duration_sec`

---

## 10) Troubleshooting

- Camera not opening:
  - run `python -m src.camera_diagnose`
  - check `/dev/video*`
  - verify `python3-picamera2` is installed
- No GUI available:
  - use `--headless`
- Slow performance:
  - use `--model-complexity 0`
  - keep `640x360` and around `20 FPS`
  - reduce stream FPS if MJPEG is enabled

---

## 11) About removing scripts

You suggested removing unnecessary scripts. Good idea, but recommended approach is:
1. mark scripts as `legacy` first,
2. verify nobody uses them in deployment/docs,
3. remove in a separate cleanup PR.

For now, this README documents the current state clearly and identifies `live_proctor_stream.py` as the main entrypoint.
