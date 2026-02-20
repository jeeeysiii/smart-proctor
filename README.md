# smart-proctor

Smart proctor MVP for in-classroom behavior detection on Raspberry Pi 4.

## Raspberry Pi setup (Bookworm, Python 3.11)

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
sudo apt update
sudo apt install -y python3-opencv python3-picamera2
pip install -r requirements/requirements-pi.txt
```

## Live proctor run (camera + real-time overlay)

Primary command:

```bash
python -m src.live_proctor
```

Recommended explicit settings on Pi:

```bash
python -m src.live_proctor --rois configs/rois_live.json --width 640 --height 360 --fps 20 --model-complexity 0
```

Headless run (no GUI calls, console summary only):

```bash
python -m src.live_proctor --headless
```

If you run without `--headless` and no display is available, the app auto-falls back to headless mode with a warning.

## ROI configuration for live mode

Live mode accepts 1..N predefined ROIs (commonly 2 or 4) in `configs/rois_live.json` or another ROI file.

Template shape:

```json
{
  "frame_size": [640, 360],
  "rois": [
    {"id": "S1", "x": 40, "y": 40, "w": 240, "h": 280},
    {"id": "S2", "x": 360, "y": 40, "w": 240, "h": 280}
  ]
}
```

How to edit:
- Keep ROI IDs stable and unique (for example `S1..S4`).
- Update `x`, `y`, `w`, `h` to match seat positions for your camera view.
- `frame_size` should match the capture resolution you run with (`--width`, `--height`).

Default 4-seat template: `configs/rois_4_default.json` (640x360, four horizontal slices).

Enable/disable ROI processing without changing the ROI file:
- Process all ROIs (default):
  - `python -m src.live_proctor --rois configs/rois_4_default.json`
- Process only one ROI:
  - `python -m src.live_proctor --rois configs/rois_4_default.json --enabled-rois S2`
- Process a subset in ROI-file order:
  - `python -m src.live_proctor --rois configs/rois_4_default.json --enabled-rois S1,S3`

Notes:
- Overlay always draws all ROI boxes. Disabled ROIs are shown in gray with `DISABLED` label.
- Console summary prints enabled ROIs only.

## What live_proctor does

- Uses libcamera camera feed through:
  - OpenCV `VideoCapture(/dev/video0)` first.
  - Falls back to `picamera2` automatically.
- Cycles processing one ROI per frame across enabled IDs in ROI-file order (for 2 ROIs this remains `S1`, `S2`, `S1`, `S2`, ...).
- Uses `mp.solutions.pose` with low-complexity model for Pi.
- Computes Layer-2 signals from landmarks:
  - `TURN`: nose offset vs shoulder midpoint normalized by shoulder width.
  - `ROT`: shoulder-line angle delta from per-student baseline.
  - `BOUND`: nose or shoulder-midpoint exits inner safe zone.
  - `REACH`: wrist near ROI boundary or wrist entering neighbor ROI.
- Applies reliability gating:
  - Low visibility nose/shoulders -> no TURN/ROT/BOUND computation, state becomes `NO_POSE` (unless already FLAG with hysteresis).
  - Low visibility wrists -> no REACH computation.
- Uses rolling evidence window (frame-based, not time-based):
  - immediate WARN on current threshold/strong signal.
  - FLAG on accumulated evidence counts/sums.
  - hysteresis to clear FLAG only when evidence decays.

## Real-time outputs

- Overlay (preview window):
  - ROI color: OK=white, WARN=yellow, FLAG=red, NO_POSE=gray.
  - Label example: `S1 OK [TURN REACH] sum=7 pts=4`.
- Console:
  - Prints one summary line about every second:
  - `S1:STATE sum=... pts=... signals=... | S2:STATE sum=... pts=... signals=...`

## Pi pitfalls / troubleshooting

- Camera busy or unavailable:
  - Ensure no other process is using camera.
  - Check camera detection: `libcamera-hello -t 2000`.
- Permissions/device:
  - Validate `/dev/video0` exists when using OpenCV path.
  - If OpenCV camera path fails, install/enable `python3-picamera2` fallback.
- No monitor / remote shell:
  - Use `--headless`.
  - Or set display forwarding properly; otherwise auto-headless fallback will trigger.
- Performance tuning:
  - Keep `--model-complexity 0`.
  - Use lower resolution/FPS first (`640x360`, `20 FPS`).
