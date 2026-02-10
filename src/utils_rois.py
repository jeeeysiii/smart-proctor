import json
import os


EXPECTED_ROI_KEYS = {"id", "x", "y", "w", "h"}


def load_rois(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ROI file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame_size = data.get("frame_size")
    if frame_size is not None:
        if not isinstance(frame_size, list) or len(frame_size) != 2:
            raise ValueError("ROI file field 'frame_size' must be [width, height]")

    rois = data.get("rois")
    if not isinstance(rois, list) or len(rois) == 0:
        raise ValueError("ROI file missing non-empty 'rois' list")

    for idx, roi in enumerate(rois):
        if not isinstance(roi, dict):
            raise ValueError(f"ROI index {idx} must be an object")
        missing = EXPECTED_ROI_KEYS - set(roi.keys())
        if missing:
            raise ValueError(f"ROI index {idx} missing keys: {sorted(missing)}")

    return frame_size, rois


def save_rois(path, frame_w, frame_h, rois):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "frame_size": [int(frame_w), int(frame_h)],
        "rois": rois,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def crop(frame_bgr, roi):
    x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
    h_frame, w_frame = frame_bgr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_frame, x + w)
    y1 = min(h_frame, y + h)
    return frame_bgr[y0:y1, x0:x1].copy()
