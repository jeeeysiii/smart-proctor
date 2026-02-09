import json
import os


def load_rois(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ROI file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rois = data.get("rois")
    if not isinstance(rois, list) or len(rois) == 0:
        raise ValueError("ROI file missing 'rois' list")
    return data.get("frame_size"), rois


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
