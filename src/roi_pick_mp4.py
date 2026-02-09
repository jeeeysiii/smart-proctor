import argparse
import os

import cv2

from utils_rois import save_rois


def parse_args():
    parser = argparse.ArgumentParser(description="Pick 3 ROIs from the first frame of an MP4.")
    parser.add_argument("--video", required=True, help="Path to MP4 video")
    parser.add_argument("--out", required=True, help="Output JSON path for ROIs")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Unable to read first frame from video")

    frame_h, frame_w = frame.shape[:2]
    rois = cv2.selectROIs("Select 3 ROIs", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if rois is None or len(rois) != 3:
        raise ValueError(f"Expected exactly 3 ROIs, got {0 if rois is None else len(rois)}")

    roi_entries = []
    for idx, (x, y, w, h) in enumerate(rois, start=1):
        roi_entries.append({"id": idx, "x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    save_rois(args.out, frame_w, frame_h, roi_entries)
    print(f"Saved {len(roi_entries)} ROIs to {args.out}")


if __name__ == "__main__":
    main()
