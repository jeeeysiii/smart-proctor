import glob
import os
import subprocess
from pathlib import Path

import cv2

from .camera_source import OpenCVCameraSource, Picamera2Source


def run_cmd(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as exc:
        return f"<failed: {exc}>"


def display_available():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def maybe_preview(frame, title):
    if not display_available():
        print("[INFO] No DISPLAY/WAYLAND detected; skipping preview window.", flush=True)
        return
    cv2.imshow(title, frame)
    print("[INFO] Preview shown for 2 seconds.", flush=True)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def save_frame(frame):
    out_path = Path("data") / "camera_test.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), frame)
    if ok:
        print(f"[INFO] Saved test frame: {out_path}", flush=True)
    else:
        print(f"[WARN] Failed to save test frame: {out_path}", flush=True)


def try_opencv_indices(width=640, height=360, fps=20):
    print("\n[DIAG] Testing OpenCV camera indices 0..4", flush=True)
    for idx in range(5):
        source = OpenCVCameraSource(device=idx, width=width, height=height, fps=fps)
        if not source.open():
            print(f"  - index {idx}: FAIL ({source.last_error})", flush=True)
            continue

        ok, frame = source.read()
        if not ok or frame is None or frame.size == 0:
            print(f"  - index {idx}: FAIL (opened but read failed)", flush=True)
            source.release()
            continue

        print(f"  - index {idx}: OK (shape={frame.shape})", flush=True)
        maybe_preview(frame, f"OpenCV index {idx}")
        save_frame(frame)
        source.release()
        return True

    return False


def try_picamera2(width=640, height=360, fps=20):
    print("\n[DIAG] Testing Picamera2", flush=True)
    source = Picamera2Source(width=width, height=height, fps=fps)
    if not source.open():
        print(f"  - Picamera2: FAIL ({source.last_error})", flush=True)
        return False

    ok, frame = source.read()
    if not ok or frame is None or frame.size == 0:
        print(f"  - Picamera2: FAIL (read failed: {source.last_error})", flush=True)
        source.release()
        return False

    print(f"  - Picamera2: OK (shape={frame.shape})", flush=True)
    maybe_preview(frame, "Picamera2")
    save_frame(frame)
    source.release()
    return True


def main():
    print("[DIAG] System info", flush=True)
    print(f"uname -a: {run_cmd(['uname', '-a'])}", flush=True)
    video_nodes = sorted(glob.glob('/dev/video*'))
    print(f"/dev/video* nodes: {video_nodes if video_nodes else '<none>'}", flush=True)
    print(f"groups: {run_cmd(['groups'])}", flush=True)

    opencv_ok = try_opencv_indices()
    picam_ok = try_picamera2()

    print("\n[DIAG] Summary", flush=True)
    print(f"OpenCV indices 0..4: {'PASS' if opencv_ok else 'FAIL'}", flush=True)
    print(f"Picamera2: {'PASS' if picam_ok else 'FAIL'}", flush=True)

    if not opencv_ok and not picam_ok:
        print(
            "[DIAG] No backend worked. Check camera enablement (raspi-config), CSI cable orientation, reboot, and video-group membership.",
            flush=True,
        )


if __name__ == "__main__":
    main()
