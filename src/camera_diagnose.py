import glob
import os
import subprocess
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

if cv2 is not None:
    from .camera_source import OpenCVCameraSource, Picamera2Source, _candidate_devices
else:
    OpenCVCameraSource = None
    Picamera2Source = None
    _candidate_devices = None


def run_cmd(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as exc:
        return f"<failed: {exc}>"


def display_available():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def maybe_preview(frame, title):
    if cv2 is None:
        return
    if not display_available():
        print("[INFO] No DISPLAY/WAYLAND detected; skipping preview window.", flush=True)
        return
    cv2.imshow(title, frame)
    print("[INFO] Preview shown for 2 seconds.", flush=True)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def save_frame(frame):
    if cv2 is None:
        print("[WARN] Cannot save frame because cv2 is not installed.", flush=True)
        return
    out_path = Path("data") / "camera_test.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), frame)
    if ok:
        print(f"[INFO] Saved test frame: {out_path}", flush=True)
    else:
        print(f"[WARN] Failed to save test frame: {out_path}", flush=True)


def try_opencv_devices(width=640, height=360, fps=20):
    if cv2 is None:
        print("\n[DIAG] OpenCV unavailable: No module named 'cv2'", flush=True)
        return False

    print("\n[DIAG] Testing OpenCV camera devices", flush=True)
    candidates = _candidate_devices(device=None, opencv_index=None)
    print(f"[DIAG] OpenCV candidates: {candidates}", flush=True)

    for dev in candidates:
        source = OpenCVCameraSource(device=dev, width=width, height=height, fps=fps)
        if not source.open():
            print(f"  - device {dev}: FAIL ({source.last_error})", flush=True)
            continue

        ok, frame = source.read()
        if not ok or frame is None or frame.size == 0:
            print(f"  - device {dev}: FAIL (opened but read failed)", flush=True)
            source.release()
            continue

        print(f"  - device {dev}: OK (shape={frame.shape})", flush=True)
        maybe_preview(frame, f"OpenCV {dev}")
        save_frame(frame)
        source.release()
        return True

    return False


def try_picamera2(width=640, height=360, fps=20):
    print("\n[DIAG] Testing Picamera2", flush=True)
    if Picamera2Source is None:
        print("  - Picamera2: SKIP (cv2 missing in this Python environment)", flush=True)
        return False

    source = Picamera2Source(width=width, height=height, fps=fps)
    if not source.open():
        print(f"  - Picamera2: FAIL ({source.last_error})", flush=True)
        if source.last_error and "No module named 'picamera2'" in source.last_error:
            print("  - Hint: install with `sudo apt install -y python3-picamera2`", flush=True)
            print("  - Hint: run with `/usr/bin/python3 -m src.camera_diagnose`", flush=True)
            print("  - Hint: if using venv, recreate with `--system-site-packages`", flush=True)
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
    print(f"python executable: {sys.executable}", flush=True)
    print(f"python version: {sys.version.split()[0]}", flush=True)
    system_check = run_cmd(['python3', '-c', 'import picamera2; print(\"ok\")'])
    print(f"system python picamera2 check: {system_check}", flush=True)
    video_nodes = sorted(glob.glob('/dev/video*'))
    print(f"/dev/video* nodes: {video_nodes if video_nodes else '<none>'}", flush=True)
    print(f"groups: {run_cmd(['groups'])}", flush=True)

    picam_ok = try_picamera2()
    opencv_ok = try_opencv_devices() if not picam_ok else False

    print("\n[DIAG] Summary", flush=True)
    print(f"Picamera2: {'PASS' if picam_ok else 'FAIL'}", flush=True)
    print(f"OpenCV: {'PASS' if opencv_ok else 'FAIL'}", flush=True)

    if not opencv_ok and not picam_ok:
        print(
            "[DIAG] No backend worked. Check camera enablement (raspi-config), CSI cable orientation, reboot, and Python environment mismatch.",
            flush=True,
        )


if __name__ == "__main__":
    main()
