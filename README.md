# smart-proctor

Smart proctor MVP for an in-classroom cheating detection thesis project. This repository currently contains only project scaffolding; application code will be added later.

## Development setup (Windows 11 laptop)

1) Create and activate a virtual environment (do **not** commit the venv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install development dependencies:

```powershell
pip install -r requirements/requirements-dev.txt
```

## Raspberry Pi test setup (Pi 4, Raspberry Pi OS 64-bit / Debian 12)

1) Clone (or pull) the repo on the Pi.
2) Create and activate a virtual environment. If you need system packages (e.g., OpenCV from apt), use `--system-site-packages`:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

3) Install system packages via apt:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-picamera2
```

4) Install Pi Python dependencies:

```bash
pip install -r requirements/requirements-pi.txt
```

## Notes

- Application code will be added later.
- MediaPipe on the Pi is optional and should be enabled only after verifying compatibility/performance.
