# biomech-eval

RGB-D camera capture and biomechanics evaluation tools.

## Camera quick start

Run these commands from the repository root. `uv` runs everything inside the
project's locked virtual environment; no global Python or `pip` installation is
required.

```powershell
uv sync
uv run biomech preview
```

The preview starts the Gemini 335L stream and shows color and depth. Press `q`
or Escape to stop it; Ctrl+C also performs a clean shutdown.

Record a session:

```powershell
uv run biomech record-session
uv run biomech record-session --output data/raw --frames 300
```

Each recording is written to a timestamped directory containing lossless PNG
color/depth frames, `frames.csv` for synchronization, and `metadata.json`.
Raw recordings are intentionally excluded from Git.

## MediaPipe pose prototype

Download Google's lightweight Pose Landmarker model once:

```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "models/pose_landmarker_lite.task"
```

Then connect the camera and run:

```powershell
uv sync
uv run biomech pose
```

The left panel overlays the 33-point MediaPipe skeleton on RGB and labels major
joints with sensor depth in metres. The right panel shows the same skeleton on
the hardware-aligned depth image. Press `q` or Escape to stop. This is an
exploratory visualization, not yet a calibrated biomechanics measurement.

## Development checks

```powershell
uv run ruff format src tests
uv run ruff check src tests
uv run pytest
```

Hardware tests should be run only on a machine with the camera connected. The
unit tests do not require camera hardware.
