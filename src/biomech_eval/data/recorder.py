"""Lossless, inspectable RGB-D session recorder."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import cv2

from biomech_eval.cameras.base import Camera, FrameSet


class SessionRecorder:
    """Write each frame and a synchronized index into one session directory."""

    def __init__(self, root: Path, *, camera_name: str = "orbbec") -> None:
        self.root = root
        self.camera_name = camera_name
        self.session_dir: Path | None = None
        self._index_file: object | None = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> Self:
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        session = self.root / f"{self.camera_name}_{stamp}"
        suffix = 1
        while session.exists():
            session = self.root / f"{self.camera_name}_{stamp}_{suffix:02d}"
            suffix += 1
        (session / "color").mkdir(parents=True, exist_ok=False)
        (session / "depth").mkdir()
        self.session_dir = session
        self._index_file = (session / "frames.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._index_file,
            fieldnames=["frame_number", "timestamp_ns", "color", "depth", "depth_scale"],
        )
        self._writer.writeheader()
        (session / "metadata.json").write_text(
            json.dumps({"camera": self.camera_name, "created_utc": stamp}, indent=2),
            encoding="utf-8",
        )
        return self

    def write(self, frame: FrameSet) -> None:
        if self.session_dir is None or self._writer is None:
            raise RuntimeError("Recorder is not open")
        stem = f"{frame.frame_number:08d}"
        color_name = depth_name = ""
        if frame.color is not None:
            color_name = f"color/{stem}.png"
            if not cv2.imwrite(str(self.session_dir / color_name), frame.color):
                raise OSError(f"Could not write {color_name}")
        if frame.depth is not None:
            depth_name = f"depth/{stem}.png"
            if not cv2.imwrite(str(self.session_dir / depth_name), frame.depth):
                raise OSError(f"Could not write {depth_name}")
        self._writer.writerow(
            {
                "frame_number": frame.frame_number,
                "timestamp_ns": frame.timestamp_ns,
                "color": color_name,
                "depth": depth_name,
                "depth_scale": frame.depth_scale,
            }
        )
        self._index_file.flush()  # type: ignore[union-attr]

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self._index_file is not None:
            self._index_file.close()  # type: ignore[union-attr]
        self._index_file = None
        self._writer = None


def record(camera: Camera, recorder: SessionRecorder, frames: int | None = None) -> Path:
    """Record until ``frames`` is reached or the caller interrupts execution."""

    with camera, recorder:
        count = 0
        while frames is None or count < frames:
            sample = camera.read()
            if sample is None:
                continue
            recorder.write(sample)
            count += 1
    assert recorder.session_dir is not None
    return recorder.session_dir
