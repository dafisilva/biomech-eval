"""Orbbec Gemini RGB-D backend.

The SDK import is intentionally local to this module. This keeps the rest of
the project testable on machines without a camera or native SDK libraries.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import cv2
import numpy as np

from .base import Camera, FrameSet, host_timestamp_ns


class OrbbecCamera(Camera):
    """Capture color and depth frames using OrbbecSDK v2."""

    def __init__(self, *, enable_color: bool = True, enable_depth: bool = True) -> None:
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self._pipeline: Any | None = None
        self._frame_number = 0
        self._depth_scale = 0.001

    def start(self) -> None:
        if self._pipeline is not None:
            return

        try:
            from pyorbbecsdk import Config, OBSensorType, Pipeline
        except ImportError as exc:  # pragma: no cover - depends on machine setup
            raise RuntimeError(
                "Orbbec SDK is unavailable. Install it with "
                "'uv add pyorbbecsdk2' and run 'uv sync'."
            ) from exc

        pipeline = Pipeline()
        config = Config()
        try:
            if self.enable_color:
                config.enable_stream(OBSensorType.COLOR_SENSOR)
            if self.enable_depth:
                config.enable_stream(OBSensorType.DEPTH_SENSOR)
            pipeline.start(config)
        except Exception:
            # A partially started native pipeline must be stopped before the
            # exception escapes, otherwise the next attempt may fail.
            with suppress(Exception):
                pipeline.stop()
            raise

        self._pipeline = pipeline
        self._frame_number = 0

    def read(self, timeout_ms: int = 1000) -> FrameSet | None:
        if self._pipeline is None:
            raise RuntimeError("Camera is not started; call start() first")

        frames = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None

        color = _color_array(frames.get_color_frame()) if self.enable_color else None
        depth = _depth_array(frames.get_depth_frame()) if self.enable_depth else None
        self._frame_number += 1
        return FrameSet(
            timestamp_ns=host_timestamp_ns(),
            color=color,
            depth=depth,
            depth_scale=self._depth_scale,
            frame_number=self._frame_number,
        )

    def stop(self) -> None:
        if self._pipeline is None:
            return
        try:
            self._pipeline.stop()
        finally:
            self._pipeline = None


def _color_array(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    video = frame.as_video_frame() if hasattr(frame, "as_video_frame") else frame
    height, width = video.get_height(), video.get_width()
    raw = np.asanyarray(video.get_data())
    fmt = str(video.get_format()).upper()
    if "MJPG" in fmt or "MJPEG" in fmt:
        image = cv2.imdecode(np.asarray(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Could not decode Orbbec MJPEG color frame")
        return image
    if raw.size == height * width * 3:
        return raw.reshape(height, width, 3).copy()
    if raw.size == height * width * 2:
        return cv2.cvtColor(raw.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
    raise RuntimeError(f"Unsupported Orbbec color frame format: {fmt}")


def _depth_array(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    video = frame.as_video_frame() if hasattr(frame, "as_video_frame") else frame
    height, width = video.get_height(), video.get_width()
    raw = np.asanyarray(video.get_data())
    return np.frombuffer(raw, dtype=np.uint16, count=height * width).reshape(height, width).copy()
