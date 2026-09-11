"""Orbbec Gemini RGB-D backend.

The SDK import is intentionally local to this module. This keeps the rest of
the project testable on machines without a camera or native SDK libraries.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import cv2
import numpy as np

from .base import Camera, FrameSet, host_received_ns


class OrbbecCamera(Camera):
    """Capture color and depth frames using OrbbecSDK v2."""

    def __init__(
        self,
        *,
        enable_color: bool = True,
        enable_depth: bool = True,
        align_depth_to_color: bool = False,
    ) -> None:
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.align_depth_to_color = align_depth_to_color
        self._pipeline: Any | None = None
        self._frame_number = 0

    def start(self) -> None:
        if self._pipeline is not None:
            return

        try:
            from pyorbbecsdk import (  # type: ignore[import-untyped]
                Config,
                OBAlignMode,
                OBSensorType,
                Pipeline,
            )
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
            if self.align_depth_to_color:
                if not (self.enable_color and self.enable_depth):
                    raise ValueError("Depth-to-color alignment requires both streams")
                config.set_align_mode(OBAlignMode.HW_MODE)
            pipeline.start(config)
            if self.align_depth_to_color:
                pipeline.enable_frame_sync()
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

        color_frame = frames.get_color_frame() if self.enable_color else None
        depth_frame = frames.get_depth_frame() if self.enable_depth else None
        color = _color_array(color_frame)
        depth = _depth_array(depth_frame)
        color_device_ts = _timestamp_us(color_frame, "get_timestamp_us")
        depth_device_ts = _timestamp_us(depth_frame, "get_timestamp_us")
        color_system_ts = _timestamp_us(color_frame, "get_system_timestamp_us")
        depth_system_ts = _timestamp_us(depth_frame, "get_system_timestamp_us")
        depth_scale = _depth_scale_m(depth_frame)
        frame_number = (
            _frame_index(depth_frame) or _frame_index(color_frame) or self._frame_number + 1
        )
        self._frame_number += 1
        return FrameSet(
            frame_number=frame_number,
            device_timestamp_us=depth_device_ts or color_device_ts,
            system_timestamp_us=depth_system_ts or color_system_ts,
            host_received_ns=host_received_ns(),
            color=color,
            depth=depth,
            depth_scale_m=depth_scale,
            color_device_timestamp_us=color_device_ts,
            depth_device_timestamp_us=depth_device_ts,
            color_system_timestamp_us=color_system_ts,
            depth_system_timestamp_us=depth_system_ts,
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


def _timestamp_us(frame: Any, method: str) -> int | None:
    """Read an SDK timestamp, returning None for missing/invalid metadata."""

    if frame is None or not hasattr(frame, method):
        return None
    try:
        value = int(getattr(frame, method)())
    except (TypeError, ValueError, RuntimeError):
        return None
    return value if value > 0 else None


def _frame_index(frame: Any) -> int | None:
    value = _timestamp_us(frame, "get_index")
    return value


def _depth_scale_m(frame: Any) -> float:
    if frame is None or not hasattr(frame, "get_depth_scale"):
        return 0.001
    try:
        # Orbbec's depth scale converts native depth values to millimetres;
        # convert that scale to metres for the project-wide contract.
        return float(frame.get_depth_scale()) * 0.001
    except (TypeError, ValueError, RuntimeError):
        return 0.001
