"""Small MediaPipe Pose wrapper and RGB-D visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TypeAlias

import cv2
import numpy as np

# MediaPipe's 33-landmark BlazePose topology. Keeping this data local lets the
# drawing and depth helpers remain unit-testable without initializing a model.
POSE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)

DEPTH_LABEL_LANDMARKS = frozenset({11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28})


@dataclass(frozen=True)
class PoseLandmark:
    """One MediaPipe landmark in normalized color-image coordinates."""

    x: float
    y: float
    z: float
    visibility: float
    presence: float
    depth_m: float | None = None


Pose: TypeAlias = list[PoseLandmark]


class PoseDetector:
    """Run MediaPipe Pose Landmarker with tracking enabled between frames."""

    def __init__(self, model_path: Path, *, min_confidence: float = 0.5) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe pose model not found: {model_path}")

        import mediapipe as mp  # type: ignore[import-untyped]

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._mp: Any = mp
        self._landmarker: Any = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, color_bgr: np.ndarray, timestamp_ms: int) -> list[Pose]:
        """Detect poses in a BGR frame using a monotonically increasing timestamp."""

        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=color_rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return [
            [
                PoseLandmark(
                    x=float(point.x),
                    y=float(point.y),
                    z=float(point.z),
                    visibility=float(point.visibility or 0.0),
                    presence=float(point.presence or 0.0),
                )
                for point in pose
            ]
            for pose in result.pose_landmarks
        ]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def add_depth_to_poses(
    poses: list[Pose],
    depth: np.ndarray | None,
    depth_scale_m: float,
    *,
    sample_radius: int = 2,
) -> list[Pose]:
    """Attach robust sensor depth to landmarks in a color-aligned depth frame."""

    if depth is None:
        return poses
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D image")
    if sample_radius < 0:
        raise ValueError("sample_radius must be non-negative")

    height, width = depth.shape
    enriched = []
    for pose in poses:
        enriched_pose = []
        for point in pose:
            x = round(point.x * (width - 1))
            y = round(point.y * (height - 1))
            sensor_depth = None
            if 0 <= x < width and 0 <= y < height:
                x0, x1 = max(0, x - sample_radius), min(width, x + sample_radius + 1)
                y0, y1 = max(0, y - sample_radius), min(height, y + sample_radius + 1)
                valid = depth[y0:y1, x0:x1]
                valid = valid[valid > 0]
                if valid.size:
                    sensor_depth = float(np.median(valid)) * depth_scale_m
            enriched_pose.append(replace(point, depth_m=sensor_depth))
        enriched.append(enriched_pose)
    return enriched


def draw_poses(
    image: np.ndarray,
    poses: list[Pose],
    *,
    min_visibility: float = 0.5,
    show_depth: bool = False,
) -> np.ndarray:
    """Return a copy of an image with visible pose landmarks overlaid."""

    output = image.copy()
    height, width = output.shape[:2]

    def visible(point: PoseLandmark) -> bool:
        return (
            point.visibility >= min_visibility
            and point.presence >= min_visibility
            and 0.0 <= point.x <= 1.0
            and 0.0 <= point.y <= 1.0
        )

    for pose in poses:
        for start, end in POSE_CONNECTIONS:
            if start >= len(pose) or end >= len(pose):
                continue
            a, b = pose[start], pose[end]
            if visible(a) and visible(b):
                a_px = (round(a.x * (width - 1)), round(a.y * (height - 1)))
                b_px = (round(b.x * (width - 1)), round(b.y * (height - 1)))
                cv2.line(output, a_px, b_px, (0, 220, 0), 2, cv2.LINE_AA)

        for index, point in enumerate(pose):
            if not visible(point):
                continue
            pixel = (round(point.x * (width - 1)), round(point.y * (height - 1)))
            cv2.circle(output, pixel, 4, (0, 255, 255), -1, cv2.LINE_AA)
            if show_depth and index in DEPTH_LABEL_LANDMARKS and point.depth_m is not None:
                label_at = (pixel[0] + 5, pixel[1] - 5)
                cv2.putText(
                    output,
                    f"{point.depth_m:.2f}m",
                    label_at,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return output
