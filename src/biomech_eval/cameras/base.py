"""Interfaces shared by physical and recorded camera backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time_ns
from typing import Self

import numpy as np


@dataclass(slots=True)
class FrameSet:
    """One synchronized RGB-D sample.

    Color is BGR (OpenCV convention), depth is uint16 in the device's native
    units, and timestamps are nanoseconds from the host monotonic clock unless
    the backend provides a more appropriate value.
    """

    timestamp_ns: int
    color: np.ndarray | None
    depth: np.ndarray | None
    depth_scale: float = 0.001
    frame_number: int = 0


class Camera(ABC):
    """Minimal lifecycle contract for every camera backend."""

    @abstractmethod
    def start(self) -> None:
        """Open the device and start streaming."""

    @abstractmethod
    def read(self, timeout_ms: int = 1000) -> FrameSet | None:
        """Return the next synchronized frame, or ``None`` on timeout."""

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release all device resources."""

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop()


def host_timestamp_ns() -> int:
    """Return a wall-clock timestamp suitable for session metadata."""

    return time_ns()
