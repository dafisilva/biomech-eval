"""Interfaces shared by physical and recorded camera backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time_ns
from typing import Self

import numpy as np


@dataclass(slots=True)
class FrameSet:
    """One RGB-D sample with explicit clock domains.

    ``device_timestamp_us`` is the primary device-clock timestamp (depth is
    preferred when available). ``system_timestamp_us`` is the SDK's host-clock
    receipt timestamp. ``host_received_ns`` is this process's wall-clock time
    immediately after receiving the frame set. None means the backend did not
    provide that timestamp. Color is BGR; depth is uint16 in native units.
    """

    frame_number: int
    device_timestamp_us: int | None
    system_timestamp_us: int | None
    host_received_ns: int
    color: np.ndarray | None
    depth: np.ndarray | None
    depth_scale_m: float = 0.001
    color_device_timestamp_us: int | None = None
    depth_device_timestamp_us: int | None = None
    color_system_timestamp_us: int | None = None
    depth_system_timestamp_us: int | None = None


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


def host_received_ns() -> int:
    """Return wall-clock receipt time in Unix nanoseconds (not monotonic)."""

    return time_ns()
