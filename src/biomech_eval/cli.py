"""Command-line entry points for camera development."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer

from biomech_eval.cameras.orbbec import OrbbecCamera
from biomech_eval.data.recorder import SessionRecorder, record

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _depth_preview(
    depth: np.ndarray,
    *,
    depth_scale: float = 0.001,
    min_distance_m: float = 0.2,
    max_distance_m: float = 4.0,
) -> np.ndarray:
    """Create a stable depth visualization using fixed metric limits.

    Per-frame min/max normalization makes the whole image flicker whenever a
    single pixel changes. The sensor's depth scale converts native uint16
    values to metres, then every frame uses the same colour mapping.
    """

    if max_distance_m <= min_distance_m:
        raise ValueError("max_distance_m must be greater than min_distance_m")
    depth_m = depth.astype(np.float32) * depth_scale
    normalized = (depth_m - min_distance_m) / (max_distance_m - min_distance_m) * 255.0
    normalized = np.clip(normalized, 0.0, 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    preview[depth == 0] = 0  # Missing depth should not appear as a false range.
    return preview


def _compose_preview(images: list[np.ndarray], height: int = 480) -> np.ndarray | None:
    """Normalize preview panels before concatenating them.

    RGB and depth streams commonly have different resolutions. OpenCV's
    ``hconcat`` requires equal row counts, data types, and channel counts, so
    normalize those properties at the display boundary only (recorded data is
    never resized).
    """

    panels = []
    for image in images:
        panel = image
        if panel is None:
            continue
        if len(panel.shape) == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        elif panel.shape[2] == 4:
            panel = cv2.cvtColor(panel, cv2.COLOR_BGRA2BGR)
        if panel.dtype != images[0].dtype:
            panel = panel.astype(images[0].dtype)
        scale = height / panel.shape[0]
        width = max(1, round(panel.shape[1] * scale))
        panels.append(cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA))
    if not panels:
        return None
    return cv2.hconcat(panels)


@app.command()
def preview() -> None:
    """Start the camera and display color/depth; press Q or Escape to stop."""

    camera = OrbbecCamera()
    try:
        with camera:
            while True:
                frame = camera.read()
                if frame is None:
                    continue
                images = []
                if frame.color is not None:
                    images.append(frame.color)
                if frame.depth is not None:
                    images.append(_depth_preview(frame.depth, depth_scale=frame.depth_scale))
                if images:
                    preview_image = _compose_preview(images)
                    if preview_image is not None:
                        cv2.imshow("biomech_eval", preview_image)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


@app.command()
def record_session(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("data/raw"),
    frames: Annotated[
        int | None,
        typer.Option(min=1, help="Stop after this many frames; otherwise press Ctrl+C."),
    ] = None,
) -> None:
    """Record lossless RGB-D frames to a new timestamped session."""

    try:
        session = record(OrbbecCamera(), SessionRecorder(output), frames=frames)
    except KeyboardInterrupt:
        typer.echo("Recording stopped.")
        return
    typer.echo(f"Session written to {session}")


if __name__ == "__main__":
    app()
