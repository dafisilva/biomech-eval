from pathlib import Path

import numpy as np

from biomech_eval.cameras.base import FrameSet
from biomech_eval.data.recorder import SessionRecorder


def test_session_recorder_writes_lossless_frames(tmp_path: Path) -> None:
    frame = FrameSet(
        frame_number=1,
        device_timestamp_us=100,
        system_timestamp_us=110,
        host_received_ns=123,
        color=np.zeros((4, 5, 3), dtype=np.uint8),
        depth=np.full((4, 5), 1200, dtype=np.uint16),
    )
    with SessionRecorder(tmp_path) as recorder:
        recorder.write(frame)
        session = recorder.session_dir
    assert session is not None
    assert (session / "color/00000001.png").exists()
    assert (session / "depth/00000001.png").exists()
    index = (session / "frames.csv").read_text(encoding="utf-8")
    assert "device_timestamp_us" in index
    assert "100" in index
    assert "123" in index
