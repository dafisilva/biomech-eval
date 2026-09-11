import numpy as np
import pytest

from biomech_eval.pose import PoseLandmark, add_depth_to_poses, draw_poses


def _point(x: float = 0.5, y: float = 0.5) -> PoseLandmark:
    return PoseLandmark(x=x, y=y, z=0.0, visibility=1.0, presence=1.0)


def test_add_depth_to_poses_uses_median_of_valid_neighborhood() -> None:
    depth = np.zeros((5, 5), dtype=np.uint16)
    depth[1:4, 1:4] = 1_000
    depth[2, 2] = 20_000

    result = add_depth_to_poses([[_point()]], depth, 0.001, sample_radius=1)

    assert result[0][0].depth_m == pytest.approx(1.0)


def test_add_depth_to_poses_leaves_out_of_frame_landmark_without_depth() -> None:
    depth = np.full((5, 5), 1_000, dtype=np.uint16)

    result = add_depth_to_poses([[_point(x=1.5)]], depth, 0.001)

    assert result[0][0].depth_m is None


def test_draw_poses_returns_annotated_copy() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    pose = [_point()] * 2

    result = draw_poses(image, [pose])

    assert np.count_nonzero(result) > 0
    assert np.count_nonzero(image) == 0
