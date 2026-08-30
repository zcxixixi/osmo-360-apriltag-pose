import numpy as np
import pytest

from tools.generate_a3_apriltags import mm_to_px
from tools.run_two_tag_synthetic_experiment import (
    PANO_ROOT,
    load_projection,
    tag_corners,
    verify_freeze,
)
@pytest.mark.skipif(
    not (PANO_ROOT / "app/core/maps.py").is_file(),
    reason="external PanoForge checkout is not present on this host",
)
def test_frozen_two_tag_locator_files_still_match():
    freeze = verify_freeze()

    assert freeze["freeze_id"] == "two-tag-locator-20260828-v1"


@pytest.mark.skipif(
    not (PANO_ROOT / "app/core/maps.py").is_file(),
    reason="external PanoForge checkout is not present on this host",
)
def test_factory_fisheye_projection_round_trip():
    project, pixels_to_rays = load_projection()
    rays = np.asarray([
        [0.0, 0.0, 1.0],
        [-0.35, -0.10, 1.20],
        [-0.20, 0.15, 1.30],
    ])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    recovered = pixels_to_rays(project(rays))

    np.testing.assert_allclose(recovered, rays, atol=1e-9)


def test_tag_corners_encode_declared_outer_size():
    corners = tag_corners((0.0, 0.0, 1.0), 0.24)

    assert np.isclose(np.linalg.norm(corners[1] - corners[0]), 0.24)
    assert np.isclose(np.linalg.norm(corners[3] - corners[0]), 0.24)
    assert mm_to_px(25.4, 300) == 300
