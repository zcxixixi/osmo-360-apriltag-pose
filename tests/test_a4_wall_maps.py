from pathlib import Path

import numpy as np

from tools.insta360_offline import load_tag_map


ROOT = Path(__file__).resolve().parents[1]


def test_new_wall_panels_have_unique_ids_and_expected_layout():
    six = load_tag_map(ROOT / "config/a4_wall_6tag_ids_128_133.json")
    four = load_tag_map(ROOT / "config/a4_wall_4tag_ids_134_137.json")

    assert six.expected_ids == [128, 129, 130, 131, 132, 133]
    assert four.expected_ids == [134, 135, 136, 137]
    assert set(six.expected_ids).isdisjoint(four.expected_ids)

    np.testing.assert_allclose(four.center(136), [-0.105, -0.1485, 0.0])
    np.testing.assert_allclose(four.center(134), [0.105, -0.1485, 0.0])
    np.testing.assert_allclose(four.center(137), [-0.105, 0.1485, 0.0])
    np.testing.assert_allclose(four.center(135), [0.105, 0.1485, 0.0])

    np.testing.assert_allclose(six.center(130), [-0.21, -0.1485, 0.0])
    np.testing.assert_allclose(six.center(129), [0.0, -0.1485, 0.0])
    np.testing.assert_allclose(six.center(128), [0.21, -0.1485, 0.0])
    np.testing.assert_allclose(six.center(133), [-0.21, 0.1485, 0.0])
    np.testing.assert_allclose(six.center(132), [0.0, 0.1485, 0.0])
    np.testing.assert_allclose(six.center(131), [0.21, 0.1485, 0.0])
