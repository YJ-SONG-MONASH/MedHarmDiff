import numpy as np

from medharmdiff.metrics import coral_distance, mmd_rbf


def test_coral_distance_zero_for_identical_arrays() -> None:
    x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    assert coral_distance(x, x) == 0.0


def test_mmd_rbf_zero_for_identical_arrays() -> None:
    x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    assert mmd_rbf(x, x) == 0.0


def test_distance_functions_validate_dimensions() -> None:
    x = np.array([[1.0], [2.0]])
    y = np.array([[1.0, 2.0], [2.0, 3.0]])

    try:
        coral_distance(x, y)
    except ValueError as exc:
        assert "same feature dimension" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
