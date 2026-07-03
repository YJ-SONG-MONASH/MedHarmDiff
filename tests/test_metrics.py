import numpy as np

from medharmdiff.metrics import coral_distance, mmd_rbf


def test_coral_distance_zero_for_identical_arrays() -> None:
    x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    assert coral_distance(x, x) == 0.0


def test_mmd_rbf_zero_for_identical_arrays() -> None:
    x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    assert mmd_rbf(x, x) == 0.0


def test_mmd_rbf_supports_deterministic_sample_cap() -> None:
    rng = np.random.default_rng(13)
    source = rng.normal(size=(300, 4))
    target = rng.normal(loc=0.2, size=(500, 4))

    first = mmd_rbf(source, target, max_samples=64)
    second = mmd_rbf(source, target, max_samples=64)

    assert isinstance(first, float)
    assert first == second


def test_distance_functions_validate_dimensions() -> None:
    x = np.array([[1.0], [2.0]])
    y = np.array([[1.0, 2.0], [2.0, 3.0]])

    try:
        coral_distance(x, y)
    except ValueError as exc:
        assert "same feature dimension" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
