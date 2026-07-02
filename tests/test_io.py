import pandas as pd

from medharmdiff.io import load_feature_csv


def test_load_feature_csv_returns_arrays_sample_ids_and_feature_columns(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "center_id": ["A", "B", "C"],
            "label": [0, 1, 0],
            "feature_0": [1.0, 2.0, 3.0],
            "feature_1": [3.0, 4.0, 5.0],
            "note": ["ignore", "ignore", "ignore"],
        }
    ).to_csv(path, index=False)

    x, site, y, sample_ids, feature_columns = load_feature_csv(path)

    assert x.shape == (3, 2)
    assert site.tolist() == ["A", "B", "C"]
    assert y.tolist() == [0, 1, 0]
    assert sample_ids == ["s1", "s2", "s3"]
    assert feature_columns == ["feature_0", "feature_1"]


def test_load_feature_csv_rejects_duplicate_sample_ids(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s3"],
            "center_id": ["A", "B", "C"],
            "label": [0, 1, 0],
            "feature_0": [1.0, 2.0, 3.0],
        }
    ).to_csv(path, index=False)

    try:
        load_feature_csv(path)
    except ValueError as exc:
        assert "duplicate sample_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_load_feature_csv_requires_feature_columns(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "center_id": ["A", "B", "C"],
            "label": [0, 1, 0],
            "not_a_feature": [1.0, 2.0, 3.0],
        }
    ).to_csv(path, index=False)

    try:
        load_feature_csv(path)
    except ValueError as exc:
        assert "at least one feature column" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_load_feature_csv_requires_core_columns(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "label": [0, 1, 0],
            "feature_0": [1.0, 2.0, 3.0],
        }
    ).to_csv(path, index=False)

    try:
        load_feature_csv(path)
    except ValueError as exc:
        assert "center_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_load_feature_csv_rejects_nan_features(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "center_id": ["A", "B", "C"],
            "label": [0, 1, 0],
            "feature_0": [1.0, None, 3.0],
        }
    ).to_csv(path, index=False)

    try:
        load_feature_csv(path)
    except ValueError as exc:
        assert "NaN" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_load_feature_csv_rejects_non_binary_labels(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "center_id": ["A", "B", "C"],
            "label": [0, 1, 2],
            "feature_0": [1.0, 2.0, 3.0],
        }
    ).to_csv(path, index=False)

    try:
        load_feature_csv(path)
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_load_feature_csv_requires_at_least_three_centers(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "center_id": ["A", "A", "B"],
            "label": [0, 1, 0],
            "feature_0": [1.0, 2.0, 3.0],
        }
    ).to_csv(path, index=False)

    try:
        load_feature_csv(path)
    except ValueError as exc:
        assert "at least three centers" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_load_feature_csv_auto_excludes_oracle_columns(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "center_id": ["A", "B", "C"],
            "label": [0, 1, 0],
            "feature_0": [1.0, 2.0, 3.0],
            "feature_1": [3.0, 4.0, 5.0],
            "oracle_clinical_latent_0": [0.1, 0.2, 0.3],
            "oracle_site_style_0": [1.1, 1.2, 1.3],
        }
    ).to_csv(path, index=False)

    x, _, _, _, feature_columns = load_feature_csv(path, feature_columns="auto")

    assert x.shape == (3, 2)
    assert feature_columns == ["feature_0", "feature_1"]
