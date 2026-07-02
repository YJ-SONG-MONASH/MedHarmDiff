import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from medharmdiff.metrics import coral_distance, mmd_rbf
from medharmdiff.synthetic import generate_synthetic_nonlinear_multicenter_features


def test_nonlinear_synthetic_generator_is_deterministic_and_schema_compatible() -> None:
    first = generate_synthetic_nonlinear_multicenter_features(
        n_centers=4,
        samples_per_center=12,
        n_features=6,
        covariance_shift_strength=0.7,
        rotation_shift_strength=0.5,
        heteroskedastic_noise_strength=0.4,
        nonlinear_warp_strength=0.8,
        random_seed=37,
    )
    second = generate_synthetic_nonlinear_multicenter_features(
        n_centers=4,
        samples_per_center=12,
        n_features=6,
        covariance_shift_strength=0.7,
        rotation_shift_strength=0.5,
        heteroskedastic_noise_strength=0.4,
        nonlinear_warp_strength=0.8,
        random_seed=37,
    )

    assert first.equals(second)
    assert len(first) == 48
    assert {"sample_id", "center_id", "label"}.issubset(first.columns)
    assert [column for column in first.columns if column.startswith("feature_")] == [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
    ]
    assert set(first["center_id"]) == {"A", "B", "C", "D"}
    assert set(first["label"]).issubset({0, 1})


def test_nonlinear_shift_strength_increases_site_separability_and_distances() -> None:
    base = generate_synthetic_nonlinear_multicenter_features(
        n_centers=3,
        samples_per_center=80,
        n_features=8,
        additive_shift_strength=0.0,
        covariance_shift_strength=0.0,
        rotation_shift_strength=0.0,
        heteroskedastic_noise_strength=0.0,
        nonlinear_warp_strength=0.0,
        random_seed=41,
    )
    shifted = generate_synthetic_nonlinear_multicenter_features(
        n_centers=3,
        samples_per_center=80,
        n_features=8,
        additive_shift_strength=0.0,
        covariance_shift_strength=1.2,
        rotation_shift_strength=1.0,
        heteroskedastic_noise_strength=0.9,
        nonlinear_warp_strength=1.1,
        random_seed=41,
    )

    assert _site_auc(base) < _site_auc(shifted)
    assert _domain_distance(base) < _domain_distance(shifted)


def test_oracle_columns_are_prefixed_and_not_counted_as_features() -> None:
    df = generate_synthetic_nonlinear_multicenter_features(
        n_centers=3,
        samples_per_center=8,
        n_features=5,
        include_oracle_columns=True,
        random_seed=43,
    )

    feature_columns = [column for column in df.columns if column.startswith("feature_")]
    oracle_columns = [column for column in df.columns if column.startswith("oracle_")]

    assert feature_columns == [f"feature_{idx}" for idx in range(5)]
    assert oracle_columns
    assert not set(feature_columns).intersection(oracle_columns)


def _site_auc(df: pd.DataFrame) -> float:
    feature_columns = [column for column in df.columns if column.startswith("feature_")]
    x = df[feature_columns].to_numpy(dtype=float)
    y = (df["center_id"].astype(str) == "C").astype(int).to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=3),
    )
    model.fit(x, y)
    return float(roc_auc_score(y, model.predict_proba(x)[:, 1]))


def _domain_distance(df: pd.DataFrame) -> float:
    feature_columns = [column for column in df.columns if column.startswith("feature_")]
    source = df[df["center_id"].isin(["A", "B"])][feature_columns].to_numpy(dtype=float)
    target = df[df["center_id"] == "C"][feature_columns].to_numpy(dtype=float)
    return float(mmd_rbf(source, target) + np.sqrt(coral_distance(source, target)))
