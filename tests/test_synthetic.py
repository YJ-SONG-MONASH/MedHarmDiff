import numpy as np

from medharmdiff.synthetic import generate_synthetic_feature_dataset


def test_synthetic_feature_dataset_is_reproducible_and_has_expected_schema() -> None:
    first = generate_synthetic_feature_dataset(
        n_centers=4,
        samples_per_center=12,
        n_features=5,
        clinical_signal_strength=1.5,
        site_shift_strength=2.0,
        site_label_confounding_strength=0.2,
        random_seed=7,
    )
    second = generate_synthetic_feature_dataset(
        n_centers=4,
        samples_per_center=12,
        n_features=5,
        clinical_signal_strength=1.5,
        site_shift_strength=2.0,
        site_label_confounding_strength=0.2,
        random_seed=7,
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
    ]
    assert set(first["center_id"]) == {"A", "B", "C", "D"}
    assert set(first["label"]).issubset({0, 1})


def test_synthetic_site_shift_strength_changes_center_separation() -> None:
    low_shift = generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=60,
        n_features=4,
        site_shift_strength=0.0,
        random_seed=11,
    )
    high_shift = generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=60,
        n_features=4,
        site_shift_strength=3.0,
        random_seed=11,
    )

    low_center_spread = low_shift.groupby("center_id")["feature_0"].mean().std()
    high_center_spread = high_shift.groupby("center_id")["feature_0"].mean().std()

    assert high_center_spread > low_center_spread + 0.5


def test_synthetic_clinical_signal_strength_changes_label_feature_association() -> None:
    weak = generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=120,
        n_features=6,
        clinical_signal_strength=0.0,
        site_shift_strength=0.0,
        noise_std=1.0,
        random_seed=17,
    )
    strong = generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=120,
        n_features=6,
        clinical_signal_strength=3.0,
        site_shift_strength=0.0,
        noise_std=1.0,
        random_seed=17,
    )

    weak_gap = abs(weak.groupby("label")["feature_0"].mean().diff().dropna().iloc[0])
    strong_gap = abs(strong.groupby("label")["feature_0"].mean().diff().dropna().iloc[0])

    assert strong_gap > weak_gap + 0.5


def test_synthetic_label_site_confounding_strength_changes_label_prevalence() -> None:
    unconfounded = generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=200,
        site_label_confounding_strength=0.0,
        random_seed=19,
    )
    confounded = generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=200,
        site_label_confounding_strength=0.8,
        random_seed=19,
    )

    unconfounded_spread = unconfounded.groupby("center_id")["label"].mean().std()
    confounded_spread = confounded.groupby("center_id")["label"].mean().std()

    assert np.isfinite(confounded_spread)
    assert confounded_spread > unconfounded_spread + 0.1


def test_synthetic_requires_at_least_three_centers() -> None:
    try:
        generate_synthetic_feature_dataset(n_centers=2)
    except ValueError as exc:
        assert "at least three centers" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_synthetic_multicenter_alias_uses_center_ids_and_padded_feature_names() -> None:
    from medharmdiff.synthetic import generate_synthetic_multicenter_features

    df = generate_synthetic_multicenter_features(
        n_centers=3,
        samples_per_center=5,
        n_features=4,
        seed=5,
    )

    assert set(df["center_id"]) == {"center_0", "center_1", "center_2"}
    assert [column for column in df.columns if column.startswith("feature_")] == [
        "feature_000",
        "feature_001",
        "feature_002",
        "feature_003",
    ]
