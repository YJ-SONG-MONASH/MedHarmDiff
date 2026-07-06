import numpy as np

from medharmdiff.latent_diffusion import (
    ClinicalPreservingTimeConditionedRidgeDiffusionHarmonizer,
    TimeConditionedRidgeDiffusionHarmonizer,
)


def _toy_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.array(
        [
            [0.0, 1.0, 0.2],
            [0.2, 1.1, 0.1],
            [2.0, 3.0, 1.2],
            [2.2, 3.1, 1.1],
            [4.0, 1.0, 2.2],
            [4.2, 1.1, 2.1],
        ],
        dtype=float,
    )
    site = np.array(["A", "A", "B", "B", "C", "C"])
    y = np.array([0, 0, 1, 1, 0, 0])
    target_x = np.array([[5.0, 4.0, 0.5], [5.1, 4.2, 0.6]], dtype=float)
    target_site = np.array(["D", "D"])
    return x, site, y, target_x, target_site


def test_latent_diffusion_fit_transform_preserves_shape() -> None:
    x, site, y, target_x, target_site = _toy_features()
    harmonizer = TimeConditionedRidgeDiffusionHarmonizer(
        num_steps=4,
        n_augments=3,
        noise_strength=0.1,
        random_seed=11,
    )

    harmonizer.fit(x, site, y, target_x=target_x + 100.0, target_site=target_site)
    transformed = harmonizer.transform(target_x, target_site)

    assert harmonizer.is_fitted
    assert harmonizer.uses_target_unlabeled is False
    assert harmonizer.uses_target_labels is False
    assert transformed.shape == target_x.shape
    assert np.isfinite(transformed).all()


def test_latent_diffusion_transform_before_fit_raises() -> None:
    harmonizer = TimeConditionedRidgeDiffusionHarmonizer()

    try:
        harmonizer.transform(np.ones((2, 3)))
    except RuntimeError as exc:
        assert "fit must be called" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_latent_diffusion_is_deterministic_with_same_random_seed() -> None:
    x, site, y, target_x, target_site = _toy_features()
    first = TimeConditionedRidgeDiffusionHarmonizer(
        num_steps=5,
        n_augments=4,
        random_seed=17,
    ).fit(x, site, y)
    second = TimeConditionedRidgeDiffusionHarmonizer(
        num_steps=5,
        n_augments=4,
        random_seed=17,
    ).fit(x, site, y)

    assert np.allclose(
        first.transform(target_x, target_site),
        second.transform(target_x, target_site),
    )


def test_latent_diffusion_num_steps_affects_finite_transform_path() -> None:
    x, site, y, target_x, target_site = _toy_features()
    short = TimeConditionedRidgeDiffusionHarmonizer(num_steps=2, random_seed=19).fit(
        x, site, y
    )
    long = TimeConditionedRidgeDiffusionHarmonizer(num_steps=7, random_seed=19).fit(
        x, site, y
    )

    short_out = short.transform(target_x, target_site)
    long_out = long.transform(target_x, target_site)

    assert np.isfinite(short_out).all()
    assert np.isfinite(long_out).all()
    assert not np.allclose(short_out, long_out)


def test_latent_diffusion_caps_training_pairs_without_float64_expansion() -> None:
    canonical_x = np.arange(80, dtype=np.float32).reshape(20, 4)
    harmonizer = TimeConditionedRidgeDiffusionHarmonizer(
        n_augments=5,
        max_training_pairs=17,
        random_seed=29,
    )
    harmonizer.feature_scale_ = np.ones(4, dtype=np.float32)
    harmonizer.site_offsets_ = {
        "A": np.zeros(4, dtype=np.float32),
        "B": np.ones(4, dtype=np.float32),
    }

    train_x, train_y = harmonizer._diffusion_training_pairs(canonical_x)

    assert train_x.shape == (17, 9)
    assert train_y.shape == (17, 4)
    assert train_x.dtype == np.float32
    assert train_y.dtype == np.float32


def test_clinical_preserving_latent_diffusion_preserves_shape() -> None:
    x, site, y, target_x, target_site = _toy_features()
    harmonizer = ClinicalPreservingTimeConditionedRidgeDiffusionHarmonizer(
        num_steps=4,
        clinical_preservation_strength=0.8,
        random_seed=23,
    )

    harmonizer.fit(x, site, y)
    transformed = harmonizer.transform(target_x, target_site)

    assert transformed.shape == target_x.shape
    assert np.isfinite(transformed).all()
