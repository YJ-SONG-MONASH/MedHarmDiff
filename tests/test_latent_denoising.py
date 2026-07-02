import numpy as np

from medharmdiff.latent_denoising import (
    ClinicalPreservingRidgeDenoisingHarmonizer,
    RidgeDenoisingHarmonizer,
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


def test_ridge_denoising_fit_transform_preserves_shape() -> None:
    x, site, y, target_x, target_site = _toy_features()
    harmonizer = RidgeDenoisingHarmonizer(
        noise_strength=0.05,
        n_augments=3,
        alpha=0.5,
        random_seed=11,
    )

    harmonizer.fit(x, site, y, target_x=target_x, target_site=target_site)
    transformed = harmonizer.transform(target_x, target_site)

    assert transformed.shape == target_x.shape
    assert np.isfinite(transformed).all()


def test_ridge_denoising_transform_before_fit_raises() -> None:
    harmonizer = RidgeDenoisingHarmonizer()

    try:
        harmonizer.transform(np.ones((2, 3)))
    except RuntimeError as exc:
        assert "fit must be called" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_ridge_denoising_is_deterministic_with_same_random_seed() -> None:
    x, site, y, target_x, target_site = _toy_features()
    first = RidgeDenoisingHarmonizer(random_seed=17, n_augments=4).fit(x, site, y)
    second = RidgeDenoisingHarmonizer(random_seed=17, n_augments=4).fit(x, site, y)

    assert np.allclose(
        first.transform(target_x, target_site),
        second.transform(target_x, target_site),
    )


def test_ridge_denoising_does_not_use_target_inputs_in_zero_shot_style_fit() -> None:
    x, site, y, target_x, target_site = _toy_features()
    source_only = RidgeDenoisingHarmonizer(random_seed=19).fit(x, site, y)
    with_target_unlabeled = RidgeDenoisingHarmonizer(random_seed=19).fit(
        x,
        site,
        y,
        target_x=target_x + 1000.0,
        target_site=target_site,
    )

    assert np.allclose(
        source_only.transform(target_x, target_site),
        with_target_unlabeled.transform(target_x, target_site),
    )


def test_clinical_preserving_variant_preserves_shape_and_uses_source_labels() -> None:
    x, site, y, target_x, target_site = _toy_features()
    harmonizer = ClinicalPreservingRidgeDenoisingHarmonizer(
        clinical_preservation_strength=0.8,
        random_seed=23,
    )

    harmonizer.fit(x, site, y)
    transformed = harmonizer.transform(target_x, target_site)

    assert transformed.shape == target_x.shape
    assert np.isfinite(transformed).all()
