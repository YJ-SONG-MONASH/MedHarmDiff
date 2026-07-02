import numpy as np

from medharmdiff.baselines import (
    CenterMeanHarmonizer,
    CoralHarmonizer,
    IdentityHarmonizer,
    MMDMeanAlignmentHarmonizer,
    StandardizeBySourceHarmonizer,
)


def test_harmonizers_preserve_feature_shape() -> None:
    source_x = np.array(
        [
            [1.0, 2.0],
            [1.5, 2.2],
            [3.0, 4.0],
            [3.2, 4.1],
        ]
    )
    source_site = np.array(["A", "A", "B", "B"])
    source_y = np.array([0, 0, 1, 1])
    target_x = np.array([[5.0, 7.0], [5.2, 7.1], [5.3, 7.4]])
    target_site = np.array(["C", "C", "C"])

    source_only = [
        IdentityHarmonizer(),
        StandardizeBySourceHarmonizer(),
        CenterMeanHarmonizer(),
    ]
    target_unlabeled = [
        CoralHarmonizer(),
        MMDMeanAlignmentHarmonizer(),
    ]

    for harmonizer in source_only:
        harmonizer.fit(source_x, source_site, source_y)
        assert harmonizer.transform(target_x, target_site).shape == target_x.shape

    for harmonizer in target_unlabeled:
        harmonizer.fit(source_x, source_site, source_y, target_x=target_x, target_site=target_site)
        assert harmonizer.transform(target_x, target_site).shape == target_x.shape
