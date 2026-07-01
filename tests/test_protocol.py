from medharmdiff.data_schema import CenterMetadata, SampleRecord, validate_sample
from medharmdiff.protocol import leave_one_center_out_splits


def test_leave_one_center_out_splits_use_each_center_as_target() -> None:
    records = [
        SampleRecord(sample_id="a1", center=CenterMetadata("A"), features=[1.0], label=0),
        SampleRecord(sample_id="a2", center=CenterMetadata("A"), features=[1.1], label=0),
        SampleRecord(sample_id="b1", center=CenterMetadata("B"), features=[2.0], label=1),
        SampleRecord(sample_id="c1", center=CenterMetadata("C"), features=[3.0], label=1),
    ]

    splits = leave_one_center_out_splits(records, val_fraction=0.25)

    assert {split.target_center for split in splits} == {"A", "B", "C"}
    assert all(split.test_ids for split in splits)
    assert all(set(split.train_ids).isdisjoint(split.test_ids) for split in splits)


def test_validate_sample_requires_features_or_path() -> None:
    record = SampleRecord(sample_id="x", center=CenterMetadata("A"))

    try:
        validate_sample(record)
    except ValueError as exc:
        assert "features or path" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
