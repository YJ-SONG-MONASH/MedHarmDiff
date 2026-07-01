from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .data_schema import SampleRecord


@dataclass(frozen=True)
class LeaveCenterOutSplit:
    target_center: str
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


def leave_one_center_out_splits(
    records: Iterable[SampleRecord],
    *,
    val_fraction: float = 0.15,
) -> list[LeaveCenterOutSplit]:
    """Create deterministic leave-one-center-out splits.

    The validation set is sampled from source centers only by simple sorted slicing.
    """

    rows = sorted(list(records), key=lambda item: item.sample_id)
    by_center: dict[str, list[SampleRecord]] = defaultdict(list)
    for row in rows:
        by_center[row.center.center_id].append(row)
    splits: list[LeaveCenterOutSplit] = []
    for target_center in sorted(by_center):
        source_rows = [row for row in rows if row.center.center_id != target_center]
        test_rows = by_center[target_center]
        val_count = max(1, int(round(len(source_rows) * val_fraction))) if source_rows else 0
        val_rows = source_rows[:val_count]
        train_rows = source_rows[val_count:]
        splits.append(
            LeaveCenterOutSplit(
                target_center=target_center,
                train_ids=[row.sample_id for row in train_rows],
                val_ids=[row.sample_id for row in val_rows],
                test_ids=[row.sample_id for row in test_rows],
            )
        )
    return splits
