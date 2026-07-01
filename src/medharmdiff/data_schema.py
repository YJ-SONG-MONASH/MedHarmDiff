from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CenterMetadata:
    """Metadata describing acquisition/source domain."""

    center_id: str
    department: str | None = None
    scanner: str | None = None
    protocol: str | None = None
    device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleRecord:
    """A single sample in a multi-center harmonization dataset."""

    sample_id: str
    center: CenterMetadata
    label: Any | None = None
    features: list[float] | None = None
    path: str | None = None
    split: str | None = None


def validate_sample(record: SampleRecord) -> None:
    if not record.sample_id:
        raise ValueError("sample_id is required")
    if not record.center.center_id:
        raise ValueError("center.center_id is required")
    if record.features is None and record.path is None:
        raise ValueError("Either features or path must be provided")
