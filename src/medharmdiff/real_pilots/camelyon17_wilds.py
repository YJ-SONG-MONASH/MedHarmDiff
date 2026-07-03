from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Camelyon17MetadataRow:
    sample_id: str
    center_id: str
    label: int
    patient_or_group_id: str | None = None
    split_group: str | None = None
    image_path: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


ALIASES = {
    "sample_id": ("sample_id", "id", "patch_id", "image_id"),
    "center_id": ("center_id", "domain_id", "hospital", "hospital_id", "metadata_hospital"),
    "label": ("label", "y", "tumor_label"),
    "patient_or_group_id": (
        "patient_id",
        "slide_id",
        "patient_or_group_id",
        "group_id",
    ),
    "split_group": ("split", "split_group"),
    "image_path": ("image_path", "path", "filepath", "file_path"),
}


def normalize_camelyon17_metadata_row(raw: Mapping[str, object]) -> Camelyon17MetadataRow:
    consumed: set[str] = set()
    sample_id = _required(raw, "sample_id", consumed)
    center_id = _required(raw, "center_id", consumed)
    label = _required(raw, "label", consumed)
    patient_or_group_id = _optional(raw, "patient_or_group_id", consumed)
    split_group = _optional(raw, "split_group", consumed)
    image_path = _optional(raw, "image_path", consumed)
    extra = {
        str(key): value
        for key, value in raw.items()
        if str(key) not in consumed and not _is_missing(value)
    }
    return Camelyon17MetadataRow(
        sample_id=str(sample_id),
        center_id=str(center_id),
        label=_label_to_int(label),
        patient_or_group_id=None
        if patient_or_group_id is None
        else str(patient_or_group_id),
        split_group=None if split_group is None else str(split_group),
        image_path=None if image_path is None else str(image_path),
        extra=extra,
    )


def validate_camelyon17_rows(rows: Sequence[Camelyon17MetadataRow]) -> dict[str, object]:
    sample_ids = [row.sample_id for row in rows]
    centers = [row.center_id for row in rows]
    warnings = []
    duplicated = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicated:
        warnings.append(f"duplicate sample_id values: {duplicated}")
    if not rows:
        warnings.append("metadata has no rows")
    if len(set(centers)) < 3:
        warnings.append("fewer than three centers/domains detected")
    if not all(row.patient_or_group_id for row in rows):
        warnings.append("patient_or_group_id is missing for at least one row")
    if not all(row.split_group for row in rows):
        warnings.append("split_group is missing for at least one row")
    return {
        "sample_count": len(rows),
        "center_count": len(set(centers)),
        "center_counts": _counts(centers),
        "label_prevalence_by_center": _label_prevalence(rows),
        "split_counts": _counts(
            [row.split_group or "missing" for row in rows],
        ),
        "has_patient_or_group_id": bool(rows and all(row.patient_or_group_id for row in rows)),
        "missing_patient_or_group_id_count": sum(
            1 for row in rows if not row.patient_or_group_id
        ),
        "warnings": warnings,
        "feature_csv_ready": bool(rows) and len(set(centers)) >= 3 and not duplicated,
    }


def build_camelyon17_feature_csv_rows(
    metadata_rows: Iterable[Camelyon17MetadataRow],
    embeddings_by_sample_id: Mapping[str, Sequence[float] | np.ndarray],
) -> pd.DataFrame:
    output_rows = []
    expected_dim: int | None = None
    for row in metadata_rows:
        if row.sample_id not in embeddings_by_sample_id:
            raise ValueError(f"Missing embedding for sample_id {row.sample_id!r}")
        embedding = np.asarray(embeddings_by_sample_id[row.sample_id], dtype=float).reshape(-1)
        if expected_dim is None:
            expected_dim = int(embedding.shape[0])
        if int(embedding.shape[0]) != expected_dim:
            raise ValueError("All embeddings must have the same feature dimension")
        output_row: dict[str, object] = {
            "sample_id": row.sample_id,
            "center_id": row.center_id,
            "label": row.label,
            "patient_or_group_id": row.patient_or_group_id,
            "split_group": row.split_group,
        }
        output_row.update(row.extra)
        output_row.update(
            {f"feature_{idx}": float(value) for idx, value in enumerate(embedding)}
        )
        output_rows.append(output_row)
    return pd.DataFrame(output_rows)


def write_camelyon17_feature_csv(
    metadata_rows: Iterable[Camelyon17MetadataRow],
    embeddings_by_sample_id: Mapping[str, Sequence[float] | np.ndarray],
    output_path: str | Path,
) -> dict[str, object]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df = build_camelyon17_feature_csv_rows(
        metadata_rows,
        embeddings_by_sample_id,
    )
    feature_df.to_csv(output_path, index=False)
    return {
        "output_csv": str(output_path),
        "sample_count": int(len(feature_df)),
        "feature_count": len(
            [column for column in feature_df.columns if column.startswith("feature_")]
        ),
    }


def normalize_camelyon17_metadata_frame(df: pd.DataFrame) -> list[Camelyon17MetadataRow]:
    return [
        normalize_camelyon17_metadata_row(record)
        for record in df.to_dict(orient="records")
    ]


def _required(
    raw: Mapping[str, object],
    canonical: str,
    consumed: set[str],
) -> object:
    value, key = _find_alias(raw, canonical)
    if key is None or _is_missing(value):
        aliases = ", ".join(ALIASES[canonical])
        raise ValueError(f"Missing required Camelyon17 field {canonical!r}; aliases: {aliases}")
    consumed.add(str(key))
    return value


def _optional(
    raw: Mapping[str, object],
    canonical: str,
    consumed: set[str],
) -> object | None:
    value, key = _find_alias(raw, canonical)
    if key is None or _is_missing(value):
        return None
    consumed.add(str(key))
    return value


def _find_alias(raw: Mapping[str, object], canonical: str) -> tuple[object | None, str | None]:
    lowered = {str(key).lower(): key for key in raw}
    for alias in ALIASES[canonical]:
        if alias.lower() in lowered:
            key = lowered[alias.lower()]
            return raw[key], str(key)
    return None, None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return bool(pd.isna(value)) if not isinstance(value, (list, tuple, dict)) else False


def _label_to_int(value: object) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "tumor", "positive", "pos"}:
            return 1
        if normalized in {"false", "normal", "negative", "neg"}:
            return 0
    label = int(value)
    if label not in {0, 1}:
        raise ValueError("Camelyon17 MVP expects binary labels encoded as 0/1")
    return label


def _counts(values: Iterable[str]) -> dict[str, int]:
    series = pd.Series(list(values), dtype=str)
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def _label_prevalence(rows: Sequence[Camelyon17MetadataRow]) -> dict[str, float]:
    if not rows:
        return {}
    df = pd.DataFrame(
        {"center_id": [row.center_id for row in rows], "label": [row.label for row in rows]}
    )
    prevalence = df.groupby("center_id")["label"].mean().sort_index()
    return {str(center): round(float(value), 6) for center, value in prevalence.items()}
