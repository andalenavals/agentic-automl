from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .schemas import DataProfile, ProjectBrief


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {suffix}. Use CSV or Parquet.")


def profile_dataset(frame: pd.DataFrame, brief: ProjectBrief) -> DataProfile:
    if brief.target_column not in frame.columns:
        raise ValueError(f"Target column '{brief.target_column}' was not found in the dataset.")

    feature_frame = frame.drop(columns=[brief.target_column])
    numeric_features = [column for column in feature_frame.columns if is_numeric_dtype(feature_frame[column])]
    categorical_features = [column for column in feature_frame.columns if column not in numeric_features]
    distinct_by_feature = {
        column: int(feature_frame[column].nunique(dropna=False))
        for column in feature_frame.columns
    }
    categorical_cardinality = {
        column: distinct_by_feature[column]
        for column in categorical_features
    }
    numeric_range_by_feature = {
        column: detect_numeric_range(feature_frame[column])
        for column in numeric_features
    }
    missing_by_feature = {
        column: float(feature_frame[column].isna().mean())
        for column in feature_frame.columns
        if feature_frame[column].isna().any()
    }
    constant_features = [
        column
        for column in feature_frame.columns
        if feature_frame[column].nunique(dropna=False) <= 1
    ]
    date_like_features = detect_date_like_features(feature_frame, categorical_features, brief)
    date_parse_failure_by_feature = detect_date_parse_failures(feature_frame, date_like_features)
    likely_identifier_features = detect_identifier_features(feature_frame, date_like_features)
    high_cardinality_categorical_features = detect_high_cardinality_categorical_features(
        rows=len(feature_frame),
        categorical_cardinality=categorical_cardinality,
        date_like_features=date_like_features,
        likely_identifier_features=likely_identifier_features,
    )
    target = frame[brief.target_column]

    missing_fraction = float(feature_frame.isna().sum().sum()) / float(max(feature_frame.size, 1))
    target_cardinality = int(target.nunique(dropna=False))
    target_skew = float(target.skew()) if brief.task_type == "regression" and is_numeric_dtype(target) else None

    class_imbalance = None
    if brief.task_type == "classification":
        counts = target.value_counts(dropna=False)
        if not counts.empty and counts.min() > 0:
            class_imbalance = float(counts.max() / counts.min())

    return DataProfile(
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categorical_cardinality=categorical_cardinality,
        missing_by_feature=missing_by_feature,
        constant_features=constant_features,
        likely_identifier_features=likely_identifier_features,
        date_like_features=date_like_features,
        date_parse_failure_by_feature=date_parse_failure_by_feature,
        high_cardinality_categorical_features=high_cardinality_categorical_features,
        missing_fraction=missing_fraction,
        target_cardinality=target_cardinality,
        target_name=brief.target_column,
        target_skew=target_skew,
        class_imbalance=class_imbalance,
        distinct_by_feature=distinct_by_feature,
        numeric_range_by_feature=numeric_range_by_feature,
    )


def format_profile_summary(profile: DataProfile) -> str:
    return (
        f"{profile.rows} rows, {profile.columns} columns, "
        f"{len(profile.numeric_features)} numeric features, "
        f"{len(profile.categorical_features)} categorical features, "
        f"{profile.missing_fraction:.1%} missing feature values."
    )


def detect_date_like_features(
    feature_frame: pd.DataFrame,
    categorical_features: list[str],
    brief: ProjectBrief,
) -> list[str]:
    detected: set[str] = set()
    if brief.date_column and brief.date_column in feature_frame.columns:
        detected.add(brief.date_column)

    for column in categorical_features:
        if column in detected:
            continue
        series = feature_frame[column].dropna()
        if series.empty:
            continue

        sample = series.astype(str).head(min(len(series), 250))
        lowered_name = column.lower()
        name_looks_like_date = any(token in lowered_name for token in ["date", "time", "timestamp"])
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        success_ratio = float(parsed.notna().mean())
        if name_looks_like_date or success_ratio >= 0.9:
            detected.add(column)

    return sorted(detected)


def detect_identifier_features(
    feature_frame: pd.DataFrame,
    date_like_features: list[str],
) -> list[str]:
    pattern = re.compile(r"(^id$|^id_|_id$|uuid|guid|identifier)")
    detected: list[str] = []
    rows = max(len(feature_frame), 1)
    for column in feature_frame.columns:
        if column in date_like_features:
            continue
        lowered_name = column.lower()
        if not pattern.search(lowered_name):
            continue
        unique_ratio = float(feature_frame[column].nunique(dropna=False)) / float(rows)
        if unique_ratio >= 0.5:
            detected.append(column)
    return detected


def detect_date_parse_failures(
    feature_frame: pd.DataFrame,
    date_like_features: list[str],
) -> dict[str, float]:
    failures: dict[str, float] = {}
    for column in date_like_features:
        series = feature_frame[column]
        if series.empty:
            continue
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        failure_fraction = float(parsed.isna().mean())
        if failure_fraction > 0:
            failures[column] = failure_fraction
    return failures


def detect_high_cardinality_categorical_features(
    rows: int,
    categorical_cardinality: dict[str, int],
    date_like_features: list[str],
    likely_identifier_features: list[str],
) -> dict[str, int]:
    if rows <= 0:
        return {}
    threshold = max(20, min(100, int(rows * 0.1)))
    high_cardinality: dict[str, int] = {}
    for column, cardinality in categorical_cardinality.items():
        if column in date_like_features or column in likely_identifier_features:
            continue
        unique_ratio = float(cardinality) / float(rows)
        if cardinality >= threshold and unique_ratio >= 0.1:
            high_cardinality[column] = cardinality
    return high_cardinality


def detect_numeric_range(series: pd.Series) -> dict[str, float | None]:
    non_null = series.dropna()
    if non_null.empty:
        return {"min": None, "max": None}
    return {
        "min": float(non_null.min()),
        "max": float(non_null.max()),
    }
