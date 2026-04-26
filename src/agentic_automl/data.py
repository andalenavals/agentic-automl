from __future__ import annotations

from pathlib import Path

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
        missing_fraction=missing_fraction,
        target_cardinality=target_cardinality,
        target_name=brief.target_column,
        target_skew=target_skew,
        class_imbalance=class_imbalance,
    )


def format_profile_summary(profile: DataProfile) -> str:
    return (
        f"{profile.rows} rows, {profile.columns} columns, "
        f"{len(profile.numeric_features)} numeric features, "
        f"{len(profile.categorical_features)} categorical features, "
        f"{profile.missing_fraction:.1%} missing feature values."
    )
