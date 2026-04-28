from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from .schemas import DataProfile


DATE_PARTS = ("year", "month", "day", "dayofweek")
PREPROCESSING_OVERRIDE_KEYS = (
    "force_categorical_features",
    "force_numeric_features",
    "force_date_features",
    "force_identifier_features",
    "force_one_hot_features",
    "force_ordinal_features",
    "drop_features",
    "keep_features",
)
PREPROCESSING_RULE_KEYS = (
    "feature_imputation_rules",
    "feature_scaling_rules",
    "feature_transform_rules",
    "derived_feature_rules",
)
SUPPORTED_NUMERIC_IMPUTATION_STRATEGIES = {"mean", "median", "constant"}
SUPPORTED_CATEGORICAL_IMPUTATION_STRATEGIES = {"most_frequent", "constant"}
SUPPORTED_SCALING_METHODS = {"standard", "minmax", "robust", "none"}
SUPPORTED_TRANSFORM_KINDS = {
    "log",
    "log1p",
    "sqrt",
    "square",
    "abs",
    "multiply",
    "divide",
    "add",
    "subtract",
}
SUPPORTED_DERIVED_FEATURE_KINDS = {"frequency_count"}


@dataclass(frozen=True, slots=True)
class PreprocessingPlan:
    option: str
    dropped_features: tuple[str, ...]
    date_features: tuple[str, ...]
    raw_numeric_features: tuple[str, ...]
    derived_numeric_features: tuple[str, ...]
    raw_categorical_features: tuple[str, ...]
    one_hot_categorical_features: tuple[str, ...]
    ordinal_categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]


def normalize_preprocessing_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {key: [] for key in PREPROCESSING_OVERRIDE_KEYS}
    normalized.update({key: [] for key in PREPROCESSING_RULE_KEYS})
    if not overrides:
        return normalized

    for key in PREPROCESSING_OVERRIDE_KEYS:
        raw_values = overrides.get(key, [])
        if isinstance(raw_values, str):
            candidates = [raw_values]
        elif isinstance(raw_values, (list, tuple, set)):
            candidates = list(raw_values)
        else:
            candidates = []
        normalized[key] = dedupe_feature_names(str(item).strip() for item in candidates if str(item).strip())

    normalized["feature_imputation_rules"] = normalize_feature_imputation_rules(
        overrides.get("feature_imputation_rules")
    )
    normalized["feature_scaling_rules"] = normalize_feature_scaling_rules(
        overrides.get("feature_scaling_rules")
    )
    normalized["feature_transform_rules"] = normalize_feature_transform_rules(
        overrides.get("feature_transform_rules")
    )
    normalized["derived_feature_rules"] = normalize_derived_feature_rules(
        overrides.get("derived_feature_rules")
    )

    return resolve_preprocessing_override_conflicts(normalized)


def stringify_rule_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return str(numeric)
    return str(value).strip()


def normalize_feature_imputation_rules(raw_rules: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rules, list):
        return []

    normalized: dict[str, dict[str, Any]] = {}
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        feature = str(raw_rule.get("feature", "")).strip()
        strategy = str(raw_rule.get("strategy", "")).strip().lower()
        if not feature or not strategy:
            continue
        value = raw_rule.get("value")
        normalized[feature] = {
            "feature": feature,
            "strategy": strategy,
            "value": value,
        }
    return list(normalized.values())


def normalize_feature_scaling_rules(raw_rules: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rules, list):
        return []

    normalized: dict[str, dict[str, Any]] = {}
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        feature = str(raw_rule.get("feature", "")).strip()
        method = str(raw_rule.get("method", "")).strip().lower()
        if not feature or not method:
            continue
        normalized[feature] = {
            "feature": feature,
            "method": method,
        }
    return list(normalized.values())


def normalize_feature_transform_rules(raw_rules: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rules, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        feature = str(raw_rule.get("feature", "")).strip()
        kind = str(raw_rule.get("kind", "")).strip().lower()
        value = raw_rule.get("value")
        if not feature or not kind:
            continue
        identity = (feature, kind, stringify_rule_value(value))
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(
            {
                "feature": feature,
                "kind": kind,
                "value": value,
            }
        )
    return normalized


def normalize_derived_feature_rules(raw_rules: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rules, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        kind = str(raw_rule.get("kind", "")).strip().lower()
        source = str(raw_rule.get("source", "")).strip()
        output = str(raw_rule.get("output", "")).strip()
        if not kind or not source:
            continue
        output_name = output or f"{source}__{kind}"
        identity = (kind, source, output_name)
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(
            {
                "kind": kind,
                "source": source,
                "output": output_name,
            }
        )
    return normalized


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def remove_feature_from_keys(overrides: dict[str, Any], feature: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        overrides[key] = [column for column in overrides[key] if column != feature]


def drop_rules_for_feature(overrides: dict[str, Any], feature: str) -> None:
    overrides["feature_imputation_rules"] = [
        rule for rule in overrides["feature_imputation_rules"] if rule.get("feature") != feature
    ]
    overrides["feature_scaling_rules"] = [
        rule for rule in overrides["feature_scaling_rules"] if rule.get("feature") != feature
    ]
    overrides["feature_transform_rules"] = [
        rule for rule in overrides["feature_transform_rules"] if rule.get("feature") != feature
    ]


def resolve_preprocessing_override_conflicts(overrides: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {key: ordered_unique(list(overrides.get(key, []))) for key in PREPROCESSING_OVERRIDE_KEYS}
    for key in PREPROCESSING_RULE_KEYS:
        resolved[key] = list(overrides.get(key, []))

    keep_features = set(resolved["keep_features"])
    if keep_features:
        resolved["drop_features"] = [
            column for column in resolved["drop_features"] if column not in keep_features
        ]
        resolved["force_identifier_features"] = [
            column for column in resolved["force_identifier_features"] if column not in keep_features
        ]

    for feature in list(resolved["drop_features"]):
        remove_feature_from_keys(
            resolved,
            feature,
            (
                "keep_features",
                "force_numeric_features",
                "force_categorical_features",
                "force_date_features",
                "force_identifier_features",
                "force_one_hot_features",
                "force_ordinal_features",
            ),
        )
        drop_rules_for_feature(resolved, feature)

    for feature in list(resolved["force_identifier_features"]):
        remove_feature_from_keys(
            resolved,
            feature,
            (
                "keep_features",
                "force_numeric_features",
                "force_categorical_features",
                "force_date_features",
                "force_one_hot_features",
                "force_ordinal_features",
            ),
        )
        drop_rules_for_feature(resolved, feature)

    for feature in list(resolved["force_one_hot_features"]):
        remove_feature_from_keys(
            resolved,
            feature,
            (
                "drop_features",
                "force_numeric_features",
                "force_date_features",
                "force_identifier_features",
                "force_ordinal_features",
            ),
        )
        resolved["feature_scaling_rules"] = [
            rule for rule in resolved["feature_scaling_rules"] if rule.get("feature") != feature
        ]
        if feature not in resolved["force_categorical_features"]:
            resolved["force_categorical_features"].append(feature)

    for feature in list(resolved["force_ordinal_features"]):
        remove_feature_from_keys(
            resolved,
            feature,
            (
                "drop_features",
                "force_numeric_features",
                "force_date_features",
                "force_identifier_features",
                "force_one_hot_features",
            ),
        )
        resolved["feature_scaling_rules"] = [
            rule for rule in resolved["feature_scaling_rules"] if rule.get("feature") != feature
        ]
        if feature not in resolved["force_categorical_features"]:
            resolved["force_categorical_features"].append(feature)

    for feature in list(resolved["force_categorical_features"]):
        remove_feature_from_keys(
            resolved,
            feature,
            (
                "drop_features",
                "force_numeric_features",
                "force_date_features",
                "force_identifier_features",
            ),
        )
        resolved["feature_scaling_rules"] = [
            rule for rule in resolved["feature_scaling_rules"] if rule.get("feature") != feature
        ]

    for feature in list(resolved["force_numeric_features"]):
        remove_feature_from_keys(
            resolved,
            feature,
            (
                "drop_features",
                "force_categorical_features",
                "force_date_features",
                "force_identifier_features",
                "force_one_hot_features",
                "force_ordinal_features",
            ),
        )

    for feature in list(resolved["force_date_features"]):
        remove_feature_from_keys(
            resolved,
            feature,
            (
                "drop_features",
                "force_numeric_features",
                "force_categorical_features",
                "force_identifier_features",
                "force_one_hot_features",
                "force_ordinal_features",
            ),
        )
        resolved["feature_scaling_rules"] = [
            rule for rule in resolved["feature_scaling_rules"] if rule.get("feature") != feature
        ]

    return {
        **{key: ordered_unique(resolved[key]) for key in PREPROCESSING_OVERRIDE_KEYS},
        "feature_imputation_rules": normalize_feature_imputation_rules(resolved["feature_imputation_rules"]),
        "feature_scaling_rules": normalize_feature_scaling_rules(resolved["feature_scaling_rules"]),
        "feature_transform_rules": normalize_feature_transform_rules(resolved["feature_transform_rules"]),
        "derived_feature_rules": normalize_derived_feature_rules(resolved["derived_feature_rules"]),
    }


def merge_preprocessing_overrides(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = normalize_preprocessing_overrides(existing)
    updates = normalize_preprocessing_overrides(incoming)

    def add_feature(key: str, feature: str) -> None:
        if feature not in merged[key]:
            merged[key].append(feature)

    for feature in updates["drop_features"]:
        remove_feature_from_keys(
            merged,
            feature,
            (
                "keep_features",
                "force_numeric_features",
                "force_categorical_features",
                "force_date_features",
                "force_identifier_features",
                "force_one_hot_features",
                "force_ordinal_features",
            ),
        )
        add_feature("drop_features", feature)

    for feature in updates["keep_features"]:
        remove_feature_from_keys(merged, feature, ("drop_features", "force_identifier_features"))
        add_feature("keep_features", feature)

    for feature in updates["force_identifier_features"]:
        remove_feature_from_keys(
            merged,
            feature,
            (
                "keep_features",
                "drop_features",
                "force_numeric_features",
                "force_categorical_features",
                "force_date_features",
                "force_one_hot_features",
                "force_ordinal_features",
            ),
        )
        add_feature("force_identifier_features", feature)

    for feature in updates["force_numeric_features"]:
        remove_feature_from_keys(
            merged,
            feature,
            (
                "drop_features",
                "force_categorical_features",
                "force_date_features",
                "force_identifier_features",
                "force_one_hot_features",
                "force_ordinal_features",
            ),
        )
        add_feature("force_numeric_features", feature)

    for feature in updates["force_date_features"]:
        remove_feature_from_keys(
            merged,
            feature,
            (
                "drop_features",
                "force_numeric_features",
                "force_categorical_features",
                "force_identifier_features",
                "force_one_hot_features",
                "force_ordinal_features",
            ),
        )
        add_feature("force_date_features", feature)

    for feature in updates["force_categorical_features"]:
        remove_feature_from_keys(
            merged,
            feature,
            (
                "drop_features",
                "force_numeric_features",
                "force_date_features",
                "force_identifier_features",
            ),
        )
        add_feature("force_categorical_features", feature)

    for feature in updates["force_one_hot_features"]:
        remove_feature_from_keys(
            merged,
            feature,
            (
                "drop_features",
                "force_numeric_features",
                "force_date_features",
                "force_identifier_features",
                "force_ordinal_features",
            ),
        )
        add_feature("force_categorical_features", feature)
        add_feature("force_one_hot_features", feature)

    for feature in updates["force_ordinal_features"]:
        remove_feature_from_keys(
            merged,
            feature,
            (
                "drop_features",
                "force_numeric_features",
                "force_date_features",
                "force_identifier_features",
                "force_one_hot_features",
            ),
        )
        add_feature("force_categorical_features", feature)
        add_feature("force_ordinal_features", feature)

    imputation_by_feature = {rule["feature"]: rule for rule in merged["feature_imputation_rules"]}
    for rule in updates["feature_imputation_rules"]:
        imputation_by_feature[rule["feature"]] = rule
    merged["feature_imputation_rules"] = list(imputation_by_feature.values())

    scaling_by_feature = {rule["feature"]: rule for rule in merged["feature_scaling_rules"]}
    for rule in updates["feature_scaling_rules"]:
        scaling_by_feature[rule["feature"]] = rule
    merged["feature_scaling_rules"] = list(scaling_by_feature.values())

    existing_transforms = {
        (rule["feature"], rule["kind"], stringify_rule_value(rule.get("value"))): rule
        for rule in merged["feature_transform_rules"]
    }
    for rule in updates["feature_transform_rules"]:
        identity = (rule["feature"], rule["kind"], stringify_rule_value(rule.get("value")))
        existing_transforms[identity] = rule
    merged["feature_transform_rules"] = list(existing_transforms.values())

    existing_derived = {
        (rule["kind"], rule["source"], rule["output"]): rule
        for rule in merged["derived_feature_rules"]
    }
    for rule in updates["derived_feature_rules"]:
        identity = (rule["kind"], rule["source"], rule["output"])
        existing_derived[identity] = rule
    merged["derived_feature_rules"] = list(existing_derived.values())

    return resolve_preprocessing_override_conflicts(merged)


def extract_preprocessing_overrides_from_feedback(feedback: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(feedback, dict):
        return normalize_preprocessing_overrides(None)
    policy_metadata = feedback.get("policy_metadata", {})
    if not isinstance(policy_metadata, dict):
        return normalize_preprocessing_overrides(None)
    return normalize_preprocessing_overrides(policy_metadata.get("preprocessing_overrides"))


def dedupe_feature_names(values) -> list[str]:  # noqa: ANN001
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


class DatasetPreparer(BaseEstimator, TransformerMixin):
    def __init__(self, plan: PreprocessingPlan, overrides: dict[str, Any] | None = None) -> None:
        self.plan = plan
        self.overrides = overrides
        self.output_columns_: list[str] = []
        self.frequency_count_maps_: dict[str, dict[Any, float]] = {}

    def _normalized_overrides(self) -> dict[str, Any]:
        return normalize_preprocessing_overrides(self.overrides)

    def fit(self, X, y=None):  # noqa: ANN001
        frame = self._as_frame(X)
        self._fit_frequency_count_maps(frame)
        prepared = self._transform_frame(frame)
        self.output_columns_ = list(prepared.columns)
        return self

    def transform(self, X):  # noqa: ANN001
        frame = self._as_frame(X)
        prepared = self._transform_frame(frame)
        if self.output_columns_:
            for column in self.output_columns_:
                if column not in prepared.columns:
                    prepared[column] = np.nan
            prepared = prepared[self.output_columns_]
        return prepared

    def _as_frame(self, X) -> pd.DataFrame:  # noqa: ANN001
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    def _fit_frequency_count_maps(self, frame: pd.DataFrame) -> None:
        overrides = self._normalized_overrides()
        self.frequency_count_maps_ = {}
        for rule in overrides["derived_feature_rules"]:
            if rule.get("kind") != "frequency_count":
                continue
            source = rule["source"]
            if source not in frame.columns:
                self.frequency_count_maps_[source] = {}
                continue
            counts = frame[source].value_counts(dropna=False)
            self.frequency_count_maps_[source] = {key: float(value) for key, value in counts.items()}

    def _apply_numeric_transform_rule(self, series: pd.Series, rule: dict[str, Any]) -> pd.Series:
        kind = rule["kind"]
        numeric = pd.to_numeric(series, errors="coerce")
        value = rule.get("value")
        if kind == "log":
            numeric = numeric.where(numeric > 0)
            return np.log(numeric)
        if kind == "log1p":
            numeric = numeric.where(numeric >= -1)
            return np.log1p(numeric)
        if kind == "sqrt":
            numeric = numeric.where(numeric >= 0)
            return np.sqrt(numeric)
        if kind == "square":
            return np.square(numeric)
        if kind == "abs":
            return np.abs(numeric)
        if kind == "multiply":
            return numeric * float(value)
        if kind == "divide":
            return numeric / float(value)
        if kind == "add":
            return numeric + float(value)
        if kind == "subtract":
            return numeric - float(value)
        return numeric

    def _transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        overrides = self._normalized_overrides()
        transformed = frame.copy()

        for rule in overrides["feature_transform_rules"]:
            feature = rule["feature"]
            if feature not in transformed.columns:
                continue
            transformed[feature] = self._apply_numeric_transform_rule(transformed[feature], rule)

        for rule in overrides["derived_feature_rules"]:
            if rule.get("kind") != "frequency_count":
                continue
            source = rule["source"]
            output = rule["output"]
            if source not in transformed.columns:
                transformed[output] = np.nan
                continue
            mapping = self.frequency_count_maps_.get(source, {})
            transformed[output] = transformed[source].map(mapping).fillna(0.0).astype(float)

        for column in self.plan.date_features:
            if column not in transformed.columns:
                for part in DATE_PARTS:
                    transformed[f"{column}__{part}"] = np.nan
                continue

            parsed = pd.to_datetime(transformed[column], errors="coerce", format="mixed")
            transformed[f"{column}__year"] = parsed.dt.year
            transformed[f"{column}__month"] = parsed.dt.month
            transformed[f"{column}__day"] = parsed.dt.day
            transformed[f"{column}__dayofweek"] = parsed.dt.dayofweek

        drop_columns = [column for column in self.plan.dropped_features if column in transformed.columns]
        if drop_columns:
            transformed = transformed.drop(columns=drop_columns)

        expected_columns = [*self.plan.numeric_features, *self.plan.categorical_features]
        for column in expected_columns:
            if column not in transformed.columns:
                transformed[column] = np.nan
        return transformed[expected_columns]


def build_preprocessing_plan(
    profile: DataProfile,
    option: str,
    overrides: dict[str, Any] | None = None,
) -> PreprocessingPlan:
    normalized_overrides = normalize_preprocessing_overrides(overrides)
    force_categorical = set(normalized_overrides["force_categorical_features"])
    force_numeric = set(normalized_overrides["force_numeric_features"])
    force_date = set(normalized_overrides["force_date_features"])
    force_identifier = set(normalized_overrides["force_identifier_features"])
    force_ordinal = set(normalized_overrides["force_ordinal_features"])
    explicit_drop = set(normalized_overrides["drop_features"])
    explicit_keep = set(normalized_overrides["keep_features"])
    derived_numeric_features = [
        rule["output"]
        for rule in normalized_overrides["derived_feature_rules"]
        if rule.get("kind") in SUPPORTED_DERIVED_FEATURE_KINDS
    ]

    dropped_features = set(profile.constant_features)
    date_features = set(profile.date_like_features)

    if option == "custom":
        dropped_features.update(profile.likely_identifier_features)
        dropped_features.update(profile.high_cardinality_categorical_features)

    dropped_features.update(force_identifier)
    dropped_features.update(explicit_drop)
    dropped_features.difference_update(explicit_keep)

    date_features.update(force_date)
    date_features.difference_update(force_identifier)
    date_features.difference_update(explicit_drop)
    date_features.update(column for column in force_date if column in explicit_keep)

    numeric_base = set(profile.numeric_features)
    categorical_base = set(profile.categorical_features)

    numeric_base.update(force_numeric)
    categorical_base.update(force_categorical)

    for column in force_categorical | force_date | force_identifier | explicit_drop:
        numeric_base.discard(column)
    for column in force_numeric | force_date | force_identifier | explicit_drop:
        categorical_base.discard(column)

    raw_date_features = tuple(sorted(date_features))
    dropped_features.update(raw_date_features)

    raw_numeric_features = [
        column
        for column in sorted(numeric_base)
        if column not in dropped_features and column not in date_features
    ]
    numeric_features = list(raw_numeric_features)
    numeric_features.extend(
        feature for feature in derived_numeric_features if feature not in numeric_features
    )
    numeric_features.extend(
        f"{column}__{part}"
        for column in raw_date_features
        for part in DATE_PARTS
    )
    raw_categorical_features = [
        column
        for column in sorted(categorical_base)
        if column not in dropped_features and column not in date_features
    ]
    ordinal_categorical_features = [
        column for column in raw_categorical_features if column in force_ordinal
    ]
    one_hot_categorical_features = [
        column for column in raw_categorical_features if column not in force_ordinal
    ]
    categorical_features = list(raw_categorical_features)

    return PreprocessingPlan(
        option=option,
        dropped_features=tuple(sorted(dropped_features)),
        date_features=raw_date_features,
        raw_numeric_features=tuple(raw_numeric_features),
        derived_numeric_features=tuple(derived_numeric_features),
        raw_categorical_features=tuple(raw_categorical_features),
        one_hot_categorical_features=tuple(one_hot_categorical_features),
        ordinal_categorical_features=tuple(ordinal_categorical_features),
        numeric_features=tuple(numeric_features),
        categorical_features=tuple(categorical_features),
    )


def build_preprocessor(
    profile: DataProfile,
    option: str,
    overrides: dict[str, Any] | None = None,
) -> Pipeline:
    normalized_overrides = normalize_preprocessing_overrides(overrides)
    plan = build_preprocessing_plan(profile, option, overrides=normalized_overrides)
    transformers: list[tuple[str, Pipeline, list[str] | tuple[str, ...]]] = []

    transformers.extend(build_numeric_transformers(profile, plan, normalized_overrides))
    transformers.extend(build_categorical_transformers(profile, plan, normalized_overrides))

    if not transformers:
        raise ValueError("The selected preprocessing policy removed every usable input feature.")

    return Pipeline(
        steps=[
            ("prepare", DatasetPreparer(plan, normalized_overrides)),
            (
                "encode",
                ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                ),
            ),
        ]
    )


def index_rules_by_feature(rules: list[dict[str, Any]], key: str = "feature") -> dict[str, dict[str, Any]]:
    return {
        str(rule.get(key, "")).strip(): rule
        for rule in rules
        if isinstance(rule, dict) and str(rule.get(key, "")).strip()
    }


def default_numeric_imputation_rule(
    profile: DataProfile,
    plan: PreprocessingPlan,
    feature: str,
) -> dict[str, Any] | None:
    if feature in profile.missing_by_feature:
        return {"feature": feature, "strategy": "median", "value": None}
    for date_feature in plan.date_features:
        prefix = f"{date_feature}__"
        if feature.startswith(prefix) and date_feature in profile.date_parse_failure_by_feature:
            return {"feature": feature, "strategy": "median", "value": None}
    return None


def default_categorical_imputation_rule(
    profile: DataProfile,
    feature: str,
) -> dict[str, Any] | None:
    if feature in profile.missing_by_feature:
        return {"feature": feature, "strategy": "most_frequent", "value": None}
    return None


def default_numeric_scaling_rule(plan: PreprocessingPlan, feature: str) -> dict[str, Any] | None:
    if feature in plan.derived_numeric_features:
        pass
    if plan.option in {"auto_tabular_preprocessing", "custom"}:
        return {"feature": feature, "method": "standard"}
    return None


def imputer_from_rule(rule: dict[str, Any] | None) -> tuple[str, object] | None:
    if not rule:
        return None
    strategy = rule.get("strategy")
    if strategy == "constant":
        return ("imputer", SimpleImputer(strategy="constant", fill_value=rule.get("value")))
    return ("imputer", SimpleImputer(strategy=strategy))


def scaler_from_rule(rule: dict[str, Any] | None) -> tuple[str, object] | None:
    if not rule:
        return None
    method = rule.get("method")
    if method == "standard":
        return ("scaler", StandardScaler())
    if method == "minmax":
        return ("scaler", MinMaxScaler())
    if method == "robust":
        return ("scaler", RobustScaler())
    return None


def build_numeric_transformers(
    profile: DataProfile,
    plan: PreprocessingPlan,
    overrides: dict[str, Any],
) -> list[tuple[str, Pipeline, list[str]]]:
    numeric_features = list(plan.numeric_features)
    if not numeric_features:
        return []

    explicit_imputation = index_rules_by_feature(overrides["feature_imputation_rules"])
    explicit_scaling = index_rules_by_feature(overrides["feature_scaling_rules"])
    grouped_columns: dict[tuple[str, str], list[str]] = {}
    grouped_steps: dict[tuple[str, str], list[tuple[str, object]]] = {}

    for feature in numeric_features:
        imputation_rule = explicit_imputation.get(feature) or default_numeric_imputation_rule(profile, plan, feature)
        scaling_rule = explicit_scaling.get(feature) or default_numeric_scaling_rule(plan, feature)
        imputer_step = imputer_from_rule(imputation_rule)
        scaler_step = scaler_from_rule(scaling_rule)
        signature = (
            f"{imputation_rule.get('strategy')}:{stringify_rule_value(imputation_rule.get('value'))}" if imputation_rule else "none",
            scaling_rule.get("method") if scaling_rule else "none",
        )
        steps: list[tuple[str, object]] = []
        if imputer_step:
            steps.append(imputer_step)
        if scaler_step:
            steps.append(scaler_step)
        if not steps:
            steps = [("identity", "passthrough")]
        grouped_columns.setdefault(signature, []).append(feature)
        grouped_steps[signature] = steps

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    for index, signature in enumerate(grouped_columns):
        transformers.append(
            (
                f"numeric_{index}",
                Pipeline(steps=grouped_steps[signature]),
                grouped_columns[signature],
            )
        )
    return transformers


def default_categorical_encoding(plan: PreprocessingPlan, feature: str) -> str:
    if feature in plan.ordinal_categorical_features:
        return "ordinal"
    return "one_hot"


def build_categorical_transformers(
    profile: DataProfile,
    plan: PreprocessingPlan,
    overrides: dict[str, Any],
) -> list[tuple[str, Pipeline, list[str]]]:
    categorical_features = list(plan.raw_categorical_features)
    if not categorical_features:
        return []

    explicit_imputation = index_rules_by_feature(overrides["feature_imputation_rules"])
    encoding_by_feature: dict[str, str] = {}
    for feature in plan.one_hot_categorical_features:
        encoding_by_feature[feature] = "one_hot"
    for feature in plan.ordinal_categorical_features:
        encoding_by_feature[feature] = "ordinal"

    grouped_columns: dict[tuple[str, str, str], list[str]] = {}
    grouped_steps: dict[tuple[str, str, str], list[tuple[str, object]]] = {}

    for feature in categorical_features:
        imputation_rule = explicit_imputation.get(feature) or default_categorical_imputation_rule(profile, feature)
        encoding = encoding_by_feature.get(feature, default_categorical_encoding(plan, feature))
        imputer_step = imputer_from_rule(imputation_rule)
        steps: list[tuple[str, object]] = []
        if imputer_step:
            steps.append(imputer_step)
        if encoding == "ordinal":
            steps.append(
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                )
            )
        else:
            steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))
        signature = (
            encoding,
            imputation_rule.get("strategy") if imputation_rule else "none",
            stringify_rule_value(imputation_rule.get("value")) if imputation_rule else "none",
        )
        grouped_columns.setdefault(signature, []).append(feature)
        grouped_steps[signature] = steps

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    for index, signature in enumerate(grouped_columns):
        transformers.append(
            (
                f"categorical_{index}",
                Pipeline(steps=grouped_steps[signature]),
                grouped_columns[signature],
            )
        )
    return transformers


def has_numeric_imputation_need(profile: DataProfile, plan: PreprocessingPlan) -> bool:
    has_numeric_missing = bool(numeric_missing_columns(profile, plan))
    has_date_parse_failures = bool(date_parse_failure_columns(profile, plan))
    return has_numeric_missing or has_date_parse_failures


def should_scale_numeric_features(plan: PreprocessingPlan) -> bool:
    return plan.option in {"auto_tabular_preprocessing", "custom"} and bool(plan.numeric_features)


def numeric_missing_columns(profile: DataProfile, plan: PreprocessingPlan) -> list[str]:
    surviving_numeric = set(plan.raw_numeric_features)
    return [
        column
        for column in plan.raw_numeric_features
        if column in surviving_numeric and column in profile.missing_by_feature
    ]


def date_parse_failure_columns(profile: DataProfile, plan: PreprocessingPlan) -> list[str]:
    return [
        column
        for column in plan.date_features
        if column in profile.date_parse_failure_by_feature
    ]


def categorical_missing_columns(
    profile: DataProfile,
    plan: PreprocessingPlan,
    columns: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    candidate_columns = list(columns or plan.raw_categorical_features)
    return [column for column in candidate_columns if column in profile.missing_by_feature]


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def format_numeric_value(value: float | None) -> str:
    if value is None:
        return "unknown"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def summarize_facts(facts: list[str], limit: int = 3) -> str:
    if not facts:
        return ""
    if len(facts) <= limit:
        return "; ".join(facts)
    return "; ".join(facts[:limit]) + f"; plus {len(facts) - limit} more checked feature(s)"


def facts_or_default(facts: list[str], fallback: str) -> str:
    summarized = summarize_facts(facts)
    return summarized or fallback


def constant_feature_facts(profile: DataProfile, columns: list[str]) -> list[str]:
    facts: list[str] = []
    for column in columns:
        distinct = profile.distinct_by_feature.get(column, 1)
        facts.append(f"`{column}` has `{distinct}` distinct value")
    return facts


def identifier_feature_facts(profile: DataProfile, columns: list[str]) -> list[str]:
    facts: list[str] = []
    rows = max(profile.rows, 1)
    for column in columns:
        distinct = profile.distinct_by_feature.get(column, profile.categorical_cardinality.get(column))
        if distinct is None:
            facts.append(f"`{column}` was flagged as identifier-like")
            continue
        uniqueness = distinct / float(rows)
        facts.append(
            f"`{column}` has `{distinct}` distinct values across `{rows}` rows ({format_percent(uniqueness)} row uniqueness)"
        )
    return facts


def high_cardinality_feature_facts(profile: DataProfile, columns: list[str]) -> list[str]:
    facts: list[str] = []
    rows = max(profile.rows, 1)
    for column in columns:
        distinct = profile.high_cardinality_categorical_features.get(
            column,
            profile.distinct_by_feature.get(column),
        )
        if distinct is None:
            facts.append(f"`{column}` was flagged as high-cardinality")
            continue
        facts.append(
            f"`{column}` has `{distinct}` distinct values across `{rows}` rows ({format_percent(distinct / float(rows))} row uniqueness)"
        )
    return facts


def date_feature_facts(
    profile: DataProfile,
    columns: list[str],
    overrides: dict[str, list[str]],
) -> list[str]:
    facts: list[str] = []
    forced_date = set(overrides["force_date_features"])
    for column in columns:
        failure_fraction = profile.date_parse_failure_by_feature.get(column, 0.0)
        if column in forced_date and column not in profile.date_like_features:
            facts.append(
                f"`{column}` was explicitly reclassified as date-like and currently shows `{format_percent(failure_fraction)}` parse failures"
            )
        else:
            facts.append(
                f"`{column}` was detected as date-like and currently shows `{format_percent(failure_fraction)}` parse failures"
            )
    return facts


def numeric_range_facts(profile: DataProfile, columns: list[str]) -> list[str]:
    facts: list[str] = []
    for column in columns:
        bounds = profile.numeric_range_by_feature.get(column, {})
        minimum = bounds.get("min")
        maximum = bounds.get("max")
        if minimum is None or maximum is None:
            facts.append(f"`{column}` is numeric but only contains missing values")
            continue
        facts.append(
            f"`{column}` spans `{format_numeric_value(minimum)}` to `{format_numeric_value(maximum)}`"
        )
    return facts


def categorical_cardinality_facts(profile: DataProfile, columns: list[str]) -> list[str]:
    facts: list[str] = []
    for column in columns:
        distinct = profile.distinct_by_feature.get(column, profile.categorical_cardinality.get(column))
        missing_fraction = profile.missing_by_feature.get(column)
        if distinct is None:
            facts.append(f"`{column}` remains categorical after inspection")
            continue
        fact = f"`{column}` has `{distinct}` distinct values"
        if missing_fraction is not None:
            fact += f" with `{format_percent(missing_fraction)}` missing values"
        facts.append(fact)
    return facts


def missing_fraction_facts(profile: DataProfile, columns: list[str]) -> list[str]:
    facts: list[str] = []
    for column in columns:
        missing_fraction = profile.missing_by_feature.get(column)
        if missing_fraction is None:
            continue
        facts.append(f"`{column}` has `{format_percent(missing_fraction)}` missing values")
    return facts


def checked_feature_snapshot_facts(profile: DataProfile, columns: list[str]) -> list[str]:
    facts: list[str] = []
    rows = max(profile.rows, 1)
    for column in columns:
        if column in profile.constant_features:
            distinct = profile.distinct_by_feature.get(column, 1)
            facts.append(f"`{column}` has `{distinct}` distinct value")
            continue
        if column in profile.likely_identifier_features:
            distinct = profile.distinct_by_feature.get(column, profile.categorical_cardinality.get(column))
            if distinct is None:
                facts.append(f"`{column}` was flagged as identifier-like")
            else:
                uniqueness = distinct / float(rows)
                facts.append(
                    f"`{column}` has `{distinct}` distinct values across `{rows}` rows ({format_percent(uniqueness)} row uniqueness)"
                )
            continue
        if column in profile.high_cardinality_categorical_features:
            distinct = profile.high_cardinality_categorical_features.get(column)
            if distinct is None:
                facts.append(f"`{column}` was flagged as high-cardinality")
            else:
                facts.append(
                    f"`{column}` has `{distinct}` distinct values across `{rows}` rows ({format_percent(distinct / float(rows))} row uniqueness)"
                )
            continue

        distinct = profile.distinct_by_feature.get(column, profile.categorical_cardinality.get(column))
        if distinct is not None:
            missing_fraction = profile.missing_by_feature.get(column)
            fact = f"`{column}` has `{distinct}` distinct values"
            if missing_fraction is not None:
                fact += f" with `{format_percent(missing_fraction)}` missing values"
            facts.append(fact)
            continue

        bounds = profile.numeric_range_by_feature.get(column, {})
        minimum = bounds.get("min")
        maximum = bounds.get("max")
        if minimum is None or maximum is None:
            facts.append(f"`{column}` is present in the inspected feature set")
        else:
            facts.append(
                f"`{column}` spans `{format_numeric_value(minimum)}` to `{format_numeric_value(maximum)}`"
            )
    return facts


def imputation_rules_by_feature(overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return index_rules_by_feature(overrides.get("feature_imputation_rules", []))


def scaling_rules_by_feature(overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return index_rules_by_feature(overrides.get("feature_scaling_rules", []))


def transform_rules_by_feature(overrides: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rule in overrides.get("feature_transform_rules", []):
        feature = rule.get("feature")
        if not feature:
            continue
        grouped.setdefault(feature, []).append(rule)
    return grouped


def derived_rules_by_source(overrides: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rule in overrides.get("derived_feature_rules", []):
        source = rule.get("source")
        if not source:
            continue
        grouped.setdefault(source, []).append(rule)
    return grouped


def describe_imputation_rule(rule: dict[str, Any]) -> str:
    strategy = rule["strategy"]
    if strategy == "mean":
        return "mean imputation"
    if strategy == "median":
        return "median imputation"
    if strategy == "most_frequent":
        return "most-frequent imputation"
    if strategy == "constant":
        return f"constant-value imputation with `{stringify_rule_value(rule.get('value'))}`"
    return f"{strategy} imputation"


def describe_scaling_method(method: str) -> str:
    if method == "standard":
        return "`StandardScaler`"
    if method == "minmax":
        return "`MinMaxScaler`"
    if method == "robust":
        return "`RobustScaler`"
    return "no scaling"


def describe_transform_rule(rule: dict[str, Any]) -> str:
    kind = rule["kind"]
    if kind == "log":
        return "a natural-log transformation"
    if kind == "log1p":
        return "a log1p transformation"
    if kind == "sqrt":
        return "a square-root transformation"
    if kind == "square":
        return "a square transformation"
    if kind == "abs":
        return "an absolute-value transformation"
    if kind == "multiply":
        return f"a multiply-by-`{stringify_rule_value(rule.get('value'))}` transformation"
    if kind == "divide":
        return f"a divide-by-`{stringify_rule_value(rule.get('value'))}` transformation"
    if kind == "add":
        return f"an add-`{stringify_rule_value(rule.get('value'))}` transformation"
    if kind == "subtract":
        return f"a subtract-`{stringify_rule_value(rule.get('value'))}` transformation"
    return f"a `{kind}` transformation"


def detect_scope_override(plan: PreprocessingPlan) -> str | None:
    if plan.raw_numeric_features and not plan.raw_categorical_features and not plan.date_features:
        return "numeric_only"
    if plan.raw_categorical_features and not plan.raw_numeric_features and not plan.date_features:
        return "categorical_only"
    if plan.date_features and not plan.raw_numeric_features and not plan.raw_categorical_features:
        return "date_only"

    return None


def detect_selected_feature_subset(
    profile: DataProfile,
    overrides: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    all_features = dedupe_feature_names([*profile.numeric_features, *profile.categorical_features])
    if not all_features:
        return [], []

    selected_features = [feature for feature in overrides["keep_features"] if feature in all_features]
    if not selected_features:
        return [], []

    excluded_features = [feature for feature in all_features if feature not in selected_features]
    explicit_exclusions = set(overrides["drop_features"]) | set(overrides["force_identifier_features"])
    if excluded_features and not set(excluded_features).issubset(explicit_exclusions):
        return [], []
    return selected_features, excluded_features


def describe_preprocessing_execution_steps(
    profile: DataProfile,
    option: str,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    normalized_overrides = normalize_preprocessing_overrides(overrides)
    plan = build_preprocessing_plan(profile, option, overrides=normalized_overrides)
    steps: list[dict[str, str]] = []
    numeric_missing_columns_list = numeric_missing_columns(profile, plan)
    date_parse_failure_columns_list = date_parse_failure_columns(profile, plan)
    categorical_missing_columns_list = categorical_missing_columns(profile, plan)
    explicit_drop_columns = set(normalized_overrides["drop_features"]) | set(normalized_overrides["force_identifier_features"])

    override_steps, consumed = describe_preprocessing_override_steps(profile, plan, normalized_overrides)
    consumed_drop_columns = consumed["drop_features"]
    steps.extend(override_steps)

    constant_drop_columns = [
        column
        for column in profile.constant_features
        if column in plan.dropped_features and column not in explicit_drop_columns
    ]
    if constant_drop_columns:
        steps.append(
            {
                "step": f"Drop constant feature(s): {', '.join(f'`{column}`' for column in constant_drop_columns)}.",
                "why": summarize_facts(constant_feature_facts(profile, constant_drop_columns))
                + ". Constant columns cannot create useful variation for the downstream models.",
            }
        )

    identifier_drop_columns = [
        column
        for column in profile.likely_identifier_features
        if column in plan.dropped_features and column not in explicit_drop_columns
    ]
    if identifier_drop_columns:
        steps.append(
            {
                "step": f"Exclude identifier-like feature(s): {', '.join(f'`{column}`' for column in identifier_drop_columns)}.",
                "why": summarize_facts(identifier_feature_facts(profile, identifier_drop_columns))
                + ". These checked values behave more like row identifiers than stable predictive signal.",
            }
        )

    high_cardinality_drop_columns = [
        column
        for column in profile.high_cardinality_categorical_features
        if column in plan.dropped_features and column not in explicit_drop_columns
    ]
    if high_cardinality_drop_columns:
        steps.append(
            {
                "step": "Remove high-cardinality categorical feature(s): "
                + ", ".join(f"`{column}`" for column in high_cardinality_drop_columns)
                + ".",
                "why": summarize_facts(high_cardinality_feature_facts(profile, high_cardinality_drop_columns))
                + ". Keeping them would create a wide sparse encoding with little repeated category support.",
            }
        )

    if plan.date_features:
        steps.append(
            {
                "step": "Parse the date-like fields "
                + ", ".join(f"`{column}`" for column in plan.date_features)
                + " into numeric calendar parts: year, month, day, and day of week.",
                "why": summarize_facts(date_feature_facts(profile, list(plan.date_features), normalized_overrides))
                + ". Converting the checked date-like content into calendar parts keeps chronology in a model-friendly numeric form.",
            }
        )

    numeric_imputation_needed = has_numeric_imputation_need(profile, plan)
    if plan.numeric_features:
        explicit_numeric_imputation_features = consumed["numeric_imputation_features"]
        explicit_scaled_features = consumed["scaled_numeric_features"]
        if option == "minimal_cleanup":
            if numeric_imputation_needed:
                column_reasons: list[str] = []
                remaining_numeric_imputation_columns = [
                    column for column in numeric_missing_columns_list if column not in explicit_numeric_imputation_features
                ]
                if remaining_numeric_imputation_columns:
                    column_reasons.append(
                        "numeric gaps in "
                        + ", ".join(f"`{column}`" for column in remaining_numeric_imputation_columns)
                    )
                remaining_date_parse_failure_columns = [
                    column for column in date_parse_failure_columns_list if column not in explicit_numeric_imputation_features
                ]
                if remaining_date_parse_failure_columns:
                    column_reasons.append(
                        "date parsing gaps in "
                        + ", ".join(f"`{column}`" for column in remaining_date_parse_failure_columns)
                    )
                if column_reasons:
                    steps.append(
                        {
                            "step": "Apply median imputation only to the incomplete numeric or date-derived features.",
                            "why": summarize_facts(column_reasons)
                            + ". Median imputation is limited to the checked incomplete numeric branch so the cleanup stays lightweight.",
                        }
                    )
            else:
                numeric_facts = numeric_range_facts(
                    profile,
                    [feature for feature in plan.raw_numeric_features if feature not in explicit_scaled_features]
                    + list(plan.derived_numeric_features),
                )
                if plan.date_features:
                    numeric_facts.append(
                        "date-derived calendar parts were added from "
                        + ", ".join(f"`{column}`" for column in plan.date_features)
                    )
                if numeric_facts:
                    steps.append(
                        {
                            "step": "Keep the numeric and date-derived features as they are without extra scaling.",
                            "why": summarize_facts(numeric_facts)
                            + ". The checked numeric branch is already complete, so extra transforms would add complexity without a stronger data-driven need.",
                        }
                    )
        else:
            if numeric_imputation_needed:
                repair_targets: list[str] = []
                remaining_numeric_imputation_columns = [
                    column for column in numeric_missing_columns_list if column not in explicit_numeric_imputation_features
                ]
                remaining_date_parse_failure_columns = [
                    column for column in date_parse_failure_columns_list if column not in explicit_numeric_imputation_features
                ]
                repair_facts = missing_fraction_facts(profile, remaining_numeric_imputation_columns)
                if remaining_numeric_imputation_columns:
                    repair_targets.append(", ".join(f"`{column}`" for column in remaining_numeric_imputation_columns))
                if remaining_date_parse_failure_columns:
                    repair_targets.append(
                        "date-derived parts from "
                        + ", ".join(f"`{column}`" for column in remaining_date_parse_failure_columns)
                    )
                    repair_facts.extend(
                        f"`{column}` has `{format_percent(profile.date_parse_failure_by_feature[column])}` date parse failures"
                        for column in remaining_date_parse_failure_columns
                    )
                if repair_targets:
                    steps.append(
                        {
                            "step": "Apply median imputation only to the incomplete numeric or date-derived features: "
                            + "; ".join(repair_targets)
                            + ".",
                            "why": summarize_facts(repair_facts)
                            + ". Median imputation repairs only the checked numeric gaps and is less sensitive to outliers than mean imputation.",
                        }
                    )
            if should_scale_numeric_features(plan):
                remaining_scaled_features = [
                    feature for feature in plan.raw_numeric_features if feature not in explicit_scaled_features
                ]
                scale_facts = numeric_range_facts(profile, remaining_scaled_features)
                if plan.date_features:
                    scale_facts.append(
                        "the numeric branch also includes calendar parts from "
                        + ", ".join(f"`{column}`" for column in plan.date_features)
                    )
                if remaining_scaled_features or plan.date_features:
                    steps.append(
                        {
                            "step": "Scale the numeric and date-derived features with `StandardScaler` before modeling.",
                            "why": summarize_facts(scale_facts)
                            + ". `StandardScaler` is used so these checked numeric ranges enter the mixed candidate set on a comparable scale.",
                        }
                    )

    if plan.raw_categorical_features:
        explicit_categorical_imputation_features = consumed["categorical_imputation_features"]
        if categorical_missing_columns_list:
            remaining_categorical_imputation_columns = [
                column for column in categorical_missing_columns_list if column not in explicit_categorical_imputation_features
            ]
        else:
            remaining_categorical_imputation_columns = []
        if remaining_categorical_imputation_columns:
            steps.append(
                {
                    "step": "Apply most-frequent imputation only to the incomplete categorical features: "
                    + ", ".join(f"`{column}`" for column in remaining_categorical_imputation_columns)
                    + ".",
                    "why": summarize_facts(missing_fraction_facts(profile, remaining_categorical_imputation_columns))
                    + ". Most-frequent imputation is limited to the checked categorical gaps before encoding.",
                }
            )
        remaining_one_hot_features = [
            column
            for column in plan.one_hot_categorical_features
            if column not in normalized_overrides["force_one_hot_features"]
        ]
        if remaining_one_hot_features:
            steps.append(
                {
                    "step": "One-hot encode the remaining categorical features with unknown-category protection.",
                    "why": summarize_facts(
                        categorical_cardinality_facts(profile, remaining_one_hot_features)
                    )
                    + ". Those checked cardinalities are manageable enough for one-hot encoding, and unknown-category protection keeps inference stable.",
                }
            )
        if plan.ordinal_categorical_features:
            steps.append(
                {
                    "step": "Ordinal-encode the categorical features assigned to the ordinal branch.",
                    "why": summarize_facts(
                        categorical_cardinality_facts(profile, list(plan.ordinal_categorical_features))
                    )
                    + ". Ordinal encoding is used here because this custom plan explicitly asked for a compact integer representation for these checked categorical values.",
                }
            )
    else:
        steps.append(
            {
                "step": "Skip categorical encoding because no categorical features remain after the preprocessing plan.",
                "why": f"After the checked drops, overrides, and date handling, `{len(plan.raw_categorical_features)}` categorical columns remain in the plan.",
            }
        )

    steps.append(
        {
            "step": "Pass the prepared feature matrix into the downstream feature-selection and modeling stages.",
            "why": "The checked preprocessing plan keeps "
            + f"`{len(plan.numeric_features)}` numeric/date-derived columns, "
            + f"`{len(plan.one_hot_categorical_features)}` one-hot categorical column group(s), and "
            + f"`{len(plan.ordinal_categorical_features)}` ordinal categorical column group(s) before model fitting.",
        }
    )
    return steps


def describe_preprocessing_override_steps(
    profile: DataProfile,
    plan: PreprocessingPlan,
    overrides: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    steps: list[dict[str, str]] = []
    consumed = {
        "drop_features": set(),
        "keep_features": set(),
        "numeric_imputation_features": set(),
        "categorical_imputation_features": set(),
        "scaled_numeric_features": set(),
    }
    selected_subset_features, excluded_subset_features = detect_selected_feature_subset(profile, overrides)
    if selected_subset_features:
        consumed["drop_features"].update(excluded_subset_features)
        consumed["keep_features"].update(selected_subset_features)
        excluded_text = (
            ", ".join(f"`{column}`" for column in excluded_subset_features)
            if excluded_subset_features
            else "no other columns"
        )
        steps.append(
            {
                "step": "Use only the selected input feature subset from the inspected file: "
                + ", ".join(f"`{column}`" for column in selected_subset_features)
                + ".",
                "why": "The custom policy keeps only these checked columns and excludes "
                + excluded_text
                + " from the modeling input.",
            }
        )

    scope_override = detect_scope_override(plan)
    if scope_override == "numeric_only" and not selected_subset_features:
        consumed["drop_features"].update(
            [column for column in overrides["drop_features"] if column in profile.categorical_features]
        )
        non_numeric_text = (
            ", ".join(f"`{column}`" for column in consumed["drop_features"])
            if consumed["drop_features"]
            else "no non-numeric columns"
        )
        steps.append(
            {
                "step": "Use only the raw numeric input features from the inspected file.",
                "why": "The checked numeric inputs are "
                + ", ".join(f"`{column}`" for column in plan.raw_numeric_features)
                + ", while "
                + non_numeric_text
                + " are excluded from this custom preprocessing path.",
            }
        )
    elif scope_override == "categorical_only" and not selected_subset_features:
        consumed["drop_features"].update(
            [column for column in overrides["drop_features"] if column in profile.numeric_features]
        )
        consumed["drop_features"].update(
            [column for column in overrides["drop_features"] if column in profile.date_like_features]
        )
        categorical_only_features = list(plan.raw_categorical_features)
        categorical_text = (
            ", ".join(f"`{column}`" for column in categorical_only_features)
            if categorical_only_features
            else "no non-date categorical columns"
        )
        steps.append(
            {
                "step": "Use only the raw categorical input features from the inspected file.",
                "why": "The checked categorical inputs are "
                + categorical_text
                + ", while numeric and date-like inputs are removed from this custom preprocessing path.",
            }
        )
    elif scope_override == "date_only" and not selected_subset_features:
        consumed["drop_features"].update(
            [column for column in overrides["drop_features"] if column in profile.numeric_features]
        )
        consumed["drop_features"].update(
            [column for column in overrides["drop_features"] if column in profile.categorical_features]
        )
        date_text = (
            ", ".join(f"`{column}`" for column in plan.date_features)
            if plan.date_features
            else "no detected date-like columns"
        )
        steps.append(
            {
                "step": "Use only the date-like input features from the inspected file.",
                "why": "The checked date-like inputs are "
                + date_text
                + ", while other numeric and non-date categorical inputs are removed from this custom preprocessing path.",
            }
        )

    for rule in overrides.get("feature_transform_rules", []):
        feature = rule["feature"]
        steps.append(
            {
                "step": "Apply " + describe_transform_rule(rule) + f" to `{feature}` before downstream modeling.",
                "why": summarize_facts(checked_feature_snapshot_facts(profile, [feature]))
                + ". This custom rule changes how the checked feature values are represented before imputation, scaling, and encoding.",
            }
        )

    for rule in overrides.get("derived_feature_rules", []):
        if rule.get("kind") != "frequency_count":
            continue
        source = rule["source"]
        output = rule["output"]
        steps.append(
            {
                "step": f"Create the derived feature `{output}` from the frequency counts of `{source}`.",
                "why": summarize_facts(checked_feature_snapshot_facts(profile, [source]))
                + ". This custom rule adds an occurrence-frequency signal derived from the checked source feature values.",
            }
        )

    explicit_one_hot = set(overrides["force_one_hot_features"])
    explicit_ordinal = set(overrides["force_ordinal_features"])
    generic_force_categorical = [
        column
        for column in overrides["force_categorical_features"]
        if column not in explicit_one_hot and column not in explicit_ordinal
    ]
    if generic_force_categorical:
        steps.append(
            {
                "step": "Treat "
                + ", ".join(f"`{column}`" for column in generic_force_categorical)
                + " as categorical feature(s) before encoding.",
                "why": facts_or_default(
                    checked_feature_snapshot_facts(profile, generic_force_categorical),
                    "These checked columns were explicitly reclassified as categorical",
                )
                + ". The custom policy therefore routes these checked values through the categorical encoding branch instead of the continuous numeric branch.",
            }
        )
    if overrides["force_one_hot_features"]:
        steps.append(
            {
                "step": "One-hot encode "
                + ", ".join(f"`{column}`" for column in overrides["force_one_hot_features"])
                + " as explicit categorical feature(s).",
                "why": facts_or_default(
                    checked_feature_snapshot_facts(profile, overrides["force_one_hot_features"]),
                    "These checked columns were explicitly selected for one-hot encoding",
                )
                + ". The custom policy therefore routes these checked values through the one-hot branch instead of treating them as continuous numeric inputs.",
            }
        )
    if overrides["force_ordinal_features"]:
        steps.append(
            {
                "step": "Ordinal-encode "
                + ", ".join(f"`{column}`" for column in overrides["force_ordinal_features"])
                + " as explicit categorical feature(s).",
                "why": facts_or_default(
                    checked_feature_snapshot_facts(profile, overrides["force_ordinal_features"]),
                    "These checked columns were explicitly selected for ordinal encoding",
                )
                + ". The custom policy therefore routes these checked values through the ordinal categorical branch instead of the default one-hot branch.",
            }
        )
    if overrides["force_numeric_features"]:
        steps.append(
            {
                "step": "Treat "
                + ", ".join(f"`{column}`" for column in overrides["force_numeric_features"])
                + " as numeric feature(s) before modeling.",
                "why": facts_or_default(
                    checked_feature_snapshot_facts(profile, overrides["force_numeric_features"]),
                    "These checked columns were explicitly reclassified as numeric",
                )
                + ". The custom policy therefore routes these checked values through the numeric branch instead of categorical encoding.",
            }
        )
    if overrides["force_date_features"]:
        steps.append(
            {
                "step": "Treat "
                + ", ".join(f"`{column}`" for column in overrides["force_date_features"])
                + " as date-like field(s) and expand them into calendar parts.",
                "why": summarize_facts(date_feature_facts(profile, overrides["force_date_features"], overrides))
                + ". The custom policy therefore derives chronology-aware calendar parts from these checked columns.",
            }
        )
    if overrides["force_identifier_features"]:
        steps.append(
            {
                "step": "Treat "
                + ", ".join(f"`{column}`" for column in overrides["force_identifier_features"])
                + " as identifier field(s) and exclude them from modeling.",
                "why": summarize_facts(identifier_feature_facts(profile, overrides["force_identifier_features"]))
                + ". The custom policy therefore excludes these checked identifier-like values from modeling.",
            }
        )

    for rule in overrides.get("feature_imputation_rules", []):
        feature = rule["feature"]
        if feature in plan.raw_categorical_features:
            consumed["categorical_imputation_features"].add(feature)
        else:
            consumed["numeric_imputation_features"].add(feature)
        steps.append(
            {
                "step": f"Apply {describe_imputation_rule(rule)} to `{feature}`.",
                "why": summarize_facts(checked_feature_snapshot_facts(profile, [feature]))
                + (
                    f". The custom rule replaces missing values in this checked feature with `{stringify_rule_value(rule.get('value'))}`."
                    if rule["strategy"] == "constant"
                    else ". The custom rule replaces missing values in this checked feature with the requested feature-specific strategy."
                ),
            }
        )

    for rule in overrides.get("feature_scaling_rules", []):
        feature = rule["feature"]
        consumed["scaled_numeric_features"].add(feature)
        steps.append(
            {
                "step": f"Scale `{feature}` with {describe_scaling_method(rule['method'])}.",
                "why": summarize_facts(checked_feature_snapshot_facts(profile, [feature]))
                + ". This custom rule changes the scaling method for this checked numeric feature before model fitting.",
            }
        )

    remaining_drop_features = [
        column for column in overrides["drop_features"] if column not in consumed["drop_features"]
    ]
    if remaining_drop_features:
        steps.append(
            {
                "step": "Drop the user-selected feature(s): "
                + ", ".join(f"`{column}`" for column in remaining_drop_features)
                + ".",
                "why": facts_or_default(
                    checked_feature_snapshot_facts(profile, remaining_drop_features),
                    "These checked columns were explicitly selected for removal",
                )
                + ". The custom policy therefore removes these checked columns from the preprocessing graph before modeling.",
            }
        )

    remaining_keep_features = [
        column for column in overrides["keep_features"] if column not in consumed["keep_features"]
    ]
    if remaining_keep_features:
        steps.append(
            {
                "step": "Keep the user-selected feature(s) in the preprocessing graph: "
                + ", ".join(f"`{column}`" for column in remaining_keep_features)
                + ".",
                "why": facts_or_default(
                    checked_feature_snapshot_facts(profile, remaining_keep_features),
                    "These checked columns were explicitly selected to stay in the preprocessing graph",
                )
                + ". The custom policy therefore preserves these checked columns even if the default heuristics would otherwise drop them.",
            }
        )
    return steps, consumed
