from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schemas import CandidateResult, DataProfile, ProjectBrief


def split_data(frame: pd.DataFrame, brief: ProjectBrief, split_option: str, test_size: float = 0.2):
    features = frame.drop(columns=[brief.target_column])
    target = frame[brief.target_column]

    if split_option == "time_ordered_holdout":
        if not brief.date_column or brief.date_column not in frame.columns:
            raise ValueError("Time-ordered holdout requires a valid date_column in the project brief.")
        ordered = frame.sort_values(by=brief.date_column)
        split_index = int(len(ordered) * (1 - test_size))
        train_frame = ordered.iloc[:split_index]
        test_frame = ordered.iloc[split_index:]
        return (
            train_frame.drop(columns=[brief.target_column]),
            test_frame.drop(columns=[brief.target_column]),
            train_frame[brief.target_column],
            test_frame[brief.target_column],
        )

    stratify = None
    if split_option == "stratified_holdout" and brief.task_type == "classification":
        if target.nunique(dropna=False) > 1 and target.value_counts().min() > 1:
            stratify = target

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )


def build_preprocessor(profile: DataProfile) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                profile.numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                profile.categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_feature_selector(option: str, task_type: str, estimated_feature_count: int):
    if option in {"skip", "auto"}:
        if option == "skip" or estimated_feature_count < 40:
            return None
        option = "select_k_best"

    if option == "variance_threshold":
        return VarianceThreshold()
    if option == "select_k_best":
        k_value = min(max(10, estimated_feature_count // 2), 50)
        return SelectKBest(score_func=f_classif if task_type == "classification" else f_regression, k=k_value)
    return None


def model_catalog(task_type: str, imbalance_ratio: float | None) -> dict[str, Any]:
    if task_type == "classification":
        class_weight = "balanced" if imbalance_ratio and imbalance_ratio >= 1.5 else None
        return {
            "linear": LogisticRegression(max_iter=1500, class_weight=class_weight),
            "random_forest": RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                n_jobs=1,
                class_weight=class_weight,
            ),
            "gradient_boosting": HistGradientBoostingClassifier(random_state=42),
        }

    return {
        "linear": Ridge(random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=1),
        "gradient_boosting": HistGradientBoostingRegressor(random_state=42),
    }


def selected_model_names(option: str) -> list[str]:
    if option == "speed_first":
        return ["linear", "gradient_boosting"]
    if option == "tree_ensemble_first":
        return ["random_forest", "gradient_boosting", "linear"]
    return ["linear", "random_forest", "gradient_boosting"]


def metric_configuration(task_type: str, selected_metric: str) -> tuple[dict[str, str], bool]:
    if task_type == "classification":
        return {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
        }, True

    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }, (selected_metric == "r2")


def cross_validation_strategy(task_type: str, option: str):
    folds = 5 if option in {"standard_cv", "thorough_cv"} else 3
    if task_type == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def build_pipeline(profile: DataProfile, task_type: str, feature_option: str, estimator: Any) -> Pipeline:
    selector = build_feature_selector(feature_option, task_type, len(profile.numeric_features) + len(profile.categorical_features))
    steps: list[tuple[str, Any]] = [("preprocessor", build_preprocessor(profile))]
    if selector is not None:
        steps.append(("feature_selector", selector))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def baseline_estimator(task_type: str):
    return DummyClassifier(strategy="most_frequent") if task_type == "classification" else DummyRegressor(strategy="median")


def convert_metric(name: str, value: float) -> float:
    return abs(float(value)) if name in {"rmse", "mae"} else float(value)


def evaluate_predictions(task_type: str, y_true: pd.Series, predictions) -> dict[str, float]:
    if task_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "f1_macro": float(f1_score(y_true, predictions, average="macro")),
        }
    return {
        "rmse": float(mean_squared_error(y_true, predictions, squared=False)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def leaderboard_row(
    pipeline: Pipeline,
    model_name: str,
    role: str,
    scoring: dict[str, str],
    task_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv,
) -> CandidateResult:
    cv_results = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return CandidateResult(
        model_name=model_name,
        role=role,
        cv_metrics={metric: convert_metric(metric, cv_results[f"test_{metric}"].mean()) for metric in scoring},
        test_metrics=evaluate_predictions(task_type, y_test, predictions),
        params={
            "model_type": pipeline.named_steps["model"].__class__.__name__,
            "pipeline_steps": [name for name, _ in pipeline.steps],
        },
        notes=[],
    )


def hpo_parameter_grid(task_type: str, model_name: str) -> dict[str, list[Any]]:
    if model_name == "linear":
        return {"model__C": [0.1, 1.0, 3.0]} if task_type == "classification" else {"model__alpha": [0.1, 1.0, 10.0]}
    if model_name == "random_forest":
        return {"model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 2, 4]}
    return {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 4, 8]}


def run_competition(
    shortlisted_names: list[str],
    task_type: str,
    profile: DataProfile,
    feature_option: str,
    scoring: dict[str, str],
    selected_metric: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv,
    imbalance_ratio: float | None,
) -> list[CandidateResult]:
    catalog = model_catalog(task_type, imbalance_ratio)
    results: list[CandidateResult] = []
    for model_name in shortlisted_names:
        pipeline = build_pipeline(profile, task_type, feature_option, catalog[model_name])
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=hpo_parameter_grid(task_type, model_name),
            scoring=scoring[selected_metric],
            cv=cv,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        predictions = search.best_estimator_.predict(X_test)
        results.append(
            CandidateResult(
                model_name=f"{model_name}_competition",
                role="competition",
                cv_metrics={selected_metric: convert_metric(selected_metric, search.best_score_)},
                test_metrics=evaluate_predictions(task_type, y_test, predictions),
                params=search.best_params_,
                notes=["Winner of the compact hyperparameter competition."],
            )
        )
    return results


def pick_best_result(results: list[CandidateResult], selected_metric: str, higher_is_better: bool) -> CandidateResult:
    key = lambda result: result.cv_metrics.get(selected_metric, result.test_metrics.get(selected_metric, float("-inf")))
    return max(results, key=key) if higher_is_better else min(results, key=key)


def export_bundle(output_dir: str | Path, trained_pipeline: Pipeline, bundle_payload: dict[str, Any]) -> tuple[str, str]:
    output_path = Path(output_dir).resolve()
    bundle_dir = output_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model_path = bundle_dir / "winning_model.joblib"
    metadata_path = bundle_dir / "bundle_metadata.joblib"
    joblib.dump(trained_pipeline, model_path)
    joblib.dump(bundle_payload, metadata_path)
    return str(bundle_dir), str(model_path)
