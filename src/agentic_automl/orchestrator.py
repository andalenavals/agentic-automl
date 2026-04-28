from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from .advisor import build_recommendations
from .data import format_profile_summary, load_dataset, profile_dataset
from .modeling import (
    build_pipeline,
    choose_baseline_strategy,
    cross_validation_strategy,
    default_model_option,
    evaluate_baseline,
    evaluate_predictions,
    hpo_search_space_config,
    humanize_model_name,
    metric_configuration,
    model_options_for_task,
    normalize_model_parameters,
    normalize_hpo_search_parameters,
    normalize_training_configuration,
    optimize_pipeline,
    split_data,
)
from .paths import resolve_repo_root
from .preprocessing import (
    build_preprocessing_plan,
    default_categorical_imputation_rule,
    default_numeric_imputation_rule,
    default_numeric_scaling_rule,
    extract_preprocessing_overrides_from_feedback,
    normalize_preprocessing_overrides,
)
from .schemas import CandidateResult, ProjectBrief, RunArtifacts, WorkflowRecommendation


def slugify(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    compact = "-".join(part for part in normalized.split("-") if part)
    return compact or "project"


def plan_project(brief: ProjectBrief, repo_root: str | Path | None = None):
    root = resolve_repo_root(repo_root)
    frame = load_dataset(brief.dataset_path)
    profile = profile_dataset(frame, brief)
    recommendations = build_recommendations(brief, profile)
    return frame, profile, recommendations


def policy_metadata_for_step(step_feedback: dict[str, dict[str, object]] | None, step_id: str) -> dict[str, Any]:
    if not isinstance(step_feedback, dict):
        return {}
    feedback = step_feedback.get(step_id, {})
    if not isinstance(feedback, dict):
        return {}
    metadata = feedback.get("policy_metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def resolve_model_option(
    brief: ProjectBrief,
    profile,
    selected_options: dict[str, str],
) -> str:
    chosen = selected_options.get("03_model_selection") or default_model_option(brief.task_type, profile)
    supported = set(model_options_for_task(brief.task_type))
    if chosen not in supported:
        return default_model_option(brief.task_type, profile)
    return chosen


def apply_training_policy(
    base_config: dict[str, Any],
    selected_option: str,
) -> dict[str, Any]:
    config = dict(base_config)
    if selected_option in {"fast_training", "fast_cv"}:
        config["cv_folds"] = 3
        if config.get("epochs"):
            config["epochs"] = max(50, int(config["epochs"] * 0.6))
    elif selected_option in {"thorough_training", "thorough_cv"}:
        config["cv_folds"] = 5
        if config.get("epochs"):
            config["epochs"] = int(config["epochs"] * 1.25)
    else:
        config["cv_folds"] = max(3, int(config.get("cv_folds", 5)))
    return config


def resolve_workflow_configuration(
    brief: ProjectBrief,
    profile,
    selected_options: dict[str, str] | None = None,
    step_feedback: dict[str, dict[str, object]] | None = None,
) -> dict[str, Any]:
    selected = dict(selected_options or {})
    preprocessing_option = selected.get("01_preprocessing", "auto_tabular_preprocessing")
    preprocessing_overrides = extract_preprocessing_overrides_from_feedback((step_feedback or {}).get("01_preprocessing"))
    split_option = selected.get("02_data_splitting", "random_holdout")
    model_option = resolve_model_option(brief, profile, selected)
    metric_option = selected.get("05_metric_selection", "f1_macro" if brief.task_type == "classification" else "rmse")
    training_option = selected.get("06_training_configuration", "standard_training")
    hpo_option = selected.get("07_hyperparameter_optimization", "skip")
    validation_option = selected.get("08_validation_and_baseline", "test_set_with_baseline")
    final_validation_option = selected.get("09_final_validation", "final_validation_dashboard")

    model_metadata = policy_metadata_for_step(step_feedback, "03_model_selection")
    training_metadata = policy_metadata_for_step(step_feedback, "06_training_configuration")
    hpo_metadata = policy_metadata_for_step(step_feedback, "07_hyperparameter_optimization")

    model_parameters = normalize_model_parameters(
        brief.task_type,
        model_option,
        profile,
        model_metadata.get("model_parameters"),
    )
    training_config = normalize_training_configuration(
        brief.task_type,
        model_option,
        profile,
        training_metadata.get("training_config"),
    )
    training_config = apply_training_policy(training_config, training_option)
    raw_hpo_config = hpo_metadata.get("hpo_config", {}) if isinstance(hpo_metadata.get("hpo_config", {}), dict) else {}
    if hpo_option == "skip":
        normalized_hpo_search_parameters: list[str] = []
        hpo_search_space = {"display_grid": {}}
    else:
        normalized_hpo_search_parameters = normalize_hpo_search_parameters(
            model_option,
            hpo_option,
            list(raw_hpo_config.get("search_parameters", []) or []),
        )
        hpo_search_space = hpo_search_space_config(
            model_option,
            hpo_option,
            requested_parameters=normalized_hpo_search_parameters,
        )
    hpo_config = {
        "search_parameters": normalized_hpo_search_parameters,
        "search_space": {
            key: [str(item) for item in values]
            for key, values in hpo_search_space["display_grid"].items()
        },
        "notes": raw_hpo_config.get("notes", ""),
    }

    return {
        "preprocessing_option": preprocessing_option,
        "preprocessing_overrides": preprocessing_overrides,
        "split_option": split_option,
        "model_option": model_option,
        "model_parameters": model_parameters,
        "metric_option": metric_option,
        "training_option": training_option,
        "training_config": training_config,
        "validation_option": validation_option,
        "hpo_option": hpo_option,
        "hpo_config": hpo_config,
        "final_validation_option": final_validation_option,
    }


def prepare_execution_context(
    brief: ProjectBrief,
    selected_options: dict[str, str] | None = None,
    step_feedback: dict[str, dict[str, object]] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    frame, profile, recommendations = plan_project(brief, root)
    selected = selected_options or {item.step_id: item.selected_option or "" for item in recommendations}
    config = resolve_workflow_configuration(brief, profile, selected, step_feedback)
    return {
        "repo_root": root,
        "brief": brief,
        "frame": frame,
        "profile": profile,
        "recommendations": recommendations,
        "selected_options": selected,
        "step_feedback": step_feedback or {},
        "config": config,
    }


def train_model_phase(context: dict[str, Any]) -> dict[str, Any]:
    brief = context["brief"]
    profile = context["profile"]
    config = context["config"]
    frame = context["frame"]

    X_train, X_test, y_train, y_test = split_data(frame, brief, config["split_option"])
    scoring, higher_is_better = metric_configuration(brief.task_type, config["metric_option"])
    cv = cross_validation_strategy(brief.task_type, config["training_config"])

    pipeline = build_pipeline(
        profile,
        brief.task_type,
        config["preprocessing_option"],
        config["model_option"],
        model_params=config["model_parameters"],
        training_config=config["training_config"],
        preprocessing_overrides=config["preprocessing_overrides"],
    )
    cv_results = cross_validate(clone(pipeline), X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
    fitted_pipeline = clone(pipeline).fit(X_train, y_train)

    return {
        "pipeline": fitted_pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cv_metrics": {metric: float(cv_results[f"test_{metric}"].mean()) for metric in scoring},
        "selected_metric": config["metric_option"],
        "higher_is_better": higher_is_better,
        "training_summary": {
            "selected_model": config["model_option"],
            "selected_model_label": humanize_model_name(config["model_option"]),
            "model_parameters": config["model_parameters"],
            "training_configuration": config["training_config"],
            "cv_metrics": {metric: float(cv_results[f"test_{metric}"].mean()) for metric in scoring},
            "train_rows": int(len(X_train)),
            "holdout_rows": int(len(X_test)),
            "split_option": config["split_option"],
            "preprocessing_option": config["preprocessing_option"],
        },
    }


def validation_phase(context: dict[str, Any], training_state: dict[str, Any], *, role: str, tuned: bool) -> dict[str, Any]:
    brief = context["brief"]
    profile = context["profile"]
    config = context["config"]
    pipeline = training_state["pipeline"]
    X_test = training_state["X_test"]
    y_test = training_state["y_test"]
    y_train = training_state["y_train"]

    predictions = pipeline.predict(X_test)
    model_metrics = evaluate_predictions(brief.task_type, y_test, predictions)
    baseline_result = evaluate_baseline(brief, profile, config["metric_option"], y_train, y_test)
    baseline_strategy, baseline_reason = choose_baseline_strategy(brief, profile, config["metric_option"])
    model_result = CandidateResult(
        model_name=f"{config['model_option']}{'_tuned' if tuned else ''}",
        role=role,
        cv_metrics=training_state.get("cv_metrics", {}),
        test_metrics=model_metrics,
        params={
            "model_parameters": training_state.get("model_parameters", config["model_parameters"]),
            "training_configuration": training_state.get("training_config", config["training_config"]),
        },
        notes=[
            "Validated on the untouched test split." if not tuned else "Final validation for the tuned model on the untouched test split."
        ],
    )

    delta = model_metrics[config["metric_option"]] - baseline_result.test_metrics[config["metric_option"]]
    summary = {
        "selected_model": model_result.model_name,
        "selected_metric": config["metric_option"],
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_result.test_metrics,
        "baseline_strategy": baseline_strategy,
        "baseline_reason": baseline_reason,
        "metric_delta_vs_baseline": delta,
        "tuned": tuned,
    }
    preview = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "prediction": pd.Series(predictions).reset_index(drop=True),
        }
    ).head(25)
    prediction_frame = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "prediction": pd.Series(predictions).reset_index(drop=True),
        }
    )
    prediction_export_frame = X_test.reset_index(drop=True).copy()
    prediction_export_frame[brief.target_column] = y_test.reset_index(drop=True)
    prediction_export_frame["prediction"] = pd.Series(predictions).reset_index(drop=True)
    return {
        "model_result": model_result,
        "baseline_result": baseline_result,
        "predictions_preview": preview,
        "prediction_frame": prediction_frame,
        "prediction_export_frame": prediction_export_frame,
        "summary": summary,
    }


def hyperparameter_optimization_phase(context: dict[str, Any], training_state: dict[str, Any]) -> dict[str, Any] | None:
    config = context["config"]
    if config["hpo_option"] == "skip":
        return None

    tuned_pipeline, optimization_summary = optimize_pipeline(
        training_state["pipeline"],
        context["brief"].task_type,
        config["model_option"],
        config["metric_option"],
        config["hpo_option"],
        config["model_parameters"],
        config["training_config"],
        training_state["X_train"],
        training_state["y_train"],
        cross_validation_strategy(context["brief"].task_type, config["training_config"]),
        hpo_config=config["hpo_config"],
    )
    return {
        "pipeline": tuned_pipeline,
        "model_parameters": optimization_summary["best_params"],
        "training_config": config["training_config"],
        "X_train": training_state["X_train"],
        "X_test": training_state["X_test"],
        "y_train": training_state["y_train"],
        "y_test": training_state["y_test"],
        "selected_metric": training_state["selected_metric"],
        "higher_is_better": training_state["higher_is_better"],
        "cv_metrics": {config["metric_option"]: float(optimization_summary["best_cv_metric"])},
        "summary": optimization_summary,
    }


def build_step_feedback_payload(
    recommendations: list[WorkflowRecommendation],
    selected_options: dict[str, str],
    step_feedback: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for recommendation in recommendations:
        raw_feedback = step_feedback.get(recommendation.step_id, {})
        payload[recommendation.step_id] = {
            "title": recommendation.title,
            "agreement": raw_feedback.get("agreement", "pending"),
            "default_option": recommendation.selected_option or "",
            "selected_option": selected_options.get(recommendation.step_id, recommendation.selected_option or ""),
            "custom_note": raw_feedback.get("custom_note", "").strip(),
            "policy_summary": raw_feedback.get("policy_summary", "").strip(),
            "policy_metadata": raw_feedback.get("policy_metadata", {}),
            "policy_confirmed": bool(raw_feedback.get("policy_confirmed", False)),
        }
    return payload


def notebook_markdown_cell(source: str) -> dict[str, Any]:
    content = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.splitlines()],
    }


def notebook_code_cell(source: str) -> dict[str, Any]:
    content = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in content.splitlines()],
    }


def format_markdown_mapping(mapping: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in mapping.items():
        if value is None or value == "":
            continue
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) if lines else "- none"


def notebook_relative_dataset_path(project_dir: str | Path, dataset_path: str | Path) -> str:
    project_path = Path(project_dir).resolve()
    dataset = Path(dataset_path).expanduser().resolve()
    return os.path.relpath(dataset, start=project_path)


def notebook_model_import_source(model_name: str) -> str:
    imports = {
        "logistic_regression": "from sklearn.linear_model import LogisticRegression",
        "ridge_regression": "from sklearn.linear_model import Ridge",
        "random_forest_classifier": "from sklearn.ensemble import RandomForestClassifier",
        "random_forest_regressor": "from sklearn.ensemble import RandomForestRegressor",
        "hist_gradient_boosting_classifier": "from sklearn.ensemble import HistGradientBoostingClassifier",
        "hist_gradient_boosting_regressor": "from sklearn.ensemble import HistGradientBoostingRegressor",
        "mlp_classifier": "from sklearn.neural_network import MLPClassifier",
        "mlp_regressor": "from sklearn.neural_network import MLPRegressor",
    }
    return imports[model_name]


def notebook_estimator_source(model_name: str) -> str:
    sources = {
        "logistic_regression": """
            def build_estimator(model_params, training_config):
                random_seed = int(training_config.get("random_seed", 42))
                epochs = training_config.get("epochs")
                optimizer = training_config.get("optimizer")
                solver = optimizer if optimizer in {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"} else "lbfgs"
                return LogisticRegression(
                    C=float(model_params.get("C", 1.0)),
                    class_weight=model_params.get("class_weight"),
                    max_iter=int(epochs or 300),
                    solver=solver,
                    random_state=random_seed,
                )
        """,
        "ridge_regression": """
            def build_estimator(model_params, training_config):
                random_seed = int(training_config.get("random_seed", 42))
                return Ridge(
                    alpha=float(model_params.get("alpha", 1.0)),
                    random_state=random_seed,
                )
        """,
        "random_forest_classifier": """
            def build_estimator(model_params, training_config):
                random_seed = int(training_config.get("random_seed", 42))
                return RandomForestClassifier(
                    n_estimators=int(model_params.get("n_estimators", 300)),
                    max_depth=model_params.get("max_depth"),
                    min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
                    class_weight=model_params.get("class_weight"),
                    random_state=random_seed,
                    n_jobs=1,
                )
        """,
        "random_forest_regressor": """
            def build_estimator(model_params, training_config):
                random_seed = int(training_config.get("random_seed", 42))
                return RandomForestRegressor(
                    n_estimators=int(model_params.get("n_estimators", 300)),
                    max_depth=model_params.get("max_depth"),
                    min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
                    random_state=random_seed,
                    n_jobs=1,
                )
        """,
        "hist_gradient_boosting_classifier": """
            def build_estimator(model_params, training_config):
                random_seed = int(training_config.get("random_seed", 42))
                learning_rate = training_config.get("learning_rate")
                epochs = training_config.get("epochs")
                early_stopping = bool(training_config.get("early_stopping", False))
                validation_fraction = float(training_config.get("validation_fraction", 0.1))
                return HistGradientBoostingClassifier(
                    learning_rate=float(learning_rate or 0.1),
                    max_iter=int(epochs or 200),
                    max_depth=model_params.get("max_depth"),
                    max_leaf_nodes=model_params.get("max_leaf_nodes"),
                    l2_regularization=float(model_params.get("l2_regularization", 0.0)),
                    early_stopping=early_stopping,
                    validation_fraction=validation_fraction,
                    random_state=random_seed,
                )
        """,
        "hist_gradient_boosting_regressor": """
            def build_estimator(model_params, training_config):
                random_seed = int(training_config.get("random_seed", 42))
                learning_rate = training_config.get("learning_rate")
                epochs = training_config.get("epochs")
                early_stopping = bool(training_config.get("early_stopping", False))
                validation_fraction = float(training_config.get("validation_fraction", 0.1))
                return HistGradientBoostingRegressor(
                    learning_rate=float(learning_rate or 0.1),
                    max_iter=int(epochs or 200),
                    max_depth=model_params.get("max_depth"),
                    max_leaf_nodes=model_params.get("max_leaf_nodes"),
                    l2_regularization=float(model_params.get("l2_regularization", 0.0)),
                    early_stopping=early_stopping,
                    validation_fraction=validation_fraction,
                    random_state=random_seed,
                )
        """,
        "mlp_classifier": """
            def build_estimator(model_params, training_config):
                random_seed = int(training_config.get("random_seed", 42))
                optimizer = training_config.get("optimizer")
                learning_rate = training_config.get("learning_rate")
                epochs = training_config.get("epochs")
                mini_batch = training_config.get("mini_batch")
                early_stopping = bool(training_config.get("early_stopping", False))
                validation_fraction = float(training_config.get("validation_fraction", 0.1))
                return MLPClassifier(
                    hidden_layer_sizes=tuple(model_params.get("hidden_layer_sizes", (64, 32))),
                    alpha=float(model_params.get("alpha", 0.0001)),
                    solver=optimizer or "adam",
                    learning_rate_init=float(learning_rate or 0.001),
                    max_iter=int(epochs or 250),
                    batch_size=int(mini_batch) if mini_batch else "auto",
                    early_stopping=early_stopping,
                    validation_fraction=validation_fraction,
                    random_state=random_seed,
                )
        """,
        "mlp_regressor": """
            def build_estimator(model_params, training_config):
                random_seed = int(training_config.get("random_seed", 42))
                optimizer = training_config.get("optimizer")
                learning_rate = training_config.get("learning_rate")
                epochs = training_config.get("epochs")
                mini_batch = training_config.get("mini_batch")
                early_stopping = bool(training_config.get("early_stopping", False))
                validation_fraction = float(training_config.get("validation_fraction", 0.1))
                return MLPRegressor(
                    hidden_layer_sizes=tuple(model_params.get("hidden_layer_sizes", (64, 32))),
                    alpha=float(model_params.get("alpha", 0.0001)),
                    solver=optimizer or "adam",
                    learning_rate_init=float(learning_rate or 0.001),
                    max_iter=int(epochs or 250),
                    batch_size=int(mini_batch) if mini_batch else "auto",
                    early_stopping=early_stopping,
                    validation_fraction=validation_fraction,
                    random_state=random_seed,
                )
        """,
    }
    return textwrap.dedent(sources[model_name]).strip()


def notebook_hpo_grid_source(
    model_name: str,
    hpo_option: str,
    search_parameters: list[str] | None = None,
) -> str:
    if hpo_option == "skip":
        return textwrap.dedent(
            """
            def hpo_parameter_grid():
                return {}
            """
        ).strip()

    notebook_grid = {
        key.replace("model__", "", 1) if key.startswith("model__") else key: values
        for key, values in hpo_search_space_config(
            model_name,
            hpo_option,
            requested_parameters=search_parameters,
        )["estimator_grid"].items()
    }
    return textwrap.dedent(
        f"""
        def hpo_parameter_grid():
            return {notebook_grid!r}
        """
    ).strip()


def notebook_split_source(split_option: str, task_type: str) -> str:
    if split_option == "time_ordered_holdout":
        return textwrap.dedent(
            """
            def apply_selected_split(transformed_data, target_column, test_size=0.2):
                split_order_values = transformed_data.attrs.get("split_order_values")
                if not split_order_values:
                    raise ValueError("Time-ordered holdout requires split-order values from preprocessing().")
                ordered = transformed_data.copy()
                ordered["__agentic_split_order__"] = pd.to_datetime(split_order_values, errors="coerce", format="mixed")
                ordered = ordered.sort_values(by="__agentic_split_order__")
                split_index = int(len(ordered) * (1 - test_size))
                train_frame = ordered.iloc[:split_index].drop(columns=["__agentic_split_order__"])
                test_frame = ordered.iloc[split_index:].drop(columns=["__agentic_split_order__"])
                return {"train": train_frame, "test": test_frame}
            """
        ).strip()
    if split_option == "stratified_holdout" and task_type == "classification":
        return textwrap.dedent(
            """
            def apply_selected_split(transformed_data, target_column, test_size=0.2):
                features = transformed_data.drop(columns=[target_column])
                target = transformed_data[target_column]
                stratify = None
                if target.nunique(dropna=False) > 1 and target.value_counts().min() > 1:
                    stratify = target
                X_train, X_test, y_train, y_test = train_test_split(
                    features,
                    target,
                    test_size=test_size,
                    random_state=42,
                    stratify=stratify,
                )
                train_frame = X_train.copy()
                train_frame[target_column] = y_train
                test_frame = X_test.copy()
                test_frame[target_column] = y_test
                return {"train": train_frame, "test": test_frame}
            """
        ).strip()
    return textwrap.dedent(
        """
        def apply_selected_split(transformed_data, target_column, test_size=0.2):
            features = transformed_data.drop(columns=[target_column])
            target = transformed_data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=test_size,
                random_state=42,
                stratify=None,
            )
            train_frame = X_train.copy()
            train_frame[target_column] = y_train
            test_frame = X_test.copy()
            test_frame[target_column] = y_test
            return {"train": train_frame, "test": test_frame}
        """
    ).strip()


def build_notebook_preprocessing_spec(
    brief: ProjectBrief,
    profile,
    config_payload: dict[str, Any],
    selected_features: list[str],
) -> dict[str, Any]:
    overrides = normalize_preprocessing_overrides(config_payload.get("preprocessing_overrides"))
    plan = build_preprocessing_plan(
        profile,
        config_payload["preprocessing_option"],
        overrides=overrides,
    )
    explicit_imputation = {rule["feature"]: dict(rule) for rule in overrides["feature_imputation_rules"]}
    explicit_scaling = {rule["feature"]: dict(rule) for rule in overrides["feature_scaling_rules"]}

    numeric_imputation_rules: dict[str, dict[str, Any]] = {}
    for feature in plan.numeric_features:
        rule = explicit_imputation.get(feature) or default_numeric_imputation_rule(profile, plan, feature)
        if rule:
            numeric_imputation_rules[feature] = {
                "strategy": rule.get("strategy"),
                "value": rule.get("value"),
            }

    categorical_imputation_rules: dict[str, dict[str, Any]] = {}
    for feature in plan.raw_categorical_features:
        rule = explicit_imputation.get(feature) or default_categorical_imputation_rule(profile, feature)
        if rule:
            categorical_imputation_rules[feature] = {
                "strategy": rule.get("strategy"),
                "value": rule.get("value"),
            }

    numeric_scaling_rules: dict[str, str] = {}
    for feature in plan.numeric_features:
        rule = explicit_scaling.get(feature) or default_numeric_scaling_rule(plan, feature)
        if rule and rule.get("method") and rule.get("method") != "none":
            numeric_scaling_rules[feature] = str(rule["method"])

    categorical_encoding = {
        feature: ("ordinal" if feature in plan.ordinal_categorical_features else "one_hot")
        for feature in plan.raw_categorical_features
    }

    return {
        "dropped_features": list(plan.dropped_features),
        "date_features": list(plan.date_features),
        "raw_categorical_features": list(plan.raw_categorical_features),
        "numeric_features": list(plan.numeric_features),
        "feature_transform_rules": list(overrides["feature_transform_rules"]),
        "derived_feature_rules": list(overrides["derived_feature_rules"]),
        "numeric_imputation_rules": numeric_imputation_rules,
        "categorical_imputation_rules": categorical_imputation_rules,
        "numeric_scaling_rules": numeric_scaling_rules,
        "categorical_encoding": categorical_encoding,
        "needs_split_order": config_payload["split_option"] == "time_ordered_holdout",
    }


def notebook_preprocessing_section_source(
    brief: ProjectBrief,
    relative_dataset_path: str,
    selected_features: list[str],
    preprocessing_spec: dict[str, Any],
) -> str:
    dataset_suffix = Path(relative_dataset_path).suffix.lower()
    has_feature_transforms = bool(preprocessing_spec.get("feature_transform_rules"))
    has_derived_features = bool(preprocessing_spec.get("derived_feature_rules"))
    has_numeric_imputation = bool(preprocessing_spec.get("numeric_imputation_rules"))
    has_categorical_imputation = bool(preprocessing_spec.get("categorical_imputation_rules"))
    has_any_imputation = has_numeric_imputation or has_categorical_imputation
    has_scaling = bool(preprocessing_spec.get("numeric_scaling_rules"))
    has_date_features = bool(preprocessing_spec.get("date_features"))
    has_drop_columns = bool(preprocessing_spec.get("dropped_features"))
    transform_kinds = sorted(
        {str(rule.get("kind")) for rule in preprocessing_spec.get("feature_transform_rules", []) if rule.get("kind")}
    )
    imputation_strategies = sorted(
        {
            str(rule.get("strategy"))
            for rule in [
                *preprocessing_spec.get("numeric_imputation_rules", {}).values(),
                *preprocessing_spec.get("categorical_imputation_rules", {}).values(),
            ]
            if rule.get("strategy")
        }
    )
    scaling_methods = sorted(
        {str(method) for method in preprocessing_spec.get("numeric_scaling_rules", {}).values() if method}
    )

    import_lines = ["from pathlib import Path", "", "import pandas as pd"]
    if has_feature_transforms:
        import_lines.insert(2, "import numpy as np")

    sections: list[str] = [
        "\n".join(import_lines),
        textwrap.dedent(
            f"""
            DATASET_RELATIVE_PATH = {relative_dataset_path!r}
            TARGET_COLUMN = {brief.target_column!r}
            SELECTED_FEATURES = {selected_features!r}
            SELECTED_PREPROCESSING_KWARGS = {preprocessing_spec!r}
            NEEDS_SPLIT_ORDER = {bool(preprocessing_spec.get("needs_split_order"))!r}
            """
        ).strip(),
        textwrap.dedent(
            (
                """
                def load_dataset(dataset_path):
                    path = Path(dataset_path).expanduser().resolve()
                    return pd.read_csv(path)
                """
                if dataset_suffix == ".csv"
                else """
                def load_dataset(dataset_path):
                    path = Path(dataset_path).expanduser().resolve()
                    return pd.read_parquet(path)
                """
            )
        ).strip(),
    ]

    if has_feature_transforms:
        transform_lines = [
            "def apply_numeric_transform(series, rule):",
            '    numeric = pd.to_numeric(series, errors="coerce")',
            '    kind = rule.get("kind")',
            '    value = rule.get("value")',
        ]
        if "log" in transform_kinds:
            transform_lines.extend(
                [
                    '    if kind == "log":',
                    "        numeric = numeric.where(numeric > 0)",
                    "        return np.log(numeric)",
                ]
            )
        if "log1p" in transform_kinds:
            transform_lines.extend(
                [
                    '    if kind == "log1p":',
                    "        numeric = numeric.where(numeric >= -1)",
                    "        return np.log1p(numeric)",
                ]
            )
        if "sqrt" in transform_kinds:
            transform_lines.extend(
                [
                    '    if kind == "sqrt":',
                    "        numeric = numeric.where(numeric >= 0)",
                    "        return np.sqrt(numeric)",
                ]
            )
        if "square" in transform_kinds:
            transform_lines.extend(['    if kind == "square":', "        return np.square(numeric)"])
        if "abs" in transform_kinds:
            transform_lines.extend(['    if kind == "abs":', "        return np.abs(numeric)"])
        if "multiply" in transform_kinds:
            transform_lines.extend(['    if kind == "multiply":', "        return numeric * float(value)"])
        if "divide" in transform_kinds:
            transform_lines.extend(['    if kind == "divide":', "        return numeric / float(value)"])
        if "add" in transform_kinds:
            transform_lines.extend(['    if kind == "add":', "        return numeric + float(value)"])
        if "subtract" in transform_kinds:
            transform_lines.extend(['    if kind == "subtract":', "        return numeric - float(value)"])
        transform_lines.append("    return numeric")
        sections.append(
            "\n".join(transform_lines)
        )

    if has_any_imputation:
        imputation_lines = [
            "def apply_imputation(series, rule):",
            '    strategy = rule.get("strategy")',
        ]
        if "mean" in imputation_strategies:
            imputation_lines.extend(
                ['    if strategy == "mean":', '        fill_value = pd.to_numeric(series, errors="coerce").mean()']
            )
        if "median" in imputation_strategies:
            imputation_lines.extend(
                ['    elif strategy == "median":' if len(imputation_lines) > 2 else '    if strategy == "median":', '        fill_value = pd.to_numeric(series, errors="coerce").median()']
            )
        if "most_frequent" in imputation_strategies:
            imputation_lines.extend(
                [
                    '    elif strategy == "most_frequent":' if len(imputation_lines) > 2 else '    if strategy == "most_frequent":',
                    "        mode = series.mode(dropna=True)",
                    "        fill_value = mode.iloc[0] if not mode.empty else None",
                ]
            )
        if "constant" in imputation_strategies:
            imputation_lines.extend(
                [
                    '    elif strategy == "constant":' if len(imputation_lines) > 2 else '    if strategy == "constant":',
                    '        fill_value = rule.get("value")',
                ]
            )
        imputation_lines.extend(["    else:", "        return series", "    return series.fillna(fill_value)"])
        sections.append(
            "\n".join(imputation_lines)
        )

    if has_scaling:
        scaling_lines = [
            "def apply_scaling(series, method):",
            '    numeric = pd.to_numeric(series, errors="coerce").astype(float)',
        ]
        if "standard" in scaling_methods:
            scaling_lines.extend(
                [
                    '    if method == "standard":',
                    "        mean = numeric.mean()",
                    "        std = numeric.std(ddof=0)",
                    "        return numeric - mean if pd.isna(std) or std == 0 else (numeric - mean) / std",
                ]
            )
        if "minmax" in scaling_methods:
            scaling_lines.extend(
                [
                    '    if method == "minmax":',
                    "        minimum = numeric.min()",
                    "        maximum = numeric.max()",
                    "        scale = maximum - minimum",
                    "        return numeric - minimum if pd.isna(scale) or scale == 0 else (numeric - minimum) / scale",
                ]
            )
        if "robust" in scaling_methods:
            scaling_lines.extend(
                [
                    '    if method == "robust":',
                    "        median = numeric.median()",
                    "        q1 = numeric.quantile(0.25)",
                    "        q3 = numeric.quantile(0.75)",
                    "        scale = q3 - q1",
                    "        return numeric - median if pd.isna(scale) or scale == 0 else (numeric - median) / scale",
                ]
            )
        scaling_lines.append("    return numeric")
        sections.append(
            "\n".join(scaling_lines)
        )

    preprocessing_lines = [
        "def preprocessing(dataset_path, target, features, transformation_kwargs):",
        "    feature_names = [str(feature) for feature in features if str(feature) != target]",
        "    spec = dict(SELECTED_PREPROCESSING_KWARGS)",
        "    if transformation_kwargs:",
        "        spec.update(transformation_kwargs)",
        "    frame = load_dataset(dataset_path)",
        "    required_columns = [*feature_names, target]",
        "    missing_columns = [column for column in required_columns if column not in frame.columns]",
        "    if missing_columns:",
        '        raise ValueError(f"Requested columns not found in the dataset: {missing_columns}")',
        "",
        "    working_frame = frame[required_columns].copy()",
        "    feature_frame = working_frame[feature_names].copy()",
    ]

    if has_feature_transforms:
        preprocessing_lines.extend(
            [
                "",
                '    for rule in spec.get("feature_transform_rules", []):',
                '        feature = rule.get("feature")',
                "        if feature in feature_frame.columns:",
                "            feature_frame[feature] = apply_numeric_transform(feature_frame[feature], rule)",
            ]
        )

    if has_derived_features:
        preprocessing_lines.extend(
            [
                "",
                '    for rule in spec.get("derived_feature_rules", []):',
                '        if rule.get("kind") != "frequency_count":',
                "            continue",
                '        source = rule.get("source")',
                '        output = rule.get("output")',
                "        if source in feature_frame.columns and output:",
                "            counts = feature_frame[source].value_counts(dropna=False)",
                '            feature_frame[output] = feature_frame[source].map(counts).fillna(0.0).astype(float)',
            ]
        )

    preprocessing_lines.extend(
        [
            "",
            "    split_order_values = None",
            f"    split_order_column = {brief.date_column!r}",
            "    if NEEDS_SPLIT_ORDER and split_order_column and split_order_column in feature_frame.columns:",
            "        split_order_values = feature_frame[split_order_column].tolist()",
        ]
    )

    if has_date_features:
        preprocessing_lines.extend(
            [
                "",
                '    for column in spec.get("date_features", []):',
                "        if column not in feature_frame.columns:",
                "            continue",
                '        parsed = pd.to_datetime(feature_frame[column], errors="coerce", format="mixed")',
                '        feature_frame[f"{column}__year"] = parsed.dt.year',
                '        feature_frame[f"{column}__month"] = parsed.dt.month',
                '        feature_frame[f"{column}__day"] = parsed.dt.day',
                '        feature_frame[f"{column}__dayofweek"] = parsed.dt.dayofweek',
            ]
        )

    if has_drop_columns:
        preprocessing_lines.extend(
            [
                "",
                '    drop_columns = [column for column in spec.get("dropped_features", []) if column in feature_frame.columns]',
                "    if drop_columns:",
                "        feature_frame = feature_frame.drop(columns=drop_columns)",
            ]
        )

    preprocessing_lines.extend(
        [
            "",
            "    numeric_output = pd.DataFrame(index=feature_frame.index)",
            '    for feature in spec.get("numeric_features", []):',
            "        if feature in feature_frame.columns:",
            "            series = feature_frame[feature]",
            "        else:",
            '            series = pd.Series(float("nan"), index=feature_frame.index, dtype=float)',
        ]
    )

    if has_numeric_imputation:
        preprocessing_lines.extend(
            [
                '        imputation_rule = spec.get("numeric_imputation_rules", {}).get(feature)',
                "        if imputation_rule:",
                "            series = apply_imputation(series, imputation_rule)",
            ]
        )

    if has_scaling:
        preprocessing_lines.extend(
            [
                '        scaling_method = spec.get("numeric_scaling_rules", {}).get(feature)',
                "        if scaling_method:",
                "            series = apply_scaling(series, scaling_method)",
            ]
        )

    preprocessing_lines.extend(
        [
            '        numeric_output[feature] = pd.to_numeric(series, errors="coerce")',
            "",
            "    categorical_parts: list[pd.DataFrame] = []",
            '    for feature in spec.get("raw_categorical_features", []):',
            "        if feature not in feature_frame.columns:",
            "            continue",
            "        series = feature_frame[feature]",
        ]
    )

    if has_categorical_imputation:
        preprocessing_lines.extend(
            [
                '        imputation_rule = spec.get("categorical_imputation_rules", {}).get(feature)',
                "        if imputation_rule:",
                "            series = apply_imputation(series, imputation_rule)",
            ]
        )

    preprocessing_lines.extend(
        [
            '        encoding = spec.get("categorical_encoding", {}).get(feature, "one_hot")',
            '        if encoding == "ordinal":',
            '            categorical = pd.Categorical(series.astype("object"))',
            "            categorical_parts.append(",
            '                pd.DataFrame({feature: categorical.codes.astype(float)}, index=feature_frame.index)',
            "            )",
            "        else:",
            '            encoded = pd.get_dummies(series.astype("object"), prefix=feature, dtype=float)',
            "            categorical_parts.append(encoded)",
            "",
            "    transformed_parts = [numeric_output] + categorical_parts",
            "    transformed_parts = [part for part in transformed_parts if not part.empty]",
            "    if not transformed_parts:",
            '        raise ValueError("The selected preprocessing plan removed every usable input feature.")',
            "",
            "    transformed = pd.concat(transformed_parts, axis=1)",
            "    transformed[target] = working_frame[target].reset_index(drop=True)",
            "    transformed = transformed.reset_index(drop=True)",
            '    transformed.attrs["target_column"] = target',
            '    transformed.attrs["split_order_values"] = split_order_values',
            "    return transformed",
        ]
    )

    sections.append("\n".join(preprocessing_lines))
    sections.append(
        textwrap.dedent(
            """
            notebook_dir = Path.cwd()
            resolved_dataset_path = (notebook_dir / DATASET_RELATIVE_PATH).resolve()
            if not resolved_dataset_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found relative to the notebook location: {resolved_dataset_path}"
                )

            transformed_dataset = preprocessing(
                resolved_dataset_path,
                TARGET_COLUMN,
                SELECTED_FEATURES,
                SELECTED_PREPROCESSING_KWARGS,
            )
            transformed_dataset.head()
            """
        ).strip()
    )

    return "\n\n".join(section for section in sections if section.strip())


def notebook_split_section_source(brief: ProjectBrief, config_payload: dict[str, Any]) -> str:
    split_imports = (
        "from sklearn.model_selection import train_test_split"
        if config_payload["split_option"] != "time_ordered_holdout"
        else ""
    )
    sections = [
        split_imports,
        f"TEST_SIZE = 0.2\nDEFAULT_TARGET_COLUMN = {brief.target_column!r}",
        notebook_split_source(config_payload["split_option"], brief.task_type),
        textwrap.dedent(
            """
            def split_data(transformed_data):
                target_column = transformed_data.attrs.get("target_column", DEFAULT_TARGET_COLUMN)
                split_state = apply_selected_split(transformed_data, target_column, test_size=TEST_SIZE)
                for part in split_state.values():
                    part.attrs.update(dict(transformed_data.attrs))
                return split_state
            """
        ).strip(),
        textwrap.dedent(
            """
            split_state = split_data(transformed_dataset)
            {
                "train_rows": len(split_state["train"]),
                "test_rows": len(split_state["test"]),
                "transformed_feature_count": int(split_state["train"].shape[1] - 1),
            }
            """
        ).strip(),
    ]
    return "\n\n".join(section for section in sections if section.strip())


def notebook_training_section_source(brief: ProjectBrief, config_payload: dict[str, Any]) -> str:
    selected_metric = config_payload["metric_option"]
    scoring_name = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }[selected_metric]
    higher_is_better = selected_metric in {"accuracy", "balanced_accuracy", "f1_macro", "r2"}
    task_is_classification = brief.task_type == "classification"
    cv_class = "StratifiedKFold" if task_is_classification else "KFold"
    sections = [
        f"from sklearn.base import clone\nfrom sklearn.model_selection import {cv_class}, cross_validate\n{notebook_model_import_source(config_payload['model_option'])}",
        f"SELECTED_METRIC = {config_payload['metric_option']!r}\nMODEL_PARAMETERS = {config_payload['model_parameters']!r}\nTRAINING_CONFIGURATION = {config_payload['training_config']!r}",
        notebook_estimator_source(config_payload["model_option"]),
        textwrap.dedent(
            f"""
            def selected_scoring():
                return ({scoring_name!r}, {higher_is_better!r})


            def selected_cv():
                folds = int(TRAINING_CONFIGURATION.get("cv_folds", 5))
                return {cv_class}(n_splits=folds, shuffle=True, random_state=42)


            def convert_metric(name, value):
                return abs(float(value)) if name in {{"rmse", "mae"}} else float(value)


            def build_selected_model():
                return build_estimator(MODEL_PARAMETERS, TRAINING_CONFIGURATION)


            def train_model(train_split):
                target_column = train_split.attrs.get("target_column", {brief.target_column!r})
                X_train = train_split.drop(columns=[target_column])
                y_train = train_split[target_column]
                scoring_name, higher_is_better = selected_scoring()
                estimator = build_selected_model()
                cv_results = cross_validate(
                    clone(estimator),
                    X_train,
                    y_train,
                    scoring={{SELECTED_METRIC: scoring_name}},
                    cv=selected_cv(),
                    n_jobs=1,
                )
                fitted_model = clone(estimator).fit(X_train, y_train)
                return {{
                    "model": fitted_model,
                    "model_name": {config_payload["model_option"]!r},
                    "y_train": y_train,
                    "selected_metric": SELECTED_METRIC,
                    "higher_is_better": higher_is_better,
                    "summary": {{
                        "selected_model": {config_payload["model_option"]!r},
                        "selected_metric": SELECTED_METRIC,
                        "train_rows": int(len(train_split)),
                        "model_parameters": MODEL_PARAMETERS,
                        "training_configuration": TRAINING_CONFIGURATION,
                        "cv_metrics": {{
                            SELECTED_METRIC: convert_metric(
                                SELECTED_METRIC,
                                cv_results[f"test_{{SELECTED_METRIC}}"].mean(),
                            )
                        }},
                    }},
                }}
            """
        ).strip(),
        'training_state = train_model(split_state["train"])\ntraining_state["summary"]',
    ]
    return "\n\n".join(section for section in sections if section.strip())


def notebook_hpo_section_source(brief: ProjectBrief, config_payload: dict[str, Any]) -> str:
    if config_payload["hpo_option"] == "skip":
        return textwrap.dedent(
            """
            def hyperparameter_optimization(train_split):
                return {
                    "skipped": True,
                    "reason": "Hyperparameter optimization was disabled for this workflow selection.",
                }


            optimization_state = hyperparameter_optimization(split_state["train"])
            optimization_state
            """
        ).strip()

    sections = [
        "from IPython.display import display",
        "from sklearn.model_selection import GridSearchCV",
        notebook_hpo_grid_source(
            config_payload["model_option"],
            config_payload["hpo_option"],
            search_parameters=list(config_payload.get("hpo_config", {}).get("search_parameters", []) or []),
        ),
        textwrap.dedent(
            f"""
            def hyperparameter_optimization(train_split):
                import matplotlib.pyplot as plt

                target_column = train_split.attrs.get("target_column", {brief.target_column!r})
                X_train = train_split.drop(columns=[target_column])
                y_train = train_split[target_column]
                scoring_name = selected_scoring()[0]
                search_grid = {{
                    key.replace("model__", "", 1) if key.startswith("model__") else key: values
                    for key, values in hpo_parameter_grid().items()
                }}
                search = GridSearchCV(
                    estimator=build_selected_model(),
                    param_grid=search_grid,
                    scoring=scoring_name,
                    cv=selected_cv(),
                    n_jobs=1,
                )
                search.fit(X_train, y_train)
                search_results = pd.DataFrame(search.cv_results_).copy()
                param_columns = [column for column in search_results.columns if column.startswith("param_")]
                dashboard_columns = ["rank_test_score", "mean_test_score", "std_test_score", *param_columns]
                competition_dashboard = search_results[dashboard_columns].copy()
                competition_dashboard.rename(
                    columns={{
                        "rank_test_score": "rank",
                        "mean_test_score": "cv_metric_raw",
                        "std_test_score": "cv_metric_std_raw",
                        **{{column: column.replace("param_", "", 1) for column in param_columns}},
                    }},
                    inplace=True,
                )
                competition_dashboard["cv_metric"] = competition_dashboard["cv_metric_raw"].map(
                    lambda value: convert_metric(SELECTED_METRIC, value)
                )
                competition_dashboard["cv_metric_std"] = competition_dashboard["cv_metric_std_raw"].map(abs)
                competition_dashboard = competition_dashboard.drop(columns=["cv_metric_raw", "cv_metric_std_raw"])
                competition_dashboard = competition_dashboard.sort_values(by="rank").reset_index(drop=True)
                top_dashboard = competition_dashboard.head(min(12, len(competition_dashboard))).copy()

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(
                    [str(int(rank)) for rank in top_dashboard["rank"]],
                    top_dashboard["cv_metric"],
                    alpha=0.85,
                )
                ax.set_title(f"Hyperparameter competition on {{SELECTED_METRIC}}")
                ax.set_xlabel("ranked trial")
                ax.set_ylabel(SELECTED_METRIC)
                plt.tight_layout()
                plt.show()
                return {{
                    "model": search.best_estimator_,
                    "model_name": {config_payload["model_option"]!r},
                    "y_train": y_train,
                    "selected_metric": SELECTED_METRIC,
                    "competition_dashboard": competition_dashboard,
                    "summary": {{
                        "selected_model": {config_payload["model_option"]!r},
                        "selected_metric": SELECTED_METRIC,
                        "tested_configurations": int(len(competition_dashboard)),
                        "search_space": {{key: [str(item) for item in values] for key, values in search_grid.items()}},
                        "best_params": search.best_params_,
                        "best_cv_metric": convert_metric(SELECTED_METRIC, search.best_score_),
                    }},
                }}
            """
        ).strip(),
        textwrap.dedent(
            """
            optimization_state = hyperparameter_optimization(split_state["train"])
            display(pd.DataFrame([optimization_state["summary"]]))
            display(optimization_state["competition_dashboard"].head(12))
            """
        ).strip(),
    ]
    return "\n\n".join(section for section in sections if section.strip())


def notebook_validation_section_source(
    brief: ProjectBrief,
    selected_metric: str,
    baseline_strategy: str,
    baseline_reason: str,
) -> str:
    metric_imports = {
        "accuracy": "from sklearn.metrics import accuracy_score",
        "balanced_accuracy": "from sklearn.metrics import balanced_accuracy_score",
        "f1_macro": "from sklearn.metrics import f1_score",
        "rmse": "from sklearn.metrics import mean_squared_error",
        "mae": "from sklearn.metrics import mean_absolute_error",
        "r2": "from sklearn.metrics import r2_score",
    }[selected_metric]
    evaluation_source = {
        "accuracy": textwrap.dedent(
            """
            def evaluate_predictions(y_true, predictions):
                return {
                    SELECTED_METRIC: float(accuracy_score(y_true, predictions)),
                }
            """
        ).strip(),
        "balanced_accuracy": textwrap.dedent(
            """
            def evaluate_predictions(y_true, predictions):
                return {
                    SELECTED_METRIC: float(balanced_accuracy_score(y_true, predictions)),
                }
            """
        ).strip(),
        "f1_macro": textwrap.dedent(
            """
            def evaluate_predictions(y_true, predictions):
                return {
                    SELECTED_METRIC: float(f1_score(y_true, predictions, average="macro")),
                }
            """
        ).strip(),
        "rmse": textwrap.dedent(
            """
            def evaluate_predictions(y_true, predictions):
                return {
                    SELECTED_METRIC: float(mean_squared_error(y_true, predictions, squared=False)),
                }
            """
        ).strip(),
        "mae": textwrap.dedent(
            """
            def evaluate_predictions(y_true, predictions):
                return {
                    SELECTED_METRIC: float(mean_absolute_error(y_true, predictions)),
                }
            """
        ).strip(),
        "r2": textwrap.dedent(
            """
            def evaluate_predictions(y_true, predictions):
                return {
                    SELECTED_METRIC: float(r2_score(y_true, predictions)),
                }
            """
        ).strip(),
    }[selected_metric]

    if baseline_strategy == "stratified_random":
        baseline_source = textwrap.dedent(
            """
            def baseline_predictions(y_train, y_test):
                class_priors = y_train.value_counts(normalize=True).sort_index()
                return np.random.default_rng(42).choice(
                    class_priors.index.to_list(),
                    size=len(y_test),
                    p=class_priors.to_numpy(dtype=float),
                )
            """
        ).strip()
    elif baseline_strategy == "most_frequent":
        baseline_source = textwrap.dedent(
            """
            def baseline_predictions(y_train, y_test):
                mode = y_train.mode(dropna=True)
                value = mode.iloc[0] if not mode.empty else None
                return np.repeat(value, len(y_test))
            """
        ).strip()
    elif baseline_strategy == "median_value":
        baseline_source = textwrap.dedent(
            """
            def baseline_predictions(y_train, y_test):
                return np.repeat(float(pd.to_numeric(y_train, errors="coerce").median()), len(y_test))
            """
        ).strip()
    else:
        baseline_source = textwrap.dedent(
            """
            def baseline_predictions(y_train, y_test):
                return np.repeat(float(pd.to_numeric(y_train, errors="coerce").mean()), len(y_test))
            """
        ).strip()

    if brief.task_type == "regression":
        plot_source = textwrap.dedent(
            """
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(prediction_frame[target_column], prediction_frame["prediction"], alpha=0.8)
            ax.set_xlabel("actual")
            ax.set_ylabel("prediction")
            ax.set_title("Winning-model predictions on the holdout rows")
            plt.tight_layout()
            plt.show()
            """
        ).strip()
    else:
        plot_source = textwrap.dedent(
            """
            confusion = (
                prediction_frame.groupby([target_column, "prediction"])
                .size()
                .reset_index(name="count")
                .sort_values([target_column, "prediction"])
            )
            print(confusion)
            """
        ).strip()
    indented_plot_source = textwrap.indent(plot_source, " " * 16)

    sections = [
        f"import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n{metric_imports}",
        f"BASELINE_STRATEGY = {baseline_strategy!r}\nBASELINE_REASON = {baseline_reason!r}\nSELECTED_METRIC = {selected_metric!r}",
        evaluation_source,
        baseline_source,
        textwrap.dedent(
            f"""
            def validate(test_split, trained_model):
                target_column = test_split.attrs.get("target_column", {brief.target_column!r})
                X_test = test_split.drop(columns=[target_column])
                y_test = test_split[target_column]
                predictions = trained_model["model"].predict(X_test)
                baseline = baseline_predictions(trained_model["y_train"], y_test)
                model_metrics = evaluate_predictions(y_test, predictions)
                baseline_metrics = evaluate_predictions(y_test, baseline)
                prediction_frame = X_test.reset_index(drop=True).copy()
                prediction_frame[target_column] = y_test.reset_index(drop=True)
                prediction_frame["prediction"] = pd.Series(predictions).reset_index(drop=True)

                metric_frame = pd.DataFrame(
                    [
                        {{"series": "model", "metric": SELECTED_METRIC, "value": model_metrics[SELECTED_METRIC]}},
                        {{"series": "baseline", "metric": SELECTED_METRIC, "value": baseline_metrics[SELECTED_METRIC]}},
                    ]
                )

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(metric_frame["series"], metric_frame["value"], alpha=0.8)
                ax.set_title(f"Model vs baseline on {{SELECTED_METRIC}}")
                ax.set_ylabel(SELECTED_METRIC)
                plt.tight_layout()
                plt.show()

{indented_plot_source}

                return {{
                    "metric_frame": metric_frame,
                    "prediction_frame": prediction_frame,
                    "summary": {{
                        "selected_model": trained_model["model_name"],
                        "selected_metric": SELECTED_METRIC,
                        "baseline_strategy": BASELINE_STRATEGY,
                        "baseline_reason": BASELINE_REASON,
                        "model_metrics": model_metrics,
                        "baseline_metrics": baseline_metrics,
                        "metric_delta_vs_baseline": model_metrics[SELECTED_METRIC] - baseline_metrics[SELECTED_METRIC],
                    }},
                }}
            """
        ).strip(),
        'winning_state = optimization_state if not optimization_state.get("skipped") else training_state\nvalidation_state = validate(split_state["test"], winning_state)\nprediction_frame = validation_state["prediction_frame"].copy()\nvalidation_state["summary"]',
    ]
    return "\n\n".join(section for section in sections if section.strip())


def build_workflow_output_notebook(
    project_dir: str | Path,
    brief: ProjectBrief,
    profile,
    recommendations: list[WorkflowRecommendation],
    selected_options: dict[str, str],
    feedback_map: dict[str, dict[str, object]],
    config_payload: dict[str, Any],
    training_summary: dict[str, Any],
    validation_summary: dict[str, Any],
    final_validation_summary: dict[str, Any],
    optimization_summary: dict[str, Any] | None,
) -> str:
    project_path = Path(project_dir).resolve()
    notebook_path = project_path / f"{slugify(brief.project_name)}_workflow_output.ipynb"
    relative_dataset_path = notebook_relative_dataset_path(project_path, brief.dataset_path)
    selected_features = (
        list(config_payload.get("preprocessing_overrides", {}).get("keep_features", []))
        or [*profile.numeric_features, *profile.categorical_features]
    )
    preprocessing_spec = build_notebook_preprocessing_spec(
        brief,
        profile,
        config_payload,
        selected_features,
    )
    optimization_mode = config_payload.get("hpo_option", "skip")
    baseline_strategy = final_validation_summary.get("baseline_strategy") or validation_summary.get("baseline_strategy", "")
    baseline_reason = final_validation_summary.get("baseline_reason") or validation_summary.get("baseline_reason", "")
    selected_feature_display = ", ".join(f"`{feature}`" for feature in selected_features) if selected_features else "all available features"
    selected_configuration_lines = [
        "## Selected Configuration",
        "",
        "Relevant input information:",
        f"- Dataset: `{relative_dataset_path}`",
        f"- Target column: `{brief.target_column}`",
        f"- Task type: `{brief.task_type}`",
        f"- Selected input features: {selected_feature_display}",
    ]
    if config_payload.get("split_option") == "time_ordered_holdout" and brief.date_column:
        selected_configuration_lines.append(f"- Time-order column: `{brief.date_column}`")

    selected_configuration_lines.extend(
        [
            "",
            "Selected policies:",
            f"- Preprocessing: `{config_payload.get('preprocessing_option')}`",
            f"- Data splitting: `{config_payload.get('split_option')}`",
            f"- Model selection: `{config_payload.get('model_option')}`",
            f"- Metric selection: `{config_payload.get('metric_option')}`",
            f"- Training: `{config_payload.get('training_option')}`",
            f"- Validation: `{config_payload.get('validation_option')}`",
            f"- Hyperparameter optimization: `{optimization_mode}`",
            f"- Final validation: `{config_payload.get('final_validation_option')}`",
            "",
            "This notebook embeds only the code needed for the selected workflow path and does not import `agentic_automl`.",
        ]
    )

    notebook = {
        "cells": [
            notebook_markdown_cell(
                f"""
                # Agentic AutoML Workflow Output

                Project: `{brief.project_name}`

                This notebook is the only exported artifact for the run.

                It captures the selected workflow as a compact runnable MVP from the original dataset path.
                """
            ),
            notebook_markdown_cell(
                "\n".join(selected_configuration_lines)
            ),
            notebook_markdown_cell("## 1. Preprocessing"),
            notebook_code_cell(
                notebook_preprocessing_section_source(
                    brief,
                    relative_dataset_path,
                    selected_features,
                    preprocessing_spec,
                )
            ),
            notebook_markdown_cell("## 2. Data Splitting"),
            notebook_code_cell(
                notebook_split_section_source(brief, config_payload)
            ),
            notebook_markdown_cell("## 3. Model Training"),
            notebook_code_cell(
                notebook_training_section_source(brief, config_payload)
            ),
            notebook_markdown_cell("## 4. Hyperparameter Optimization"),
            notebook_code_cell(
                notebook_hpo_section_source(brief, config_payload)
            ),
            notebook_markdown_cell("## 5. Validation"),
            notebook_code_cell(
                notebook_validation_section_source(
                    brief,
                    config_payload["metric_option"],
                    baseline_strategy,
                    baseline_reason,
                )
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return str(notebook_path)


def build_workflow_payload(
    brief: ProjectBrief,
    profile,
    recommendations: list[WorkflowRecommendation],
    selected_options: dict[str, str],
    feedback_map: dict[str, dict[str, object]],
    model_result: CandidateResult,
    baseline_result: CandidateResult,
    optimization_summary: dict[str, Any] | None,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "brief": brief.to_dict(),
        "profile": profile.to_dict(),
        "workflow_decisions": {item.step_id: selected_options.get(item.step_id, item.selected_option or "") for item in recommendations},
        "workflow_step_feedback": feedback_map,
        "validation_result": model_result.to_dict(),
        "baseline_result": baseline_result.to_dict(),
        "optimization_summary": optimization_summary or {},
        "generated_at": generated_at,
    }


def render_report(
    brief: ProjectBrief,
    profile,
    recommendations: list[WorkflowRecommendation],
    training_summary: dict[str, Any],
    validation_summary: dict[str, Any],
    final_validation_summary: dict[str, Any],
    optimization_summary: dict[str, Any] | None,
    feedback_map: dict[str, dict[str, object]],
) -> str:
    lines = [
        f"# Agentic AutoML Report: {brief.project_name}",
        "",
        "## Project brief",
        f"- Dataset: `{brief.dataset_path}`",
        f"- Problem: {brief.problem_description}",
        f"- Task type: `{brief.task_type}`",
        f"- Target column: `{brief.target_column}`",
        "",
        "## Data profile",
        f"- {format_profile_summary(profile)}",
        "",
        "## Workflow decisions",
    ]
    for recommendation in recommendations:
        feedback = feedback_map.get(recommendation.step_id, {})
        lines.append(
            f"- Step {recommendation.step_number}: {recommendation.title} -> "
            f"`{feedback.get('selected_option', recommendation.selected_option)}`"
        )
        if feedback.get("policy_summary"):
            lines.append(f"  Policy summary: {feedback['policy_summary']}")

    lines.extend(
        [
            "",
            "## Training summary",
            f"- Selected model: `{training_summary.get('selected_model')}`",
            f"- Model parameters: `{training_summary.get('model_parameters')}`",
            f"- Training configuration: `{training_summary.get('training_configuration')}`",
            f"- CV metrics: `{training_summary.get('cv_metrics')}`",
            "",
            "## Validation summary",
            f"- Selected metric: `{validation_summary.get('selected_metric')}`",
            f"- Baseline strategy: `{validation_summary.get('baseline_strategy')}`",
            f"- Baseline reason: {validation_summary.get('baseline_reason')}",
            f"- Model metrics: `{validation_summary.get('model_metrics')}`",
            f"- Baseline metrics: `{validation_summary.get('baseline_metrics')}`",
            "",
            "## Hyperparameter optimization",
        ]
    )
    if optimization_summary:
        lines.extend(
            [
                f"- Best params: `{optimization_summary.get('best_params')}`",
                f"- Best CV metric: `{optimization_summary.get('best_cv_metric')}`",
                f"- Search space: `{optimization_summary.get('search_space')}`",
            ]
        )
    else:
        lines.append("- Skipped.")

    lines.extend(
        [
            "",
            "## Final validation",
            f"- Model metrics: `{final_validation_summary.get('model_metrics')}`",
            f"- Baseline metrics: `{final_validation_summary.get('baseline_metrics')}`",
            f"- Metric delta vs baseline: `{final_validation_summary.get('metric_delta_vs_baseline')}`",
        ]
    )
    return "\n".join(lines)


def materialize_workflow_artifacts(
    context: dict[str, Any],
    training_state: dict[str, Any],
    validation_state: dict[str, Any],
    optimization_state: dict[str, Any] | None = None,
    final_state: dict[str, Any] | None = None,
    output_root: str | Path | None = None,
    build_output_notebook: bool = True,
) -> RunArtifacts:
    brief = context["brief"]
    root = context["repo_root"]
    final_validation_state = final_state or validation_state

    (root / "projects").mkdir(parents=True, exist_ok=True)
    project_slug = slugify(brief.project_name)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    project_dir = Path(output_root or root / "projects" / f"{project_slug}-{now}").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    feedback_map = build_step_feedback_payload(
        context["recommendations"],
        context["selected_options"],
        context["step_feedback"],
    )
    workflow_payload = build_workflow_payload(
        brief,
        context["profile"],
        context["recommendations"],
        context["selected_options"],
        feedback_map,
        final_validation_state["model_result"],
        final_validation_state["baseline_result"],
        optimization_state["summary"] if optimization_state else None,
        now,
    )

    output_notebook_path = None
    if build_output_notebook:
        output_notebook_path = build_workflow_output_notebook(
            project_dir,
            brief,
            context["profile"],
            context["recommendations"],
            context["selected_options"],
            feedback_map,
            context["config"],
            training_state["training_summary"],
            validation_state["summary"],
            final_validation_state["summary"],
            optimization_state["summary"] if optimization_state else None,
        )

    report_markdown = render_report(
        brief,
        context["profile"],
        context["recommendations"],
        training_state["training_summary"],
        validation_state["summary"],
        final_validation_state["summary"],
        optimization_state["summary"] if optimization_state else None,
        feedback_map,
    )

    leaderboard = [validation_state["baseline_result"], validation_state["model_result"]]
    if optimization_state or final_validation_state is not validation_state:
        leaderboard.append(final_validation_state["model_result"])

    metrics_summary = {
        "winner_test_metric": final_validation_state["model_result"].test_metrics.get(
            context["config"]["metric_option"],
            float("nan"),
        ),
        "baseline_test_metric": final_validation_state["baseline_result"].test_metrics.get(
            context["config"]["metric_option"],
            float("nan"),
        ),
        "initial_validation_metric": validation_state["model_result"].test_metrics.get(
            context["config"]["metric_option"],
            float("nan"),
        ),
    }

    return RunArtifacts(
        project_dir=str(project_dir),
        selected_metric=context["config"]["metric_option"],
        higher_is_better=training_state["higher_is_better"],
        winner=final_validation_state["model_result"].model_name,
        baseline=final_validation_state["baseline_result"],
        leaderboard=leaderboard,
        report_markdown=report_markdown,
        metrics_summary=metrics_summary,
        predictions_preview=final_validation_state["predictions_preview"].to_dict(orient="records"),
        workflow_decisions=workflow_payload["workflow_decisions"],
        workflow_step_feedback=feedback_map,
        training_summary=training_state["training_summary"],
        validation_summary=validation_state["summary"],
        optimization_summary=optimization_state["summary"] if optimization_state else {},
        final_validation_summary=final_validation_state["summary"],
        output_notebook_path=output_notebook_path,
    )


def execute_workflow(
    brief: ProjectBrief,
    selected_options: dict[str, str] | None = None,
    step_feedback: dict[str, dict[str, object]] | None = None,
    repo_root: str | Path | None = None,
    output_root: str | Path | None = None,
    build_output_notebook: bool = True,
) -> RunArtifacts:
    context = prepare_execution_context(
        brief,
        selected_options=selected_options,
        step_feedback=step_feedback,
        repo_root=repo_root,
    )

    training_state = train_model_phase(context)
    validation_state = validation_phase(context, training_state, role="candidate", tuned=False)
    optimization_state = hyperparameter_optimization_phase(context, training_state)
    final_state = validation_phase(
        context,
        optimization_state or training_state,
        role="competition" if optimization_state else "candidate",
        tuned=optimization_state is not None,
    )
    return materialize_workflow_artifacts(
        context,
        training_state,
        validation_state,
        optimization_state=optimization_state,
        final_state=final_state,
        output_root=output_root,
        build_output_notebook=build_output_notebook,
    )
