from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from .paths import resolve_repo_root
from . import final_validation_actions
from . import hpo_actions
from . import metric_actions
from . import model_actions
from . import preprocessing_actions
from . import split_actions
from . import training_actions
from . import validation_actions
from .modeling import (
    default_hpo_search_parameters,
    hpo_parameter_reason_map,
    hpo_supported_parameters,
)
from .preprocessing import merge_preprocessing_overrides, normalize_preprocessing_overrides
from .schemas import ProjectBrief
from .step_knowledge import append_limit_request, knowledge_file, load_capability_keys, section_items


STARTER_MESSAGE = """Explain the project in a short structured brief:

project_name: customer-churn-mvp
dataset_path: /absolute/path/to/data.csv
target_column: churned
task_type: classification
problem_description: predict which customers are likely to churn next month
competition_enabled: yes
date_column:
"""

AGREEMENT_LABELS = {
    "agree": "I agree with the default procedure",
    "different": "I want something different",
}

FIELD_ALIASES = {
    "project": "project_name",
    "project name": "project_name",
    "project_name": "project_name",
    "name": "project_name",
    "dataset": "dataset_path",
    "dataset path": "dataset_path",
    "dataset_path": "dataset_path",
    "data": "dataset_path",
    "path": "dataset_path",
    "target": "target_column",
    "target column": "target_column",
    "target_column": "target_column",
    "label": "target_column",
    "task": "task_type",
    "task type": "task_type",
    "task_type": "task_type",
    "problem type": "task_type",
    "problem": "problem_description",
    "problem description": "problem_description",
    "problem_description": "problem_description",
    "description": "problem_description",
    "objective": "problem_description",
    "competition": "competition_enabled",
    "competition enabled": "competition_enabled",
    "competition_enabled": "competition_enabled",
    "baseline metric": "baseline_metric",
    "baseline_metric": "baseline_metric",
    "date column": "date_column",
    "date_column": "date_column",
}

REQUIRED_FIELDS = ["dataset_path", "target_column", "task_type", "problem_description"]
QUESTION_WORDS = {
    "what",
    "how",
    "where",
    "why",
    "when",
    "which",
    "can",
    "could",
    "should",
    "would",
    "do",
    "does",
    "did",
    "is",
    "are",
    "will",
}
DATASET_PATTERN = re.compile(r"(?P<path>(?:/|\.{1,2}/)?[\w./-]+\.(?:csv|parquet|pq))", re.IGNORECASE)
TARGET_PATTERN = re.compile(r"(?:target(?:\s+column)?|label)\s*(?:is|=|:)\s*([A-Za-z_][\w-]*)", re.IGNORECASE)
PROJECT_PATTERN = re.compile(r"(?:project(?:\s+name)?)\s*(?:is|=|:)\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)

STEP_PURPOSES = {
    "00_intake": "what the workflow needs to know before it can plan the project",
    "01_preprocessing": "how the raw table is cleaned and encoded before modeling",
    "02_data_splitting": "how training and final validation data are separated",
    "03_model_selection": "which specific model should be used as the starting point",
    "05_metric_selection": "which primary score decides the winner",
    "06_training_configuration": "which concrete training parameters should be used for the selected model",
    "08_validation_and_baseline": "how the current trained model is validated against a no-model baseline",
    "07_hyperparameter_optimization": "which hyperparameters should be optimized for the selected model",
    "09_final_validation": "how the final tuned model is validated and summarized for export",
}

OPTION_DESCRIPTIONS = {
    "00_intake": {
        "structured_brief": "Use the standard project brief with dataset path, target column, task type, and success goal.",
    },
    "01_preprocessing": {
        "auto_tabular_preprocessing": "Automatic tabular cleanup with imputation, one-hot encoding, and scaling where needed.",
        "minimal_cleanup": "Keep preprocessing lightweight and avoid unnecessary transformations.",
        "custom": "Use a custom preprocessing path with explicit feature, scope, and encoding overrides.",
    },
    "02_data_splitting": {
        "stratified_holdout": "Keep class proportions similar across train and test splits.",
        "random_holdout": "Randomly split rows into train and test data.",
        "time_ordered_holdout": "Keep earlier rows for training and later rows for validation to preserve chronology.",
    },
    "03_model_selection": {
        "logistic_regression": "Use logistic regression as the initial classification model.",
        "random_forest_classifier": "Use a random forest classifier as the initial classification model.",
        "hist_gradient_boosting_classifier": "Use histogram gradient boosting as the initial classification model.",
        "mlp_classifier": "Use a multilayer perceptron classifier as the initial classification model.",
        "ridge_regression": "Use ridge regression as the initial regression model.",
        "random_forest_regressor": "Use a random forest regressor as the initial regression model.",
        "hist_gradient_boosting_regressor": "Use histogram gradient boosting as the initial regression model.",
        "mlp_regressor": "Use a multilayer perceptron regressor as the initial regression model.",
    },
    "05_metric_selection": {
        "balanced_accuracy": "Reward performance across classes more evenly when imbalance matters.",
        "f1_macro": "Balance precision and recall across classes.",
        "accuracy": "Optimize for overall correctness across all predictions.",
        "rmse": "Penalize larger regression errors more strongly.",
        "mae": "Measure average absolute regression error in a more robust way.",
        "r2": "Optimize for explained variance in the target.",
    },
    "06_training_configuration": {
        "fast_training": "Use a lighter training budget for speed.",
        "standard_training": "Use the default reproducible training setup.",
        "thorough_training": "Spend more training budget for a stronger estimate.",
    },
    "07_hyperparameter_optimization": {
        "skip": "Skip the extra tuning competition and validate the best baseline candidate directly.",
        "small_competition": "Run a compact tuning competition on the strongest candidates.",
        "expanded_competition": "Run a broader tuning competition with more search budget.",
    },
    "08_validation_and_baseline": {
        "test_set_with_baseline": "Compare the current model to a no-model baseline on the untouched test split.",
    },
    "09_final_validation": {
        "final_validation_dashboard": "Repeat validation for the final model and build the export dashboard story.",
    },
}

OPTION_KEYWORDS = {
    "01_preprocessing": {
        "auto_tabular_preprocessing": ["auto", "automatic", "one hot", "standard preprocessing"],
        "minimal_cleanup": ["minimal", "lightweight", "simple cleanup", "minimal cleanup", "keep it simple"],
        "custom": ["custom", "manual", "domain rule", "special rule", "business rule", "special preprocessing"],
    },
    "02_data_splitting": {
        "stratified_holdout": ["stratified", "imbalance", "class balance", "same class mix"],
        "random_holdout": ["random", "shuffle", "random split"],
        "time_ordered_holdout": ["time", "chronolog", "future", "date split", "temporal", "sequence", "ordered"],
    },
    "03_model_selection": {
        "logistic_regression": ["logistic", "logistic regression"],
        "random_forest_classifier": ["random forest classifier", "random forest", "forest classifier"],
        "hist_gradient_boosting_classifier": ["hist gradient boosting classifier", "gradient boosting classifier", "boosting classifier"],
        "mlp_classifier": ["mlp classifier", "neural network classifier", "perceptron classifier"],
        "ridge_regression": ["ridge regression", "ridge model"],
        "random_forest_regressor": ["random forest regressor", "forest regressor"],
        "hist_gradient_boosting_regressor": ["hist gradient boosting regressor", "gradient boosting regressor", "boosting regressor"],
        "mlp_regressor": ["mlp regressor", "neural network regressor", "perceptron regressor"],
    },
    "05_metric_selection": {
        "balanced_accuracy": ["balanced accuracy", "imbalance", "class balance", "minority class"],
        "f1_macro": ["f1", "precision and recall", "macro f1", "recall and precision"],
        "accuracy": ["accuracy", "overall correctness", "percent correct"],
        "rmse": ["rmse", "root mean square", "large errors"],
        "mae": ["mae", "absolute error", "robust to outliers"],
        "r2": ["r2", "explained variance", "variance explained"],
    },
    "06_training_configuration": {
        "fast_training": ["fast", "quick", "cheap", "lighter training"],
        "standard_training": ["standard", "default", "normal training"],
        "thorough_training": ["thorough", "careful", "deeper training", "more epochs", "stronger estimate"],
    },
    "07_hyperparameter_optimization": {
        "skip": ["skip", "no tuning", "no hpo"],
        "small_competition": ["small", "compact", "light tuning", "small competition"],
        "expanded_competition": ["expanded", "bigger", "broader", "full competition", "deeper search"],
    },
    "08_validation_and_baseline": {
        "test_set_with_baseline": ["baseline", "holdout", "test set", "compare to baseline"],
    },
    "09_final_validation": {
        "final_validation_dashboard": ["final validation", "dashboard", "tuned model validation"],
    },
}

STEP_REFERENCE_KEYWORDS = {
    "00_intake": ["intake", "brief", "project brief"],
    "01_preprocessing": ["preprocessing", "preprocess", "cleaning", "encoding"],
    "02_data_splitting": ["data splitting", "split", "holdout", "train test split", "validation split"],
    "03_model_selection": ["model selection", "models", "selected model", "model family"],
    "05_metric_selection": ["metric selection", "metric", "score", "objective metric"],
    "06_training_configuration": ["training", "training configuration", "optimizer", "epochs", "learning rate", "batch size"],
    "08_validation_and_baseline": ["validation", "baseline", "test set", "model validation"],
    "07_hyperparameter_optimization": ["hyperparameter", "competition", "hpo", "tuning"],
    "09_final_validation": ["final validation", "tuned validation", "final dashboard"],
}

PREFERENCE_HINTS = ["i want", "prefer", "instead", "use ", "please", "let's", "lets", "do ", "keep "]
MAIN_CHAT_HEIGHT = 520
STEP_CHAT_HEIGHT = 260
DISCUSSION_HISTORY_KEY = "discussion_history"
ACTION_HISTORY_KEY = "action_history"
CAPITAL_CITY_LOOKUP = {
    "france": "Paris",
    "germany": "Berlin",
    "spain": "Madrid",
    "italy": "Rome",
    "portugal": "Lisbon",
    "united kingdom": "London",
    "uk": "London",
    "england": "London",
    "united states": "Washington, D.C.",
    "usa": "Washington, D.C.",
}

def normalize_field_name(raw_key: str) -> str | None:
    key = raw_key.strip().lower().replace("-", " ").replace("_", " ")
    return FIELD_ALIASES.get(key)


def infer_task_type(text: str) -> str | None:
    lowered = text.lower()
    if any(token in lowered for token in ["classification", "classify", "classifier"]):
        return "classification"
    if any(token in lowered for token in ["regression", "regress", "predict a number", "continuous target"]):
        return "regression"
    return None


def is_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    first_word = stripped.split(maxsplit=1)[0].lower().strip(".,:;!")
    return first_word in QUESTION_WORDS


def extract_brief_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    stripped = text.strip()

    for raw_line in stripped.splitlines():
        line = raw_line.strip().lstrip("-*").strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = normalize_field_name(key)
        clean_value = value.strip()
        if normalized_key and clean_value:
            fields[normalized_key] = clean_value

    if "dataset_path" not in fields:
        match = DATASET_PATTERN.search(stripped)
        if match:
            fields["dataset_path"] = match.group("path")

    if "target_column" not in fields:
        match = TARGET_PATTERN.search(stripped)
        if match:
            fields["target_column"] = match.group(1)

    if "project_name" not in fields:
        match = PROJECT_PATTERN.search(stripped)
        if match:
            fields["project_name"] = match.group(1)

    if "task_type" not in fields:
        task_type = infer_task_type(stripped)
        if task_type:
            fields["task_type"] = task_type

    if "problem_description" not in fields and not is_question(stripped):
        lowered = stripped.lower()
        if any(token in lowered for token in ["predict", "forecast", "classif", "regress", "estimate", "detect", "optimiz"]):
            fields["problem_description"] = stripped

    return fields


def build_brief_from_fields(fields: dict[str, str]) -> tuple[ProjectBrief | None, list[str]]:
    working = dict(fields)
    dataset_path = working.get("dataset_path")
    if dataset_path and not working.get("project_name"):
        working["project_name"] = Path(dataset_path).stem.replace("_", "-") or "project"

    missing = [field for field in REQUIRED_FIELDS if not working.get(field)]
    if missing:
        return None, missing

    task_type = infer_task_type(working["task_type"]) or working["task_type"].strip().lower()
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be classification or regression.")

    brief = ProjectBrief(
        project_name=working["project_name"],
        dataset_path=working["dataset_path"],
        target_column=working["target_column"],
        task_type=task_type,
        problem_description=working["problem_description"],
        date_column=working.get("date_column") or None,
        baseline_metric=working.get("baseline_metric") or None,
        competition_enabled=working.get("competition_enabled", "no").lower() in {"yes", "true", "1"},
    )
    return brief, []


def build_initial_step_feedback(recommendations: list[dict]) -> dict[str, dict[str, Any]]:
    return {
        recommendation["step_id"]: {
            "agreement": "agree",
            "custom_note": "",
            "policy_summary": "",
            "policy_metadata": {},
            "policy_confirmed": True,
            DISCUSSION_HISTORY_KEY: [],
            ACTION_HISTORY_KEY: [],
        }
        for recommendation in recommendations
    }


def normalize_step_feedback_entry(feedback: dict[str, Any], recommendation: dict) -> dict[str, Any]:
    feedback.setdefault("agreement", "agree")
    if feedback.get("agreement") == "pending":
        feedback["agreement"] = "agree"
    feedback.setdefault("custom_note", "")
    feedback.setdefault("policy_summary", "")
    feedback.setdefault("policy_metadata", {})
    has_legacy_custom_state = bool(
        feedback.get("discussion")
        or feedback.get(ACTION_HISTORY_KEY)
        or feedback.get("custom_note")
        or feedback.get("policy_summary")
    )
    feedback.setdefault("policy_confirmed", not has_legacy_custom_state)
    if "discussion" in feedback and ACTION_HISTORY_KEY not in feedback:
        feedback[ACTION_HISTORY_KEY] = feedback.pop("discussion")
    feedback.setdefault(DISCUSSION_HISTORY_KEY, [])
    feedback.setdefault(ACTION_HISTORY_KEY, [])
    return feedback


def validate_step_feedback(
    recommendations: list[dict],
    step_feedback: dict[str, dict[str, Any]],
    selected_options: dict[str, str],
) -> list[str]:
    issues: list[str] = []
    for recommendation in recommendations:
        step_id = recommendation["step_id"]
        title = recommendation["title"]
        default_option = recommendation.get("selected_option") or ""
        current_feedback = step_feedback.get(step_id, {})
        agreement = current_feedback.get("agreement", "agree")
        custom_note = current_feedback.get("custom_note", "").strip()
        policy_summary = current_feedback.get("policy_summary", "").strip()
        policy_confirmed = bool(current_feedback.get("policy_confirmed", False))
        chosen_option = selected_options.get(step_id, default_option)

        if agreement == "pending":
            issues.append(f"{title}: choose whether you agree with the default or want something different.")
            continue

        if agreement == "different" and chosen_option == default_option and not custom_note and not policy_summary:
            issues.append(f"{title}: describe what should be different so I can build a working execution policy.")

        if agreement == "different" and (custom_note or policy_summary) and not policy_confirmed:
            issues.append(f"{title}: confirm the custom policy before running the workflow.")

    return issues


def summarize_step_feedback(step_feedback: dict[str, dict[str, Any]], step_id: str) -> str:
    agreement = step_feedback.get(step_id, {}).get("agreement", "agree")
    if agreement == "agree":
        return "agreed"
    if agreement == "different":
        return "customized"
    return "pending"


def build_workflow_step_feedback_payload(
    recommendations: list[dict],
    step_feedback: dict[str, dict[str, Any]],
    selected_options: dict[str, str],
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for recommendation in recommendations:
        step_id = recommendation["step_id"]
        payload[step_id] = {
            "title": recommendation["title"],
            "agreement": step_feedback.get(step_id, {}).get("agreement", "agree"),
            "default_option": recommendation.get("selected_option") or "",
            "selected_option": selected_options.get(step_id, recommendation.get("selected_option") or ""),
            "custom_note": step_feedback.get(step_id, {}).get("custom_note", "").strip(),
            "policy_summary": step_feedback.get(step_id, {}).get("policy_summary", "").strip(),
            "policy_metadata": step_feedback.get(step_id, {}).get("policy_metadata", {}),
            "policy_confirmed": bool(step_feedback.get(step_id, {}).get("policy_confirmed", False)),
        }
    return payload


def humanize_option(option: str) -> str:
    return option.replace("_", " ")


def describe_option(step_id: str, option: str) -> str:
    return OPTION_DESCRIPTIONS.get(step_id, {}).get(option, humanize_option(option).capitalize())


def current_model_selection(
    selected_options: dict[str, str] | None,
    recommendation: dict[str, Any] | None = None,
) -> str | None:
    if isinstance(selected_options, dict):
        selected_model = selected_options.get("03_model_selection")
        if selected_model:
            return selected_model
    if isinstance(recommendation, dict):
        metadata = recommendation.get("metadata", {})
        if isinstance(metadata, dict):
            model_name = metadata.get("model_option")
            if model_name:
                return str(model_name)
    return None


def hpo_knowledge_heading_for_model(model_name: str | None) -> str | None:
    if not model_name:
        return None
    return model_name.replace("_", " ").title()


def build_unsupported_action_reply(
    step_id: str,
    user_message: str,
    reasons: list[str] | None = None,
    keep_current_policy: bool = False,
) -> str:
    limit_path = append_limit_request(step_id, user_message)
    knowledge_path = knowledge_file(step_id)
    lines = [
        "Sorry I can not perform this task yet. I checked the current step knowledge and this request is outside the executable action space, so I stored it in LIMITS.md.",
    ]
    if reasons:
        lines.extend(["", "Why I could not apply it yet:"])
        for reason in reasons:
            lines.append(f"- {reason}")
    if keep_current_policy:
        lines.extend(["", "The current working policy is unchanged."])
    lines.extend(
        [
            "",
            f"- Knowledge: `{knowledge_path}`",
            f"- Limits: `{limit_path}`",
        ]
    )
    return "\n".join(lines)


def dedupe_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        ordered.append(message)
    return ordered


def detect_scope_change_summary(
    overrides: dict[str, Any] | None,
    profile: dict[str, Any] | None,
) -> tuple[str | None, set[str]]:
    normalized = normalize_preprocessing_overrides(overrides)
    feature_groups = preprocessing_actions.extract_profile_feature_groups(profile)
    drop_features = set(normalized["drop_features"])
    numeric_features = set(feature_groups["numeric"])
    categorical_features = set(feature_groups["categorical"])
    date_like_features = set(feature_groups["date_like"])
    numeric_reclassified = (
        set(normalized["force_categorical_features"])
        | set(normalized["force_date_features"])
        | set(normalized["force_one_hot_features"])
        | set(normalized["force_ordinal_features"])
    ) & numeric_features
    categorical_reclassified = set(normalized["force_numeric_features"]) & categorical_features

    if (
        categorical_features
        and categorical_features.issubset(drop_features)
        and not (numeric_features & drop_features)
        and not numeric_reclassified
    ):
        return (
            "Use only the numeric input features from the inspected file.",
            categorical_features,
        )

    if (
        numeric_features
        and numeric_features.issubset(drop_features)
        and date_like_features.issubset(drop_features)
        and not categorical_reclassified
    ):
        remaining_categorical = (categorical_features - date_like_features) - drop_features
        if remaining_categorical:
            return (
                "Use only the categorical input features from the inspected file.",
                numeric_features | date_like_features,
            )

    if (
        date_like_features
        and numeric_features.issubset(drop_features)
        and (categorical_features - date_like_features).issubset(drop_features)
        and not categorical_reclassified
    ):
        return (
            "Use only the date-like input features from the inspected file.",
            numeric_features | (categorical_features - date_like_features),
        )

    return None, set()


def detect_feature_subset_change_summary(
    overrides: dict[str, Any] | None,
    profile: dict[str, Any] | None,
) -> tuple[str | None, set[str], set[str]]:
    normalized = normalize_preprocessing_overrides(overrides)
    all_features = preprocessing_actions.extract_profile_feature_names(profile)
    if not all_features:
        return None, set(), set()

    selected_features = [feature for feature in normalized["keep_features"] if feature in all_features]
    if not selected_features:
        return None, set(), set()

    excluded_features = [feature for feature in all_features if feature not in selected_features]
    explicit_exclusions = set(normalized["drop_features"]) | set(normalized["force_identifier_features"])
    if excluded_features and not set(excluded_features).issubset(explicit_exclusions):
        return None, set(), set()

    summary = "Use only the selected input feature subset: " + ", ".join(
        f"`{feature}`" for feature in selected_features
    ) + "."
    return summary, set(excluded_features), set(selected_features)

def summarize_preprocessing_override_changes(overrides: dict[str, Any] | None) -> list[str]:
    normalized = normalize_preprocessing_overrides(overrides)
    changes: list[str] = []
    if normalized["feature_transform_rules"]:
        for rule in normalized["feature_transform_rules"]:
            changes.append(
                "Apply "
                + rule["kind"]
                + " to "
                + f"`{rule['feature']}`"
                + (
                    f" with value `{rule.get('value')}`"
                    if rule.get("value") is not None and rule["kind"] in {"multiply", "divide", "add", "subtract"}
                    else ""
                )
                + "."
            )
    if normalized["derived_feature_rules"]:
        for rule in normalized["derived_feature_rules"]:
            changes.append(
                f"Create `{rule['output']}` from the {rule['kind']} signal of `{rule['source']}`."
            )
    explicit_one_hot = set(normalized["force_one_hot_features"])
    explicit_ordinal = set(normalized["force_ordinal_features"])
    generic_categorical = [
        column
        for column in normalized["force_categorical_features"]
        if column not in explicit_one_hot and column not in explicit_ordinal
    ]
    if normalized["force_one_hot_features"]:
        changes.append(
            "One-hot encode "
            + ", ".join(f"`{column}`" for column in normalized["force_one_hot_features"])
            + " by routing those feature(s) through the categorical encoding branch."
        )
    if normalized["force_ordinal_features"]:
        changes.append(
            "Ordinal-encode "
            + ", ".join(f"`{column}`" for column in normalized["force_ordinal_features"])
            + " with a compact categorical representation."
        )
    if generic_categorical:
        changes.append(
            "Treat "
            + ", ".join(f"`{column}`" for column in generic_categorical)
            + " as categorical feature(s) instead of numeric/other raw roles."
        )
    if normalized["force_numeric_features"]:
        changes.append(
            "Treat "
            + ", ".join(f"`{column}`" for column in normalized["force_numeric_features"])
            + " as numeric feature(s)."
        )
    if normalized["force_date_features"]:
        changes.append(
            "Treat "
            + ", ".join(f"`{column}`" for column in normalized["force_date_features"])
            + " as date-like field(s) and expand them into calendar parts."
        )
    if normalized["force_identifier_features"]:
        changes.append(
            "Treat "
            + ", ".join(f"`{column}`" for column in normalized["force_identifier_features"])
            + " as identifier field(s) and exclude them from modeling."
        )
    if normalized["feature_imputation_rules"]:
        for rule in normalized["feature_imputation_rules"]:
            if rule["strategy"] == "constant":
                changes.append(
                    f"Impute `{rule['feature']}` with the constant value `{rule.get('value')}`."
                )
            else:
                changes.append(f"Impute `{rule['feature']}` with `{rule['strategy']}`.")
    if normalized["feature_scaling_rules"]:
        for rule in normalized["feature_scaling_rules"]:
            changes.append(f"Scale `{rule['feature']}` with `{rule['method']}`.")
    if normalized["drop_features"]:
        changes.append(
            "Drop " + ", ".join(f"`{column}`" for column in normalized["drop_features"]) + " from the preprocessing graph."
        )
    if normalized["keep_features"]:
        changes.append(
            "Keep " + ", ".join(f"`{column}`" for column in normalized["keep_features"]) + " in the preprocessing graph."
        )
    return changes


def summarize_preprocessing_policy_changes(
    overrides: dict[str, Any] | None,
    profile: dict[str, Any] | None = None,
) -> list[str]:
    normalized = normalize_preprocessing_overrides(overrides)
    changes: list[str] = []
    subset_change, consumed_subset_drop_features, consumed_subset_keep_features = detect_feature_subset_change_summary(
        normalized,
        profile,
    )
    if subset_change:
        changes.append(subset_change)

    scope_change, consumed_drop_features = detect_scope_change_summary(normalized, profile)
    if scope_change:
        changes.append(scope_change)
    consumed_drop_features = set(consumed_drop_features) | set(consumed_subset_drop_features)

    if normalized["feature_transform_rules"]:
        for rule in normalized["feature_transform_rules"]:
            changes.append(
                "Apply "
                + rule["kind"]
                + " to "
                + f"`{rule['feature']}`"
                + (
                    f" with value `{rule.get('value')}`"
                    if rule.get("value") is not None and rule["kind"] in {"multiply", "divide", "add", "subtract"}
                    else ""
                )
                + "."
            )
    if normalized["derived_feature_rules"]:
        for rule in normalized["derived_feature_rules"]:
            changes.append(
                f"Create `{rule['output']}` from the {rule['kind']} signal of `{rule['source']}`."
            )

    explicit_one_hot = set(normalized["force_one_hot_features"])
    explicit_ordinal = set(normalized["force_ordinal_features"])
    generic_categorical = [
        column
        for column in normalized["force_categorical_features"]
        if column not in explicit_one_hot and column not in explicit_ordinal
    ]
    if normalized["force_one_hot_features"]:
        changes.append(
            "One-hot encode "
            + ", ".join(f"`{column}`" for column in normalized["force_one_hot_features"])
            + " by routing those feature(s) through the categorical encoding branch."
        )
    if normalized["force_ordinal_features"]:
        changes.append(
            "Ordinal-encode "
            + ", ".join(f"`{column}`" for column in normalized["force_ordinal_features"])
            + " with a compact categorical representation."
        )
    if generic_categorical:
        changes.append(
            "Treat "
            + ", ".join(f"`{column}`" for column in generic_categorical)
            + " as categorical feature(s) instead of numeric/other raw roles."
        )
    if normalized["force_numeric_features"]:
        changes.append(
            "Treat "
            + ", ".join(f"`{column}`" for column in normalized["force_numeric_features"])
            + " as numeric feature(s)."
        )
    if normalized["force_date_features"]:
        changes.append(
            "Treat "
            + ", ".join(f"`{column}`" for column in normalized["force_date_features"])
            + " as date-like field(s) and expand them into calendar parts."
        )
    if normalized["force_identifier_features"]:
        changes.append(
            "Treat "
            + ", ".join(f"`{column}`" for column in normalized["force_identifier_features"])
            + " as identifier field(s) and exclude them from modeling."
        )
    if normalized["feature_imputation_rules"]:
        for rule in normalized["feature_imputation_rules"]:
            if rule["strategy"] == "constant":
                changes.append(
                    f"Impute `{rule['feature']}` with the constant value `{rule.get('value')}`."
                )
            else:
                changes.append(f"Impute `{rule['feature']}` with `{rule['strategy']}`.")
    if normalized["feature_scaling_rules"]:
        for rule in normalized["feature_scaling_rules"]:
            changes.append(f"Scale `{rule['feature']}` with `{rule['method']}`.")
    remaining_drop_features = [
        column for column in normalized["drop_features"] if column not in consumed_drop_features
    ]
    if remaining_drop_features:
        changes.append(
            "Drop " + ", ".join(f"`{column}`" for column in remaining_drop_features) + " from the preprocessing graph."
        )
    remaining_keep_features = [
        column for column in normalized["keep_features"] if column not in consumed_subset_keep_features
    ]
    if remaining_keep_features:
        changes.append(
            "Keep " + ", ".join(f"`{column}`" for column in remaining_keep_features) + " in the preprocessing graph."
        )
    return dedupe_messages(changes)


def build_preprocessing_policy_summary(
    selected_option: str,
    overrides: dict[str, Any],
    recognized_changes: list[str] | None = None,
) -> str:
    changes = recognized_changes or summarize_preprocessing_override_changes(overrides)
    if not changes:
        return f"Keep `{selected_option}` as the executable preprocessing base."
    return f"Use `{selected_option}` for preprocessing with these execution changes: " + " ".join(changes)


def step_discussion_is_question(text: str) -> bool:
    lowered = text.strip().lower()
    if is_question(text):
        return True
    return any(
        lowered.startswith(prefix)
        for prefix in (
            "explain ",
            "explain why",
            "help me understand",
            "i do not understand",
            "what does ",
            "why ",
            "difference between ",
        )
    )


def infer_execution_option(step_id: str, options: list[str], user_message: str, default_option: str) -> str | None:
    lowered = user_message.lower()
    normalized = lowered.replace("-", " ").replace("_", " ")

    for option in options:
        option_phrase = option.replace("_", " ")
        if option_phrase in normalized:
            return option

    for option, keywords in OPTION_KEYWORDS.get(step_id, {}).items():
        if option not in options:
            continue
        if any(keyword in normalized for keyword in keywords):
            return option

    if any(hint in normalized for hint in PREFERENCE_HINTS) and default_option in options:
        return default_option

    return None


def question_requests_option_menu(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in [
            "what are the options",
            "which options",
            "available policies",
            "available executable policies",
            "difference between",
            "what is the difference",
            "compare the options",
        ]
    )


def format_percent(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.1%}"


def answer_preprocessing_question(
    user_message: str,
    profile: dict[str, Any] | None,
    selected_option: str | None = None,
) -> str | None:
    if not profile:
        return None

    lowered = user_message.lower()
    missing_fraction = float(profile.get("missing_fraction") or 0.0)
    numeric_features = profile.get("numeric_features", [])
    categorical_features = profile.get("categorical_features", [])
    numeric_count = len(numeric_features)
    categorical_count = len(categorical_features)
    categorical_cardinality = profile.get("categorical_cardinality", {})
    date_like_count = len(profile.get("date_like_features", []))
    date_parse_failures = profile.get("date_parse_failure_by_feature", {})
    missing_by_feature = profile.get("missing_by_feature", {})
    numeric_missing_columns = [column for column in numeric_features if column in missing_by_feature]
    categorical_missing_columns = [column for column in categorical_features if column in missing_by_feature]

    if any(token in lowered for token in ["categor", "unique"]) and any(
        token in lowered for token in ["how many", "count", "number"]
    ):
        if not categorical_cardinality:
            if categorical_count == 0:
                return "This dataset does not have categorical input features in the current profile."
            return "I can see categorical input features, but I do not have their category counts available in the current profile."

        lines = [
            f"The dataset has `{categorical_count}` categorical feature(s).",
            "",
            "Category counts by feature:",
        ]
        for column, count in categorical_cardinality.items():
            lines.append(f"- `{column}`: `{count}` categories")
        return "\n".join(lines)

    if "imput" in lowered or "missing" in lowered:
        if missing_fraction <= 0 and not date_parse_failures:
            return (
                "No default imputation is needed from the current dataset profile.\n\n"
                f"I found `{format_percent(missing_fraction)}` missing feature values and no date parsing gaps, so the preprocessing step does not need to fill anything in unless you want a custom rule."
            )
        repair_reasons: list[str] = []
        if numeric_missing_columns:
            repair_reasons.append(
                "numeric gaps in " + ", ".join(f"`{column}`" for column in numeric_missing_columns)
            )
        if categorical_missing_columns:
            repair_reasons.append(
                "categorical gaps in " + ", ".join(f"`{column}`" for column in categorical_missing_columns)
            )
        if date_parse_failures:
            repair_reasons.append(
                "date parsing gaps in "
                + ", ".join(f"`{column}`" for column in date_parse_failures)
            )
        if not repair_reasons and missing_fraction > 0:
            repair_reasons.append(f"`{format_percent(missing_fraction)}` missing feature values across the table")

        imputation_actions: list[str] = []
        if numeric_missing_columns or date_parse_failures:
            imputation_actions.append("median imputation for the incomplete numeric or date-derived features")
        if categorical_missing_columns:
            imputation_actions.append("most-frequent imputation for the incomplete categorical features")
        return (
            "Yes, the dataset needs imputation in the default workflow.\n\n"
            f"I found {' and '.join(repair_reasons)}. "
            f"Because this project has `{numeric_count}` numeric and `{categorical_count}` categorical features, "
            + (
                "the default policy uses " + " and ".join(imputation_actions) + "."
                if imputation_actions
                else "the default policy only repairs the columns that actually contain gaps."
            )
        )

    if "scal" in lowered or "standardscaler" in lowered:
        if selected_option == "minimal_cleanup":
            return (
                "No default scaling is planned for this preprocessing choice.\n\n"
                "The current policy is `minimal_cleanup`, which leaves the numeric features on their original scale because the file already looks clean enough to avoid extra transforms."
            )
        if numeric_count == 0 and date_like_count == 0:
            return (
                "No scaling is needed in the default preprocessing path.\n\n"
                "The current profile does not expose numeric or date-derived input features that would benefit from scaling."
            )
        return (
            "The default scaling method is `StandardScaler`.\n\n"
            "`StandardScaler` centers each numeric feature and scales it to unit variance. "
            "That is the chosen default because it keeps numeric ranges comparable across the broader candidate set, especially for linear and distance-sensitive models."
        )

    if "encoding" in lowered or "one hot" in lowered:
        if categorical_count == 0:
            return (
                "No one-hot encoding is needed in the default preprocessing path.\n\n"
                "The current profile shows no categorical input features, so preprocessing can stay numeric-only."
            )
        return (
            "Yes, one-hot encoding is part of the default preprocessing path.\n\n"
            f"The current profile shows `{categorical_count}` categorical features, so the default workflow encodes them before modeling."
        )

    return None


def answer_contextual_step_question(
    recommendation: dict,
    user_message: str,
    current_option: str | None = None,
    profile: dict[str, Any] | None = None,
    selected_options: dict[str, str] | None = None,
    current_policy_metadata: dict[str, Any] | None = None,
    allow_recommendation_fallback: bool = True,
) -> str | None:
    step_id = recommendation["step_id"]
    lowered = user_message.lower()
    active_option = active_step_option(recommendation, current_option)

    if question_requests_option_menu(user_message):
        return explain_step_choice(recommendation)

    knowledge_answer = answer_step_knowledge_question(
        step_id,
        lowered,
        selected_model=current_model_selection(selected_options, recommendation),
    )
    if knowledge_answer:
        return knowledge_answer

    if step_id == "01_preprocessing":
        preprocessing_answer = answer_preprocessing_question(
            user_message,
            profile,
            selected_option=active_option,
        )
        if preprocessing_answer:
            return preprocessing_answer

    if step_id == "07_hyperparameter_optimization":
        hpo_answer = answer_hpo_question(
            user_message,
            current_model_selection(selected_options, recommendation),
            current_policy_metadata=current_policy_metadata,
        )
        if hpo_answer:
            return hpo_answer

    reasoning = recommendation.get("reasoning", [])
    if reasoning and allow_recommendation_fallback:
        lines = [
            f"For this project, the current active policy is `{active_option}` for {recommendation['title'].lower()}.",
            "",
            "Why:",
        ]
        lines.extend(f"- {line}" for line in reasoning[:3])
        lines.extend(
            [
                "",
                "If you want, ask about a specific tradeoff or describe the behavior you want instead, and I will turn that into a working execution policy.",
            ]
        )
        return "\n".join(lines)

    return None


def answer_hpo_question(
    user_message: str,
    model_name: str | None,
    current_policy_metadata: dict[str, Any] | None = None,
) -> str | None:
    if not model_name:
        return None

    lowered = user_message.lower()
    requested_recommendation = any(
        phrase in lowered
        for phrase in [
            "which hyperparameters would you recommend",
            "what hyperparameters would you recommend",
            "which hyperparameters should",
            "what hyperparameters should",
            "what should we tune",
            "what should i tune",
            "recommend to tune",
            "recommend tuning",
        ]
    )
    if requested_recommendation:
        parameter_reasons = hpo_parameter_reason_map(model_name)
        recommended_parameters = default_hpo_search_parameters(model_name, "expanded_competition")
        lines = [
            f"For this `{model_name}`, I'd tune a focused, high-leverage set rather than everything.",
            "",
        ]
        for parameter_name in recommended_parameters:
            reason = parameter_reasons.get(parameter_name, "is part of the packaged HPO support for this model")
            lines.append(f"- `{parameter_name}`: {reason}.")
        lines.extend(
            [
                "",
                "If you want, tell me the exact list you want to optimize and I will turn it into the executable HPO search scope.",
            ]
        )
        return "\n".join(lines)

    if any(phrase in lowered for phrase in ["current search scope", "what are we tuning", "which hyperparameters are we tuning"]):
        hpo_config = {}
        if isinstance(current_policy_metadata, dict):
            hpo_config = dict(current_policy_metadata.get("hpo_config", {}))
        search_parameters = list(hpo_config.get("search_parameters", []) or [])
        if search_parameters:
            return "The current HPO search scope is: " + ", ".join(f"`{item}`" for item in search_parameters)
        supported = hpo_supported_parameters(model_name)
        if supported:
            return (
                "There is no explicit custom HPO list yet. "
                f"For `{model_name}`, the packaged tuning support covers: "
                + ", ".join(f"`{item}`" for item in supported)
            )
    return None


def format_knowledge_section_answer(step_id: str, heading: str, intro: str) -> str | None:
    items = section_items(step_id, heading)
    if not items:
        return None
    lines = [intro, ""]
    lines.extend(f"- {item}" for item in items)
    lines.extend(["", f"This comes from the current `{heading}` section of `{knowledge_file(step_id).name}`."])
    return "\n".join(lines)


def format_multi_section_knowledge_answer(
    step_id: str,
    intro: str,
    section_map: list[tuple[str, str]],
) -> str | None:
    lines = [intro, ""]
    added_any = False
    for heading, label in section_map:
        items = section_items(step_id, heading)
        if not items:
            continue
        added_any = True
        if label:
            lines.append(f"{label}:")
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    if not added_any:
        return None
    lines.append(f"This comes from the current knowledge sections of `{knowledge_file(step_id).name}`.")
    return "\n".join(lines)


def message_asks_about_support(lowered_message: str) -> bool:
    return any(
        phrase in lowered_message
        for phrase in [
            "supported",
            "support",
            "available",
            "can you do",
            "can i use",
            "do you support",
            "what can you do",
            "what is available",
            "which can i use",
        ]
    )


def section_contains_any_term(step_id: str, heading: str, terms: list[str]) -> bool:
    items = [item.replace("`", "").lower() for item in section_items(step_id, heading)]
    return any(term in item for item in items for term in terms)


def answer_specific_preprocessing_encoding_support_question(lowered_message: str) -> str | None:
    if not message_asks_about_support(lowered_message):
        return None

    encoding_aliases = {
        "one-hot encoding": ["one hot", "one-hot", "dummy encoding", "dummy variable"],
        "ordinal encoding": ["ordinal", "label encoding", "integer encoding"],
        "frequency encoding": ["frequency encoding", "frequency-encoding"],
        "count encoding": ["count encoding", "count-encoding"],
        "target encoding": ["target encoding", "mean encoding", "leave one out encoding", "catboost encoding"],
        "binary encoding": ["binary encoding", "binary-encoding"],
        "hashing encoding": ["hashing encoding", "hash encoding", "feature hashing"],
        "weight-of-evidence encoding": ["weight of evidence", "woe encoding", "woe"],
    }

    for label, aliases in encoding_aliases.items():
        if not any(alias in lowered_message for alias in aliases):
            continue
        supported = section_contains_any_term("01_preprocessing", "Encoding", aliases)
        if supported:
            section_answer = format_knowledge_section_answer(
                "01_preprocessing",
                "Encoding",
                f"Yes, {label} is currently supported in preprocessing. The encoding actions listed in the current knowledge are:",
            )
            return section_answer
        section_answer = format_knowledge_section_answer(
            "01_preprocessing",
            "Encoding",
            f"No, {label} is not part of the current preprocessing encoding support. The encoding actions listed in the current knowledge are:",
        )
        return section_answer
    return None


def answer_step_knowledge_question(
    step_id: str,
    lowered_message: str,
    selected_model: str | None = None,
) -> str | None:
    support_query = message_asks_about_support(lowered_message)

    if step_id == "01_preprocessing":
        specific_encoding_answer = answer_specific_preprocessing_encoding_support_question(lowered_message)
        if specific_encoding_answer:
            return specific_encoding_answer

        if support_query and any(
            phrase in lowered_message
            for phrase in [
                "encoding",
                "encodings",
                "encoder",
                "encoders",
                "one hot",
                "one-hot",
                "ordinal",
                "dummy encoding",
            ]
        ):
            return format_knowledge_section_answer(
                step_id,
                "Encoding",
                "The encodings currently supported in preprocessing are:",
            )
        if support_query and any(
            phrase in lowered_message
            for phrase in [
                "imputation",
                "imputations",
                "impute",
                "missing value",
                "fill missing",
            ]
        ):
            return format_knowledge_section_answer(
                step_id,
                "Imputation",
                "The imputation rules currently supported in preprocessing are:",
            )
        if support_query and any(
            phrase in lowered_message
            for phrase in [
                "scaling",
                "scale",
                "scaler",
                "scalers",
                "normalization",
                "normalisation",
                "standardization",
                "standardisation",
                "minmax",
                "robust",
            ]
        ):
            return format_knowledge_section_answer(
                step_id,
                "Scaling",
                "The scaling methods currently supported in preprocessing are:",
            )
        if support_query and any(
            phrase in lowered_message
            for phrase in [
                "transformation",
                "transformations",
                "transform",
                "transforms",
                "log",
                "log1p",
                "sqrt",
                "square",
                "absolute value",
            ]
        ):
            return format_knowledge_section_answer(
                step_id,
                "Numeric Transformations",
                "The numeric transformations currently supported in preprocessing are:",
            )
        if support_query and any(
            phrase in lowered_message
            for phrase in [
                "derived feature",
                "derived features",
                "feature engineering",
                "frequency count",
                "new column",
                "new feature",
            ]
        ):
            return format_knowledge_section_answer(
                step_id,
                "Derived Features",
                "The derived-feature actions currently supported in preprocessing are:",
            )

    if step_id == "03_model_selection" and support_query and any(
        phrase in lowered_message
        for phrase in ["model", "models", "classifier", "classifiers", "regressor", "regressors"]
    ):
        if "classification" in lowered_message or "classifier" in lowered_message:
            return format_knowledge_section_answer(
                step_id,
                "Classification",
                "The classification models currently supported in model selection are:",
            )
        if "regression" in lowered_message or "regressor" in lowered_message:
            return format_knowledge_section_answer(
                step_id,
                "Regression",
                "The regression models currently supported in model selection are:",
            )
        return format_multi_section_knowledge_answer(
            step_id,
            "The models currently supported in model selection are:",
            [("Classification", "Classification"), ("Regression", "Regression")],
        )

    if step_id == "05_metric_selection" and support_query and any(
        phrase in lowered_message
        for phrase in ["metric", "metrics", "score", "scores"]
    ):
        if "classification" in lowered_message:
            return format_knowledge_section_answer(
                step_id,
                "Classification",
                "The classification metrics currently supported in metric selection are:",
            )
        if "regression" in lowered_message:
            return format_knowledge_section_answer(
                step_id,
                "Regression",
                "The regression metrics currently supported in metric selection are:",
            )
        return format_multi_section_knowledge_answer(
            step_id,
            "The metrics currently supported in metric selection are:",
            [("Classification", "Classification"), ("Regression", "Regression")],
        )

    if step_id == "06_training_configuration" and support_query and any(
        phrase in lowered_message
        for phrase in [
            "training",
            "optimizer",
            "optimizers",
            "learning rate",
            "epochs",
            "batch",
            "cv folds",
            "training parameter",
            "training parameters",
        ]
    ):
        return format_multi_section_knowledge_answer(
            step_id,
            "The training capabilities currently supported in this step are:",
            [
                ("Supported Base Policies", "Supported base policies"),
                ("Supported Action-Mode Changes", "Supported action-mode changes"),
            ],
        )

    if step_id == "07_hyperparameter_optimization" and support_query and any(
        phrase in lowered_message
        for phrase in ["hyperparameter", "hyperparameters", "parameter", "parameters", "hpo", "tuning", "competition", "search"]
    ):
        if any(
            phrase in lowered_message
            for phrase in ["hyperparameter", "hyperparameters", "parameter", "parameters", "search scope", "search space"]
        ):
            heading = hpo_knowledge_heading_for_model(selected_model)
            if heading:
                model_answer = format_knowledge_section_answer(
                    step_id,
                    heading,
                    f"For `{selected_model}`, the hyperparameters currently supported in HPO are:",
                )
                if model_answer:
                    return model_answer
        return format_multi_section_knowledge_answer(
            step_id,
            "The hyperparameter-optimization capabilities currently supported in this step are:",
            [
                ("Supported Base Policies", "Supported base policies"),
                ("Supported Action-Mode Changes", "Supported action-mode changes"),
            ],
        )

    if step_id == "08_validation_and_baseline" and support_query and any(
        phrase in lowered_message
        for phrase in ["validation", "baseline", "output package", "output notebook", "predictions csv", "bundle"]
    ):
        return format_multi_section_knowledge_answer(
            step_id,
            "The validation capabilities currently supported in this step are:",
            [
                ("Supported Base Policies", "Supported base policies"),
                ("Supported Action-Mode Changes", "Supported action-mode changes"),
            ],
        )

    if step_id == "09_final_validation" and support_query and any(
        phrase in lowered_message
        for phrase in ["final validation", "dashboard", "output package", "output notebook", "predictions csv", "bundle", "tuned"]
    ):
        return format_multi_section_knowledge_answer(
            step_id,
            "The final-validation capabilities currently supported in this step are:",
            [
                ("Supported Base Policies", "Supported base policies"),
                ("Supported Action-Mode Changes", "Supported action-mode changes"),
            ],
        )

    if support_query and any(
        phrase in lowered_message
        for phrase in [
            "what is supported",
            "what can you do",
            "which actions are supported",
            "supported actions",
            "action mode changes",
            "what changes can you make",
            "what can action mode do",
        ]
    ):
        action_answer = format_knowledge_section_answer(
            step_id,
            "Supported Action-Mode Changes" if step_id != "00_intake" else "Supported Actions",
            "The current step knowledge says these actions are supported:",
        )
        if action_answer:
            return action_answer

    if support_query and any(
        phrase in lowered_message
        for phrase in [
            "which base policies are supported",
            "what base policies are supported",
            "which policies are supported",
            "which policy options are supported",
            "available policies",
            "available base policies",
        ]
    ):
        base_policy_answer = format_knowledge_section_answer(
            step_id,
            "Supported Base Policies",
            "The current step knowledge lists these supported base policies:",
        )
        if base_policy_answer:
            return base_policy_answer

    return None


def answer_project_context_question(
    user_message: str,
    brief: ProjectBrief | None = None,
    profile: dict[str, Any] | None = None,
) -> str | None:
    lowered = user_message.lower()

    if brief and any(token in lowered for token in ["dataset path", "where is the data", "where is the dataset", "dataset located", "data located"]):
        return f"The current dataset path is `{brief.dataset_path}`."

    if brief and "target" in lowered:
        return f"The current target column is `{brief.target_column}`."

    if brief and any(token in lowered for token in ["task type", "problem type", "what problem", "what task"]):
        return f"The current task type is `{brief.task_type}`.\n\nProblem description: {brief.problem_description}"

    if brief and any(token in lowered for token in ["summarize", "summary", "project overview", "what are we doing"]):
        return (
            f"Project: `{brief.project_name}`\n"
            f"- Dataset: `{brief.dataset_path}`\n"
            f"- Target: `{brief.target_column}`\n"
            f"- Task type: `{brief.task_type}`\n"
            f"- Problem: {brief.problem_description}"
        )

    if not profile:
        return None

    if "rows" in lowered and "column" in lowered:
        return f"The dataset profile shows `{profile.get('rows')}` rows and `{profile.get('columns')}` columns."

    if "rows" in lowered:
        return f"The dataset profile shows `{profile.get('rows')}` rows."

    if "columns" in lowered or "features" in lowered:
        return (
            f"The current profile shows `{profile.get('columns')}` total columns, "
            f"`{len(profile.get('numeric_features', []))}` numeric features, and "
            f"`{len(profile.get('categorical_features', []))}` categorical features."
        )

    if "missing" in lowered:
        return f"The current feature table has `{format_percent(profile.get('missing_fraction'))}` missing values."

    if "imbalance" in lowered or "class balance" in lowered:
        imbalance = profile.get("class_imbalance")
        if imbalance:
            return f"The current class imbalance ratio is about `{imbalance:.2f}`."
        return "The current profile does not indicate a notable class imbalance signal."

    return None


def answer_general_discussion_question(user_message: str) -> str | None:
    lowered = user_message.lower()
    now = datetime.now().astimezone()
    weekday = now.strftime("%A")
    calendar_date = f"{now:%B} {now.day}, {now:%Y}"
    local_time = now.strftime("%H:%M")

    if any(phrase in lowered for phrase in ["what day is today", "which day is today", "what weekday is today", "which weekday is today"]):
        return f"Today is {weekday}."

    if any(phrase in lowered for phrase in ["what is today's date", "what is the date today", "which date is today", "date today"]):
        return f"Today is {calendar_date}."

    if any(phrase in lowered for phrase in ["what time is it", "current time", "time now"]):
        return f"The current local time is {local_time}."

    capital_match = re.search(r"what(?:'s| is)\s+the\s+capital\s+of\s+([a-zA-Z .-]+)\??", lowered)
    if capital_match:
        country = capital_match.group(1).strip(" ?.!").lower()
        capital = CAPITAL_CITY_LOOKUP.get(country)
        if capital:
            return f"The capital of {country.title()} is {capital}."

    if "streamlit" in lowered and any(token in lowered for token in ["typescript", "react", "frontend", "ui stack"]):
        return (
            "TypeScript would usually be the stronger choice for a more professional product UI.\n\n"
            "Streamlit is excellent for rapid internal MVPs, but TypeScript plus React gives you finer control over state, routing, layout, reusable components, and richer chat interactions. "
            "Streamlit is still reasonable for fast iteration, but the long-term polish ceiling is lower."
        )

    if "overfitting" in lowered:
        return (
            "Overfitting means a model learns the training data too specifically and stops generalizing well to unseen data.\n\n"
            "Typical signs are strong training scores but weaker validation or test scores."
        )

    if "underfitting" in lowered:
        return (
            "Underfitting means the model is too simple or too constrained to capture the useful signal in the data.\n\n"
            "You usually see weak performance on both training and validation data."
        )

    if "leakage" in lowered:
        return (
            "Data leakage happens when information from the future, the target, or the test side slips into training.\n\n"
            "It can make validation look unrealistically good and then fail in real use."
        )

    if "cross validation" in lowered or re.search(r"\bcv\b", lowered):
        return (
            "Cross-validation estimates model quality by repeating train/validation splits inside the training data.\n\n"
            "It gives a more stable quality estimate than a single split, especially when datasets are not very large."
        )

    if "baseline" in lowered:
        return (
            "A baseline is a simple reference model used to check whether the more complex workflow is actually adding value.\n\n"
            "If the tuned model does not beat the baseline, the workflow has not earned its complexity yet."
        )

    if "hyperparameter" in lowered or "tuning" in lowered:
        return (
            "Hyperparameter optimization searches over training settings that are not learned directly from data, such as tree depth or regularization strength.\n\n"
            "It can improve performance, but it also adds compute cost and complexity."
        )

    if "precision" in lowered or "recall" in lowered or "f1" in lowered:
        return (
            "Precision measures how often predicted positives are correct, recall measures how many real positives are found, and F1 balances the two.\n\n"
            "They are especially important when false positives and false negatives have different business costs."
        )

    if "balanced accuracy" in lowered:
        return (
            "Balanced accuracy averages recall across classes, so it is less biased toward the majority class than raw accuracy.\n\n"
            "It is often a better choice when class imbalance matters."
        )

    if "rmse" in lowered or "mae" in lowered or "r2" in lowered:
        return (
            "RMSE penalizes large errors more strongly, MAE measures average absolute error more robustly, and R2 measures explained variance.\n\n"
            "The best choice depends on whether you care more about large misses, average deviation, or explanatory power."
        )

    return None


def build_discussion_intake_context(brief: ProjectBrief | None) -> str:
    if not brief:
        return "No intake information is available yet."

    lines = [
        f"project_name: {brief.project_name}",
        f"dataset_path: {brief.dataset_path}",
        f"target_column: {brief.target_column}",
        f"task_type: {brief.task_type}",
        f"problem_description: {brief.problem_description}",
    ]
    if brief.date_column:
        lines.append(f"date_column: {brief.date_column}")
    if brief.baseline_metric:
        lines.append(f"baseline_metric: {brief.baseline_metric}")
    lines.append(f"competition_enabled: {'yes' if brief.competition_enabled else 'no'}")
    return "\n".join(lines)


def build_step_discussion_context_block(
    recommendation: dict,
    feedback: dict[str, Any] | None = None,
    current_option: str | None = None,
) -> str:
    feedback = feedback or {}
    default_option = recommendation.get("selected_option") or ""
    selected_option = current_option or default_option
    agreement = feedback.get("agreement", "agree")
    reasons = recommendation.get("reasoning", [])

    lines = [
        f"step_id: {recommendation['step_id']}",
        f"step_title: {recommendation['title']}",
        f"step_purpose: {STEP_PURPOSES.get(recommendation['step_id'], '')}",
        f"default_execution_policy: {default_option}",
        f"current_execution_policy: {selected_option}",
        f"agreement_status: {agreement}",
    ]

    policy_summary = feedback.get("policy_summary", "").strip()
    custom_note = feedback.get("custom_note", "").strip()
    recommendation_text = recommendation.get("recommendation", "").strip()

    if policy_summary:
        lines.append(f"policy_summary: {policy_summary}")
    if custom_note:
        lines.append(f"custom_note: {custom_note}")
    if recommendation_text:
        lines.extend(["recommendation_summary:", recommendation_text])
    if reasons:
        lines.append("recommendation_reasons:")
        lines.extend(f"- {reason}" for reason in reasons)

    return "\n".join(lines)


def build_previous_steps_context(
    current_step_id: str,
    recommendations: list[dict] | None = None,
    step_feedback: dict[str, dict[str, Any]] | None = None,
    selected_options: dict[str, str] | None = None,
) -> str:
    if not recommendations:
        return "No previous workflow steps are available yet."

    lines: list[str] = []
    for recommendation in recommendations:
        step_id = recommendation["step_id"]
        if step_id == current_step_id:
            break

        feedback = normalize_step_feedback_entry(
            dict((step_feedback or {}).get(step_id, {})),
            recommendation,
        )
        agreement = feedback.get("agreement", "agree")
        selected_option = (selected_options or {}).get(step_id, recommendation.get("selected_option") or "")

        lines.append(f"{recommendation['title']}:")
        lines.append(f"- current_execution_policy: {selected_option}")
        lines.append(f"- agreement_status: {agreement}")

        policy_summary = feedback.get("policy_summary", "").strip()
        if policy_summary:
            lines.append(f"- policy_summary: {policy_summary}")
        elif agreement == "agree" and recommendation.get("recommendation"):
            lines.append(f"- recommendation_summary: {recommendation['recommendation']}")

    if not lines:
        return "No previous workflow steps are available yet."
    return "\n".join(lines)


def build_discussion_context(
    brief: ProjectBrief | None,
    recommendation: dict | None = None,
    recommendations: list[dict] | None = None,
    step_feedback: dict[str, dict[str, Any]] | None = None,
    selected_options: dict[str, str] | None = None,
) -> str:
    sections = [f"Intake context:\n{build_discussion_intake_context(brief)}"]

    if recommendation:
        sections.append(
            "Workflow context from previous steps:\n"
            + build_previous_steps_context(
                recommendation["step_id"],
                recommendations=recommendations,
                step_feedback=step_feedback,
                selected_options=selected_options,
            )
        )
        current_option = (selected_options or {}).get(recommendation["step_id"], recommendation.get("selected_option") or "")
        sections.append(
            "Current workflow step context:\n"
            + build_step_discussion_context_block(
                recommendation,
                feedback=(step_feedback or {}).get(recommendation["step_id"], {}),
                current_option=current_option,
            )
        )

    return "\n\n".join(sections)


def format_discussion_history(history: list[dict[str, str]], limit: int = 12) -> str:
    trimmed_history = history[-limit:]
    if not trimmed_history:
        return "No previous discussion messages."
    return "\n".join(f"{item['role'].title()}: {item['content']}" for item in trimmed_history)


def parse_codex_exec_json_events(raw_stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw_stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type"):
            events.append(payload)
    return events


def build_codex_discussion_activity(events: list[dict[str, Any]], final_reply: str) -> list[dict[str, str]]:
    activity: list[dict[str, str]] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item", {})
        item_type = item.get("type")
        if item_type == "agent_message":
            text = (item.get("text") or "").strip()
            if text and text != final_reply:
                activity.append({"role": "assistant", "content": f"_Codex status_: {text}"})
        elif item_type == "command_execution":
            command = (item.get("command") or "").strip()
            output = (item.get("aggregated_output") or "").strip()
            exit_code = item.get("exit_code")
            command_lines = [f"_Codex activity_: ran `{command}`"]
            if exit_code is not None:
                command_lines.append(f"Exit code: `{exit_code}`")
            if output:
                preview = "\n".join(output.splitlines()[:8])
                command_lines.append(f"Output:\n```text\n{preview}\n```")
            activity.append({"role": "assistant", "content": "\n".join(command_lines)})
    return activity


def maybe_call_local_codex_discussion(
    user_message: str,
    history: list[dict[str, str]],
    brief: ProjectBrief | None = None,
    recommendation: dict | None = None,
    recommendations: list[dict] | None = None,
    step_feedback: dict[str, dict[str, Any]] | None = None,
    selected_options: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    if os.getenv("AGENTIC_AUTOML_DISABLE_LOCAL_CODEX_DISCUSSION", "").lower() in {"1", "true", "yes"}:
        return None

    codex_path = shutil.which("codex")
    if not codex_path:
        return None

    discussion_context = build_discussion_context(
        brief,
        recommendation=recommendation,
        recommendations=recommendations,
        step_feedback=step_feedback,
        selected_options=selected_options,
    )
    repo_root = resolve_repo_root()
    output_path: Path | None = None
    events: list[dict[str, Any]] = []
    prompt = (
        "You are the Discussion mode assistant for Agentic AutoML.\n"
        "This is an open-ended general chat.\n"
        "You may answer out-of-scope questions normally.\n"
        "Do not build or modify workflow policy in this mode.\n"
        "You may use the project and workflow context below to explain why recommendations were made.\n\n"
        f"Project and workflow context:\n{discussion_context}\n\n"
        f"Previous discussion history:\n{format_discussion_history(history)}\n\n"
        f"Current user message:\n{user_message}"
    )

    with tempfile.NamedTemporaryFile(prefix="agentic-automl-discussion-", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)
    try:
        command = [
            codex_path,
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--json",
            "-C",
            str(repo_root),
            "-o",
            str(output_path),
            prompt,
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if result.returncode != 0:
            return None
        reply = output_path.read_text(encoding="utf-8").strip()
        events = parse_codex_exec_json_events(result.stdout)
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return None
    finally:
        if output_path is not None:
            output_path.unlink(missing_ok=True)

    if not reply:
        return None
    return {
        "reply": reply,
        "activity": build_codex_discussion_activity(events, reply),
    }


def respond_to_discussion_message(
    recommendation: dict,
    user_message: str,
    brief: ProjectBrief | None = None,
    profile: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
    recommendations: list[dict] | None = None,
    step_feedback: dict[str, dict[str, Any]] | None = None,
    selected_options: dict[str, str] | None = None,
    allow_local_codex: bool = True,
) -> str:
    if allow_local_codex:
        model_answer = maybe_call_local_codex_discussion(
            user_message,
            history or [],
            brief=brief,
            recommendation=recommendation,
            recommendations=recommendations,
            step_feedback=step_feedback,
            selected_options=selected_options,
        )
        if model_answer:
            return model_answer["reply"]

    general_answer = answer_general_discussion_question(user_message)
    if general_answer:
        return general_answer

    project_answer = answer_project_context_question(
        user_message,
        brief=brief,
        profile=profile,
    )
    if project_answer:
        return project_answer

    return (
        "Discussion mode is open-ended, but the local Codex bridge is not available right now.\n\n"
        f"I do not have a local built-in answer for: `{user_message}` yet. "
        "If the local Codex CLI is available, this discussion tab will use it with the current workflow context."
    )


def build_policy_summary(
    recommendation: dict,
    selected_option: str,
    user_message: str,
    used_default_base: bool,
) -> str:
    title = recommendation["title"].lower()
    option_description = describe_option(recommendation["step_id"], selected_option)
    guidance = user_message.strip()
    if used_default_base:
        return f"Keep `{selected_option}` as the executable base for {title}, and apply this extra guidance: {guidance}"
    return f"Use `{selected_option}` for {title}. That means: {option_description} Extra guidance: {guidance}"


def explain_step_choice(recommendation: dict) -> str:
    lines = [
        f"This step controls {STEP_PURPOSES.get(recommendation['step_id'], recommendation['title'].lower())}.",
        "",
        f"Default executable policy: `{recommendation.get('selected_option')}`",
        f"- {describe_option(recommendation['step_id'], recommendation.get('selected_option') or '')}",
    ]
    options = recommendation.get("options", [])
    if options:
        lines.extend(["", "Available executable policies:"])
        for option in options:
            lines.append(f"- `{option}`: {describe_option(recommendation['step_id'], option)}")
    lines.extend(
        [
            "",
            "Tell me the behavior you want, and I will turn it into a working execution policy automatically.",
        ]
    )
    return "\n".join(lines)


def parse_hpo_policy_request(
    recommendation: dict,
    user_message: str,
    current_option: str,
    selected_options: dict[str, str] | None = None,
    current_policy_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    model_name = current_model_selection(selected_options, recommendation)
    if not model_name:
        return None
    existing_metadata = {}
    if isinstance(current_policy_metadata, dict):
        existing_metadata = dict(current_policy_metadata.get("hpo_config", {}))
    parse_result = hpo_actions.parse_hpo_action_request(
        user_message,
        model_name,
        current_option=current_option or recommendation.get("selected_option") or "",
        current_hpo_config=existing_metadata,
    )
    if parse_result is None:
        return None
    chosen_option = parse_result.selected_option or current_option or recommendation.get("selected_option") or ""
    return {
        "selected_option": chosen_option,
        "model_name": model_name,
        "hpo_config": parse_result.hpo_config,
        "requested_capability_keys": dedupe_messages(parse_result.requested_capability_keys),
        "unsupported_reasons": list(parse_result.unsupported_reasons),
    }


def format_parameter_overrides(parameters: dict[str, Any]) -> str:
    if not parameters:
        return "no parameter overrides yet"
    return ", ".join(f"`{key}`={value!r}" for key, value in parameters.items())


def active_step_option(
    recommendation: dict[str, Any],
    current_option: str | None,
) -> str:
    return current_option or recommendation.get("selected_option") or ""


def summarize_current_action_context(
    recommendation: dict[str, Any],
    current_option: str,
    profile: dict[str, Any] | None = None,
    selected_options: dict[str, str] | None = None,
    current_policy_metadata: dict[str, Any] | None = None,
) -> list[str]:
    step_id = recommendation["step_id"]
    active_option = active_step_option(recommendation, current_option)
    lines = [f"- Active policy: `{active_option}`"]

    if step_id == "01_preprocessing":
        metadata = current_policy_metadata or {}
        overrides = metadata.get("preprocessing_overrides", {})
        recognized_changes = metadata.get("recognized_changes") or summarize_preprocessing_override_changes(
            overrides
        )
        if recognized_changes:
            lines.append("- Current execution changes: " + " ".join(recognized_changes[:3]))
        else:
            lines.append("- Current execution changes: none yet")
        return lines

    if step_id == "03_model_selection":
        metadata = current_policy_metadata or {}
        lines.append(
            "- Current initial parameters: "
            + format_parameter_overrides(metadata.get("model_parameters", {}))
        )
        return lines

    if step_id == "06_training_configuration":
        model_name = current_model_selection(selected_options, recommendation)
        if model_name:
            lines.append(f"- Selected model context: `{model_name}`")
        metadata = current_policy_metadata or {}
        lines.append(
            "- Current training parameters: "
            + format_parameter_overrides(metadata.get("training_config", {}))
        )
        return lines

    if step_id == "07_hyperparameter_optimization":
        model_name = current_model_selection(selected_options, recommendation)
        if model_name:
            lines.append(f"- Selected model context: `{model_name}`")
        metadata = current_policy_metadata or {}
        search_parameters = list(metadata.get("hpo_config", {}).get("search_parameters", []) or [])
        if search_parameters:
            lines.append(
                "- Current search scope: " + ", ".join(f"`{item}`" for item in search_parameters)
            )
        else:
            lines.append("- Current search scope: packaged default for the active HPO policy")
        return lines

    return lines


def build_action_mode_suggestions(
    recommendation: dict[str, Any],
    current_option: str,
    selected_options: dict[str, str] | None = None,
    current_policy_metadata: dict[str, Any] | None = None,
) -> list[str]:
    step_id = recommendation["step_id"]
    active_option = active_step_option(recommendation, current_option)
    options = list(recommendation.get("options", []))
    alternative_options = [option for option in options if option != active_option]

    if step_id == "01_preprocessing":
        suggestions = [
            "switch preprocessing to `" + option + "`" for option in alternative_options[:2]
        ]
        suggestions.extend(
            [
                "drop or keep a named feature subset",
                "change encoding, imputation, scaling, or feature-role rules for specific columns",
            ]
        )
        return dedupe_messages(suggestions)

    if step_id == "02_data_splitting":
        suggestions = [
            "switch the holdout strategy to `" + option + "`" for option in alternative_options[:2]
        ]
        suggestions.append("explain whether chronology or leakage argues for a different holdout strategy")
        return dedupe_messages(suggestions)

    if step_id == "03_model_selection":
        suggestions = [
            "switch the starting model to `" + option + "`" for option in alternative_options[:2]
        ]
        suggestions.extend(
            [
                "override supported initial model parameters for the active model",
                "ask which packaged models are still available for this task",
            ]
        )
        return dedupe_messages(suggestions)

    if step_id == "05_metric_selection":
        suggestions = [
            "switch the winner metric to `" + option + "`" for option in alternative_options[:3]
        ]
        suggestions.append("ask about the tradeoff between the current metric and another supported metric")
        return dedupe_messages(suggestions)

    if step_id == "06_training_configuration":
        model_name = current_model_selection(selected_options, recommendation)
        suggestions = [
            "switch the training budget to `" + option + "`" for option in alternative_options[:2]
        ]
        suggestions.append("change `cv_folds` or `random_seed`")
        if model_name in {"logistic_regression", "mlp_classifier", "mlp_regressor"}:
            suggestions.append("change the supported optimizer for the selected model")
        if model_name in {
            "hist_gradient_boosting_classifier",
            "hist_gradient_boosting_regressor",
            "mlp_classifier",
            "mlp_regressor",
        }:
            suggestions.append("change `learning_rate` or `epochs` for the selected model")
        if model_name in {"mlp_classifier", "mlp_regressor"}:
            suggestions.append("change `mini_batch` or `early_stopping` for the selected model")
        return dedupe_messages(suggestions)

    if step_id == "07_hyperparameter_optimization":
        metadata = current_policy_metadata or {}
        search_parameters = list(metadata.get("hpo_config", {}).get("search_parameters", []) or [])
        suggestions = [
            "switch the HPO mode to `" + option + "`" for option in alternative_options[:2]
        ]
        if search_parameters:
            suggestions.extend(
                [
                    "add supported hyperparameters to the current search scope",
                    "remove supported hyperparameters from the current search scope",
                ]
            )
        else:
            suggestions.extend(
                [
                    "name the exact hyperparameters you want to optimize",
                    "use the top recommended hyperparameters for the selected model",
                ]
            )
        return dedupe_messages(suggestions)

    if step_id == "08_validation_and_baseline":
        return [
            "ask which baseline strategy the current validation step is using",
            "ask what the output notebook will include after validation",
        ]

    if step_id == "09_final_validation":
        return [
            "ask what the tuned-versus-untuned dashboard will compare",
            "ask what optimization results are included in the final notebook output",
        ]

    return []


def build_action_mode_consultation_reply(
    answer: str,
    recommendation: dict[str, Any],
    current_option: str,
    profile: dict[str, Any] | None = None,
    selected_options: dict[str, str] | None = None,
    current_policy_metadata: dict[str, Any] | None = None,
) -> str:
    lines = [
        answer.strip(),
        "",
        "This is a consultation answer only. I did not change the execution policy.",
        "",
        "Current action context:",
    ]
    lines.extend(
        summarize_current_action_context(
            recommendation,
            current_option=current_option,
            profile=profile,
            selected_options=selected_options,
            current_policy_metadata=current_policy_metadata,
        )
    )
    suggestions = build_action_mode_suggestions(
        recommendation,
        current_option=current_option,
        selected_options=selected_options,
        current_policy_metadata=current_policy_metadata,
    )
    if suggestions:
        lines.extend(["", "Useful next moves:"])
        lines.extend(f"- {suggestion}" for suggestion in suggestions[:4])
    lines.extend(
        [
            "",
            "If you ask for something outside this step's current knowledge, I will keep the policy unchanged and store it in `LIMITS.md`.",
        ]
    )
    return "\n".join(lines)


def respond_to_step_discussion(
    recommendation: dict,
    user_message: str,
    current_option: str,
    profile: dict[str, Any] | None = None,
    selected_options: dict[str, str] | None = None,
    current_policy_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if step_discussion_is_question(user_message):
        contextual_answer = answer_contextual_step_question(
            recommendation,
            user_message,
            current_option=current_option,
            profile=profile,
            selected_options=selected_options,
            current_policy_metadata=current_policy_metadata,
        )
        return {
            "reply": build_action_mode_consultation_reply(
                contextual_answer or explain_step_choice(recommendation),
                recommendation,
                current_option=current_option,
                profile=profile,
                selected_options=selected_options,
                current_policy_metadata=current_policy_metadata,
            ),
            "selected_option": None,
            "policy_summary": "",
            "custom_note": "",
            "updated_policy": False,
        }

    default_option = current_option or recommendation.get("selected_option") or ""
    options = recommendation.get("options", [])
    if recommendation["step_id"] == "01_preprocessing":
        action_metadata = preprocessing_actions.parse_preprocessing_action_request(user_message, profile)
        supported_capability_keys = load_capability_keys(recommendation["step_id"])
        unsupported_capability_keys = [
            key
            for key in action_metadata.requested_capability_keys
            if key not in supported_capability_keys
        ]
        if action_metadata.unsupported_reasons or unsupported_capability_keys:
            return {
                "reply": build_unsupported_action_reply(
                    recommendation["step_id"],
                    user_message,
                    reasons=action_metadata.unsupported_reasons,
                    keep_current_policy=True,
                ),
                "selected_option": None,
                "policy_summary": "",
                "custom_note": "",
                "updated_policy": False,
            }
        if action_metadata.has_executable_overrides:
            existing_overrides = {}
            if isinstance(current_policy_metadata, dict):
                existing_overrides = current_policy_metadata.get("preprocessing_overrides", {})
            merged_overrides = merge_preprocessing_overrides(
                existing_overrides,
                action_metadata.preprocessing_overrides,
            )
            merged_changes = summarize_preprocessing_policy_changes(merged_overrides, profile=profile)
            selected_option = "custom" if "custom" in options else (current_option or default_option)
            policy_summary = build_preprocessing_policy_summary(
                selected_option,
                merged_overrides,
                recognized_changes=merged_changes,
            )
            reply_lines = [
                "I updated the preprocessing execution policy.",
                "",
                f"- Executable policy: `{selected_option}`",
            ]
            for change in merged_changes:
                reply_lines.append(f"- {change}")
            reply_lines.extend(
                [
                    "",
                    "This will change the actual preprocessing graph used during training and export.",
                ]
            )
            return {
                "reply": "\n".join(reply_lines),
                "selected_option": selected_option,
                "policy_summary": policy_summary,
                "custom_note": user_message.strip(),
                "policy_metadata": {
                    "preprocessing_overrides": merged_overrides,
                    "recognized_changes": merged_changes,
                },
                "updated_policy": True,
            }
    if recommendation["step_id"] == "02_data_splitting":
        action_metadata = split_actions.parse_split_action_request(
            user_message,
            current_option=current_option or recommendation.get("selected_option") or "",
        )
        if action_metadata:
            supported_capability_keys = load_capability_keys(recommendation["step_id"])
            unsupported_capability_keys = [
                key
                for key in action_metadata.requested_capability_keys
                if key not in supported_capability_keys
            ]
            if action_metadata.unsupported_reasons or unsupported_capability_keys:
                return {
                    "reply": build_unsupported_action_reply(
                        recommendation["step_id"],
                        user_message,
                        reasons=list(action_metadata.unsupported_reasons)
                        + (
                            [f"unsupported capability key(s): {', '.join(unsupported_capability_keys)}"]
                            if unsupported_capability_keys
                            else []
                        ),
                        keep_current_policy=True,
                    ),
                    "selected_option": None,
                    "policy_summary": "",
                    "custom_note": "",
                    "updated_policy": False,
                }
            selected_option = action_metadata.selected_option
            policy_summary = (
                f"Use `{selected_option}` as the final holdout strategy for workflow execution."
            )
            return {
                "reply": (
                    "I updated the data-splitting execution policy.\n\n"
                    f"- Final holdout strategy: `{selected_option}`"
                ),
                "selected_option": selected_option,
                "policy_summary": policy_summary,
                "custom_note": user_message.strip(),
                "policy_metadata": {},
                "updated_policy": True,
            }
    if recommendation["step_id"] == "03_model_selection":
        inferred_option = infer_execution_option(
            recommendation["step_id"],
            recommendation.get("options", []),
            user_message,
            recommendation.get("selected_option") or current_option,
        )
        action_metadata = model_actions.parse_model_action_request(
            user_message,
            selected_option=inferred_option,
            current_option=current_option or recommendation.get("selected_option") or "",
            current_policy_metadata=current_policy_metadata,
        )
        if action_metadata:
            supported_capability_keys = load_capability_keys(recommendation["step_id"])
            unsupported_capability_keys = [
                key
                for key in action_metadata.requested_capability_keys
                if key not in supported_capability_keys
            ]
            if action_metadata.unsupported_reasons or unsupported_capability_keys:
                return {
                    "reply": build_unsupported_action_reply(
                        recommendation["step_id"],
                        user_message,
                        reasons=list(action_metadata.unsupported_reasons)
                        + (
                            [f"unsupported capability key(s): {', '.join(unsupported_capability_keys)}"]
                            if unsupported_capability_keys
                            else []
                        ),
                        keep_current_policy=True,
                    ),
                    "selected_option": None,
                    "policy_summary": "",
                    "custom_note": "",
                    "updated_policy": False,
                }
            selected_option = action_metadata.selected_option
            model_parameters = action_metadata.model_parameters
            policy_summary = (
                f"Use `{selected_option}` as the selected model with these initial parameters: "
                f"{format_parameter_overrides(model_parameters)}."
            )
            return {
                "reply": (
                    "I updated the model-selection execution policy.\n\n"
                    f"- Selected model: `{selected_option}`\n"
                    f"- Initial parameters: {format_parameter_overrides(model_parameters)}"
                ),
                "selected_option": selected_option,
                "policy_summary": policy_summary,
                "custom_note": user_message.strip(),
                "policy_metadata": {"model_parameters": model_parameters},
                "updated_policy": True,
            }
    if recommendation["step_id"] == "05_metric_selection":
        action_metadata = metric_actions.parse_metric_action_request(
            user_message,
            current_option=current_option or recommendation.get("selected_option") or "",
        )
        if action_metadata:
            supported_capability_keys = load_capability_keys(recommendation["step_id"])
            unsupported_capability_keys = [
                key
                for key in action_metadata.requested_capability_keys
                if key not in supported_capability_keys
            ]
            if action_metadata.unsupported_reasons or unsupported_capability_keys:
                return {
                    "reply": build_unsupported_action_reply(
                        recommendation["step_id"],
                        user_message,
                        reasons=list(action_metadata.unsupported_reasons)
                        + (
                            [f"unsupported capability key(s): {', '.join(unsupported_capability_keys)}"]
                            if unsupported_capability_keys
                            else []
                        ),
                        keep_current_policy=True,
                    ),
                    "selected_option": None,
                    "policy_summary": "",
                    "custom_note": "",
                    "updated_policy": False,
                }
            selected_option = action_metadata.selected_option
            policy_summary = (
                f"Use `{selected_option}` as the primary winner metric for workflow ranking and validation comparison."
            )
            return {
                "reply": (
                    "I updated the metric-selection execution policy.\n\n"
                    f"- Primary metric: `{selected_option}`"
                ),
                "selected_option": selected_option,
                "policy_summary": policy_summary,
                "custom_note": user_message.strip(),
                "policy_metadata": {},
                "updated_policy": True,
            }
    if recommendation["step_id"] == "06_training_configuration":
        model_name = current_model_selection(selected_options, recommendation)
        if model_name:
            inferred_option = infer_execution_option(
                recommendation["step_id"],
                recommendation.get("options", []),
                user_message,
                recommendation.get("selected_option") or current_option,
            )
            action_metadata = training_actions.parse_training_action_request(
                user_message,
                model_name=model_name,
                selected_option=inferred_option,
                current_option=current_option or recommendation.get("selected_option") or "",
                current_policy_metadata=current_policy_metadata,
            )
        else:
            action_metadata = None
        if action_metadata:
            supported_capability_keys = load_capability_keys(recommendation["step_id"])
            unsupported_capability_keys = [
                key
                for key in action_metadata.requested_capability_keys
                if key not in supported_capability_keys
            ]
            if action_metadata.unsupported_reasons or unsupported_capability_keys:
                return {
                    "reply": build_unsupported_action_reply(
                        recommendation["step_id"],
                        user_message,
                        reasons=list(action_metadata.unsupported_reasons)
                        + (
                            [f"unsupported capability key(s): {', '.join(unsupported_capability_keys)}"]
                            if unsupported_capability_keys
                            else []
                        ),
                        keep_current_policy=True,
                    ),
                    "selected_option": None,
                    "policy_summary": "",
                    "custom_note": "",
                    "updated_policy": False,
                }
            selected_option = action_metadata.selected_option
            training_config = action_metadata.training_config
            policy_summary = (
                f"Use `{selected_option}` with these training parameters for `{model_name}`: "
                f"{format_parameter_overrides(training_config)}."
            )
            return {
                "reply": (
                    "I updated the training execution policy.\n\n"
                    f"- Selected model: `{model_name}`\n"
                    f"- Training policy: `{selected_option}`\n"
                    f"- Training parameters: {format_parameter_overrides(training_config)}"
                ),
                "selected_option": selected_option,
                "policy_summary": policy_summary,
                "custom_note": user_message.strip(),
                "policy_metadata": {"training_config": training_config},
                "updated_policy": True,
            }
    if recommendation["step_id"] == "07_hyperparameter_optimization":
        action_metadata = parse_hpo_policy_request(
            recommendation,
            user_message,
            current_option=current_option,
            selected_options=selected_options,
            current_policy_metadata=current_policy_metadata,
        )
        if action_metadata:
            if action_metadata.get("unsupported_reasons"):
                return {
                    "reply": build_unsupported_action_reply(
                        recommendation["step_id"],
                        user_message,
                        reasons=action_metadata["unsupported_reasons"],
                        keep_current_policy=True,
                    ),
                    "selected_option": None,
                    "policy_summary": "",
                    "custom_note": "",
                    "updated_policy": False,
                }
            supported_capability_keys = load_capability_keys(recommendation["step_id"])
            unsupported_capability_keys = [
                key for key in action_metadata["requested_capability_keys"] if key not in supported_capability_keys
            ]
            if unsupported_capability_keys:
                return {
                    "reply": build_unsupported_action_reply(
                        recommendation["step_id"],
                        user_message,
                        reasons=[f"unsupported capability key(s): {', '.join(unsupported_capability_keys)}"],
                        keep_current_policy=True,
                    ),
                    "selected_option": None,
                    "policy_summary": "",
                    "custom_note": "",
                    "updated_policy": False,
                }
            selected_option = action_metadata["selected_option"]
            hpo_config = action_metadata["hpo_config"]
            search_parameters = hpo_config.get("search_parameters", [])
            model_name = action_metadata.get("model_name") or current_model_selection(selected_options, recommendation) or "selected_model"
            summary_tail = (
                "no explicit hyperparameter list yet"
                if not search_parameters
                else ", ".join(f"`{name}`" for name in search_parameters)
            )
            policy_summary = (
                f"Use `{selected_option}` for HPO on `{model_name}` with this focused search scope: {summary_tail}."
            )
            return {
                "reply": (
                    "I updated the hyperparameter-optimization policy.\n\n"
                    f"- Selected model: `{model_name}`\n"
                    f"- HPO policy: `{selected_option}`\n"
                    f"- Search scope: {summary_tail}"
                ),
                "selected_option": selected_option,
                "policy_summary": policy_summary,
                "custom_note": user_message.strip(),
                "policy_metadata": {"hpo_config": hpo_config},
                "updated_policy": True,
            }
    if recommendation["step_id"] == "08_validation_and_baseline":
        action_metadata = validation_actions.parse_validation_action_request(
            user_message,
            current_option=current_option or recommendation.get("selected_option") or "",
        )
        if action_metadata:
            supported_capability_keys = load_capability_keys(recommendation["step_id"])
            unsupported_capability_keys = [
                key
                for key in action_metadata.requested_capability_keys
                if key not in supported_capability_keys
            ]
            if action_metadata.unsupported_reasons or unsupported_capability_keys:
                return {
                    "reply": build_unsupported_action_reply(
                        recommendation["step_id"],
                        user_message,
                        reasons=list(action_metadata.unsupported_reasons)
                        + (
                            [f"unsupported capability key(s): {', '.join(unsupported_capability_keys)}"]
                            if unsupported_capability_keys
                            else []
                        ),
                        keep_current_policy=True,
                    ),
                    "selected_option": None,
                    "policy_summary": "",
                    "custom_note": "",
                    "updated_policy": False,
                }
            selected_option = action_metadata.selected_option
            policy_summary = (
                f"Keep `{selected_option}` as the validation flow, including the untouched test-set comparison and baseline check."
            )
            return {
                "reply": (
                    "I kept the validation execution policy aligned with the packaged holdout-and-baseline flow.\n\n"
                    f"- Validation policy: `{selected_option}`"
                ),
                "selected_option": selected_option,
                "policy_summary": policy_summary,
                "custom_note": user_message.strip(),
                "policy_metadata": {},
                "updated_policy": True,
            }
    if recommendation["step_id"] == "09_final_validation":
        action_metadata = final_validation_actions.parse_final_validation_action_request(
            user_message,
            current_option=current_option or recommendation.get("selected_option") or "",
        )
        if action_metadata:
            supported_capability_keys = load_capability_keys(recommendation["step_id"])
            unsupported_capability_keys = [
                key
                for key in action_metadata.requested_capability_keys
                if key not in supported_capability_keys
            ]
            if action_metadata.unsupported_reasons or unsupported_capability_keys:
                return {
                    "reply": build_unsupported_action_reply(
                        recommendation["step_id"],
                        user_message,
                        reasons=list(action_metadata.unsupported_reasons)
                        + (
                            [f"unsupported capability key(s): {', '.join(unsupported_capability_keys)}"]
                            if unsupported_capability_keys
                            else []
                        ),
                        keep_current_policy=True,
                    ),
                    "selected_option": None,
                    "policy_summary": "",
                    "custom_note": "",
                    "updated_policy": False,
                }
            selected_option = action_metadata.selected_option
            policy_summary = (
                f"Keep `{selected_option}` as the final validation and dashboard flow for the tuned workflow outcome."
            )
            return {
                "reply": (
                    "I kept the final-validation execution policy aligned with the packaged dashboard flow.\n\n"
                    f"- Final validation policy: `{selected_option}`"
                ),
                "selected_option": selected_option,
                "policy_summary": policy_summary,
                "custom_note": user_message.strip(),
                "policy_metadata": {},
                "updated_policy": True,
            }

    inferred_option = infer_execution_option(
        recommendation["step_id"],
        options,
        user_message,
        default_option,
    )
    unsupported_action_request = False
    if recommendation["step_id"] == "01_preprocessing":
        unsupported_action_request = preprocessing_actions.looks_like_action_request(user_message)
    elif recommendation["step_id"] == "02_data_splitting":
        unsupported_action_request = split_actions.looks_like_split_action_request(user_message)
    elif recommendation["step_id"] == "03_model_selection":
        unsupported_action_request = model_actions.looks_like_model_action_request(user_message)
    elif recommendation["step_id"] == "05_metric_selection":
        unsupported_action_request = metric_actions.looks_like_metric_action_request(user_message)
    elif recommendation["step_id"] == "06_training_configuration":
        unsupported_action_request = training_actions.looks_like_training_action_request(user_message)
    elif recommendation["step_id"] == "07_hyperparameter_optimization":
        unsupported_action_request = hpo_actions.looks_like_hpo_action_request(user_message)
    elif recommendation["step_id"] == "08_validation_and_baseline":
        unsupported_action_request = validation_actions.looks_like_validation_action_request(user_message)
    elif recommendation["step_id"] == "09_final_validation":
        unsupported_action_request = final_validation_actions.looks_like_final_validation_action_request(user_message)

    if inferred_option is None and unsupported_action_request:
        return {
            "reply": build_unsupported_action_reply(recommendation["step_id"], user_message),
            "selected_option": None,
            "policy_summary": "",
            "custom_note": "",
            "updated_policy": False,
        }
    selected_option = inferred_option or current_option or default_option
    used_default_base = selected_option == default_option and inferred_option is None
    policy_summary = build_policy_summary(
        recommendation,
        selected_option,
        user_message,
        used_default_base,
    )
    if used_default_base:
        reply = (
            f"I stored your guidance for {recommendation['title'].lower()}.\n\n"
            f"- Executable base policy: `{selected_option}`\n"
            f"- Stored guidance: {user_message.strip()}\n\n"
            "I did not detect a concrete executable override from this message yet, so the runtime plan itself is unchanged. "
            "If you want the execution graph to change, describe a direct rule such as `treat <column> as categorical`, `drop <column>`, `keep <column>`, or `treat <column> as date`."
        )
    else:
        reply = (
            f"I built a new working policy for {recommendation['title'].lower()}.\n\n"
            f"- Executable policy: `{selected_option}`\n"
            f"- Working policy: {policy_summary}\n\n"
            "Keep chatting if you want to refine this step further."
        )

    return {
        "reply": reply,
        "selected_option": selected_option,
        "policy_summary": policy_summary,
        "custom_note": user_message.strip(),
        "policy_metadata": {},
        "updated_policy": True,
    }


def append_chat_history_entry(feedback: dict[str, Any], history_key: str, role: str, content: str) -> None:
    feedback.setdefault(history_key, []).append({"role": role, "content": content})


def find_step_for_custom_chat(
    user_message: str,
    recommendations: list[dict],
    step_feedback: dict[str, dict[str, Any]],
) -> str | None:
    lowered = user_message.lower()
    if any(
        token in lowered
        for token in ["what do i do now", "what next", "next step", "now what", "workflow tab", "run the bundle", "bundle build"]
    ):
        return None

    explicit_matches: list[str] = []
    for recommendation in recommendations:
        step_id = recommendation["step_id"]
        keywords = STEP_REFERENCE_KEYWORDS.get(step_id, [])
        if (
            any(keyword in lowered for keyword in keywords)
            or step_id.replace("_", " ") in lowered
            or f"step {recommendation['step_number']}" in lowered
        ):
            explicit_matches.append(step_id)

    if len(explicit_matches) == 1:
        return explicit_matches[0]

    custom_steps = [
        recommendation["step_id"]
        for recommendation in recommendations
        if step_feedback.get(recommendation["step_id"], {}).get("agreement") == "different"
    ]
    if len(custom_steps) == 1:
        return custom_steps[0]

    return None


def format_missing_fields_message(fields: dict[str, str], missing: list[str]) -> str:
    captured_lines = []
    for key in ["project_name", "dataset_path", "target_column", "task_type", "problem_description"]:
        value = fields.get(key)
        if value:
            captured_lines.append(f"- {key}: `{value}`")

    lines = ["You do not need to send everything at once."]
    if captured_lines:
        lines.extend(["", "I already captured:"])
        lines.extend(captured_lines)

    lines.extend(
        [
            "",
            "I still need:",
            *[f"- {field}" for field in missing],
            "",
            "You can reply with just the missing pieces, for example:",
            "`target_column: churned`",
        ]
    )
    return "\n".join(lines)
