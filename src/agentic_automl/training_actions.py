from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


TRAINING_BASE_POLICY_CAPABILITIES = {
    "fast_training": "training_fast",
    "standard_training": "training_standard",
    "thorough_training": "training_thorough",
}

TRAINING_PARAMETER_CAPABILITY_KEYS = {
    "cv_folds": "training_override_cv_folds",
    "random_seed": "training_override_random_seed",
    "optimizer": "training_override_optimizer",
    "learning_rate": "training_override_learning_rate",
    "epochs": "training_override_epochs",
    "mini_batch": "training_override_mini_batch",
    "early_stopping": "training_override_early_stopping",
}

TRAINING_PARAMETER_SUPPORT = {
    "logistic_regression": {"cv_folds", "random_seed", "optimizer", "epochs"},
    "ridge_regression": {"cv_folds", "random_seed"},
    "random_forest_classifier": {"cv_folds", "random_seed"},
    "random_forest_regressor": {"cv_folds", "random_seed"},
    "hist_gradient_boosting_classifier": {
        "cv_folds",
        "random_seed",
        "learning_rate",
        "epochs",
        "early_stopping",
    },
    "hist_gradient_boosting_regressor": {
        "cv_folds",
        "random_seed",
        "learning_rate",
        "epochs",
        "early_stopping",
    },
    "mlp_classifier": {
        "cv_folds",
        "random_seed",
        "optimizer",
        "learning_rate",
        "epochs",
        "mini_batch",
        "early_stopping",
    },
    "mlp_regressor": {
        "cv_folds",
        "random_seed",
        "optimizer",
        "learning_rate",
        "epochs",
        "mini_batch",
        "early_stopping",
    },
}

VALID_OPTIMIZERS_BY_MODEL = {
    "logistic_regression": {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"},
    "mlp_classifier": {"adam", "sgd", "lbfgs"},
    "mlp_regressor": {"adam", "sgd", "lbfgs"},
}


@dataclass(frozen=True, slots=True)
class TrainingActionParseResult:
    selected_option: str
    training_config: dict[str, Any]
    requested_capability_keys: list[str]
    unsupported_reasons: list[str]


def parse_numeric_literal(raw_value: str) -> int | float:
    return float(raw_value) if "." in raw_value else int(raw_value)


def looks_like_training_action_request(text: str) -> bool:
    lowered = text.lower()
    return any(
        hint in lowered
        for hint in [
            "training",
            "cv ",
            "folds",
            "cross-validation",
            "random seed",
            "seed",
            "optimizer",
            "solver",
            "learning rate",
            "epochs",
            "iterations",
            "max iter",
            "mini batch",
            "batch size",
            "early stopping",
        ]
    )


def parse_training_action_request(
    user_message: str,
    model_name: str,
    selected_option: str | None,
    current_option: str,
    current_policy_metadata: dict[str, Any] | None = None,
) -> TrainingActionParseResult | None:
    chosen_option = selected_option or current_option
    if not chosen_option:
        return None

    existing_config = {}
    if isinstance(current_policy_metadata, dict):
        existing_config = dict(current_policy_metadata.get("training_config", {}))

    supported_parameters = set(TRAINING_PARAMETER_SUPPORT.get(model_name, set()))
    updated_config = {
        key: value for key, value in existing_config.items() if key in supported_parameters
    }
    requested_capability_keys: list[str] = []
    unsupported_reasons: list[str] = []

    base_capability_key = TRAINING_BASE_POLICY_CAPABILITIES.get(chosen_option)
    if base_capability_key:
        requested_capability_keys.append(base_capability_key)

    integer_patterns = {
        "cv_folds": r"(?:cv folds|cross[- ]validation folds|folds)\s*(?:=|to)?\s*(\d+)",
        "random_seed": r"(?:random seed|seed)\s*(?:=|to)?\s*(\d+)",
        "epochs": r"(?:epochs|max iterations|max iter|max_iter)\s*(?:=|to)?\s*(\d+)",
        "mini_batch": r"(?:mini[_ -]?batch|batch size|mini batch)\s*(?:=|to)?\s*(\d+)",
    }
    for parameter_name, pattern in integer_patterns.items():
        match = re.search(pattern, user_message, re.IGNORECASE)
        if not match:
            continue
        if parameter_name not in supported_parameters:
            unsupported_reasons.append(
                f"`{parameter_name}` is not an exposed training parameter for the selected model `{model_name}`"
            )
            continue
        updated_config[parameter_name] = int(match.group(1))
        requested_capability_keys.append(TRAINING_PARAMETER_CAPABILITY_KEYS[parameter_name])

    learning_rate_match = re.search(
        r"(?:learning rate|learning_rate)\s*(?:=|to)?\s*([-+]?\d*\.?\d+)",
        user_message,
        re.IGNORECASE,
    )
    if learning_rate_match:
        if "learning_rate" not in supported_parameters:
            unsupported_reasons.append(
                f"`learning_rate` is not an exposed training parameter for the selected model `{model_name}`"
            )
        else:
            updated_config["learning_rate"] = float(learning_rate_match.group(1))
            requested_capability_keys.append(TRAINING_PARAMETER_CAPABILITY_KEYS["learning_rate"])

    optimizer_match = re.search(
        r"(?:optimizer|solver)\s*(?:=|to)?\s*(adam|sgd|lbfgs|saga|liblinear|sag|newton-cg|newton-cholesky)",
        user_message,
        re.IGNORECASE,
    )
    if optimizer_match:
        raw_value = optimizer_match.group(1).lower()
        if "optimizer" not in supported_parameters:
            unsupported_reasons.append(
                f"`optimizer` is not an exposed training parameter for the selected model `{model_name}`"
            )
        else:
            valid_optimizers = VALID_OPTIMIZERS_BY_MODEL.get(model_name, set())
            if raw_value not in valid_optimizers:
                supported_rendered = ", ".join(f"`{item}`" for item in sorted(valid_optimizers)) or "none"
                unsupported_reasons.append(
                    f"`{raw_value}` is not a supported optimizer for `{model_name}`. Supported values here are: {supported_rendered}"
                )
            else:
                updated_config["optimizer"] = raw_value
                requested_capability_keys.append(TRAINING_PARAMETER_CAPABILITY_KEYS["optimizer"])

    if re.search(r"\bearly stopping\b", user_message, re.IGNORECASE):
        if "early_stopping" not in supported_parameters:
            unsupported_reasons.append(
                f"`early_stopping` is not an exposed training parameter for the selected model `{model_name}`"
            )
        else:
            updated_config["early_stopping"] = not bool(
                re.search(r"\b(off|disable|false|no)\b", user_message, re.IGNORECASE)
            )
            requested_capability_keys.append(TRAINING_PARAMETER_CAPABILITY_KEYS["early_stopping"])

    if not selected_option and updated_config == existing_config and not unsupported_reasons:
        return None

    return TrainingActionParseResult(
        selected_option=chosen_option,
        training_config=updated_config,
        requested_capability_keys=requested_capability_keys,
        unsupported_reasons=unsupported_reasons,
    )
