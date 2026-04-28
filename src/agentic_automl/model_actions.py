from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


MODEL_CAPABILITY_KEYS = {
    "logistic_regression": "model_logistic_regression",
    "random_forest_classifier": "model_random_forest_classifier",
    "hist_gradient_boosting_classifier": "model_hist_gradient_boosting_classifier",
    "mlp_classifier": "model_mlp_classifier",
    "ridge_regression": "model_ridge_regression",
    "random_forest_regressor": "model_random_forest_regressor",
    "hist_gradient_boosting_regressor": "model_hist_gradient_boosting_regressor",
    "mlp_regressor": "model_mlp_regressor",
}


MODEL_PARAMETER_CAPABILITY_KEYS = {
    "C": "model_override_c",
    "class_weight": "model_override_class_weight",
    "alpha": "model_override_alpha",
    "n_estimators": "model_override_n_estimators",
    "max_depth": "model_override_max_depth",
    "min_samples_leaf": "model_override_min_samples_leaf",
    "max_leaf_nodes": "model_override_max_leaf_nodes",
    "l2_regularization": "model_override_l2_regularization",
    "hidden_layer_sizes": "model_override_hidden_layer_sizes",
}


MODEL_PARAMETER_SUPPORT = {
    "logistic_regression": {"C", "class_weight"},
    "random_forest_classifier": {"n_estimators", "max_depth", "min_samples_leaf", "class_weight"},
    "hist_gradient_boosting_classifier": {"max_depth", "max_leaf_nodes", "l2_regularization"},
    "mlp_classifier": {"hidden_layer_sizes", "alpha"},
    "ridge_regression": {"alpha"},
    "random_forest_regressor": {"n_estimators", "max_depth", "min_samples_leaf"},
    "hist_gradient_boosting_regressor": {"max_depth", "max_leaf_nodes", "l2_regularization"},
    "mlp_regressor": {"hidden_layer_sizes", "alpha"},
}


@dataclass(frozen=True, slots=True)
class ModelActionParseResult:
    selected_option: str
    model_parameters: dict[str, Any]
    requested_capability_keys: list[str]
    unsupported_reasons: list[str]


def parse_numeric_literal(raw_value: str) -> int | float:
    return float(raw_value) if "." in raw_value else int(raw_value)


def looks_like_model_action_request(text: str) -> bool:
    lowered = text.lower()
    return any(
        hint in lowered
        for hint in [
            "use ",
            "switch ",
            "change model",
            "set ",
            "override ",
            "parameter",
            "hyperparameter",
            "n_estimators",
            "max depth",
            "min samples leaf",
            "hidden layers",
            "regularization",
            "alpha",
            "class weight",
        ]
    )


def parse_model_action_request(
    user_message: str,
    selected_option: str | None,
    current_option: str,
    current_policy_metadata: dict[str, Any] | None = None,
) -> ModelActionParseResult | None:
    chosen_model = selected_option or current_option
    if not chosen_model:
        return None

    existing_parameters = {}
    if isinstance(current_policy_metadata, dict):
        existing_parameters = dict(current_policy_metadata.get("model_parameters", {}))

    supported_parameters = set(MODEL_PARAMETER_SUPPORT.get(chosen_model, set()))
    updated_parameters = {
        key: value for key, value in existing_parameters.items() if key in supported_parameters
    }
    requested_capability_keys: list[str] = []
    unsupported_reasons: list[str] = []

    model_capability = MODEL_CAPABILITY_KEYS.get(chosen_model)
    if model_capability:
        requested_capability_keys.append(model_capability)

    parameter_patterns = {
        "n_estimators": r"(?:n_estimators|estimators|number of trees)\s*(?:=|to)?\s*(\d+)",
        "max_depth": r"(?:max_depth|max depth|tree depth)\s*(?:=|to)?\s*(\d+)",
        "min_samples_leaf": r"(?:min_samples_leaf|min samples leaf|leaf size)\s*(?:=|to)?\s*(\d+)",
        "max_leaf_nodes": r"(?:max_leaf_nodes|max leaf nodes|leaf nodes)\s*(?:=|to)?\s*(\d+)",
        "l2_regularization": r"(?:l2_regularization|l2 regularization)\s*(?:=|to)?\s*([-+]?\d*\.?\d+)",
        "alpha": r"(?:alpha|regularization)\s*(?:=|to)?\s*([-+]?\d*\.?\d+)",
        "C": r"(?:\bC\b|inverse regularization)\s*(?:=|to)?\s*([-+]?\d*\.?\d+)",
    }

    for parameter_name, pattern in parameter_patterns.items():
        match = re.search(pattern, user_message, re.IGNORECASE)
        if not match:
            continue
        if parameter_name not in supported_parameters:
            unsupported_reasons.append(
                f"`{parameter_name}` is not an exposed initial parameter for the selected model `{chosen_model}`"
            )
            continue
        updated_parameters[parameter_name] = parse_numeric_literal(match.group(1))
        capability_key = MODEL_PARAMETER_CAPABILITY_KEYS.get(parameter_name)
        if capability_key:
            requested_capability_keys.append(capability_key)

    class_weight_match = re.search(
        r"(?:class weight|class_weight)\s*(?:=|to)?\s*(balanced_subsample|balanced|none)",
        user_message,
        re.IGNORECASE,
    )
    if class_weight_match:
        if "class_weight" not in supported_parameters:
            unsupported_reasons.append(
                f"`class_weight` is not an exposed initial parameter for the selected model `{chosen_model}`"
            )
        else:
            raw_value = class_weight_match.group(1).lower()
            updated_parameters["class_weight"] = None if raw_value == "none" else raw_value
            requested_capability_keys.append(MODEL_PARAMETER_CAPABILITY_KEYS["class_weight"])

    hidden_layers_match = re.search(
        r"(?:hidden[_ ]layer[_ ]sizes|hidden layers|layers)\s*(?:=|to)?\s*([0-9,\sx]+)",
        user_message,
        re.IGNORECASE,
    )
    if hidden_layers_match:
        if "hidden_layer_sizes" not in supported_parameters:
            unsupported_reasons.append(
                f"`hidden_layer_sizes` is not an exposed initial parameter for the selected model `{chosen_model}`"
            )
        else:
            raw_layers = hidden_layers_match.group(1).replace("x", ",")
            layers = tuple(int(part.strip()) for part in raw_layers.split(",") if part.strip().isdigit())
            if layers:
                updated_parameters["hidden_layer_sizes"] = layers
                requested_capability_keys.append(MODEL_PARAMETER_CAPABILITY_KEYS["hidden_layer_sizes"])

    if not selected_option and updated_parameters == existing_parameters and not unsupported_reasons:
        return None

    return ModelActionParseResult(
        selected_option=chosen_model,
        model_parameters=updated_parameters,
        requested_capability_keys=requested_capability_keys,
        unsupported_reasons=unsupported_reasons,
    )
