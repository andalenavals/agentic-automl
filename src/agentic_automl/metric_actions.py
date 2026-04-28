from __future__ import annotations

from dataclasses import dataclass


METRIC_CAPABILITY_KEYS = {
    "balanced_accuracy": "metric_balanced_accuracy",
    "f1_macro": "metric_f1_macro",
    "accuracy": "metric_accuracy",
    "rmse": "metric_rmse",
    "mae": "metric_mae",
    "r2": "metric_r2",
}

UNSUPPORTED_METRIC_REASONS = {
    "auc": "AUC-style ranking metrics are not packaged yet in this metric-selection step",
    "roc": "ROC-AUC is not packaged yet in this metric-selection step",
    "precision": "precision-only winner metrics are not packaged yet in this metric-selection step",
    "recall": "recall-only winner metrics are not packaged yet in this metric-selection step",
    "logloss": "log-loss is not packaged yet in this metric-selection step",
    "log loss": "log-loss is not packaged yet in this metric-selection step",
    "mse": "MSE is not packaged yet as a primary winner metric in this step",
    "mape": "MAPE is not packaged yet as a primary winner metric in this step",
    "msle": "MSLE is not packaged yet as a primary winner metric in this step",
}


@dataclass(frozen=True, slots=True)
class MetricActionParseResult:
    selected_option: str
    requested_capability_keys: list[str]
    unsupported_reasons: list[str]


def looks_like_metric_action_request(text: str) -> bool:
    lowered = text.lower()
    return any(
        hint in lowered
        for hint in [
            "metric",
            "score",
            "optimize for",
            "optimise for",
            "winner",
            "balanced accuracy",
            "f1",
            "accuracy",
            "rmse",
            "mae",
            "r2",
            "auc",
            "precision",
            "recall",
            "mape",
        ]
    )


def infer_metric_option(user_message: str) -> str | None:
    lowered = user_message.lower()
    if "balanced accuracy" in lowered:
        return "balanced_accuracy"
    if "f1 macro" in lowered or "macro f1" in lowered:
        return "f1_macro"
    if "accuracy" in lowered and "balanced accuracy" not in lowered:
        return "accuracy"
    if "rmse" in lowered or "root mean squared error" in lowered:
        return "rmse"
    if "mae" in lowered or "mean absolute error" in lowered:
        return "mae"
    if "r2" in lowered or "r-squared" in lowered or "explained variance" in lowered:
        return "r2"
    return None


def parse_metric_action_request(
    user_message: str,
    current_option: str,
) -> MetricActionParseResult | None:
    selected_option = infer_metric_option(user_message)
    requested_capability_keys: list[str] = []
    unsupported_reasons: list[str] = []

    if selected_option:
        capability_key = METRIC_CAPABILITY_KEYS.get(selected_option)
        if capability_key:
            requested_capability_keys.append(capability_key)

    lowered = user_message.lower()
    for token, reason in UNSUPPORTED_METRIC_REASONS.items():
        if token in lowered:
            unsupported_reasons.append(reason)

    if not selected_option and not unsupported_reasons:
        return None

    return MetricActionParseResult(
        selected_option=selected_option or current_option,
        requested_capability_keys=requested_capability_keys,
        unsupported_reasons=unsupported_reasons,
    )
