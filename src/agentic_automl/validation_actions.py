from __future__ import annotations

from dataclasses import dataclass


VALIDATION_CAPABILITY_KEYS = {
    "test_set_with_baseline": "validation_test_set_with_baseline",
}


@dataclass(frozen=True, slots=True)
class ValidationActionParseResult:
    selected_option: str
    requested_capability_keys: list[str]
    unsupported_reasons: list[str]


def looks_like_validation_action_request(text: str) -> bool:
    lowered = text.lower()
    return any(
        hint in lowered
        for hint in [
            "baseline",
            "validate",
            "validation",
            "holdout",
            "test set",
            "notebook",
            "output",
            "prediction",
            "export",
            "plot",
        ]
    )


def parse_validation_action_request(
    user_message: str,
    current_option: str,
) -> ValidationActionParseResult | None:
    lowered = user_message.lower()
    requested_capability_keys: list[str] = []
    unsupported_reasons: list[str] = []

    if any(
        token in lowered
        for token in [
            "keep the untouched holdout",
            "keep holdout",
            "compare against baseline",
            "with baseline",
            "test set with baseline",
        ]
    ):
        capability_key = VALIDATION_CAPABILITY_KEYS.get(current_option)
        if capability_key:
            requested_capability_keys.append(capability_key)

    if any(
        token in lowered
        for token in [
            "skip baseline",
            "skip the baseline",
            "without baseline",
            "no baseline",
            "disable baseline",
        ]
    ):
        unsupported_reasons.append("removing the baseline comparison is not supported in this validation step")
    if any(token in lowered for token in ["package output", "python package", "wheel", "module export"]):
        unsupported_reasons.append("the validation step only prepares the notebook-based output flow")
    if any(token in lowered for token in ["write csv", "save csv", "_withpred.csv", "prediction csv"]):
        unsupported_reasons.append("standalone CSV export is not part of the current validation output contract")

    if not requested_capability_keys and not unsupported_reasons:
        return None

    return ValidationActionParseResult(
        selected_option=current_option,
        requested_capability_keys=requested_capability_keys,
        unsupported_reasons=unsupported_reasons,
    )
