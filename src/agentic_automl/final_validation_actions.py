from __future__ import annotations

from dataclasses import dataclass


FINAL_VALIDATION_CAPABILITY_KEYS = {
    "final_validation_dashboard": "final_validation_dashboard",
}


@dataclass(frozen=True, slots=True)
class FinalValidationActionParseResult:
    selected_option: str
    requested_capability_keys: list[str]
    unsupported_reasons: list[str]


def looks_like_final_validation_action_request(text: str) -> bool:
    lowered = text.lower()
    return any(
        hint in lowered
        for hint in [
            "final validation",
            "final dashboard",
            "tuned",
            "untuned",
            "optimization summary",
            "notebook",
            "output",
            "prediction",
            "export",
        ]
    )


def parse_final_validation_action_request(
    user_message: str,
    current_option: str,
) -> FinalValidationActionParseResult | None:
    lowered = user_message.lower()
    requested_capability_keys: list[str] = []
    unsupported_reasons: list[str] = []

    if any(
        token in lowered
        for token in [
            "keep the final dashboard",
            "keep final dashboard",
            "show tuned versus untuned",
            "include optimization summary",
            "final validation dashboard",
        ]
    ):
        capability_key = FINAL_VALIDATION_CAPABILITY_KEYS.get(current_option)
        if capability_key:
            requested_capability_keys.append(capability_key)

    if any(token in lowered for token in ["skip final dashboard", "without dashboard", "no dashboard"]):
        unsupported_reasons.append("removing the final dashboard flow is not supported in this step")
    if any(
        token in lowered
        for token in [
            "hide tuning",
            "drop optimization summary",
            "omit optimization summary",
            "omit the optimization summary",
        ]
    ):
        unsupported_reasons.append("removing the optimization summary is not supported in this final validation step")
    if any(token in lowered for token in ["write csv", "save csv", "_withpred.csv", "prediction csv"]):
        unsupported_reasons.append("standalone CSV export is not part of the current final validation output contract")

    if not requested_capability_keys and not unsupported_reasons:
        return None

    return FinalValidationActionParseResult(
        selected_option=current_option,
        requested_capability_keys=requested_capability_keys,
        unsupported_reasons=unsupported_reasons,
    )
