from __future__ import annotations

from dataclasses import dataclass


SPLIT_CAPABILITY_KEYS = {
    "stratified_holdout": "split_stratified_holdout",
    "random_holdout": "split_random_holdout",
    "time_ordered_holdout": "split_time_ordered_holdout",
}


@dataclass(frozen=True, slots=True)
class SplitActionParseResult:
    selected_option: str
    requested_capability_keys: list[str]
    unsupported_reasons: list[str]


def looks_like_split_action_request(text: str) -> bool:
    lowered = text.lower()
    return any(
        hint in lowered
        for hint in [
            "split",
            "holdout",
            "train/test",
            "test set",
            "validation set",
            "chronolog",
            "time order",
            "temporal",
            "stratif",
            "shuffle",
            "group split",
            "test size",
            "80/20",
            "70/30",
            "90/10",
        ]
    )


def infer_split_option(user_message: str, current_option: str) -> str | None:
    lowered = user_message.lower()
    if any(token in lowered for token in ["time ordered", "time-order", "time split", "chronolog", "temporal", "future rows", "preserve chronology"]):
        return "time_ordered_holdout"
    if any(token in lowered for token in ["stratified", "class balance", "class proportions"]):
        return "stratified_holdout"
    if any(token in lowered for token in ["random holdout", "random split", "shuffle split", "shuffled split"]):
        return "random_holdout"
    return None


def parse_split_action_request(
    user_message: str,
    current_option: str,
) -> SplitActionParseResult | None:
    selected_option = infer_split_option(user_message, current_option)
    requested_capability_keys: list[str] = []
    unsupported_reasons: list[str] = []

    if selected_option:
        capability_key = SPLIT_CAPABILITY_KEYS.get(selected_option)
        if capability_key:
            requested_capability_keys.append(capability_key)

    lowered = user_message.lower()
    if any(token in lowered for token in ["group split", "grouped split", "group kfold", "group-kfold", "leave one group out"]):
        unsupported_reasons.append("group-aware final holdout strategies are not supported yet in this step")
    if any(token in lowered for token in ["k-fold", "cross-validation split", "cross validation split"]):
        unsupported_reasons.append("cross-validation split design is handled in training configuration, not in this final-holdout step")
    if any(token in lowered for token in ["test size", "validation fraction", "80/20", "70/30", "90/10", "% split"]):
        unsupported_reasons.append("custom holdout ratios are not exposed yet in this packaged data-splitting step")

    if not selected_option and not unsupported_reasons:
        return None

    return SplitActionParseResult(
        selected_option=selected_option or current_option,
        requested_capability_keys=requested_capability_keys,
        unsupported_reasons=unsupported_reasons,
    )
