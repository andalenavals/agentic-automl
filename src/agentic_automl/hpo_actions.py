from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .modeling import (
    default_hpo_search_parameters,
    hpo_parameter_alias_lookup,
    hpo_supported_parameters,
)


BASE_POLICY_CAPABILITIES = {
    "skip": "hpo_skip",
    "small_competition": "hpo_small_competition",
    "expanded_competition": "hpo_expanded_competition",
}

HPO_PARAMETER_CAPABILITIES = {
    "C": "hpo_tune_c",
    "alpha": "hpo_tune_alpha",
    "class_weight": "hpo_tune_class_weight",
    "optimizer": "hpo_tune_optimizer",
    "learning_rate": "hpo_tune_learning_rate",
    "n_estimators": "hpo_tune_n_estimators",
    "max_depth": "hpo_tune_max_depth",
    "min_samples_leaf": "hpo_tune_min_samples_leaf",
    "min_samples_split": "hpo_tune_min_samples_split",
    "max_features": "hpo_tune_max_features",
    "max_leaf_nodes": "hpo_tune_max_leaf_nodes",
    "l2_regularization": "hpo_tune_l2_regularization",
    "epochs": "hpo_tune_epochs",
    "hidden_layer_sizes": "hpo_tune_hidden_layer_sizes",
    "mini_batch": "hpo_tune_mini_batch",
}

GENERIC_CANDIDATE_TOKENS = {
    "",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "for",
    "with",
    "of",
    "on",
    "in",
    "please",
    "only",
    "just",
    "these",
    "this",
    "those",
    "parameter",
    "parameters",
    "hyperparameter",
    "hyperparameters",
    "search",
    "scope",
    "competition",
    "model",
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass(frozen=True, slots=True)
class HpoActionParseResult:
    selected_option: str | None
    hpo_config: dict[str, Any]
    requested_capability_keys: list[str]
    unsupported_reasons: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.selected_option or self.hpo_config)


def looks_like_hpo_action_request(text: str) -> bool:
    lowered = text.lower()
    return any(
        hint in lowered
        for hint in [
            "tune ",
            "optimi",
            "hyperparameter",
            "search over",
            "search across",
            "competition",
            "hpo",
            "add ",
            "remove ",
            "drop ",
            "exclude ",
            "include ",
        ]
    )


def infer_hpo_policy_option(user_message: str, current_option: str, explicit_parameters: list[str]) -> str | None:
    lowered = user_message.lower()
    if any(token in lowered for token in ["skip tuning", "skip hpo", "no tuning", "no hpo", "disable tuning"]):
        return "skip"
    if any(token in lowered for token in ["expanded competition", "broader tuning", "broader search", "deeper search", "full competition", "wider search"]):
        return "expanded_competition"
    if any(token in lowered for token in ["small competition", "compact tuning", "compact search", "light tuning"]):
        return "small_competition"
    if any(token in lowered for token in ["run hpo", "run tuning", "enable tuning", "optimize the model", "tune the model"]):
        return "small_competition"
    if explicit_parameters and current_option in {"", "skip"}:
        return "small_competition"
    return None


def normalize_candidate_phrase(candidate: str) -> str:
    lowered = candidate.strip().lower().strip("`'\".?!:;()[]{}")
    lowered = lowered.replace("-", " ").replace("/", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"\b(?:i want|i wanna|please|instead|focus on|search over|search across|optimize|optimise|tune|use)\b", " ", lowered)
    lowered = re.sub(r"\b(?:only|just|these|those|this|the|a|an)\b", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def extract_parameter_candidate_phrases(user_message: str) -> list[str]:
    lowered = user_message.lower()
    segments: list[str] = []
    patterns = [
        r"(?:tune|optimi[sz]e|search over|search across|focus on|run hpo on|run tuning on)\s+(.+)",
        r"(?:hyperparameters?|parameters?)\s*(?:to\s*(?:tune|optimi[sz]e))?\s*[:\-]?\s*(.+)",
        r"(?:use|keep|select|optimize|optimise|tune)\s+only\s+(.+?)\s+as\s+(?:an?\s+)?hyperparameter",
        r"(?:use|keep|select)\s+(.+?)\s+as\s+(?:an?\s+)?hyperparameter",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            segment = re.split(r"[.?!\n]", match.group(1), maxsplit=1)[0]
            segments.append(segment)
    if not segments:
        fallback_mentions = []
        for phrase in re.split(r",|;|\band\b|\bplus\b", lowered):
            normalized = normalize_candidate_phrase(phrase)
            if normalized and normalized not in GENERIC_CANDIDATE_TOKENS:
                fallback_mentions.append(normalized)
        return fallback_mentions if "hyperparameter" in lowered or "tuning" in lowered else []

    candidates: list[str] = []
    for segment in segments:
        split_candidates = re.split(r",|;|\band\b|\bplus\b", segment)
        for candidate in split_candidates:
            normalized = normalize_candidate_phrase(candidate)
            if normalized and normalized not in GENERIC_CANDIDATE_TOKENS:
                candidates.append(normalized)
    return candidates


def match_supported_parameters(candidate_phrases: list[str], model_name: str) -> tuple[list[str], list[str]]:
    alias_lookup = hpo_parameter_alias_lookup(model_name)
    matched: list[str] = []
    unsupported: list[str] = []

    for phrase in candidate_phrases:
        matched_parameter: str | None = None
        if phrase == "c":
            matched_parameter = "C" if "C" in hpo_supported_parameters(model_name) else None
        if matched_parameter is None:
            for alias in sorted(alias_lookup, key=len, reverse=True):
                if phrase == alias or phrase.endswith(alias) or alias in phrase:
                    matched_parameter = alias_lookup[alias]
                    break
        if matched_parameter:
            if matched_parameter not in matched:
                matched.append(matched_parameter)
            continue
        unsupported.append(phrase)
    return matched, unsupported


def extract_direct_supported_parameter_mentions(user_message: str, model_name: str) -> list[str]:
    lowered = user_message.lower()
    alias_lookup = hpo_parameter_alias_lookup(model_name)
    matched_parameters: list[tuple[str, int]] = []

    for alias, parameter_name in alias_lookup.items():
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        match = re.search(pattern, lowered)
        if not match:
            continue
        if parameter_name not in [item[0] for item in matched_parameters]:
            matched_parameters.append((parameter_name, match.start()))

    matched_parameters.sort(key=lambda item: item[1])
    return [parameter_name for parameter_name, _ in matched_parameters]


def message_likely_refines_hpo_scope(user_message: str, current_option: str) -> bool:
    lowered = user_message.lower()
    if current_option and current_option != "skip":
        return True
    return any(
        token in lowered
        for token in [
            "focus on",
            "prioritize",
            "prioritise",
            "use only",
            "keep only",
            "select",
            "include",
            "add",
            "remove",
            "drop",
            "exclude",
            "competition",
            "search scope",
            "search space",
        ]
    )


def should_add_to_existing_scope(user_message: str) -> bool:
    lowered = user_message.lower()
    return any(token in lowered for token in ["add ", "also ", "include ", "plus ", "along with"])


def should_remove_from_existing_scope(user_message: str) -> bool:
    lowered = user_message.lower()
    return any(token in lowered for token in ["remove ", "drop ", "exclude ", "without "])


def extract_priority_count_request(user_message: str) -> int | None:
    lowered = user_message.lower()
    if not any(
        phrase in lowered
        for phrase in [
            "most important hyperparameter",
            "most important parameters",
            "top hyperparameter",
            "top parameters",
            "highest leverage hyperparameter",
            "highest leverage parameters",
            "strongest hyperparameter",
            "strongest parameters",
        ]
    ):
        return None

    numeric_match = re.search(
        r"\b(?:top|only|just|use|keep|select)?\s*(?:the\s+)?(\d+)\s+(?:most important|top|highest leverage|strongest)?\s*hyperparameters?\b",
        lowered,
    )
    if numeric_match:
        return int(numeric_match.group(1))

    word_match = re.search(
        r"\b(?:top|only|just|use|keep|select)?\s*(?:the\s+)?(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:most important|top|highest leverage|strongest)?\s*hyperparameters?\b",
        lowered,
    )
    if word_match:
        return NUMBER_WORDS.get(word_match.group(1))

    reverse_numeric_match = re.search(
        r"\b(?:most important|top|highest leverage|strongest)\s+(\d+)\s+hyperparameters?\b",
        lowered,
    )
    if reverse_numeric_match:
        return int(reverse_numeric_match.group(1))

    reverse_word_match = re.search(
        r"\b(?:most important|top|highest leverage|strongest)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+hyperparameters?\b",
        lowered,
    )
    if reverse_word_match:
        return NUMBER_WORDS.get(reverse_word_match.group(1))

    if any(
        phrase in lowered
        for phrase in [
            "most important hyperparameters",
            "top hyperparameters",
            "highest leverage hyperparameters",
            "strongest hyperparameters",
        ]
    ):
        return 0
    return None


def parse_hpo_action_request(
    user_message: str,
    model_name: str,
    current_option: str,
    current_hpo_config: dict[str, Any] | None = None,
) -> HpoActionParseResult | None:
    existing_config = dict(current_hpo_config or {})
    existing_search_parameters = list(existing_config.get("search_parameters", []) or [])
    explicit_candidates = extract_parameter_candidate_phrases(user_message)
    matched_parameters, unsupported_candidates = match_supported_parameters(explicit_candidates, model_name)
    priority_count = extract_priority_count_request(user_message)
    if priority_count is not None and not matched_parameters:
        ranked_defaults = default_hpo_search_parameters(model_name, "expanded_competition")
        if priority_count <= 0:
            matched_parameters = list(default_hpo_search_parameters(model_name, "small_competition"))
        else:
            matched_parameters = list(ranked_defaults[:priority_count])
        unsupported_candidates = []
    if not matched_parameters and message_likely_refines_hpo_scope(user_message, current_option):
        matched_parameters = extract_direct_supported_parameter_mentions(user_message, model_name)
    selected_option = infer_hpo_policy_option(user_message, current_option, matched_parameters)

    updated_search_parameters = list(existing_search_parameters)
    replace_scope = bool(matched_parameters) and (
        "only" in user_message.lower() or not should_add_to_existing_scope(user_message)
    )
    if matched_parameters:
        if should_remove_from_existing_scope(user_message):
            base_scope = updated_search_parameters or default_hpo_search_parameters(model_name, current_option or "small_competition")
            updated_search_parameters = [item for item in base_scope if item not in matched_parameters]
        elif should_add_to_existing_scope(user_message) and not replace_scope:
            base_scope = updated_search_parameters or default_hpo_search_parameters(model_name, current_option or "small_competition")
            updated_search_parameters = list(base_scope)
            for parameter_name in matched_parameters:
                if parameter_name not in updated_search_parameters:
                    updated_search_parameters.append(parameter_name)
        else:
            updated_search_parameters = list(matched_parameters)

    requested_capability_keys: list[str] = []
    if selected_option in BASE_POLICY_CAPABILITIES:
        requested_capability_keys.append(BASE_POLICY_CAPABILITIES[selected_option])
    if matched_parameters:
        requested_capability_keys.append("hpo_select_search_parameters")
    for parameter_name in matched_parameters:
        capability_key = HPO_PARAMETER_CAPABILITIES.get(parameter_name)
        if capability_key:
            requested_capability_keys.append(capability_key)

    unsupported_reasons: list[str] = []
    if unsupported_candidates:
        unsupported_rendered = ", ".join(f"`{item}`" for item in unsupported_candidates)
        unsupported_reasons.append(
            f"these requested hyperparameters are not supported for the current selected model `{model_name}`: {unsupported_rendered}"
        )

    updated_config = dict(existing_config)
    if matched_parameters:
        updated_config["search_parameters"] = updated_search_parameters

    if not selected_option and updated_config == existing_config and not unsupported_reasons:
        return None

    return HpoActionParseResult(
        selected_option=selected_option,
        hpo_config=updated_config,
        requested_capability_keys=requested_capability_keys,
        unsupported_reasons=unsupported_reasons,
    )
