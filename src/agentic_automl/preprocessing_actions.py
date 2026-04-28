from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .preprocessing import normalize_preprocessing_overrides


PREPROCESSING_CAPABILITIES = {
    "keep_subset": "scope_keep_subset",
    "drop_named_feature": "scope_drop_named_features",
    "keep_named_feature": "scope_keep_named_features",
    "only_numeric": "scope_only_numeric",
    "only_categorical": "scope_only_categorical",
    "only_date_like": "scope_only_date_like",
    "exclude_numeric": "scope_exclude_numeric",
    "exclude_categorical": "scope_exclude_categorical",
    "exclude_date_like": "scope_exclude_date_like",
    "as_categorical": "role_as_categorical",
    "as_numeric": "role_as_numeric",
    "as_date_like": "role_as_date_like",
    "as_identifier": "role_as_identifier",
    "one_hot": "encoding_one_hot",
    "ordinal": "encoding_ordinal",
    "frequency_encoding": "encoding_frequency",
    "target_encoding": "encoding_target",
    "binary_encoding": "encoding_binary",
    "hashing_encoding": "encoding_hashing",
    "count_encoding": "encoding_count",
    "woe_encoding": "encoding_weight_of_evidence",
    "impute_mean": "impute_mean",
    "impute_median": "impute_median",
    "impute_most_frequent": "impute_most_frequent",
    "impute_constant": "impute_constant",
    "scale_standard": "scale_standard",
    "scale_minmax": "scale_minmax",
    "scale_robust": "scale_robust",
    "scale_none": "scale_none",
    "transform_log": "transform_log",
    "transform_log1p": "transform_log1p",
    "transform_sqrt": "transform_sqrt",
    "transform_square": "transform_square",
    "transform_abs": "transform_abs",
    "transform_multiply_constant": "transform_multiply_constant",
    "transform_divide_constant": "transform_divide_constant",
    "transform_add_constant": "transform_add_constant",
    "transform_subtract_constant": "transform_subtract_constant",
    "derive_frequency_count": "derive_frequency_count",
    "derive_multi_source_expression": "derive_multi_source_expression",
}


@dataclass(frozen=True, slots=True)
class PreprocessingActionParseResult:
    preprocessing_overrides: dict[str, Any]
    requested_capability_keys: list[str]
    unsupported_reasons: list[str]

    @property
    def has_executable_overrides(self) -> bool:
        return any(bool(value) for value in self.preprocessing_overrides.values())


def feature_aliases(feature_name: str) -> set[str]:
    lowered = feature_name.lower()
    return {
        lowered,
        lowered.replace("_", " "),
        lowered.replace(" ", "_"),
        lowered.replace("-", " "),
        lowered.replace(" ", "-"),
    }


def text_mentions_feature(text: str, feature_name: str) -> bool:
    lowered = text.lower()
    for alias in feature_aliases(feature_name):
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        if re.search(pattern, lowered):
            return True
    return False


def extract_profile_feature_names(profile: dict[str, Any] | None) -> list[str]:
    if not profile:
        return []
    ordered: list[str] = []
    for group in [profile.get("numeric_features", []), profile.get("categorical_features", [])]:
        for feature in group:
            if feature not in ordered:
                ordered.append(feature)
    return ordered


def extract_profile_feature_groups(profile: dict[str, Any] | None) -> dict[str, list[str]]:
    if not profile:
        return {
            "numeric": [],
            "categorical": [],
            "date_like": [],
            "categorical_non_date": [],
        }
    numeric_features = list(profile.get("numeric_features", []))
    date_like_features = list(profile.get("date_like_features", []))
    categorical_features = list(profile.get("categorical_features", []))
    categorical_non_date = [feature for feature in categorical_features if feature not in date_like_features]
    return {
        "numeric": numeric_features,
        "categorical": categorical_features,
        "date_like": date_like_features,
        "categorical_non_date": categorical_non_date,
    }


def detect_clause_features(clause: str, feature_names: list[str]) -> list[str]:
    return [feature for feature in feature_names if text_mentions_feature(clause, feature)]


def find_feature_mentions(clause: str, feature_names: list[str]) -> list[tuple[str, int, int]]:
    mentions: list[tuple[str, int, int]] = []
    lowered = clause.lower()
    for feature in feature_names:
        matches: list[tuple[int, int]] = []
        for alias in feature_aliases(feature):
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            match = re.search(pattern, lowered)
            if match:
                matches.append((match.start(), match.end()))
        if matches:
            start, end = min(matches, key=lambda item: item[0])
            mentions.append((feature, start, end))
    mentions.sort(key=lambda item: item[1])
    return mentions


def clause_requests_role_change(clause: str, role: str) -> bool:
    if role == "categorical":
        return bool(re.search(r"\b(categorical|category|categories)\b", clause))
    if role == "numeric":
        return bool(re.search(r"\b(numeric|numerical|continuous)\b", clause))
    if role == "date":
        return bool(re.search(r"\b(date|datetime|timestamp|temporal|time)\b", clause))
    if role == "identifier":
        return bool(re.search(r"\b(identifier|identifiers|id|ids|key|keys)\b", clause))
    return False


def clause_requests_feature_encoding(clause: str, encoding: str) -> bool:
    if encoding == "one_hot":
        return bool(
            re.search(r"\bone[\s-]?hot\b", clause)
            or re.search(r"\bdummy(?:\s+variable|\s+encode|\s+encoding)?\b", clause)
        )
    if encoding == "ordinal":
        return bool(
            re.search(r"\bordinal\b", clause)
            or re.search(r"\blabel(?:\s+encode|\s+encoding)?\b", clause)
            or re.search(r"\binteger(?:\s+encode|\s+encoding)?\b", clause)
        )
    return False


def unsupported_encoding_requests(clause: str) -> list[tuple[str, str]]:
    requests: list[tuple[str, str]] = []
    checks = [
        (
            PREPROCESSING_CAPABILITIES["frequency_encoding"],
            r"\bfrequency[\s-]?encoding\b",
            "frequency encoding is not supported yet",
        ),
        (
            PREPROCESSING_CAPABILITIES["count_encoding"],
            r"\bcount[\s-]?encoding\b",
            "count encoding is not supported yet",
        ),
        (
            PREPROCESSING_CAPABILITIES["target_encoding"],
            r"\b(target|mean|leave[\s-]?one[\s-]?out|catboost)[\s-]?encoding\b",
            "target-style categorical encoding is not supported yet",
        ),
        (
            PREPROCESSING_CAPABILITIES["binary_encoding"],
            r"\bbinary[\s-]?encoding\b",
            "binary encoding is not supported yet",
        ),
        (
            PREPROCESSING_CAPABILITIES["hashing_encoding"],
            r"\b(hash|hashing)[\s-]?encoding\b",
            "hashing encoding is not supported yet",
        ),
        (
            PREPROCESSING_CAPABILITIES["woe_encoding"],
            r"\b(woe|weight[\s-]?of[\s-]?evidence)[\s-]?encoding\b",
            "weight-of-evidence encoding is not supported yet",
        ),
    ]
    for capability_key, pattern, reason in checks:
        if re.search(pattern, clause):
            requests.append((capability_key, reason))
    return requests


def message_requests_only_role(text: str, role: str) -> bool:
    patterns = {
        "numeric": r"(numeric|numerical)",
        "categorical": r"(categorical|category|categories)",
        "date": r"(date|datetime|timestamp|temporal|time)",
    }
    role_pattern = patterns[role]
    return bool(
        re.search(rf"\b(?:use|keep|select|retain|work with|model with)\b.*\bonly\b.*\b{role_pattern}\b", text)
        or re.search(rf"\bonly\b.*\b{role_pattern}\b.*\b(features?|columns?|variables?)\b", text)
        or re.search(rf"\bjust\b.*\b{role_pattern}\b.*\b(features?|columns?|variables?)\b", text)
    )


def message_requests_drop_role(text: str, role: str) -> bool:
    patterns = {
        "numeric": r"(numeric|numerical)",
        "categorical": r"(categorical|category|categories)",
        "date": r"(date|datetime|timestamp|temporal|time)",
    }
    role_pattern = patterns[role]
    return bool(
        re.search(rf"\b(?:drop|remove|exclude|ignore|skip)\b.*\b{role_pattern}\b", text)
        or re.search(rf"\bwithout\b.*\b{role_pattern}\b", text)
        or re.search(rf"\bno\b.*\b{role_pattern}\b.*\b(features?|columns?|variables?)\b", text)
    )


def clause_requests_feature_subset(clause: str) -> bool:
    return bool(
        re.search(r"\b(?:use|keep|select|retain|include|work with|model with)\b.*\bonly\b", clause)
        or re.search(r"\bkeep\b.*\bjust\b", clause)
        or re.search(r"\bonly\b.*\b(features?|columns?|variables?)\b", clause)
        or re.search(r"\bjust\b.*\b(features?|columns?|variables?)\b", clause)
    )


def looks_like_action_request(text: str) -> bool:
    lowered = text.lower()
    return any(
        hint in lowered
        for hint in [
            "apply ",
            "use ",
            "keep ",
            "drop ",
            "remove ",
            "exclude ",
            "add ",
            "create ",
            "extend ",
            "derive ",
            "generate ",
            "encode ",
            "imput",
            "fill ",
            "transform",
            "scale ",
            "normalize",
            "standardiz",
            "add a new column",
            "create a new column",
            "feature called",
            "column called",
            "equal to",
            "feature engineering",
            "multiply ",
            "divide ",
            "subtract ",
            "frequency count",
        ]
    )


def parse_literal_value(raw_value: str) -> Any:
    value = raw_value.strip().strip(".,;")
    if not value:
        return None
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"zero", "zeros"}:
        return 0
    if lowered in {"unknown", "missing"}:
        return lowered
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?\d*\.?\d+", value):
        return float(value)
    return value


def extract_fill_value_from_text(text: str) -> Any:
    quoted_match = re.search(r"(?:with|value)\s+([\"'][^\"']+[\"'])", text)
    if quoted_match:
        return parse_literal_value(quoted_match.group(1))

    numeric_match = re.search(r"(?:with|value)\s+([-+]?\d*\.?\d+)", text)
    if numeric_match:
        return parse_literal_value(numeric_match.group(1))

    token_match = re.search(r"(?:with|value)\s+([a-zA-Z_][\w-]*)", text)
    if token_match:
        return parse_literal_value(token_match.group(1))
    return None


def parse_feature_imputation_rule(
    clause: str,
    feature: str,
    numeric_feature_set: set[str],
    categorical_feature_set: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    if not re.search(r"\b(imput\w*|fill\w*|missing)\b", clause):
        return None, None

    if "mean" in clause:
        if feature not in numeric_feature_set:
            return None, "mean imputation is only supported for numeric features"
        return {"feature": feature, "strategy": "mean", "value": None}, None
    if "median" in clause:
        if feature not in numeric_feature_set:
            return None, "median imputation is only supported for numeric features"
        return {"feature": feature, "strategy": "median", "value": None}, None
    if "most frequent" in clause or "mode" in clause:
        return {"feature": feature, "strategy": "most_frequent", "value": None}, None
    if "zero" in clause or "zeros" in clause:
        return {"feature": feature, "strategy": "constant", "value": 0}, None

    if "out of range" in clause or "constant" in clause or "specific value" in clause or "fill" in clause:
        fill_value = extract_fill_value_from_text(clause)
        if fill_value is None:
            return None, "constant-value imputation requires an explicit replacement value"
        return {"feature": feature, "strategy": "constant", "value": fill_value}, None

    return None, "that imputation strategy is not supported yet"


def parse_feature_scaling_rule(
    clause: str,
    feature: str,
    numeric_feature_set: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    if feature not in numeric_feature_set and not re.search(r"\b(scale|normaliz|standardiz|robust)\b", clause):
        return None, None
    if not re.search(r"\b(scale|normaliz|standardiz|robust|min[\s-]?max|unscaled)\b", clause):
        return None, None
    if feature not in numeric_feature_set:
        return None, "feature-specific scaling is only supported for numeric features"
    if "standard" in clause or "z-score" in clause or "z score" in clause:
        return {"feature": feature, "method": "standard"}, None
    if "normaliz" in clause or "min max" in clause or "minmax" in clause:
        return {"feature": feature, "method": "minmax"}, None
    if "robust" in clause:
        return {"feature": feature, "method": "robust"}, None
    if "unscaled" in clause or "no scaling" in clause:
        return {"feature": feature, "method": "none"}, None
    return None, "that scaling method is not supported yet"


def parse_feature_transform_rule(
    clause: str,
    feature: str,
    numeric_feature_set: set[str],
) -> tuple[list[dict[str, Any]], str | None]:
    if feature not in numeric_feature_set:
        return [], None
    if "log1p" in clause:
        return [{"feature": feature, "kind": "log1p", "value": None}], None
    if re.search(r"\blog\b|\blogarithm", clause):
        return [{"feature": feature, "kind": "log", "value": None}], None
    if "sqrt" in clause or "square root" in clause:
        return [{"feature": feature, "kind": "sqrt", "value": None}], None
    if "square" in clause and "square root" not in clause:
        return [{"feature": feature, "kind": "square", "value": None}], None
    if "absolute value" in clause or re.search(r"\babs\b", clause):
        return [{"feature": feature, "kind": "abs", "value": None}], None

    multiply_match = re.search(r"\bmultiply\b.*\bby\b\s*([-+]?\d*\.?\d+)", clause)
    if multiply_match:
        return [{"feature": feature, "kind": "multiply", "value": parse_literal_value(multiply_match.group(1))}], None
    divide_match = re.search(r"\bdivide\b.*\bby\b\s*([-+]?\d*\.?\d+)", clause)
    if divide_match:
        return [{"feature": feature, "kind": "divide", "value": parse_literal_value(divide_match.group(1))}], None
    add_match = re.search(r"\badd\b\s*([-+]?\d*\.?\d+)\b", clause)
    if add_match:
        return [{"feature": feature, "kind": "add", "value": parse_literal_value(add_match.group(1))}], None
    subtract_match = re.search(r"\bsubtract\b\s*([-+]?\d*\.?\d+)\b", clause)
    if subtract_match:
        return [{"feature": feature, "kind": "subtract", "value": parse_literal_value(subtract_match.group(1))}], None

    if re.search(r"\btransform\w*\b", clause):
        return [], "that feature transformation is not supported yet"
    return [], None


def parse_frequency_count_rule(clause: str, feature: str) -> tuple[dict[str, Any] | None, str | None]:
    if not re.search(r"\b(frequency count|frequency counts|count column|count feature)\b", clause):
        return None, None
    return {
        "kind": "frequency_count",
        "source": feature,
        "output": f"{feature}__frequency_count",
    }, None


def clause_requests_new_feature(clause: str) -> bool:
    return bool(
        re.search(r"\b(add|create|derive|generate|engineer|build|extend)\b.*\b(feature|column|variable)\b", clause)
        or re.search(r"\bfeature\s+called\b", clause)
        or re.search(r"\bcolumn\s+called\b", clause)
        or re.search(r"\bextend\b.*\bfeature set\b", clause)
    )


def clause_requests_unsupported_derived_feature(
    clause: str,
    feature_names: list[str],
) -> bool:
    if not clause_requests_new_feature(clause):
        return False
    if re.search(r"\b(frequency count|frequency counts|count column|count feature)\b", clause):
        return False
    feature_mentions = detect_clause_features(clause, feature_names)
    if len(feature_mentions) < 2:
        return False
    return bool(
        re.search(r"[=:+\-*/]", clause)
        or re.search(r"\b(equal to|equals|operation|interaction|cross feature|combined with|combine)\b", clause)
        or re.search(r"\b(sum|difference|product|ratio|multiply|divide|add|subtract)\b", clause)
    )


def capability_key_for_imputation_rule(rule: dict[str, Any]) -> str | None:
    strategy = str(rule.get("strategy", "")).strip().lower()
    mapping = {
        "mean": PREPROCESSING_CAPABILITIES["impute_mean"],
        "median": PREPROCESSING_CAPABILITIES["impute_median"],
        "most_frequent": PREPROCESSING_CAPABILITIES["impute_most_frequent"],
        "constant": PREPROCESSING_CAPABILITIES["impute_constant"],
    }
    return mapping.get(strategy)


def capability_key_for_scaling_rule(rule: dict[str, Any]) -> str | None:
    method = str(rule.get("method", "")).strip().lower()
    mapping = {
        "standard": PREPROCESSING_CAPABILITIES["scale_standard"],
        "minmax": PREPROCESSING_CAPABILITIES["scale_minmax"],
        "robust": PREPROCESSING_CAPABILITIES["scale_robust"],
        "none": PREPROCESSING_CAPABILITIES["scale_none"],
    }
    return mapping.get(method)


def capability_keys_for_transform_rules(rules: list[dict[str, Any]]) -> list[str]:
    mapping = {
        "log": PREPROCESSING_CAPABILITIES["transform_log"],
        "log1p": PREPROCESSING_CAPABILITIES["transform_log1p"],
        "sqrt": PREPROCESSING_CAPABILITIES["transform_sqrt"],
        "square": PREPROCESSING_CAPABILITIES["transform_square"],
        "abs": PREPROCESSING_CAPABILITIES["transform_abs"],
        "multiply": PREPROCESSING_CAPABILITIES["transform_multiply_constant"],
        "divide": PREPROCESSING_CAPABILITIES["transform_divide_constant"],
        "add": PREPROCESSING_CAPABILITIES["transform_add_constant"],
        "subtract": PREPROCESSING_CAPABILITIES["transform_subtract_constant"],
    }
    capability_keys: list[str] = []
    for rule in rules:
        capability_key = mapping.get(str(rule.get("kind", "")).strip().lower())
        if capability_key:
            capability_keys.append(capability_key)
    return capability_keys


def dedupe_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        ordered.append(message)
    return ordered


def parse_preprocessing_action_request(
    user_message: str,
    profile: dict[str, Any] | None,
) -> PreprocessingActionParseResult:
    overrides = normalize_preprocessing_overrides(None)
    feature_names = extract_profile_feature_names(profile)
    feature_groups = extract_profile_feature_groups(profile)
    numeric_feature_set = set(feature_groups["numeric"])
    categorical_feature_set = set(feature_groups["categorical"])
    date_feature_set = set(feature_groups["date_like"])
    if not feature_names:
        return PreprocessingActionParseResult(
            preprocessing_overrides=overrides,
            requested_capability_keys=[],
            unsupported_reasons=[],
        )

    lowered_message = user_message.lower()
    unsupported_reasons: list[str] = []
    requested_capability_keys: list[str] = []
    explicit_feature_subset = False
    clauses = [clause.strip().lower() for clause in re.split(r"[\n.;]+", user_message) if clause.strip()]
    for clause in clauses:
        clause_supported_action = False
        unsupported_count_before = len(unsupported_reasons)
        for capability_key, reason in unsupported_encoding_requests(clause):
            requested_capability_keys.append(capability_key)
            unsupported_reasons.append(reason)
        if clause_requests_unsupported_derived_feature(clause, feature_names):
            requested_capability_keys.append(PREPROCESSING_CAPABILITIES["derive_multi_source_expression"])
            unsupported_reasons.append(
                "creating a new derived feature from multiple source columns is not supported yet"
            )
        mentions = detect_clause_features(clause, feature_names)
        if mentions and clause_requests_feature_subset(clause):
            selected_features = list(dict.fromkeys(mentions))
            excluded_features = [feature for feature in feature_names if feature not in selected_features]
            overrides["keep_features"].extend(selected_features)
            overrides["drop_features"].extend(excluded_features)
            explicit_feature_subset = True
            clause_supported_action = True
            requested_capability_keys.append(PREPROCESSING_CAPABILITIES["keep_subset"])

        if not explicit_feature_subset and message_requests_only_role(lowered_message, "numeric"):
            overrides["drop_features"].extend(feature_groups["categorical"])
            clause_supported_action = True
            requested_capability_keys.append(PREPROCESSING_CAPABILITIES["only_numeric"])
        elif not explicit_feature_subset and message_requests_only_role(lowered_message, "categorical"):
            overrides["drop_features"].extend(feature_groups["numeric"])
            overrides["drop_features"].extend(feature_groups["date_like"])
            clause_supported_action = True
            requested_capability_keys.append(PREPROCESSING_CAPABILITIES["only_categorical"])
        elif not explicit_feature_subset and message_requests_only_role(lowered_message, "date"):
            overrides["drop_features"].extend(feature_groups["numeric"])
            overrides["drop_features"].extend(feature_groups["categorical_non_date"])
            clause_supported_action = True
            requested_capability_keys.append(PREPROCESSING_CAPABILITIES["only_date_like"])

        if not explicit_feature_subset and message_requests_drop_role(lowered_message, "categorical"):
            overrides["drop_features"].extend(feature_groups["categorical"])
            clause_supported_action = True
            requested_capability_keys.append(PREPROCESSING_CAPABILITIES["exclude_categorical"])
        if not explicit_feature_subset and message_requests_drop_role(lowered_message, "numeric"):
            overrides["drop_features"].extend(feature_groups["numeric"])
            clause_supported_action = True
            requested_capability_keys.append(PREPROCESSING_CAPABILITIES["exclude_numeric"])
        if not explicit_feature_subset and message_requests_drop_role(lowered_message, "date"):
            overrides["drop_features"].extend(feature_groups["date_like"])
            clause_supported_action = True
            requested_capability_keys.append(PREPROCESSING_CAPABILITIES["exclude_date_like"])

        for feature, start, end in find_feature_mentions(clause, feature_names):
            window = clause[max(0, start - 60): min(len(clause), end + 60)]
            matched_supported_action = False
            if re.search(r"\b(drop|remove|exclude|ignore|skip)\b", window):
                overrides["drop_features"].append(feature)
                matched_supported_action = True
                requested_capability_keys.append(PREPROCESSING_CAPABILITIES["drop_named_feature"])
            if re.search(r"\b(keep|retain|include|preserve)\b", window):
                overrides["keep_features"].append(feature)
                matched_supported_action = True
                requested_capability_keys.append(PREPROCESSING_CAPABILITIES["keep_named_feature"])
            if clause_requests_feature_encoding(window, "one_hot"):
                overrides["force_one_hot_features"].append(feature)
                matched_supported_action = True
                requested_capability_keys.append(PREPROCESSING_CAPABILITIES["one_hot"])
            if clause_requests_feature_encoding(window, "ordinal"):
                overrides["force_ordinal_features"].append(feature)
                matched_supported_action = True
                requested_capability_keys.append(PREPROCESSING_CAPABILITIES["ordinal"])
            if clause_requests_role_change(window, "categorical") and feature not in categorical_feature_set:
                overrides["force_categorical_features"].append(feature)
                matched_supported_action = True
                requested_capability_keys.append(PREPROCESSING_CAPABILITIES["as_categorical"])
            if clause_requests_role_change(window, "numeric") and feature not in numeric_feature_set:
                overrides["force_numeric_features"].append(feature)
                matched_supported_action = True
                requested_capability_keys.append(PREPROCESSING_CAPABILITIES["as_numeric"])
            if clause_requests_role_change(window, "date") and feature not in date_feature_set:
                overrides["force_date_features"].append(feature)
                matched_supported_action = True
                requested_capability_keys.append(PREPROCESSING_CAPABILITIES["as_date_like"])
            if clause_requests_role_change(window, "identifier"):
                overrides["force_identifier_features"].append(feature)
                matched_supported_action = True
                requested_capability_keys.append(PREPROCESSING_CAPABILITIES["as_identifier"])

            imputation_rule, imputation_error = parse_feature_imputation_rule(
                window,
                feature,
                numeric_feature_set,
                categorical_feature_set,
            )
            if imputation_rule:
                overrides["feature_imputation_rules"].append(imputation_rule)
                matched_supported_action = True
                capability_key = capability_key_for_imputation_rule(imputation_rule)
                if capability_key:
                    requested_capability_keys.append(capability_key)
            elif imputation_error:
                unsupported_reasons.append(imputation_error)

            scaling_rule, scaling_error = parse_feature_scaling_rule(
                window,
                feature,
                numeric_feature_set,
            )
            if scaling_rule:
                overrides["feature_scaling_rules"].append(scaling_rule)
                matched_supported_action = True
                capability_key = capability_key_for_scaling_rule(scaling_rule)
                if capability_key:
                    requested_capability_keys.append(capability_key)
            elif scaling_error:
                unsupported_reasons.append(scaling_error)

            transform_rules, transform_error = parse_feature_transform_rule(
                window,
                feature,
                numeric_feature_set,
            )
            if transform_rules:
                overrides["feature_transform_rules"].extend(transform_rules)
                matched_supported_action = True
                requested_capability_keys.extend(capability_keys_for_transform_rules(transform_rules))
            elif transform_error:
                unsupported_reasons.append(transform_error)

            derived_rule, derived_error = parse_frequency_count_rule(window, feature)
            if derived_rule:
                overrides["derived_feature_rules"].append(derived_rule)
                matched_supported_action = True
                requested_capability_keys.append(PREPROCESSING_CAPABILITIES["derive_frequency_count"])
            elif derived_error:
                unsupported_reasons.append(derived_error)

            if matched_supported_action:
                clause_supported_action = True

        if (
            mentions
            and looks_like_action_request(clause)
            and not clause_supported_action
            and len(unsupported_reasons) == unsupported_count_before
        ):
            unsupported_reasons.append("that preprocessing action is not supported yet")

    return PreprocessingActionParseResult(
        preprocessing_overrides=normalize_preprocessing_overrides(overrides),
        requested_capability_keys=dedupe_messages(requested_capability_keys),
        unsupported_reasons=dedupe_messages(unsupported_reasons),
    )
