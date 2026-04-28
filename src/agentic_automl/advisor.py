from __future__ import annotations

from .modeling import (
    choose_baseline_strategy,
    default_model_option,
    default_hpo_search_parameters,
    default_model_parameters,
    default_training_configuration,
    hpo_search_space_config,
    humanize_model_name,
    model_options_for_task,
)
from .schemas import DataProfile, ProjectBrief, WorkflowRecommendation
from .workflow import STEP_DEFINITIONS


def build_recommendations(
    brief: ProjectBrief,
    profile: DataProfile,
) -> list[WorkflowRecommendation]:
    step_map = {step.step_id: step for step in STEP_DEFINITIONS}

    preprocessing_option = choose_preprocessing_option(profile)
    split_option = choose_split_option(brief)
    model_option = default_model_option(brief.task_type, profile)
    model_parameters = default_model_parameters(brief.task_type, model_option, profile)
    metric_option = choose_metric_option(brief, profile)
    training_option = "thorough_training" if profile.rows > 10_000 else "standard_training"
    training_config = default_training_configuration(brief.task_type, model_option, profile)
    baseline_strategy, baseline_reason = choose_baseline_strategy(brief, profile, metric_option)
    hpo_option = "small_competition" if brief.competition_enabled else "skip"

    return [
        WorkflowRecommendation(
            step_id="00_intake",
            step_number=step_map["00_intake"].step_number,
            title=step_map["00_intake"].title,
            recommendation="Ask the human for the dataset path, target column, task type, and what success means.",
            reasoning=reasoning_lines(
                "This module is AI-first, so the human only needs to explain the data and the problem.",
                "All downstream recommendations should come from the structured brief plus observed data.",
            ),
            options=["structured_brief"],
            selected_option="structured_brief",
        ),
        WorkflowRecommendation(
            step_id="01_preprocessing",
            step_number=step_map["01_preprocessing"].step_number,
            title=step_map["01_preprocessing"].title,
            recommendation=preprocessing_recommendation_text(profile, preprocessing_option),
            reasoning=reasoning_lines(
                f"The file inspection found {len(profile.numeric_features)} numeric and {len(profile.categorical_features)} categorical features.",
                preprocessing_reasoning_signal(profile),
                preprocessing_risk_signal(profile),
            ),
            options=["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
            selected_option=preprocessing_option,
        ),
        WorkflowRecommendation(
            step_id="02_data_splitting",
            step_number=step_map["02_data_splitting"].step_number,
            title=step_map["02_data_splitting"].title,
            recommendation=split_recommendation_text(split_option, brief.task_type),
            reasoning=reasoning_lines(
                f"The dataset contains {profile.rows} rows, so we need a clean final holdout.",
                "All later training and tuning must stay inside the training partition to avoid leakage.",
            ),
            options=["stratified_holdout", "random_holdout", "time_ordered_holdout"],
            selected_option=split_option,
        ),
        WorkflowRecommendation(
            step_id="03_model_selection",
            step_number=step_map["03_model_selection"].step_number,
            title=step_map["03_model_selection"].title,
            recommendation=model_recommendation_text(model_option, model_parameters),
            reasoning=reasoning_lines(
                f"The problem is `{brief.task_type}`, so the workflow should start from one concrete model instead of a multi-model sweep.",
                f"Dataset scale and feature mix point to `{model_option}` as the best initial model.",
            ),
            options=model_options_for_task(brief.task_type),
            selected_option=model_option,
            metadata={"model_parameters": model_parameters},
        ),
        WorkflowRecommendation(
            step_id="05_metric_selection",
            step_number=step_map["05_metric_selection"].step_number,
            title=step_map["05_metric_selection"].title,
            recommendation=metric_recommendation_text(brief.task_type, metric_option),
            reasoning=reasoning_lines(
                f"The workflow will optimize for `{metric_option}` and still log supporting metrics.",
                "The same primary metric must be used for validation, baseline comparison, and final dashboard ranking.",
            ),
            options=metric_options_for_task(brief.task_type),
            selected_option=metric_option,
        ),
        WorkflowRecommendation(
            step_id="06_training_configuration",
            step_number=step_map["06_training_configuration"].step_number,
            title=step_map["06_training_configuration"].title,
            recommendation=training_recommendation_text(training_option, model_option, training_config),
            reasoning=reasoning_lines(
                "Training should expose the parameters that are actually relevant for the chosen model.",
                "The initial configuration should stay reproducible while leaving room for Action-mode overrides.",
            ),
            options=["fast_training", "standard_training", "thorough_training"],
            selected_option=training_option,
            metadata={"training_config": training_config},
        ),
        WorkflowRecommendation(
            step_id="08_validation_and_baseline",
            step_number=step_map["08_validation_and_baseline"].step_number,
            title=step_map["08_validation_and_baseline"].title,
            recommendation=validation_recommendation_text(baseline_strategy),
            reasoning=reasoning_lines(
                "Validation should compare the trained model against a no-model baseline on the untouched test set.",
                baseline_reason,
            ),
            options=["test_set_with_baseline"],
            selected_option="test_set_with_baseline",
            metadata={"baseline_strategy": baseline_strategy},
        ),
        WorkflowRecommendation(
            step_id="07_hyperparameter_optimization",
            step_number=step_map["07_hyperparameter_optimization"].step_number,
            title=step_map["07_hyperparameter_optimization"].title,
            recommendation=hpo_recommendation_text(hpo_option, model_option),
            reasoning=reasoning_lines(
                "The optimization stage should tune the already selected model instead of reopening the whole model search space.",
                "This keeps the workflow explainable while still allowing a focused competition on relevant hyperparameters.",
            ),
            options=["skip", "small_competition", "expanded_competition"],
            selected_option=hpo_option,
            metadata={
                "model_option": model_option,
                "search_parameters": [] if hpo_option == "skip" else default_hpo_search_parameters(model_option, hpo_option),
                "search_space": (
                    {}
                    if hpo_option == "skip"
                    else {
                        key: [str(item) for item in values]
                        for key, values in hpo_search_space_config(
                            model_option,
                            hpo_option,
                            requested_parameters=default_hpo_search_parameters(model_option, hpo_option),
                        )["display_grid"].items()
                    }
                ),
            },
        ),
        WorkflowRecommendation(
            step_id="09_final_validation",
            step_number=step_map["09_final_validation"].step_number,
            title=step_map["09_final_validation"].title,
            recommendation=(
                "Repeat validation with the tuned model, compare it again against the baseline, "
                "and build the leading dashboard plus the final rerunnable notebook export."
            ),
            reasoning=reasoning_lines(
                "The final dashboard should make the untuned-versus-tuned story explicit.",
                "If hyperparameter optimization was skipped, this step should still confirm the final notebook-ready workflow path.",
            ),
            options=["final_validation_dashboard"],
            selected_option="final_validation_dashboard",
        ),
    ]


def reasoning_lines(*lines: str) -> list[str]:
    return [line for line in lines if line]


def choose_split_option(brief: ProjectBrief) -> str:
    if brief.date_column:
        return "time_ordered_holdout"
    if brief.task_type == "classification":
        return "stratified_holdout"
    return "random_holdout"


def choose_preprocessing_option(profile: DataProfile) -> str:
    if (
        profile.likely_identifier_features
        or profile.date_like_features
        or profile.constant_features
        or profile.high_cardinality_categorical_features
    ):
        return "custom"
    if profile.missing_fraction == 0:
        return "minimal_cleanup"
    return "auto_tabular_preprocessing"


def choose_metric_option(brief: ProjectBrief, profile: DataProfile) -> str:
    if brief.baseline_metric:
        return brief.baseline_metric
    if brief.task_type == "classification":
        if profile.class_imbalance and profile.class_imbalance >= 1.5:
            return "balanced_accuracy"
        return "f1_macro"
    if profile.target_skew and abs(profile.target_skew) >= 1.0:
        return "mae"
    return "rmse"


def format_feature_list(columns: list[str]) -> str:
    return ", ".join(f"`{column}`" for column in columns)


def format_cardinality_summary(cardinality_map: dict[str, int]) -> str:
    return ", ".join(f"`{column}` ({count})" for column, count in cardinality_map.items())


def format_missing_summary(missing_map: dict[str, float]) -> str:
    ordered = sorted(missing_map.items(), key=lambda item: item[1], reverse=True)
    preview = ordered[:3]
    return ", ".join(f"`{column}` ({fraction:.1%})" for column, fraction in preview)


def format_parse_failure_summary(parse_failures: dict[str, float]) -> str:
    ordered = sorted(parse_failures.items(), key=lambda item: item[1], reverse=True)
    preview = ordered[:3]
    return ", ".join(f"`{column}` ({fraction:.1%})" for column, fraction in preview)


def preprocessing_recommendation_text(profile: DataProfile, option: str) -> str:
    actions: list[str] = []
    if profile.likely_identifier_features:
        actions.append(
            f"exclude likely identifier columns such as {format_feature_list(profile.likely_identifier_features)}"
        )
    if profile.date_like_features:
        actions.append(
            f"parse date-like columns such as {format_feature_list(profile.date_like_features)} into chronological features before modeling"
        )
    if profile.constant_features:
        actions.append(
            f"drop constant columns such as {format_feature_list(profile.constant_features)} because they add no signal"
        )
    repair_signals: list[str] = []
    if profile.missing_by_feature:
        repair_signals.append(f"missing values in {format_missing_summary(profile.missing_by_feature)}")
    if profile.date_parse_failure_by_feature:
        repair_signals.append(f"date parsing gaps in {format_parse_failure_summary(profile.date_parse_failure_by_feature)}")
    if repair_signals:
        actions.append("impute only the features that actually need repair, especially " + " and ".join(repair_signals))
    else:
        actions.append("skip imputation because the inspected file has no missing feature values or date parsing gaps")
    if option in {"auto_tabular_preprocessing", "custom"} and (profile.numeric_features or profile.date_like_features):
        actions.append("scale the surviving numeric and date-derived features with `StandardScaler` so the chosen model sees comparable numeric ranges")
    if profile.high_cardinality_categorical_features:
        actions.append(
            f"review high-cardinality categoricals like {format_cardinality_summary(profile.high_cardinality_categorical_features)} before one-hot encoding"
        )
    elif profile.categorical_features:
        actions.append(f"one-hot encode the manageable categorical fields {format_feature_list(profile.categorical_features[:4])}")
    else:
        actions.append("skip categorical encoding because the file is numeric-only")
    actions.append("keep any feature pruning, feature subsetting, or feature-role overrides inside preprocessing")

    if option == "custom":
        lead = "Use targeted preprocessing driven by the inspected file instead of the generic default:"
    elif option == "minimal_cleanup":
        lead = "Use a light preprocessing pass because the inspected file already looks clean:"
    else:
        lead = "Use automatic tabular preprocessing, but tailor it to the inspected file:"
    return f"{lead} " + "; ".join(actions) + "."


def preprocessing_reasoning_signal(profile: DataProfile) -> str:
    if profile.missing_by_feature and profile.date_parse_failure_by_feature:
        return (
            f"Missing values are concentrated in {format_missing_summary(profile.missing_by_feature)}, "
            f"and date parsing gaps were detected in {format_parse_failure_summary(profile.date_parse_failure_by_feature)}."
        )
    if profile.missing_by_feature:
        return f"Missing values are concentrated in {format_missing_summary(profile.missing_by_feature)}."
    if profile.date_parse_failure_by_feature:
        return f"Date parsing gaps were detected in {format_parse_failure_summary(profile.date_parse_failure_by_feature)}."
    return "The inspected file has no missing feature values."


def preprocessing_risk_signal(profile: DataProfile) -> str:
    signals: list[str] = []
    if profile.likely_identifier_features:
        signals.append(f"likely IDs: {format_feature_list(profile.likely_identifier_features)}")
    if profile.date_like_features:
        signals.append(f"date-like fields: {format_feature_list(profile.date_like_features)}")
    if profile.constant_features:
        signals.append(f"constant fields: {format_feature_list(profile.constant_features)}")
    if profile.high_cardinality_categorical_features:
        signals.append(f"high-cardinality categoricals: {format_cardinality_summary(profile.high_cardinality_categorical_features)}")
    if profile.date_parse_failure_by_feature:
        signals.append(f"date parsing gaps: {format_parse_failure_summary(profile.date_parse_failure_by_feature)}")
    if not signals:
        return "The file does not show obvious preprocessing risks beyond the standard tabular pipeline."
    return "File-specific preprocessing risks were detected: " + "; ".join(signals) + "."


def split_recommendation_text(option: str, task_type: str) -> str:
    if option == "time_ordered_holdout":
        return "Use a time-ordered holdout split so future rows stay in the validation window."
    if option == "stratified_holdout":
        return f"Use an 80/20 stratified holdout split for the {task_type} target."
    return "Use an 80/20 random holdout split so model training, HPO, and validation stay clearly separated."


def format_parameter_map(parameters: dict[str, object]) -> str:
    if not parameters:
        return "no explicit parameters yet"
    formatted: list[str] = []
    for key, value in parameters.items():
        if value is None:
            continue
        formatted.append(f"`{key}`={value!r}")
    return ", ".join(formatted) if formatted else "no explicit parameters yet"


def model_recommendation_text(model_option: str, model_parameters: dict[str, Any]) -> str:
    return (
        f"Start with one specific model: `{model_option}` ({humanize_model_name(model_option)}). "
        f"Initial parameters: {format_parameter_map(model_parameters)}."
    )


def metric_recommendation_text(task_type: str, metric_option: str) -> str:
    if task_type == "classification":
        return f"Optimize for `{metric_option}` while still reporting accuracy and F1 macro for context."
    return f"Optimize for `{metric_option}` while still reporting MAE and R2 as supporting metrics."


def training_recommendation_text(training_option: str, model_option: str, training_config: dict[str, Any]) -> str:
    return (
        f"Use `{training_option}` for `{model_option}` with these initial training parameters: "
        f"{format_parameter_map(training_config)}."
    )


def validation_recommendation_text(baseline_strategy: str) -> str:
    return (
        "Validate the trained model on the untouched test split and compare it against a no-model baseline. "
        f"The recommended baseline strategy is `{baseline_strategy}`."
    )


def hpo_recommendation_text(option: str, model_option: str) -> str:
    if option == "skip":
        return f"Skip tuning for now and carry `{model_option}` directly into validation."
    if option == "expanded_competition":
        return f"Run an expanded search over the key hyperparameters of `{model_option}`."
    return f"Run a compact gym-style search over the most important hyperparameters of `{model_option}`."


def metric_options_for_task(task_type: str) -> list[str]:
    if task_type == "classification":
        return ["balanced_accuracy", "f1_macro", "accuracy"]
    return ["rmse", "mae", "r2"]
