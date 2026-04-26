from __future__ import annotations

from .schemas import DataProfile, ProjectBrief, WorkflowRecommendation
from .workflow import STEP_DEFINITIONS


def build_recommendations(
    brief: ProjectBrief,
    profile: DataProfile,
    memories: dict[str, str],
) -> list[WorkflowRecommendation]:
    step_map = {step.step_id: step for step in STEP_DEFINITIONS}
    model_option = "balanced_candidate_set"
    if profile.rows > 100_000:
        model_option = "speed_first"
    if len(profile.categorical_features) == 0 and profile.rows > 10_000:
        model_option = "tree_ensemble_first"

    metric_option = choose_metric_option(brief, profile)
    split_option = choose_split_option(brief)
    feature_option = choose_feature_option(profile)
    training_option = "thorough_cv" if profile.rows > 2_000 else "standard_cv"
    hpo_option = "small_competition" if brief.competition_enabled else "skip"

    return [
        WorkflowRecommendation(
            step_id="00_intake",
            step_number=step_map["00_intake"].step_number,
            title=step_map["00_intake"].title,
            recommendation="Ask the human for the dataset path, target column, task type, and what success means.",
            reasoning=[
                "This module is AI-first, so the human only needs to explain the data and the problem.",
                "All downstream recommendations should come from the structured brief plus observed data.",
                memory_hint(memories.get("00_intake", "")),
            ],
            options=["structured_brief"],
            selected_option="structured_brief",
        ),
        WorkflowRecommendation(
            step_id="01_preprocessing",
            step_number=step_map["01_preprocessing"].step_number,
            title=step_map["01_preprocessing"].title,
            recommendation=(
                "Use automatic tabular preprocessing: median imputation for numeric features, "
                "most-frequent imputation for categoricals, one-hot encoding, and scaling for linear models."
            ),
            reasoning=[
                f"The dataset has {len(profile.numeric_features)} numeric and {len(profile.categorical_features)} categorical features.",
                f"Missing feature values account for {profile.missing_fraction:.1%} of the table.",
                memory_hint(memories.get("01_preprocessing", "")),
            ],
            options=["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
            selected_option="auto_tabular_preprocessing",
        ),
        WorkflowRecommendation(
            step_id="02_data_splitting",
            step_number=step_map["02_data_splitting"].step_number,
            title=step_map["02_data_splitting"].title,
            recommendation=split_recommendation_text(split_option, brief.task_type),
            reasoning=[
                f"The dataset contains {profile.rows} rows, so we need a clean final holdout.",
                "Cross-validation should stay inside the training split to avoid leakage.",
                memory_hint(memories.get("02_data_splitting", "")),
            ],
            options=["stratified_holdout", "random_holdout", "time_ordered_holdout"],
            selected_option=split_option,
        ),
        WorkflowRecommendation(
            step_id="03_model_selection",
            step_number=step_map["03_model_selection"].step_number,
            title=step_map["03_model_selection"].title,
            recommendation=model_recommendation_text(brief.task_type, model_option),
            reasoning=[
                f"The problem is {brief.task_type}, so candidate models should reflect that.",
                f"Dataset scale and feature mix point to the '{model_option}' strategy.",
                memory_hint(memories.get("03_model_selection", "")),
            ],
            options=["balanced_candidate_set", "speed_first", "tree_ensemble_first"],
            selected_option=model_option,
        ),
        WorkflowRecommendation(
            step_id="04_feature_selection",
            step_number=step_map["04_feature_selection"].step_number,
            title=step_map["04_feature_selection"].title,
            recommendation=feature_recommendation_text(feature_option),
            reasoning=[
                f"The dataset exposes {profile.columns - 1} input features before encoding.",
                "Feature selection only helps if width is likely to dominate signal.",
                memory_hint(memories.get("04_feature_selection", "")),
            ],
            options=["auto", "skip", "select_k_best", "variance_threshold"],
            selected_option=feature_option,
        ),
        WorkflowRecommendation(
            step_id="05_metric_selection",
            step_number=step_map["05_metric_selection"].step_number,
            title=step_map["05_metric_selection"].title,
            recommendation=metric_recommendation_text(brief.task_type, metric_option),
            reasoning=[
                f"The workflow will optimize for '{metric_option}' and still log supporting metrics.",
                "The final winner and the baseline must be compared using the same primary metric.",
                memory_hint(memories.get("05_metric_selection", "")),
            ],
            options=metric_options_for_task(brief.task_type),
            selected_option=metric_option,
        ),
        WorkflowRecommendation(
            step_id="06_training_configuration",
            step_number=step_map["06_training_configuration"].step_number,
            title=step_map["06_training_configuration"].title,
            recommendation=training_recommendation_text(training_option),
            reasoning=[
                "The initial MVP should be reproducible and easy to inspect.",
                "Cross-validation depth should scale with iteration budget and data size.",
                memory_hint(memories.get("06_training_configuration", "")),
            ],
            options=["fast_cv", "standard_cv", "thorough_cv"],
            selected_option=training_option,
        ),
        WorkflowRecommendation(
            step_id="07_hyperparameter_optimization",
            step_number=step_map["07_hyperparameter_optimization"].step_number,
            title=step_map["07_hyperparameter_optimization"].title,
            recommendation=hpo_recommendation_text(hpo_option),
            reasoning=[
                "The competition stage should only spend extra budget on the strongest candidates.",
                "This keeps the workflow explainable while still allowing a winner-takes-all round.",
                memory_hint(memories.get("07_hyperparameter_optimization", "")),
            ],
            options=["skip", "small_competition", "expanded_competition"],
            selected_option=hpo_option,
        ),
        WorkflowRecommendation(
            step_id="08_validation_and_baseline",
            step_number=step_map["08_validation_and_baseline"].step_number,
            title=step_map["08_validation_and_baseline"].title,
            recommendation=(
                "Validate the winning model on the untouched test split, compare it against a dummy baseline, "
                "and export the full model bundle with decisions, metrics, and artifacts."
            ),
            reasoning=[
                "The operator should be able to see both the score and the decision path.",
                "The exported bundle should be reusable by future agent runs.",
                memory_hint(memories.get("08_validation_and_baseline", "")),
            ],
            options=["holdout_vs_baseline"],
            selected_option="holdout_vs_baseline",
        ),
    ]


def choose_split_option(brief: ProjectBrief) -> str:
    if brief.date_column:
        return "time_ordered_holdout"
    if brief.task_type == "classification":
        return "stratified_holdout"
    return "random_holdout"


def choose_feature_option(profile: DataProfile) -> str:
    return "auto" if (profile.columns - 1) >= 40 else "skip"


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


def memory_hint(content: str) -> str:
    lines = [line.strip("- ").strip() for line in content.splitlines() if line.startswith("- ")]
    if lines:
        return f"Memory signal: {lines[-1]}"
    return "Memory signal: no prior preference recorded yet."


def split_recommendation_text(option: str, task_type: str) -> str:
    if option == "time_ordered_holdout":
        return "Use a time-ordered holdout split so future rows stay in the validation window."
    if option == "stratified_holdout":
        return f"Use an 80/20 stratified holdout split for the {task_type} target."
    return "Use an 80/20 random holdout split with cross-validation only on the training partition."


def model_recommendation_text(task_type: str, option: str) -> str:
    if option == "speed_first":
        return f"Start with two fast {task_type} candidates and optimize the winner later."
    if option == "tree_ensemble_first":
        return f"Favor tree ensembles first, then keep a simple linear {task_type} baseline."
    return f"Use a balanced candidate set with linear, random-forest, and gradient-boosting {task_type} models."


def feature_recommendation_text(option: str) -> str:
    if option == "skip":
        return "Skip feature selection in the first pass and let the models consume the full encoded feature space."
    if option == "variance_threshold":
        return "Use a lightweight variance threshold to remove dead features after preprocessing."
    if option == "select_k_best":
        return "Use SelectKBest after preprocessing to keep the most predictive transformed features."
    return "Keep feature selection on automatic mode and only activate it if the encoded feature space becomes wide."


def metric_recommendation_text(task_type: str, metric_option: str) -> str:
    if task_type == "classification":
        return f"Optimize for {metric_option} while still reporting accuracy and F1 macro for context."
    return f"Optimize for {metric_option} while still reporting MAE and R2 as supporting metrics."


def training_recommendation_text(option: str) -> str:
    if option == "fast_cv":
        return "Use 3-fold cross-validation to keep iteration speed high."
    if option == "thorough_cv":
        return "Use 5-fold cross-validation with a slightly larger candidate sweep."
    return "Use a reproducible standard CV setup with 5 folds for medium/large data and 3 folds for small data."


def hpo_recommendation_text(option: str) -> str:
    if option == "skip":
        return "Skip the competition stage for now and move directly to validation with the best baseline candidate."
    if option == "expanded_competition":
        return "Run an expanded parameter competition on the top candidate families."
    return "Run a compact gym-style competition on the top two candidate families with small parameter grids."


def metric_options_for_task(task_type: str) -> list[str]:
    if task_type == "classification":
        return ["balanced_accuracy", "f1_macro", "accuracy"]
    return ["rmse", "mae", "r2"]
