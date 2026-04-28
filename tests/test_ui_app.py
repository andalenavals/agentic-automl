from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentic_automl.ui_app import (
    answer_contextual_step_question,
    answer_general_discussion_question,
    build_codex_discussion_activity,
    build_discussion_context,
    build_initial_step_feedback,
    build_preprocessing_policy_details,
    build_step_discussion_context_block,
    build_brief_from_fields,
    build_workflow_step_feedback_payload,
    extract_brief_fields,
    format_missing_fields_message,
    infer_execution_option,
    is_question,
    normalize_step_feedback_entry,
    parse_codex_exec_json_events,
    respond_to_discussion_message,
    respond_to_step_discussion,
    summarize_step_feedback,
    validate_step_feedback,
)


class UiAppParsingTest(unittest.TestCase):
    def test_extract_brief_fields_accepts_aliases(self) -> None:
        fields = extract_brief_fields(
            "\n".join(
                [
                    "dataset: examples/customer_churn_demo.csv",
                    "target: churned",
                    "task: classification",
                    "problem: predict churn next month",
                ]
            )
        )
        self.assertEqual(fields["dataset_path"], "examples/customer_churn_demo.csv")
        self.assertEqual(fields["target_column"], "churned")
        self.assertEqual(fields["task_type"], "classification")
        self.assertEqual(fields["problem_description"], "predict churn next month")

    def test_extract_brief_fields_handles_natural_language(self) -> None:
        fields = extract_brief_fields(
            "My dataset is examples/customer_churn_demo.csv and the target is churned. "
            "This is a classification problem and I want to predict churn next month."
        )
        self.assertEqual(fields["dataset_path"], "examples/customer_churn_demo.csv")
        self.assertEqual(fields["target_column"], "churned")
        self.assertEqual(fields["task_type"], "classification")
        self.assertIn("predict churn next month", fields["problem_description"].lower())

    def test_build_brief_from_fields_defaults_project_name_from_dataset(self) -> None:
        brief, missing = build_brief_from_fields(
            {
                "dataset_path": "examples/customer_churn_demo.csv",
                "target_column": "churned",
                "task_type": "classification",
                "problem_description": "Predict churn next month.",
            }
        )
        self.assertEqual(missing, [])
        self.assertIsNotNone(brief)
        assert brief is not None
        self.assertEqual(brief.project_name, "customer-churn-demo")

    def test_format_missing_fields_message_is_incremental(self) -> None:
        message = format_missing_fields_message(
            {"dataset_path": "/tmp/data.csv", "target_column": "label"},
            ["task_type", "problem_description"],
        )
        self.assertIn("I already captured", message)
        self.assertIn("`/tmp/data.csv`", message)
        self.assertIn("- task_type", message)
        self.assertIn("- problem_description", message)

    def test_is_question_detects_short_follow_up(self) -> None:
        self.assertTrue(is_question("What do I do now?"))
        self.assertFalse(is_question("target_column: churned"))

    def test_step_feedback_starts_as_agreed(self) -> None:
        recommendations = [
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
            },
            {"step_id": "02_data_splitting", "title": "Data splitting", "selected_option": "stratified_holdout"},
        ]
        feedback = build_initial_step_feedback(recommendations)
        self.assertEqual(feedback["01_preprocessing"]["agreement"], "agree")
        self.assertTrue(feedback["01_preprocessing"]["policy_confirmed"])
        self.assertEqual(feedback["01_preprocessing"]["discussion_history"], [])
        self.assertEqual(feedback["01_preprocessing"]["action_history"], [])
        self.assertEqual(summarize_step_feedback(feedback, "02_data_splitting"), "agreed")

    def test_intake_feedback_starts_as_agreed_after_brief_load(self) -> None:
        feedback = build_initial_step_feedback(
            [
                {
                    "step_id": "00_intake",
                    "title": "Intake",
                    "selected_option": "structured_brief",
                }
            ]
        )
        self.assertEqual(feedback["00_intake"]["agreement"], "agree")
        self.assertTrue(feedback["00_intake"]["policy_confirmed"])
        self.assertEqual(summarize_step_feedback(feedback, "00_intake"), "agreed")

    def test_validate_step_feedback_rejects_pending_review_state(self) -> None:
        recommendations = [
            {"step_id": "01_preprocessing", "title": "Preprocessing", "selected_option": "auto_tabular_preprocessing"}
        ]
        issues = validate_step_feedback(
            recommendations,
            {"01_preprocessing": {"agreement": "pending", "custom_note": ""}},
            {"01_preprocessing": "auto_tabular_preprocessing"},
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("agree", issues[0].lower())

    def test_normalize_step_feedback_converts_pending_to_agree(self) -> None:
        recommendation = {
            "step_id": "01_preprocessing",
            "title": "Preprocessing",
            "selected_option": "auto_tabular_preprocessing",
        }
        feedback = normalize_step_feedback_entry({"agreement": "pending"}, recommendation)
        self.assertEqual(feedback["agreement"], "agree")
        self.assertTrue(feedback["policy_confirmed"])

    def test_validate_step_feedback_accepts_customized_step_with_note(self) -> None:
        recommendations = [
            {"step_id": "02_data_splitting", "title": "Data splitting", "selected_option": "stratified_holdout"}
        ]
        issues = validate_step_feedback(
            recommendations,
            {"02_data_splitting": {"agreement": "different", "custom_note": "Use a time split.", "policy_confirmed": True}},
            {"02_data_splitting": "stratified_holdout"},
        )
        self.assertEqual(issues, [])

    def test_validate_step_feedback_accepts_generated_policy_summary(self) -> None:
        recommendations = [
            {"step_id": "03_model_selection", "title": "Model selection", "selected_option": "logistic_regression"}
        ]
        issues = validate_step_feedback(
            recommendations,
            {
                "03_model_selection": {
                    "agreement": "different",
                    "custom_note": "",
                    "policy_summary": "Use `random_forest_classifier` as the selected model with stronger tree capacity.",
                    "policy_confirmed": True,
                }
            },
            {"03_model_selection": "random_forest_classifier"},
        )
        self.assertEqual(issues, [])

    def test_validate_step_feedback_requires_confirmation_for_custom_policy(self) -> None:
        recommendations = [
            {"step_id": "03_model_selection", "title": "Model selection", "selected_option": "logistic_regression"}
        ]
        issues = validate_step_feedback(
            recommendations,
            {
                "03_model_selection": {
                    "agreement": "different",
                    "custom_note": "Use a tree model instead.",
                    "policy_summary": "Use `random_forest_classifier` as the selected model with stronger tree capacity.",
                    "policy_confirmed": False,
                }
            },
            {"03_model_selection": "random_forest_classifier"},
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("confirm", issues[0].lower())

    def test_build_workflow_step_feedback_payload_keeps_selected_and_default(self) -> None:
        recommendations = [
            {"step_id": "05_metric_selection", "title": "Metric selection", "selected_option": "f1_macro"}
        ]
        payload = build_workflow_step_feedback_payload(
            recommendations,
            {
                "05_metric_selection": {
                    "agreement": "different",
                    "custom_note": "Prefer recall for the positive class.",
                    "policy_summary": "Use balanced accuracy for a more stable class-sensitive winner rule.",
                    "policy_confirmed": True,
                }
            },
            {"05_metric_selection": "balanced_accuracy"},
        )
        self.assertEqual(payload["05_metric_selection"]["agreement"], "different")
        self.assertEqual(payload["05_metric_selection"]["default_option"], "f1_macro")
        self.assertEqual(payload["05_metric_selection"]["selected_option"], "balanced_accuracy")
        self.assertEqual(payload["05_metric_selection"]["policy_summary"], "Use balanced accuracy for a more stable class-sensitive winner rule.")
        self.assertEqual(payload["05_metric_selection"]["policy_metadata"], {})
        self.assertTrue(payload["05_metric_selection"]["policy_confirmed"])

    def test_infer_execution_option_maps_time_split_language(self) -> None:
        selected = infer_execution_option(
            "02_data_splitting",
            ["stratified_holdout", "random_holdout", "time_ordered_holdout"],
            "Please keep time order because future rows must stay in validation.",
            "stratified_holdout",
        )
        self.assertEqual(selected, "time_ordered_holdout")

    def test_respond_to_step_discussion_builds_policy(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "03_model_selection",
                "title": "Model selection",
                "selected_option": "logistic_regression",
                "options": [
                    "logistic_regression",
                    "random_forest_classifier",
                    "hist_gradient_boosting_classifier",
                    "mlp_classifier",
                ],
            },
            "Use random forest classifier with n_estimators 500 and max_depth 12.",
            current_option="logistic_regression",
        )
        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "random_forest_classifier")
        self.assertIn("random_forest_classifier", response["policy_summary"])
        self.assertEqual(response["policy_metadata"]["model_parameters"]["n_estimators"], 500)
        self.assertEqual(response["policy_metadata"]["model_parameters"]["max_depth"], 12)

    def test_respond_to_step_discussion_rejects_model_override_not_supported_by_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            skill_dir = repo_root / "src" / "agentic_automl" / "assets" / "skills" / "03-model-selection"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "KNOWLEDGE.md").write_text(
                "\n".join(
                    [
                        "# Model Selection Knowledge",
                        "",
                        "## Capability Keys",
                        "",
                        "- `model_logistic_regression`: use logistic regression",
                        "- `model_override_c`: override C",
                        "- `model_override_class_weight`: override class_weight",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"AGENTIC_AUTOML_HOME": str(repo_root)}):
                response = respond_to_step_discussion(
                    {
                        "step_id": "03_model_selection",
                        "title": "Model selection",
                        "selected_option": "logistic_regression",
                        "options": [
                            "logistic_regression",
                            "random_forest_classifier",
                            "hist_gradient_boosting_classifier",
                            "mlp_classifier",
                        ],
                    },
                    "Use logistic regression with n_estimators 500.",
                    current_option="logistic_regression",
                )

            self.assertFalse(response["updated_policy"])
            self.assertIn("not an exposed initial parameter", response["reply"])
            limits_path = skill_dir / "LIMITS.md"
            self.assertTrue(limits_path.exists())
            self.assertIn("Use logistic regression with n_estimators 500.", limits_path.read_text(encoding="utf-8"))

    def test_respond_to_step_discussion_supports_logistic_regression_c_override(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "03_model_selection",
                "title": "Model selection",
                "selected_option": "logistic_regression",
                "options": [
                    "logistic_regression",
                    "random_forest_classifier",
                    "hist_gradient_boosting_classifier",
                    "mlp_classifier",
                ],
            },
            "Use logistic regression with C 0.5 and class_weight balanced.",
            current_option="logistic_regression",
        )
        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "logistic_regression")
        self.assertEqual(response["policy_metadata"]["model_parameters"]["C"], 0.5)
        self.assertEqual(response["policy_metadata"]["model_parameters"]["class_weight"], "balanced")

    def test_model_action_supports_multiple_policy_revisions(self) -> None:
        recommendation = {
            "step_id": "03_model_selection",
            "title": "Model selection",
            "selected_option": "logistic_regression",
            "options": [
                "logistic_regression",
                "random_forest_classifier",
                "hist_gradient_boosting_classifier",
                "mlp_classifier",
            ],
        }
        first_response = respond_to_step_discussion(
            recommendation,
            "Use random forest classifier with n_estimators 500 and max_depth 12.",
            current_option="logistic_regression",
        )
        second_response = respond_to_step_discussion(
            recommendation,
            "Actually switch to logistic regression with C 0.5 and class_weight balanced.",
            current_option=first_response["selected_option"],
            current_policy_metadata=first_response["policy_metadata"],
        )
        self.assertTrue(second_response["updated_policy"])
        self.assertEqual(second_response["selected_option"], "logistic_regression")
        self.assertEqual(
            second_response["policy_metadata"]["model_parameters"],
            {"C": 0.5, "class_weight": "balanced"},
        )

    def test_respond_to_step_discussion_rejects_training_override_not_supported_by_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            skill_dir = repo_root / "src" / "agentic_automl" / "assets" / "skills" / "06-training-configuration"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "KNOWLEDGE.md").write_text(
                "\n".join(
                    [
                        "# Training Configuration Knowledge",
                        "",
                        "## Capability Keys",
                        "",
                        "- `training_standard`: standard training",
                        "- `training_override_optimizer`: override optimizer",
                        "- `training_override_epochs`: override epochs",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"AGENTIC_AUTOML_HOME": str(repo_root)}):
                response = respond_to_step_discussion(
                    {
                        "step_id": "06_training_configuration",
                        "title": "Training configuration",
                        "selected_option": "standard_training",
                        "options": ["fast_training", "standard_training", "thorough_training"],
                    },
                    "Set optimizer to adam and epochs to 100.",
                    current_option="standard_training",
                    selected_options={"03_model_selection": "random_forest_classifier"},
                )

            self.assertFalse(response["updated_policy"])
            self.assertIn("not an exposed training parameter", response["reply"])
            limits_path = skill_dir / "LIMITS.md"
            self.assertTrue(limits_path.exists())
            self.assertIn("Set optimizer to adam and epochs to 100.", limits_path.read_text(encoding="utf-8"))

    def test_respond_to_step_discussion_supports_mlp_training_override(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "06_training_configuration",
                "title": "Training configuration",
                "selected_option": "standard_training",
                "options": ["fast_training", "standard_training", "thorough_training"],
            },
            "Set optimizer to sgd, learning rate to 0.01, epochs to 300, and mini batch to 64.",
            current_option="standard_training",
            selected_options={"03_model_selection": "mlp_classifier"},
        )
        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "standard_training")
        self.assertEqual(response["policy_metadata"]["training_config"]["optimizer"], "sgd")
        self.assertEqual(response["policy_metadata"]["training_config"]["learning_rate"], 0.01)
        self.assertEqual(response["policy_metadata"]["training_config"]["epochs"], 300)
        self.assertEqual(response["policy_metadata"]["training_config"]["mini_batch"], 64)

    def test_training_action_supports_multiple_policy_revisions(self) -> None:
        recommendation = {
            "step_id": "06_training_configuration",
            "title": "Training configuration",
            "selected_option": "standard_training",
            "options": ["fast_training", "standard_training", "thorough_training"],
        }
        first_response = respond_to_step_discussion(
            recommendation,
            "Set optimizer to sgd and learning rate to 0.01.",
            current_option="standard_training",
            selected_options={"03_model_selection": "mlp_classifier"},
        )
        second_response = respond_to_step_discussion(
            recommendation,
            "Also use epochs 300 and mini batch 64.",
            current_option=first_response["selected_option"],
            selected_options={"03_model_selection": "mlp_classifier"},
            current_policy_metadata=first_response["policy_metadata"],
        )
        self.assertTrue(second_response["updated_policy"])
        self.assertEqual(second_response["selected_option"], "standard_training")
        self.assertEqual(second_response["policy_metadata"]["training_config"]["optimizer"], "sgd")
        self.assertEqual(second_response["policy_metadata"]["training_config"]["learning_rate"], 0.01)
        self.assertEqual(second_response["policy_metadata"]["training_config"]["epochs"], 300)
        self.assertEqual(second_response["policy_metadata"]["training_config"]["mini_batch"], 64)

    def test_respond_to_step_discussion_builds_preprocessing_override(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "minimal_cleanup",
                "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
            },
            "I want to treat support_tickets as a categorical feature.",
            current_option="minimal_cleanup",
            profile={
                "numeric_features": ["support_tickets", "tenure_months"],
                "categorical_features": ["region"],
            },
        )
        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "custom")
        self.assertIn("support_tickets", response["policy_summary"])
        self.assertIn("actual preprocessing graph", response["reply"])
        self.assertEqual(
            response["policy_metadata"]["preprocessing_overrides"]["force_categorical_features"],
            ["support_tickets"],
        )

    def test_respond_to_step_discussion_updates_data_split_policy(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "02_data_splitting",
                "title": "Data splitting",
                "selected_option": "random_holdout",
                "options": ["stratified_holdout", "random_holdout", "time_ordered_holdout"],
            },
            "Please use a time ordered holdout because chronology matters.",
            current_option="random_holdout",
        )
        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "time_ordered_holdout")
        self.assertIn("time_ordered_holdout", response["policy_summary"])

    def test_data_splitting_action_supports_multiple_policy_revisions(self) -> None:
        recommendation = {
            "step_id": "02_data_splitting",
            "title": "Data splitting",
            "selected_option": "random_holdout",
            "options": ["stratified_holdout", "random_holdout", "time_ordered_holdout"],
        }
        first_response = respond_to_step_discussion(
            recommendation,
            "Please use a time ordered holdout because chronology matters.",
            current_option="random_holdout",
        )
        second_response = respond_to_step_discussion(
            recommendation,
            "Actually keep class balance and use a stratified holdout.",
            current_option=first_response["selected_option"],
        )
        self.assertTrue(second_response["updated_policy"])
        self.assertEqual(second_response["selected_option"], "stratified_holdout")
        self.assertIn("stratified_holdout", second_response["policy_summary"])

    def test_respond_to_step_discussion_rejects_unsupported_group_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            skill_dir = repo_root / "src" / "agentic_automl" / "assets" / "skills" / "02-data-splitting"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "KNOWLEDGE.md").write_text(
                "\n".join(
                    [
                        "# Data Splitting Knowledge",
                        "",
                        "## Capability Keys",
                        "",
                        "- `split_stratified_holdout`: stratified holdout",
                        "- `split_random_holdout`: random holdout",
                        "- `split_time_ordered_holdout`: time ordered holdout",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"AGENTIC_AUTOML_HOME": str(repo_root)}):
                response = respond_to_step_discussion(
                    {
                        "step_id": "02_data_splitting",
                        "title": "Data splitting",
                        "selected_option": "random_holdout",
                        "options": ["stratified_holdout", "random_holdout", "time_ordered_holdout"],
                    },
                    "Use a group split by customer_id.",
                    current_option="random_holdout",
                )

            self.assertFalse(response["updated_policy"])
            self.assertIn("group-aware final holdout strategies are not supported yet", response["reply"])
            limits_path = skill_dir / "LIMITS.md"
            self.assertTrue(limits_path.exists())
            self.assertIn("Use a group split by customer_id.", limits_path.read_text(encoding="utf-8"))

    def test_respond_to_step_discussion_updates_metric_policy(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "05_metric_selection",
                "title": "Metric selection",
                "selected_option": "accuracy",
                "options": ["balanced_accuracy", "f1_macro", "accuracy"],
            },
            "Use balanced accuracy as the winner metric.",
            current_option="accuracy",
        )
        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "balanced_accuracy")
        self.assertIn("balanced_accuracy", response["policy_summary"])

    def test_metric_action_supports_multiple_policy_revisions(self) -> None:
        recommendation = {
            "step_id": "05_metric_selection",
            "title": "Metric selection",
            "selected_option": "accuracy",
            "options": ["balanced_accuracy", "f1_macro", "accuracy"],
        }
        first_response = respond_to_step_discussion(
            recommendation,
            "Use balanced accuracy as the winner metric.",
            current_option="accuracy",
        )
        second_response = respond_to_step_discussion(
            recommendation,
            "Actually use f1 macro.",
            current_option=first_response["selected_option"],
        )
        self.assertTrue(second_response["updated_policy"])
        self.assertEqual(second_response["selected_option"], "f1_macro")
        self.assertIn("f1_macro", second_response["policy_summary"])

    def test_respond_to_step_discussion_rejects_unsupported_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            skill_dir = repo_root / "src" / "agentic_automl" / "assets" / "skills" / "05-metric-selection"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "KNOWLEDGE.md").write_text(
                "\n".join(
                    [
                        "# Metric Selection Knowledge",
                        "",
                        "## Capability Keys",
                        "",
                        "- `metric_balanced_accuracy`: use balanced accuracy",
                        "- `metric_f1_macro`: use f1 macro",
                        "- `metric_accuracy`: use accuracy",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"AGENTIC_AUTOML_HOME": str(repo_root)}):
                response = respond_to_step_discussion(
                    {
                        "step_id": "05_metric_selection",
                        "title": "Metric selection",
                        "selected_option": "accuracy",
                        "options": ["balanced_accuracy", "f1_macro", "accuracy"],
                    },
                    "Use AUC as the winner metric.",
                    current_option="accuracy",
                )

            self.assertFalse(response["updated_policy"])
            self.assertIn("not packaged yet", response["reply"])
            limits_path = skill_dir / "LIMITS.md"
            self.assertTrue(limits_path.exists())
            self.assertIn("Use AUC as the winner metric.", limits_path.read_text(encoding="utf-8"))

    def test_respond_to_step_discussion_rejects_validation_baseline_removal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            skill_dir = repo_root / "src" / "agentic_automl" / "assets" / "skills" / "08-validation-and-baseline"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "KNOWLEDGE.md").write_text(
                "\n".join(
                    [
                        "# Validation And Baseline Knowledge",
                        "",
                        "## Capability Keys",
                        "",
                        "- `validation_test_set_with_baseline`: keep validation with baseline",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"AGENTIC_AUTOML_HOME": str(repo_root)}):
                response = respond_to_step_discussion(
                    {
                        "step_id": "08_validation_and_baseline",
                        "title": "Validation and baseline",
                        "selected_option": "test_set_with_baseline",
                        "options": ["test_set_with_baseline"],
                    },
                    "Validate on the test set but skip the baseline.",
                    current_option="test_set_with_baseline",
                )

            self.assertFalse(response["updated_policy"])
            self.assertIn("removing the baseline comparison is not supported", response["reply"])
            limits_path = skill_dir / "LIMITS.md"
            self.assertTrue(limits_path.exists())
            self.assertIn("Validate on the test set but skip the baseline.", limits_path.read_text(encoding="utf-8"))

    def test_respond_to_step_discussion_rejects_final_validation_dashboard_removal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            skill_dir = repo_root / "src" / "agentic_automl" / "assets" / "skills" / "09-final-validation"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "KNOWLEDGE.md").write_text(
                "\n".join(
                    [
                        "# Final Validation Knowledge",
                        "",
                        "## Capability Keys",
                        "",
                        "- `final_validation_dashboard`: keep the final dashboard flow",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"AGENTIC_AUTOML_HOME": str(repo_root)}):
                response = respond_to_step_discussion(
                    {
                        "step_id": "09_final_validation",
                        "title": "Final validation",
                        "selected_option": "final_validation_dashboard",
                        "options": ["final_validation_dashboard"],
                    },
                    "Run final validation but omit the optimization summary.",
                    current_option="final_validation_dashboard",
                )

            self.assertFalse(response["updated_policy"])
            self.assertIn("optimization summary", response["reply"])
            limits_path = skill_dir / "LIMITS.md"
            self.assertTrue(limits_path.exists())
            self.assertIn("Run final validation but omit the optimization summary.", limits_path.read_text(encoding="utf-8"))

    def test_respond_to_step_discussion_understands_one_hot_request_as_custom_override(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
                "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
            },
            "I want to apply one hot encoding to the feature support_tickets.",
            current_option="auto_tabular_preprocessing",
            profile={
                "numeric_features": ["support_tickets", "tenure_months"],
                "categorical_features": ["region"],
                "distinct_by_feature": {"support_tickets": 6, "region": 4},
            },
        )
        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "custom")
        self.assertIn("One-hot encode `support_tickets`", response["policy_summary"])
        self.assertEqual(
            response["policy_metadata"]["preprocessing_overrides"]["force_one_hot_features"],
            ["support_tickets"],
        )
        self.assertEqual(
            response["policy_metadata"]["preprocessing_overrides"]["force_categorical_features"],
            ["support_tickets"],
        )

    def test_respond_to_step_discussion_understands_numeric_only_scope(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
            },
            "I want to use only the numerical features.",
            current_option="custom",
            profile={
                "numeric_features": ["age", "monthly_spend", "support_tickets", "tenure_months"],
                "categorical_features": ["customer_id", "contract_type", "region"],
                "date_like_features": [],
            },
        )
        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "custom")
        self.assertIn("only the numeric input features", response["policy_summary"])
        self.assertIn("customer_id", response["policy_metadata"]["preprocessing_overrides"]["drop_features"])
        self.assertIn("contract_type", response["policy_metadata"]["preprocessing_overrides"]["drop_features"])
        self.assertIn("region", response["policy_metadata"]["preprocessing_overrides"]["drop_features"])

    def test_respond_to_step_discussion_understands_specific_feature_subset(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
            },
            "I want to use only two numerico features: age and tenure_months",
            current_option="custom",
            profile={
                "numeric_features": ["age", "monthly_spend", "support_tickets", "tenure_months"],
                "categorical_features": ["customer_id", "contract_type", "region"],
                "date_like_features": [],
            },
        )
        overrides = response["policy_metadata"]["preprocessing_overrides"]
        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "custom")
        self.assertIn("selected input feature subset", response["policy_summary"])
        self.assertEqual(overrides["keep_features"], ["age", "tenure_months"])
        self.assertEqual(
            set(overrides["drop_features"]),
            {"customer_id", "contract_type", "region", "monthly_spend", "support_tickets"},
        )

    def test_respond_to_step_discussion_supports_feature_specific_mean_imputation(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
            },
            "For age apply mean imputation.",
            current_option="custom",
            profile={
                "numeric_features": ["age", "tenure_months"],
                "categorical_features": ["region"],
                "date_like_features": [],
            },
        )
        self.assertTrue(response["updated_policy"])
        overrides = response["policy_metadata"]["preprocessing_overrides"]
        self.assertEqual(overrides["feature_imputation_rules"][0]["feature"], "age")
        self.assertEqual(overrides["feature_imputation_rules"][0]["strategy"], "mean")
        self.assertIn("Impute `age` with `mean`", response["policy_summary"])

    def test_respond_to_step_discussion_rejects_supported_code_path_when_knowledge_disallows_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            skill_dir = repo_root / "src" / "agentic_automl" / "assets" / "skills" / "01-preprocessing"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "KNOWLEDGE.md").write_text(
                "\n".join(
                    [
                        "# Preprocessing Knowledge",
                        "",
                        "## Capability Keys",
                        "",
                        "- `scope_drop_named_features`: drop named features",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"AGENTIC_AUTOML_HOME": str(repo_root)}):
                response = respond_to_step_discussion(
                    {
                        "step_id": "01_preprocessing",
                        "title": "Preprocessing",
                        "selected_option": "custom",
                        "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
                    },
                    "For age apply mean imputation.",
                    current_option="custom",
                    profile={
                        "numeric_features": ["age", "tenure_months"],
                        "categorical_features": ["region"],
                        "date_like_features": [],
                    },
                )

            self.assertFalse(response["updated_policy"])
            self.assertIn("outside the executable action space", response["reply"])
            limits_path = skill_dir / "LIMITS.md"
            self.assertTrue(limits_path.exists())
            self.assertIn("For age apply mean imputation.", limits_path.read_text(encoding="utf-8"))

    def test_respond_to_step_discussion_supports_frequency_count_column(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
            },
            "Add a new column that results from the frequency counts of region.",
            current_option="custom",
            profile={
                "numeric_features": ["age"],
                "categorical_features": ["region"],
                "date_like_features": [],
            },
        )
        self.assertTrue(response["updated_policy"])
        overrides = response["policy_metadata"]["preprocessing_overrides"]
        self.assertEqual(overrides["derived_feature_rules"][0]["source"], "region")
        self.assertEqual(overrides["derived_feature_rules"][0]["output"], "region__frequency_count")

    def test_respond_to_step_discussion_stores_unsupported_preprocessing_action_in_todo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            skill_dir = repo_root / "src" / "agentic_automl" / "assets" / "skills" / "01-preprocessing"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "KNOWLEDGE.md").write_text("# Preprocessing Knowledge\n", encoding="utf-8")

            with patch.dict(os.environ, {"AGENTIC_AUTOML_HOME": str(repo_root)}):
                response = respond_to_step_discussion(
                    {
                        "step_id": "01_preprocessing",
                        "title": "Preprocessing",
                        "selected_option": "custom",
                        "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
                    },
                    "Apply winsorization to monthly_spend.",
                    current_option="custom",
                    profile={
                        "numeric_features": ["monthly_spend", "age"],
                        "categorical_features": ["region"],
                        "date_like_features": [],
                    },
                )

            self.assertFalse(response["updated_policy"])
            self.assertIn("Sorry I can not perform this task yet", response["reply"])
            limits_path = skill_dir / "LIMITS.md"
            self.assertTrue(limits_path.exists())
            self.assertIn("Apply winsorization to monthly_spend.", limits_path.read_text(encoding="utf-8"))

    def test_respond_to_step_discussion_stores_unsupported_multi_feature_derivation_in_todo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            skill_dir = repo_root / "src" / "agentic_automl" / "assets" / "skills" / "01-preprocessing"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "KNOWLEDGE.md").write_text("# Preprocessing Knowledge\n", encoding="utf-8")

            request = (
                "Extend the feature set with a feature called merge, "
                "which is equal to the operation age*tenure_months."
            )
            with patch.dict(os.environ, {"AGENTIC_AUTOML_HOME": str(repo_root)}):
                response = respond_to_step_discussion(
                    {
                        "step_id": "01_preprocessing",
                        "title": "Preprocessing",
                        "selected_option": "custom",
                        "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
                    },
                    request,
                    current_option="custom",
                    profile={
                        "numeric_features": ["age", "tenure_months", "monthly_spend"],
                        "categorical_features": ["region"],
                        "date_like_features": [],
                    },
                )

            self.assertFalse(response["updated_policy"])
            self.assertIn("Sorry I can not perform this task yet", response["reply"])
            limits_path = skill_dir / "LIMITS.md"
            self.assertTrue(limits_path.exists())
            self.assertIn(request, limits_path.read_text(encoding="utf-8"))

    def test_respond_to_step_discussion_merges_preprocessing_changes_across_turns(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
            },
            "I want to apply one hot encoding to the feature support_tickets.",
            current_option="custom",
            current_policy_metadata={
                "preprocessing_overrides": {
                    "drop_features": ["customer_id", "contract_type", "region"]
                }
            },
            profile={
                "numeric_features": ["age", "monthly_spend", "support_tickets", "tenure_months"],
                "categorical_features": ["customer_id", "contract_type", "region"],
                "date_like_features": [],
                "distinct_by_feature": {
                    "support_tickets": 6,
                    "customer_id": 24,
                    "contract_type": 3,
                    "region": 4,
                },
            },
        )
        overrides = response["policy_metadata"]["preprocessing_overrides"]
        self.assertEqual(response["selected_option"], "custom")
        self.assertIn("customer_id", overrides["drop_features"])
        self.assertIn("contract_type", overrides["drop_features"])
        self.assertIn("region", overrides["drop_features"])
        self.assertIn("support_tickets", overrides["force_one_hot_features"])
        self.assertIn("support_tickets", overrides["force_categorical_features"])
        self.assertIn("One-hot encode `support_tickets`", response["policy_summary"])

    def test_respond_to_step_discussion_rejects_unsupported_follow_up_without_overwriting_current_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            skill_dir = repo_root / "src" / "agentic_automl" / "assets" / "skills" / "01-preprocessing"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "KNOWLEDGE.md").write_text(
                "\n".join(
                    [
                        "# Preprocessing Knowledge",
                        "",
                        "## Capability Keys",
                        "",
                        "- `scope_keep_subset`: keep only a named subset of features",
                        "- `encoding_one_hot`: one-hot encode a named feature",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            profile = {
                "numeric_features": ["age", "tenure_months", "monthly_spend", "support_tickets"],
                "categorical_features": ["customer_id", "contract_type", "region"],
                "date_like_features": [],
                "distinct_by_feature": {
                    "region": 4,
                    "contract_type": 3,
                    "customer_id": 24,
                },
            }

            with patch.dict(os.environ, {"AGENTIC_AUTOML_HOME": str(repo_root)}):
                first_response = respond_to_step_discussion(
                    {
                        "step_id": "01_preprocessing",
                        "title": "Preprocessing",
                        "selected_option": "custom",
                        "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
                    },
                    "I want to use only the region as a feature.",
                    current_option="custom",
                    profile=profile,
                )
                second_response = respond_to_step_discussion(
                    {
                        "step_id": "01_preprocessing",
                        "title": "Preprocessing",
                        "selected_option": "custom",
                        "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
                    },
                    "I wanna use Frequency encoding instead of one-hot encoding.",
                    current_option="custom",
                    current_policy_metadata=first_response["policy_metadata"],
                    profile=profile,
                )

            self.assertTrue(first_response["updated_policy"])
            self.assertFalse(second_response["updated_policy"])
            self.assertIsNone(second_response["selected_option"])
            self.assertIn("frequency encoding is not supported yet", second_response["reply"].lower())
            self.assertIn("current working policy is unchanged", second_response["reply"].lower())
            limits_path = skill_dir / "LIMITS.md"
            self.assertTrue(limits_path.exists())
            self.assertIn(
                "I wanna use Frequency encoding instead of one-hot encoding.",
                limits_path.read_text(encoding="utf-8"),
            )

    def test_normalize_step_feedback_entry_migrates_legacy_discussion(self) -> None:
        feedback = normalize_step_feedback_entry(
            {"discussion": [{"role": "user", "content": "legacy"}]},
            {"step_id": "01_preprocessing"},
        )
        self.assertEqual(feedback["action_history"][0]["content"], "legacy")
        self.assertEqual(feedback["discussion_history"], [])
        self.assertEqual(feedback["policy_metadata"], {})
        self.assertFalse(feedback["policy_confirmed"])

    def test_answer_contextual_step_question_handles_imputation(self) -> None:
        reply = answer_contextual_step_question(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
                "reasoning": [
                    "The dataset has 3 numeric and 2 categorical features.",
                    "Missing feature values account for 8.0% of the table.",
                ],
            },
            "Does the dataset need imputation?",
            profile={
                "missing_fraction": 0.08,
                "numeric_features": ["a", "b", "c"],
                "categorical_features": ["x", "y"],
            },
        )
        assert reply is not None
        self.assertIn("needs imputation", reply.lower())
        self.assertIn("8.0%", reply)

    def test_answer_contextual_step_question_limits_imputation_to_present_gap_types(self) -> None:
        reply = answer_contextual_step_question(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
                "reasoning": [],
            },
            "Does the dataset need imputation?",
            profile={
                "missing_fraction": 0.10,
                "numeric_features": ["age", "income"],
                "categorical_features": ["segment", "region"],
                "missing_by_feature": {"age": 0.10},
                "date_like_features": [],
                "date_parse_failure_by_feature": {},
            },
        )
        assert reply is not None
        self.assertIn("median imputation", reply)
        self.assertNotIn("most-frequent imputation", reply)

    def test_answer_contextual_step_question_handles_scaling_method(self) -> None:
        reply = answer_contextual_step_question(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
                "reasoning": [],
            },
            "Which scaling method are you planning to use?",
            profile={
                "missing_fraction": 0.0,
                "numeric_features": ["a", "b", "c"],
                "categorical_features": ["segment"],
                "date_like_features": [],
                "date_parse_failure_by_feature": {},
            },
        )
        assert reply is not None
        self.assertIn("StandardScaler", reply)
        self.assertIn("unit variance", reply)

    def test_answer_contextual_step_question_handles_categorical_cardinality(self) -> None:
        reply = answer_contextual_step_question(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
                "reasoning": [],
            },
            "How many categories are there in each of the two categorical features?",
            profile={
                "missing_fraction": 0.0,
                "numeric_features": ["a", "b", "c"],
                "categorical_features": ["segment", "region"],
                "categorical_cardinality": {"segment": 4, "region": 3},
            },
        )
        assert reply is not None
        self.assertIn("`segment`: `4` categories", reply)
        self.assertIn("`region`: `3` categories", reply)

    def test_answer_contextual_step_question_uses_knowledge_for_supported_encodings(self) -> None:
        reply = answer_contextual_step_question(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "reasoning": [
                    "The dataset has 3 numeric and 2 categorical features.",
                    "Missing feature values account for 0.0% of the table.",
                ],
            },
            "Which encodings are supported?",
            profile={
                "missing_fraction": 0.0,
                "numeric_features": ["a", "b", "c"],
                "categorical_features": ["segment", "region"],
            },
        )
        assert reply is not None
        self.assertIn("The encodings currently supported in preprocessing are", reply)
        self.assertIn("one-hot encode a named feature", reply)
        self.assertIn("ordinal encode a named feature", reply)
        self.assertNotIn("Yes, one-hot encoding is part of the default preprocessing path.", reply)

    def test_respond_to_step_discussion_answers_question_from_profile(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
                "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
                "reasoning": [
                    "The dataset has 3 numeric and 2 categorical features.",
                    "Missing feature values account for 8.0% of the table.",
                ],
            },
            "Does the dataset need imputation?",
            current_option="auto_tabular_preprocessing",
            profile={
                "missing_fraction": 0.08,
                "numeric_features": ["a", "b", "c"],
                "categorical_features": ["x", "y"],
            },
        )
        self.assertFalse(response["updated_policy"])
        self.assertIn("needs imputation", response["reply"].lower())
        self.assertIn("This is a consultation answer only", response["reply"])
        self.assertIn("Current action context:", response["reply"])
        self.assertIn("Useful next moves:", response["reply"])

    def test_respond_to_step_discussion_answers_supported_encodings_from_knowledge(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
                "reasoning": [
                    "The dataset has 3 numeric and 2 categorical features.",
                ],
            },
            "Which encodings are supported?",
            current_option="custom",
            profile={
                "numeric_features": ["a", "b", "c"],
                "categorical_features": ["segment", "region"],
            },
        )
        self.assertFalse(response["updated_policy"])
        self.assertIn("one-hot encode a named feature", response["reply"])
        self.assertIn("ordinal encode a named feature", response["reply"])
        self.assertNotIn("default preprocessing path", response["reply"])

    def test_respond_to_step_discussion_question_shows_active_hpo_context_and_suggestions(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "07_hyperparameter_optimization",
                "title": "Hyperparameter optimization",
                "selected_option": "skip",
                "options": ["skip", "small_competition", "expanded_competition"],
                "metadata": {"model_option": "random_forest_classifier"},
            },
            "Which hyperparameters are supported?",
            current_option="small_competition",
            selected_options={"03_model_selection": "random_forest_classifier"},
            current_policy_metadata={"hpo_config": {"search_parameters": ["max_depth", "class_weight"]}},
        )
        self.assertFalse(response["updated_policy"])
        self.assertIn("This is a consultation answer only", response["reply"])
        self.assertIn("Selected model context: `random_forest_classifier`", response["reply"])
        self.assertIn("Current search scope: `max_depth`, `class_weight`", response["reply"])
        self.assertIn("Useful next moves:", response["reply"])

    def test_respond_to_step_discussion_question_uses_current_active_policy_in_context(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "05_metric_selection",
                "title": "Metric selection",
                "selected_option": "accuracy",
                "options": ["balanced_accuracy", "f1_macro", "accuracy"],
                "reasoning": ["The current classification problem should prefer a balanced metric."],
            },
            "Why?",
            current_option="balanced_accuracy",
        )
        self.assertFalse(response["updated_policy"])
        self.assertIn("current active policy is `balanced_accuracy`", response["reply"])
        self.assertIn("Active policy: `balanced_accuracy`", response["reply"])

    def test_respond_to_step_discussion_answers_supported_encodings_without_question_mark(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
                "reasoning": [
                    "The dataset has 3 numeric and 2 categorical features.",
                ],
            },
            "Which encodings are supported",
            current_option="custom",
            profile={
                "numeric_features": ["a", "b", "c"],
                "categorical_features": ["segment", "region"],
            },
        )
        self.assertFalse(response["updated_policy"])
        self.assertIn("The encodings currently supported in preprocessing are", response["reply"])
        self.assertIn("one-hot encode a named feature", response["reply"])
        self.assertIn("ordinal encode a named feature", response["reply"])
        self.assertNotIn("default preprocessing path", response["reply"])

    def test_respond_to_step_discussion_answers_unsupported_encoding_question_from_knowledge(self) -> None:
        response = respond_to_step_discussion(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "options": ["auto_tabular_preprocessing", "minimal_cleanup", "custom"],
                "reasoning": [
                    "The dataset has 3 numeric and 2 categorical features.",
                ],
            },
            "Do you support frequency encoding?",
            current_option="custom",
            profile={
                "numeric_features": ["a", "b", "c"],
                "categorical_features": ["segment", "region"],
            },
        )
        self.assertFalse(response["updated_policy"])
        self.assertIn("frequency encoding is not part of the current preprocessing encoding support", response["reply"])
        self.assertIn("one-hot encode a named feature", response["reply"])
        self.assertIn("ordinal encode a named feature", response["reply"])
        self.assertNotIn("default preprocessing path", response["reply"])

    def test_respond_to_discussion_message_answers_stack_question(self) -> None:
        reply = respond_to_discussion_message(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
                "reasoning": [],
            },
            "Is TypeScript better than Streamlit for this kind of product?",
            allow_local_codex=False,
        )
        self.assertIn("TypeScript", reply)
        self.assertIn("Streamlit", reply)

    def test_respond_to_discussion_message_handles_general_question_without_workflow_fallback(self) -> None:
        reply = respond_to_discussion_message(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
                "reasoning": [
                    "The dataset has 5 numeric and 2 categorical features.",
                    "Missing feature values account for 0.0% of the table.",
                ],
            },
            "What is the capital of France?",
            allow_local_codex=False,
        )
        self.assertIn("Paris", reply)
        self.assertNotIn("auto_tabular_preprocessing", reply)

    def test_answer_general_discussion_question_handles_current_day(self) -> None:
        reply = answer_general_discussion_question("Which day is today?")
        assert reply is not None
        self.assertIn(datetime.now().astimezone().strftime("%A"), reply)

    def test_respond_to_discussion_message_can_use_intake_context(self) -> None:
        brief, _ = build_brief_from_fields(
            {
                "dataset_path": "examples/customer_churn_demo.csv",
                "target_column": "churned",
                "task_type": "classification",
                "problem_description": "Predict churn next month.",
            }
        )
        assert brief is not None
        reply = respond_to_discussion_message(
            {
                "step_id": "03_model_selection",
                "title": "Model selection",
                "selected_option": "logistic_regression",
                "reasoning": [],
            },
            "What is the current target?",
            brief=brief,
            allow_local_codex=False,
        )
        self.assertIn("`churned`", reply)

    def test_build_step_discussion_context_block_includes_policy_and_reasons(self) -> None:
        block = build_step_discussion_context_block(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
                "recommendation": "Use automatic preprocessing because the file is mostly clean.",
                "reasoning": [
                    "The dataset has 3 numeric and 2 categorical features.",
                    "Missing feature values account for 8.0% of the table.",
                ],
            },
            feedback={
                "agreement": "different",
                "policy_summary": "Keep the default policy but exclude customer_id.",
                "custom_note": "Exclude the identifier column before encoding.",
            },
            current_option="custom",
        )
        self.assertIn("default_execution_policy: auto_tabular_preprocessing", block)
        self.assertIn("current_execution_policy: custom", block)
        self.assertIn("policy_summary: Keep the default policy but exclude customer_id.", block)
        self.assertIn("- The dataset has 3 numeric and 2 categorical features.", block)

    def test_build_discussion_context_includes_previous_step_and_current_reasons(self) -> None:
        brief, _ = build_brief_from_fields(
            {
                "dataset_path": "examples/customer_churn_demo.csv",
                "target_column": "churned",
                "task_type": "classification",
                "problem_description": "Predict churn next month.",
            }
        )
        assert brief is not None
        recommendations = [
            {
                "step_id": "00_intake",
                "title": "Intake",
                "selected_option": "structured_brief",
                "recommendation": "Use the standard intake brief.",
                "reasoning": ["The workflow needs the dataset path, target, and problem framing."],
            },
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
                "recommendation": "Use automatic preprocessing because the file has mixed feature types.",
                "reasoning": [
                    "The dataset has 5 numeric and 2 categorical features.",
                    "Missing feature values account for 0.0% of the table.",
                ],
            },
        ]
        context = build_discussion_context(
            brief,
            recommendation=recommendations[1],
            recommendations=recommendations,
            step_feedback={
                "00_intake": {"agreement": "agree", "policy_summary": "", "custom_note": ""},
                "01_preprocessing": {"agreement": "agree", "policy_summary": "", "custom_note": ""},
            },
            selected_options={
                "00_intake": "structured_brief",
                "01_preprocessing": "auto_tabular_preprocessing",
            },
        )
        self.assertIn("project_name: customer-churn-demo", context)
        self.assertIn("Workflow context from previous steps:", context)
        self.assertIn("Intake:", context)
        self.assertIn("current_execution_policy: structured_brief", context)
        self.assertIn("Current workflow step context:", context)
        self.assertIn("default_execution_policy: auto_tabular_preprocessing", context)
        self.assertIn("- The dataset has 5 numeric and 2 categorical features.", context)

    def test_build_preprocessing_policy_details_exposes_execution_steps(self) -> None:
        details = build_preprocessing_policy_details(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "reasoning": [
                    "The file inspection found 2 numeric and 2 categorical features.",
                    "File-specific preprocessing risks were detected: likely IDs and date-like fields.",
                ],
            },
            {
                "rows": 50,
                "columns": 5,
                "numeric_features": ["age", "income"],
                "categorical_features": ["customer_id", "event_date"],
                "categorical_cardinality": {"customer_id": 50, "event_date": 30},
                "missing_by_feature": {},
                "constant_features": [],
                "likely_identifier_features": ["customer_id"],
                "date_like_features": ["event_date"],
                "date_parse_failure_by_feature": {},
                "high_cardinality_categorical_features": {},
                "missing_fraction": 0.0,
                "target_cardinality": 2,
                "target_name": "target",
                "target_skew": None,
                "class_imbalance": 1.2,
            },
        )
        self.assertEqual(details["selected_option"], "custom")
        self.assertTrue(any("Exclude identifier-like feature(s)" in entry["step"] for entry in details["steps"]))
        self.assertTrue(any("Parse the date-like fields" in entry["step"] for entry in details["steps"]))
        self.assertFalse(any("imputation" in entry["step"].lower() for entry in details["steps"]))
        self.assertTrue(any("`StandardScaler`" in entry["step"] for entry in details["steps"]))
        self.assertTrue(all(entry.get("why") for entry in details["steps"]))
        self.assertTrue(any("`customer_id` has `50` distinct values across `50` rows" in entry["why"] for entry in details["steps"]))

    def test_build_preprocessing_policy_details_skips_categorical_imputation_when_not_needed(self) -> None:
        details = build_preprocessing_policy_details(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "auto_tabular_preprocessing",
                "reasoning": [],
            },
            {
                "rows": 50,
                "columns": 5,
                "numeric_features": ["age", "income"],
                "categorical_features": ["region", "plan_type"],
                "categorical_cardinality": {"region": 4, "plan_type": 3},
                "missing_by_feature": {},
                "constant_features": [],
                "likely_identifier_features": [],
                "date_like_features": [],
                "date_parse_failure_by_feature": {},
                "high_cardinality_categorical_features": {},
                "missing_fraction": 0.0,
                "target_cardinality": 2,
                "target_name": "target",
                "target_skew": None,
                "class_imbalance": 1.2,
            },
        )
        self.assertFalse(any("most-frequent imputation" in entry["step"].lower() for entry in details["steps"]))
        self.assertTrue(any("one-hot encode" in entry["step"].lower() for entry in details["steps"]))

    def test_build_preprocessing_policy_details_uses_feature_role_overrides(self) -> None:
        details = build_preprocessing_policy_details(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "minimal_cleanup",
                "reasoning": [],
            },
            {
                "rows": 20,
                "columns": 4,
                "numeric_features": ["support_tickets", "tenure_months"],
                "categorical_features": ["region"],
                "categorical_cardinality": {"region": 3},
                "missing_by_feature": {},
                "constant_features": [],
                "likely_identifier_features": [],
                "date_like_features": [],
                "date_parse_failure_by_feature": {},
                "high_cardinality_categorical_features": {},
                "missing_fraction": 0.0,
                "target_cardinality": 2,
                "target_name": "target",
                "target_skew": None,
                "class_imbalance": 1.0,
            },
            selected_option="custom",
            feedback={
                "policy_metadata": {
                    "preprocessing_overrides": {"force_categorical_features": ["support_tickets"]}
                }
            },
        )
        self.assertEqual(details["selected_option"], "custom")
        self.assertTrue(any("Treat `support_tickets` as categorical" in entry["step"] for entry in details["steps"]))

    def test_build_preprocessing_policy_details_handles_numeric_only_scope(self) -> None:
        details = build_preprocessing_policy_details(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "reasoning": [],
            },
            {
                "rows": 24,
                "columns": 8,
                "numeric_features": ["age", "monthly_spend", "support_tickets", "tenure_months"],
                "categorical_features": ["customer_id", "contract_type", "region"],
                "categorical_cardinality": {"customer_id": 24, "contract_type": 3, "region": 4},
                "missing_by_feature": {},
                "constant_features": [],
                "likely_identifier_features": ["customer_id"],
                "date_like_features": [],
                "date_parse_failure_by_feature": {},
                "high_cardinality_categorical_features": {},
                "missing_fraction": 0.0,
                "target_cardinality": 2,
                "target_name": "target",
                "target_skew": None,
                "class_imbalance": 1.0,
                "distinct_by_feature": {
                    "customer_id": 24,
                    "contract_type": 3,
                    "region": 4,
                    "age": 20,
                    "monthly_spend": 24,
                    "support_tickets": 6,
                    "tenure_months": 24,
                },
                "numeric_range_by_feature": {
                    "age": {"min": 24.0, "max": 57.0},
                    "monthly_spend": {"min": 31.0, "max": 115.0},
                    "support_tickets": {"min": 0.0, "max": 5.0},
                    "tenure_months": {"min": 1.0, "max": 24.0},
                },
            },
            selected_option="custom",
            feedback={
                "policy_metadata": {
                    "preprocessing_overrides": {
                        "drop_features": ["customer_id", "contract_type", "region"]
                    }
                }
            },
        )
        self.assertTrue(any("Use only the raw numeric input features" in entry["step"] for entry in details["steps"]))
        self.assertTrue(any("Skip categorical encoding" in entry["step"] for entry in details["steps"]))
        self.assertFalse(any("One-hot encode the remaining categorical features" in entry["step"] for entry in details["steps"]))

    def test_build_preprocessing_policy_details_handles_specific_feature_subset(self) -> None:
        details = build_preprocessing_policy_details(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "reasoning": [],
            },
            {
                "rows": 24,
                "columns": 8,
                "numeric_features": ["age", "monthly_spend", "support_tickets", "tenure_months"],
                "categorical_features": ["customer_id", "contract_type", "region"],
                "categorical_cardinality": {"customer_id": 24, "contract_type": 3, "region": 4},
                "missing_by_feature": {},
                "constant_features": [],
                "likely_identifier_features": ["customer_id"],
                "date_like_features": [],
                "date_parse_failure_by_feature": {},
                "high_cardinality_categorical_features": {},
                "missing_fraction": 0.0,
                "target_cardinality": 2,
                "target_name": "target",
                "target_skew": None,
                "class_imbalance": 1.0,
                "distinct_by_feature": {
                    "customer_id": 24,
                    "contract_type": 3,
                    "region": 4,
                    "age": 20,
                    "monthly_spend": 24,
                    "support_tickets": 6,
                    "tenure_months": 24,
                },
                "numeric_range_by_feature": {
                    "age": {"min": 24.0, "max": 57.0},
                    "monthly_spend": {"min": 31.0, "max": 115.0},
                    "support_tickets": {"min": 0.0, "max": 5.0},
                    "tenure_months": {"min": 1.0, "max": 24.0},
                },
            },
            selected_option="custom",
            feedback={
                "policy_metadata": {
                    "preprocessing_overrides": {
                        "keep_features": ["age", "tenure_months"],
                        "drop_features": ["customer_id", "contract_type", "region", "monthly_spend", "support_tickets"],
                    }
                }
            },
        )
        self.assertTrue(any("selected input feature subset" in entry["step"] for entry in details["steps"]))
        self.assertTrue(any("`age` spans `24` to `57`" in entry["why"] for entry in details["steps"]))
        self.assertTrue(any("`tenure_months` spans `1` to `24`" in entry["why"] for entry in details["steps"]))
        self.assertTrue(any("Skip categorical encoding" in entry["step"] for entry in details["steps"]))
        self.assertFalse(any("monthly_spend" in entry["why"] for entry in details["steps"] if "StandardScaler" in entry["step"]))

    def test_build_preprocessing_policy_details_uses_merged_scope_and_feature_encoding(self) -> None:
        details = build_preprocessing_policy_details(
            {
                "step_id": "01_preprocessing",
                "title": "Preprocessing",
                "selected_option": "custom",
                "reasoning": [],
            },
            {
                "rows": 24,
                "columns": 8,
                "numeric_features": ["age", "monthly_spend", "support_tickets", "tenure_months"],
                "categorical_features": ["customer_id", "contract_type", "region"],
                "categorical_cardinality": {"customer_id": 24, "contract_type": 3, "region": 4},
                "missing_by_feature": {},
                "constant_features": [],
                "likely_identifier_features": ["customer_id"],
                "date_like_features": [],
                "date_parse_failure_by_feature": {},
                "high_cardinality_categorical_features": {},
                "missing_fraction": 0.0,
                "target_cardinality": 2,
                "target_name": "target",
                "target_skew": None,
                "class_imbalance": 1.0,
                "distinct_by_feature": {
                    "customer_id": 24,
                    "contract_type": 3,
                    "region": 4,
                    "age": 20,
                    "monthly_spend": 24,
                    "support_tickets": 6,
                    "tenure_months": 24,
                },
                "numeric_range_by_feature": {
                    "age": {"min": 24.0, "max": 57.0},
                    "monthly_spend": {"min": 31.0, "max": 115.0},
                    "support_tickets": {"min": 0.0, "max": 5.0},
                    "tenure_months": {"min": 1.0, "max": 24.0},
                },
            },
            selected_option="custom",
            feedback={
                "policy_metadata": {
                    "preprocessing_overrides": {
                        "drop_features": ["customer_id", "contract_type", "region"],
                        "force_one_hot_features": ["support_tickets"],
                        "force_categorical_features": ["support_tickets"],
                    }
                }
            },
        )
        self.assertFalse(any("Use only the raw numeric input features" in entry["step"] for entry in details["steps"]))
        self.assertTrue(any("One-hot encode `support_tickets`" in entry["step"] for entry in details["steps"]))
        self.assertFalse(any("One-hot encode the remaining categorical features" in entry["step"] for entry in details["steps"]))

    def test_parse_codex_exec_json_events_filters_non_json_lines(self) -> None:
        events = parse_codex_exec_json_events(
            "\n".join(
                [
                    "Reading additional input from stdin...",
                    '{"type":"turn.started"}',
                    "2026-04-26 WARN something",
                    '{"type":"item.completed","item":{"type":"agent_message","text":"Hello"}}',
                ]
            )
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1]["type"], "item.completed")

    def test_build_codex_discussion_activity_keeps_safe_status_and_command_messages(self) -> None:
        activity = build_codex_discussion_activity(
            [
                {"type": "item.completed", "item": {"type": "agent_message", "text": "Checking the date."}},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "date '+%A'",
                        "aggregated_output": "Sunday\n",
                        "exit_code": 0,
                    },
                },
                {"type": "item.completed", "item": {"type": "agent_message", "text": "Sunday."}},
            ],
            final_reply="Sunday.",
        )
        self.assertEqual(len(activity), 2)
        self.assertIn("Codex status", activity[0]["content"])
        self.assertIn("Codex activity", activity[1]["content"])


if __name__ == "__main__":
    unittest.main()
