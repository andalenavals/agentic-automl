from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_automl.advisor import build_recommendations
from agentic_automl.modeling import hpo_search_space_config
from agentic_automl.orchestrator import resolve_workflow_configuration
from agentic_automl.schemas import DataProfile, ProjectBrief
from agentic_automl.ui_app import respond_to_step_discussion


class HpoActionTest(unittest.TestCase):
    def make_hpo_recommendation(self) -> dict:
        return {
            "step_id": "07_hyperparameter_optimization",
            "title": "Hyperparameter optimization",
            "selected_option": "skip",
            "options": ["skip", "small_competition", "expanded_competition"],
            "metadata": {"model_option": "random_forest_classifier"},
        }

    def test_hpo_question_returns_model_specific_recommendations(self) -> None:
        response = respond_to_step_discussion(
            self.make_hpo_recommendation(),
            "Which hyperparameters would you recommend to tune?",
            current_option="skip",
            selected_options={"03_model_selection": "random_forest_classifier"},
        )

        self.assertFalse(response["updated_policy"])
        self.assertIn("`max_depth`", response["reply"])
        self.assertIn("`min_samples_leaf`", response["reply"])
        self.assertIn("`class_weight`", response["reply"])

    def test_hpo_question_returns_model_specific_supported_hyperparameters(self) -> None:
        response = respond_to_step_discussion(
            self.make_hpo_recommendation(),
            "Which hyperparameters are supported?",
            current_option="skip",
            selected_options={"03_model_selection": "random_forest_classifier"},
        )

        self.assertFalse(response["updated_policy"])
        self.assertIn("For `random_forest_classifier`", response["reply"])
        self.assertIn("`max_depth`", response["reply"])
        self.assertIn("`n_estimators`", response["reply"])

    def test_hpo_action_request_builds_explicit_search_scope(self) -> None:
        response = respond_to_step_discussion(
            self.make_hpo_recommendation(),
            "Tune max_depth, min_samples_leaf, and class_weight.",
            current_option="skip",
            selected_options={"03_model_selection": "random_forest_classifier"},
            current_policy_metadata={},
        )

        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "small_competition")
        self.assertEqual(
            response["policy_metadata"]["hpo_config"]["search_parameters"],
            ["max_depth", "min_samples_leaf", "class_weight"],
        )
        self.assertIn("`random_forest_classifier`", response["reply"])

    def test_hpo_action_request_keeps_active_competition_when_restricting_scope(self) -> None:
        recommendation = self.make_hpo_recommendation()
        first_response = respond_to_step_discussion(
            recommendation,
            "I want to run a small competitin",
            current_option="skip",
            selected_options={"03_model_selection": "random_forest_classifier"},
        )

        second_response = respond_to_step_discussion(
            recommendation,
            "I want to use only max_depth as an hyperparameter",
            current_option=first_response["selected_option"],
            selected_options={
                "03_model_selection": "random_forest_classifier",
                "07_hyperparameter_optimization": first_response["selected_option"],
            },
            current_policy_metadata=first_response.get("policy_metadata", {}),
        )

        self.assertTrue(second_response["updated_policy"])
        self.assertEqual(second_response["selected_option"], "small_competition")
        self.assertEqual(
            second_response["policy_metadata"]["hpo_config"]["search_parameters"],
            ["max_depth"],
        )

    def test_hpo_action_request_supports_top_two_most_important_hyperparameters(self) -> None:
        response = respond_to_step_discussion(
            self.make_hpo_recommendation(),
            "I want a small competition with only the two most important hyperparameters.",
            current_option="skip",
            selected_options={"03_model_selection": "random_forest_classifier"},
            current_policy_metadata={},
        )

        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "small_competition")
        self.assertEqual(
            response["policy_metadata"]["hpo_config"]["search_parameters"],
            ["max_depth", "min_samples_leaf"],
        )

    def test_hpo_action_request_supports_top_hyperparameters_without_explicit_count(self) -> None:
        response = respond_to_step_discussion(
            self.make_hpo_recommendation(),
            "Use the most important hyperparameters for the competition.",
            current_option="skip",
            selected_options={"03_model_selection": "random_forest_classifier"},
            current_policy_metadata={},
        )

        self.assertTrue(response["updated_policy"])
        self.assertEqual(response["selected_option"], "small_competition")
        self.assertEqual(
            response["policy_metadata"]["hpo_config"]["search_parameters"],
            ["max_depth", "min_samples_leaf", "max_features", "class_weight"],
        )

    def test_hpo_follow_up_can_refine_scope_without_repeating_hyperparameter_word(self) -> None:
        recommendation = self.make_hpo_recommendation()
        first_response = respond_to_step_discussion(
            recommendation,
            "I want to run a small competition.",
            current_option="skip",
            selected_options={"03_model_selection": "random_forest_classifier"},
        )

        second_response = respond_to_step_discussion(
            recommendation,
            "Focus on max_depth and class_weight.",
            current_option=first_response["selected_option"],
            selected_options={
                "03_model_selection": "random_forest_classifier",
                "07_hyperparameter_optimization": first_response["selected_option"],
            },
            current_policy_metadata=first_response.get("policy_metadata", {}),
        )

        self.assertTrue(second_response["updated_policy"])
        self.assertEqual(second_response["selected_option"], "small_competition")
        self.assertEqual(
            second_response["policy_metadata"]["hpo_config"]["search_parameters"],
            ["max_depth", "class_weight"],
        )

    def test_hpo_follow_up_can_add_supported_parameter_by_name(self) -> None:
        recommendation = self.make_hpo_recommendation()
        first_response = respond_to_step_discussion(
            recommendation,
            "Tune max_depth and class_weight.",
            current_option="skip",
            selected_options={"03_model_selection": "random_forest_classifier"},
            current_policy_metadata={},
        )

        second_response = respond_to_step_discussion(
            recommendation,
            "Also include min_samples_leaf.",
            current_option=first_response["selected_option"],
            selected_options={
                "03_model_selection": "random_forest_classifier",
                "07_hyperparameter_optimization": first_response["selected_option"],
            },
            current_policy_metadata=first_response.get("policy_metadata", {}),
        )

        self.assertTrue(second_response["updated_policy"])
        self.assertEqual(second_response["selected_option"], "small_competition")
        self.assertEqual(
            second_response["policy_metadata"]["hpo_config"]["search_parameters"],
            ["max_depth", "class_weight", "min_samples_leaf"],
        )

    def test_hpo_action_request_rejects_unsupported_parameter_and_logs_limit(self) -> None:
        request = "Tune gamma and max_depth."
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            with patch.dict("os.environ", {"AGENTIC_AUTOML_HOME": str(repo_root)}):
                response = respond_to_step_discussion(
                    self.make_hpo_recommendation(),
                    request,
                    current_option="skip",
                    selected_options={"03_model_selection": "random_forest_classifier"},
                    current_policy_metadata={},
                )

            self.assertFalse(response["updated_policy"])
            self.assertIn("Sorry I can not perform this task yet", response["reply"])
            limits_path = (
                repo_root
                / "src"
                / "agentic_automl"
                / "assets"
                / "skills"
                / "07-hyperparameter-optimization"
                / "LIMITS.md"
            )
            self.assertTrue(limits_path.exists())
            self.assertIn(request, limits_path.read_text(encoding="utf-8"))


class HpoRuntimeTest(unittest.TestCase):
    def make_profile(self) -> DataProfile:
        return DataProfile(
            rows=200,
            columns=6,
            numeric_features=["age", "tenure_months", "monthly_spend"],
            categorical_features=["region", "contract_type"],
            categorical_cardinality={"region": 4, "contract_type": 3},
            missing_by_feature={},
            constant_features=[],
            likely_identifier_features=[],
            date_like_features=[],
            date_parse_failure_by_feature={},
            high_cardinality_categorical_features={},
            missing_fraction=0.0,
            target_cardinality=2,
            target_name="churned",
            class_imbalance=1.3,
            distinct_by_feature={},
            numeric_range_by_feature={},
        )

    def test_hpo_search_space_config_filters_to_requested_parameters(self) -> None:
        search_config = hpo_search_space_config(
            "random_forest_classifier",
            "expanded_competition",
            ["max_depth", "class_weight"],
        )

        self.assertEqual(set(search_config["display_grid"]), {"max_depth", "class_weight"})
        self.assertIn("model__max_depth", search_config["estimator_grid"])
        self.assertIn("model__class_weight", search_config["estimator_grid"])
        self.assertNotIn("model__n_estimators", search_config["estimator_grid"])

    def test_resolve_workflow_configuration_keeps_nested_hpo_scope(self) -> None:
        brief = ProjectBrief(
            project_name="hpo-config",
            dataset_path="synthetic.csv",
            target_column="churned",
            task_type="classification",
            problem_description="Predict churn.",
            competition_enabled=True,
        )
        profile = self.make_profile()
        config = resolve_workflow_configuration(
            brief,
            profile,
            {
                "03_model_selection": "random_forest_classifier",
                "07_hyperparameter_optimization": "small_competition",
            },
            {
                "07_hyperparameter_optimization": {
                    "policy_metadata": {
                        "hpo_config": {
                            "search_parameters": ["max_depth", "class_weight"],
                        }
                    }
                }
            },
        )

        self.assertEqual(config["hpo_config"]["search_parameters"], ["max_depth", "class_weight"])
        self.assertEqual(set(config["hpo_config"]["search_space"]), {"max_depth", "class_weight"})

    def test_resolve_workflow_configuration_uses_empty_hpo_scope_when_skipped(self) -> None:
        brief = ProjectBrief(
            project_name="hpo-skip-config",
            dataset_path="synthetic.csv",
            target_column="churned",
            task_type="classification",
            problem_description="Predict churn.",
            competition_enabled=False,
        )
        profile = self.make_profile()
        config = resolve_workflow_configuration(
            brief,
            profile,
            {
                "03_model_selection": "random_forest_classifier",
                "07_hyperparameter_optimization": "skip",
            },
            {},
        )

        self.assertEqual(config["hpo_config"]["search_parameters"], [])
        self.assertEqual(config["hpo_config"]["search_space"], {})

    def test_hpo_recommendation_metadata_is_empty_when_tuning_is_skipped(self) -> None:
        brief = ProjectBrief(
            project_name="hpo-skip-recommendation",
            dataset_path="synthetic.csv",
            target_column="churned",
            task_type="classification",
            problem_description="Predict churn.",
            competition_enabled=False,
        )
        profile = self.make_profile()
        recommendation = next(
            item
            for item in build_recommendations(brief, profile)
            if item.step_id == "07_hyperparameter_optimization"
        )

        self.assertEqual(recommendation.selected_option, "skip")
        self.assertEqual(recommendation.metadata["search_parameters"], [])
        self.assertEqual(recommendation.metadata["search_space"], {})
