from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from agentic_automl.advisor import build_recommendations
from agentic_automl.data import profile_dataset
from agentic_automl.modeling import choose_baseline_strategy
from agentic_automl.preprocessing import (
    DatasetPreparer,
    build_preprocessing_plan,
    build_preprocessor,
    describe_preprocessing_execution_steps,
    merge_preprocessing_overrides,
    normalize_preprocessing_overrides,
)
from agentic_automl.schemas import ProjectBrief


class PreprocessingRecommendationTest(unittest.TestCase):
    def test_preprocessing_recommendation_uses_file_specific_signals(self) -> None:
        frame = pd.DataFrame(
            {
                "customer_id": [f"c-{index:03d}" for index in range(40)],
                "event_date": [f"2026-01-{(index % 28) + 1:02d}" for index in range(40)],
                "zip_code": [f"zip-{index:03d}" for index in range(30)] + ["zip-000"] * 10,
                "age": [None if index % 5 == 0 else 20 + index for index in range(40)],
                "constant_flag": ["same"] * 40,
                "region": ["north", "south", "east", "west"] * 10,
                "target": ["yes" if index % 2 == 0 else "no" for index in range(40)],
            }
        )
        brief = ProjectBrief(
            project_name="file-driven-preprocessing",
            dataset_path="synthetic.csv",
            target_column="target",
            task_type="classification",
            problem_description="Predict the binary target.",
        )
        profile = profile_dataset(frame, brief)
        recommendation = next(
            item
            for item in build_recommendations(brief, profile)
            if item.step_id == "01_preprocessing"
        )

        self.assertIn("customer_id", profile.likely_identifier_features)
        self.assertIn("event_date", profile.date_like_features)
        self.assertIn("constant_flag", profile.constant_features)
        self.assertIn("zip_code", profile.high_cardinality_categorical_features)
        self.assertIn("age", profile.missing_by_feature)
        self.assertEqual(recommendation.selected_option, "custom")
        self.assertIn("customer_id", recommendation.recommendation)
        self.assertIn("event_date", recommendation.recommendation)
        self.assertIn("zip_code", recommendation.recommendation)

        plan = build_preprocessing_plan(profile, recommendation.selected_option)
        self.assertIn("customer_id", plan.dropped_features)
        self.assertIn("constant_flag", plan.dropped_features)
        self.assertIn("event_date", plan.date_features)
        self.assertNotIn("zip_code", plan.categorical_features)

        prepared = DatasetPreparer(plan).fit_transform(frame.drop(columns=["target"]))
        prepared_columns = set(prepared.columns)
        self.assertIn("event_date__year", prepared_columns)
        self.assertIn("event_date__month", prepared_columns)
        self.assertNotIn("customer_id", prepared_columns)
        self.assertNotIn("constant_flag", prepared_columns)
        self.assertNotIn("zip_code", prepared_columns)

        execution_steps = describe_preprocessing_execution_steps(profile, recommendation.selected_option)
        self.assertTrue(any("median imputation" in entry["step"].lower() for entry in execution_steps))
        self.assertTrue(any("`StandardScaler`" in entry["step"] for entry in execution_steps))
        self.assertTrue(any("`customer_id` has `40` distinct values across `40` rows" in entry["why"] for entry in execution_steps))
        self.assertTrue(any("`region` has `4` distinct values" in entry["why"] for entry in execution_steps))

    def test_preprocessing_recommendation_uses_light_cleanup_for_clean_file(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [21, 24, 30, 44, 55, 62],
                "income": [32000, 40000, 51000, 58000, 61000, 72000],
                "region": ["north", "south", "north", "west", "east", "south"],
                "target": [0, 1, 0, 1, 1, 0],
            }
        )
        brief = ProjectBrief(
            project_name="clean-file-preprocessing",
            dataset_path="clean.csv",
            target_column="target",
            task_type="classification",
            problem_description="Predict the binary target.",
        )
        profile = profile_dataset(frame, brief)
        recommendation = next(
            item
            for item in build_recommendations(brief, profile)
            if item.step_id == "01_preprocessing"
        )

        self.assertEqual(profile.missing_by_feature, {})
        self.assertEqual(profile.likely_identifier_features, [])
        self.assertEqual(profile.date_like_features, [])
        self.assertEqual(profile.date_parse_failure_by_feature, {})
        self.assertEqual(profile.constant_features, [])
        self.assertEqual(profile.high_cardinality_categorical_features, {})
        self.assertEqual(recommendation.selected_option, "minimal_cleanup")
        self.assertIn("file already looks clean", recommendation.recommendation.lower())
        self.assertIn("skip imputation", recommendation.recommendation.lower())

        execution_steps = describe_preprocessing_execution_steps(profile, recommendation.selected_option)
        self.assertFalse(any("imputation" in entry["step"].lower() for entry in execution_steps))
        self.assertFalse(any("StandardScaler" in entry["step"] for entry in execution_steps))
        numeric_step = next(entry for entry in execution_steps if "without extra scaling" in entry["step"].lower())
        self.assertIn("`age` spans `21` to `62`", numeric_step["why"])
        self.assertIn("`income` spans `32,000` to `72,000`", numeric_step["why"])

    def test_date_parse_failures_trigger_targeted_numeric_repair(self) -> None:
        frame = pd.DataFrame(
            {
                "event_date": ["2026-01-01", "2026-01-02", "not-a-date", "2026-01-04"],
                "amount": [12.0, 15.5, 10.2, 16.7],
                "target": [0, 1, 0, 1],
            }
        )
        brief = ProjectBrief(
            project_name="date-parse-failure",
            dataset_path="dates.csv",
            target_column="target",
            task_type="classification",
            problem_description="Predict the binary target.",
        )

        profile = profile_dataset(frame, brief)
        execution_steps = describe_preprocessing_execution_steps(profile, "custom")

        self.assertIn("event_date", profile.date_parse_failure_by_feature)
        imputation_step = next(
            entry for entry in execution_steps if "median imputation" in entry["step"].lower()
        )
        self.assertIn("date-derived parts from `event_date`", imputation_step["step"])

    def test_categorical_imputation_is_not_recommended_without_categorical_gaps(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [21, 24, 30, 44, 55, 62],
                "income": [32000, 40000, 51000, 58000, 61000, 72000],
                "region": ["north", "south", "north", "west", "east", "south"],
                "plan_type": ["basic", "pro", "basic", "team", "pro", "team"],
                "target": [0, 1, 0, 1, 1, 0],
            }
        )
        brief = ProjectBrief(
            project_name="clean-categorical-file",
            dataset_path="clean_categorical.csv",
            target_column="target",
            task_type="classification",
            problem_description="Predict the binary target.",
        )

        profile = profile_dataset(frame, brief)
        execution_steps = describe_preprocessing_execution_steps(profile, "auto_tabular_preprocessing")

        self.assertFalse(profile.missing_by_feature)
        self.assertTrue(profile.categorical_features)
        self.assertFalse(any("most-frequent imputation" in entry["step"].lower() for entry in execution_steps))
        one_hot_step = next(entry for entry in execution_steps if "one-hot encode" in entry["step"].lower())
        self.assertIn("`region` has `4` distinct values", one_hot_step["why"])
        self.assertIn("`plan_type` has `3` distinct values", one_hot_step["why"])

    def test_preprocessing_plan_can_reclassify_numeric_feature_as_categorical(self) -> None:
        frame = pd.DataFrame(
            {
                "support_tickets": [0, 1, 2, 0, 1, 3],
                "tenure_months": [2, 5, 8, 12, 15, 18],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        brief = ProjectBrief(
            project_name="feature-role-override",
            dataset_path="roles.csv",
            target_column="target",
            task_type="classification",
            problem_description="Predict the binary target.",
        )

        profile = profile_dataset(frame, brief)
        overrides = {"force_categorical_features": ["support_tickets"]}
        plan = build_preprocessing_plan(profile, "custom", overrides=overrides)
        execution_steps = describe_preprocessing_execution_steps(profile, "custom", overrides=overrides)

        self.assertNotIn("support_tickets", plan.numeric_features)
        self.assertIn("support_tickets", plan.categorical_features)
        self.assertTrue(any("Treat `support_tickets` as categorical" in entry["step"] for entry in execution_steps))
        self.assertTrue(any("one-hot encode" in entry["step"].lower() for entry in execution_steps))

    def test_preprocessing_plan_can_use_only_numeric_features(self) -> None:
        frame = pd.DataFrame(
            {
                "customer_id": [f"c-{index:03d}" for index in range(24)],
                "contract_type": ["basic", "plus", "pro"] * 8,
                "region": ["north", "south", "east", "west"] * 6,
                "age": list(range(24, 48)),
                "monthly_spend": [31 + index * 3 for index in range(24)],
                "support_tickets": [index % 6 for index in range(24)],
                "tenure_months": list(range(1, 25)),
                "target": [index % 2 for index in range(24)],
            }
        )
        brief = ProjectBrief(
            project_name="numeric-only-scope",
            dataset_path="numeric_only.csv",
            target_column="target",
            task_type="classification",
            problem_description="Predict the binary target.",
        )

        profile = profile_dataset(frame, brief)
        overrides = {"drop_features": ["customer_id", "contract_type", "region"]}
        plan = build_preprocessing_plan(profile, "custom", overrides=overrides)
        execution_steps = describe_preprocessing_execution_steps(profile, "custom", overrides=overrides)

        self.assertEqual(set(plan.raw_categorical_features), set())
        self.assertIn("age", plan.raw_numeric_features)
        self.assertIn("monthly_spend", plan.raw_numeric_features)
        self.assertTrue(any("Use only the raw numeric input features" in entry["step"] for entry in execution_steps))
        self.assertTrue(any("Skip categorical encoding" in entry["step"] for entry in execution_steps))

    def test_preprocessing_plan_can_merge_scope_and_one_hot_feature_override(self) -> None:
        frame = pd.DataFrame(
            {
                "customer_id": [f"c-{index:03d}" for index in range(24)],
                "contract_type": ["basic", "plus", "pro"] * 8,
                "region": ["north", "south", "east", "west"] * 6,
                "age": list(range(24, 48)),
                "monthly_spend": [31 + index * 3 for index in range(24)],
                "support_tickets": [index % 6 for index in range(24)],
                "tenure_months": list(range(1, 25)),
                "target": [index % 2 for index in range(24)],
            }
        )
        brief = ProjectBrief(
            project_name="merged-preprocessing-overrides",
            dataset_path="merged_overrides.csv",
            target_column="target",
            task_type="classification",
            problem_description="Predict the binary target.",
        )

        profile = profile_dataset(frame, brief)
        overrides = merge_preprocessing_overrides(
            {"drop_features": ["customer_id", "contract_type", "region"]},
            {"force_one_hot_features": ["support_tickets"]},
        )
        plan = build_preprocessing_plan(profile, "custom", overrides=overrides)
        execution_steps = describe_preprocessing_execution_steps(profile, "custom", overrides=overrides)

        self.assertEqual(set(plan.raw_numeric_features), {"age", "monthly_spend", "tenure_months"})
        self.assertEqual(set(plan.one_hot_categorical_features), {"support_tickets"})
        self.assertFalse(any("Use only the raw numeric input features" in entry["step"] for entry in execution_steps))
        self.assertTrue(any("One-hot encode `support_tickets`" in entry["step"] for entry in execution_steps))
        self.assertFalse(any("One-hot encode the remaining categorical features" in entry["step"] for entry in execution_steps))

    def test_preprocessing_plan_can_use_only_selected_feature_subset(self) -> None:
        frame = pd.DataFrame(
            {
                "customer_id": [f"c-{index:03d}" for index in range(24)],
                "contract_type": ["basic", "plus", "pro"] * 8,
                "region": ["north", "south", "east", "west"] * 6,
                "age": list(range(24, 48)),
                "monthly_spend": [31 + index * 3 for index in range(24)],
                "support_tickets": [index % 6 for index in range(24)],
                "tenure_months": list(range(1, 25)),
                "target": [index % 2 for index in range(24)],
            }
        )
        brief = ProjectBrief(
            project_name="selected-feature-subset",
            dataset_path="feature_subset.csv",
            target_column="target",
            task_type="classification",
            problem_description="Predict the binary target.",
        )

        profile = profile_dataset(frame, brief)
        overrides = {
            "keep_features": ["age", "tenure_months"],
            "drop_features": ["customer_id", "contract_type", "region", "monthly_spend", "support_tickets"],
        }
        plan = build_preprocessing_plan(profile, "custom", overrides=overrides)
        execution_steps = describe_preprocessing_execution_steps(profile, "custom", overrides=overrides)

        self.assertEqual(set(plan.raw_numeric_features), {"age", "tenure_months"})
        self.assertEqual(set(plan.raw_categorical_features), set())
        self.assertTrue(any("selected input feature subset" in entry["step"] for entry in execution_steps))
        scaling_step = next(entry for entry in execution_steps if "StandardScaler" in entry["step"])
        self.assertIn("`age` spans `24` to `47`", scaling_step["why"])
        self.assertIn("`tenure_months` spans `1` to `24`", scaling_step["why"])
        self.assertNotIn("monthly_spend", scaling_step["why"])
        self.assertTrue(any("Skip categorical encoding" in entry["step"] for entry in execution_steps))

    def test_preprocessor_can_apply_feature_specific_mean_imputation_and_minmax_scaling(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [10.0, None, 30.0],
                "target": [0, 1, 0],
            }
        )
        brief = ProjectBrief(
            project_name="numeric-imputation-and-scaling",
            dataset_path="numeric_imputation.csv",
            target_column="target",
            task_type="classification",
            problem_description="Predict the binary target.",
        )
        profile = profile_dataset(frame, brief)
        overrides = normalize_preprocessing_overrides(
            {
                "feature_imputation_rules": [{"feature": "age", "strategy": "mean"}],
                "feature_scaling_rules": [{"feature": "age", "method": "minmax"}],
            }
        )
        preprocessor = build_preprocessor(profile, "custom", overrides=overrides)
        transformed = preprocessor.fit_transform(frame.drop(columns=["target"]))

        self.assertEqual(transformed.shape[1], 1)
        self.assertAlmostEqual(float(transformed[0, 0]), 0.0)
        self.assertAlmostEqual(float(transformed[1, 0]), 0.5)
        self.assertAlmostEqual(float(transformed[2, 0]), 1.0)

    def test_dataset_preparer_can_apply_transform_and_frequency_count_rules(self) -> None:
        frame = pd.DataFrame(
            {
                "monthly_spend": [9.0, 99.0, 999.0, 99.0],
                "region": ["north", "south", "north", "south"],
                "target": [0, 1, 0, 1],
            }
        )
        brief = ProjectBrief(
            project_name="transform-and-frequency-count",
            dataset_path="transform_frequency.csv",
            target_column="target",
            task_type="classification",
            problem_description="Predict the binary target.",
        )
        profile = profile_dataset(frame, brief)
        overrides = normalize_preprocessing_overrides(
            {
                "feature_transform_rules": [{"feature": "monthly_spend", "kind": "log1p"}],
                "derived_feature_rules": [{"kind": "frequency_count", "source": "region", "output": "region__frequency_count"}],
            }
        )
        plan = build_preprocessing_plan(profile, "custom", overrides=overrides)
        prepared = DatasetPreparer(plan, overrides=overrides).fit_transform(frame.drop(columns=["target"]))

        self.assertIn("region__frequency_count", prepared.columns)
        self.assertAlmostEqual(float(prepared.loc[0, "monthly_spend"]), float(np.log1p(9.0)))
        self.assertEqual(list(prepared["region__frequency_count"]), [2.0, 2.0, 2.0, 2.0])


class BaselineStrategyTest(unittest.TestCase):
    def test_classification_uses_class_prior_baseline_for_hard_label_metric(self) -> None:
        frame = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5, 6],
                "target": [0, 0, 0, 1, 1, 1],
            }
        )
        brief = ProjectBrief(
            project_name="classification-baseline",
            dataset_path="classification.csv",
            target_column="target",
            task_type="classification",
            problem_description="Predict the class label.",
        )
        profile = profile_dataset(frame, brief)

        strategy, reason = choose_baseline_strategy(brief, profile, "f1_macro")

        self.assertEqual(strategy, "stratified_random")
        self.assertIn("training target distribution", reason)
        self.assertIn("strongest simple reference", reason)
        self.assertIn("most_frequent", reason)
        self.assertIn("uniform_random", reason)

    def test_regression_prefers_mean_for_squared_error_metrics(self) -> None:
        frame = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5, 6],
                "target": [10.0, 12.0, 13.0, 15.0, 100.0, 120.0],
            }
        )
        brief = ProjectBrief(
            project_name="regression-mean-baseline",
            dataset_path="regression.csv",
            target_column="target",
            task_type="regression",
            problem_description="Predict a continuous target with visible outliers.",
        )
        profile = profile_dataset(frame, brief)

        strategy, reason = choose_baseline_strategy(brief, profile, "rmse")

        self.assertEqual(strategy, "mean_value")
        self.assertIn("training target distribution", reason)
        self.assertIn("squared-error style metric", reason)

    def test_regression_prefers_median_for_absolute_error_metrics(self) -> None:
        frame = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5, 6],
                "target": [10.0, 12.0, 13.0, 15.0, 100.0, 120.0],
            }
        )
        brief = ProjectBrief(
            project_name="regression-median-baseline",
            dataset_path="regression.csv",
            target_column="target",
            task_type="regression",
            problem_description="Predict a continuous target.",
        )
        profile = profile_dataset(frame, brief)

        strategy, reason = choose_baseline_strategy(brief, profile, "mae")

        self.assertEqual(strategy, "median_value")
        self.assertIn("training target distribution", reason)
        self.assertIn("absolute-error metric", reason)


if __name__ == "__main__":
    unittest.main()
