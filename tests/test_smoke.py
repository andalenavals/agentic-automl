from __future__ import annotations

import json
import pandas as pd
import tempfile
import unittest
from pathlib import Path

from agentic_automl.orchestrator import execute_workflow, plan_project, prepare_execution_context, train_model_phase
from agentic_automl.paths import resolve_repo_root
from agentic_automl.schemas import ProjectBrief
from agentic_automl.workflow import get_workflow_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "examples" / "customer_churn_demo.csv"


class AgenticAutomlSmokeTest(unittest.TestCase):
    def test_canonical_workflow_comes_from_packaged_asset(self) -> None:
        canonical_path = REPO_ROOT / "src" / "agentic_automl" / "assets" / "automl_workflow.md"
        legacy_path = REPO_ROOT / "workflow" / "automl_workflow.md"
        self.assertFalse(legacy_path.exists())
        self.assertEqual(get_workflow_markdown(REPO_ROOT), canonical_path.read_text(encoding="utf-8"))

    def test_workflow_skill_files_exist_and_are_referenced(self) -> None:
        workflow_markdown = get_workflow_markdown(REPO_ROOT)
        expected_paths = [
            "skills/00-intake/SKILLS.md",
            "skills/00-intake/KNOWLEDGE.md",
            "skills/00-intake/LIMITS.md",
            "skills/01-preprocessing/SKILLS.md",
            "skills/01-preprocessing/KNOWLEDGE.md",
            "skills/01-preprocessing/LIMITS.md",
            "skills/02-data-splitting/SKILLS.md",
            "skills/02-data-splitting/KNOWLEDGE.md",
            "skills/02-data-splitting/LIMITS.md",
            "skills/03-model-selection/SKILLS.md",
            "skills/03-model-selection/KNOWLEDGE.md",
            "skills/03-model-selection/LIMITS.md",
            "skills/05-metric-selection/SKILLS.md",
            "skills/05-metric-selection/KNOWLEDGE.md",
            "skills/05-metric-selection/LIMITS.md",
            "skills/06-training-configuration/SKILLS.md",
            "skills/06-training-configuration/KNOWLEDGE.md",
            "skills/06-training-configuration/LIMITS.md",
            "skills/07-hyperparameter-optimization/SKILLS.md",
            "skills/07-hyperparameter-optimization/KNOWLEDGE.md",
            "skills/07-hyperparameter-optimization/LIMITS.md",
            "skills/08-validation-and-baseline/SKILLS.md",
            "skills/08-validation-and-baseline/KNOWLEDGE.md",
            "skills/08-validation-and-baseline/LIMITS.md",
            "skills/09-final-validation/SKILLS.md",
            "skills/09-final-validation/KNOWLEDGE.md",
            "skills/09-final-validation/LIMITS.md",
        ]
        for relative_path in expected_paths:
            asset_path = REPO_ROOT / "src" / "agentic_automl" / "assets" / relative_path
            self.assertTrue(asset_path.exists(), relative_path)
            self.assertIn(relative_path, workflow_markdown)

        preprocessing_skill = (
            REPO_ROOT
            / "src"
            / "agentic_automl"
            / "assets"
            / "skills"
            / "01-preprocessing"
            / "SKILLS.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Cardinality Reasoning", preprocessing_skill)
        self.assertIn("high-cardinality", preprocessing_skill)
        self.assertIn("low-cardinality", preprocessing_skill)
        self.assertIn("Sparsity Reasoning", preprocessing_skill)
        self.assertIn("sparse", preprocessing_skill.lower())
        self.assertIn("date parsing failures", preprocessing_skill)
        self.assertIn("scaling method", preprocessing_skill)
        self.assertIn("checked feature facts", preprocessing_skill)
        self.assertIn("There is no separate workflow step for feature selection anymore", preprocessing_skill)

        removed_skill_dir = REPO_ROOT / "src" / "agentic_automl" / "assets" / "skills" / "04-feature-selection"
        self.assertFalse(removed_skill_dir.exists())
        self.assertNotIn("skills/04-feature-selection", workflow_markdown)

        for step_slug in [
            "00-intake",
            "01-preprocessing",
            "02-data-splitting",
            "03-model-selection",
            "05-metric-selection",
            "06-training-configuration",
            "07-hyperparameter-optimization",
            "08-validation-and-baseline",
            "09-final-validation",
        ]:
            knowledge_text = (
                REPO_ROOT / "src" / "agentic_automl" / "assets" / "skills" / step_slug / "KNOWLEDGE.md"
            ).read_text(encoding="utf-8")
            limits_text = (
                REPO_ROOT / "src" / "agentic_automl" / "assets" / "skills" / step_slug / "LIMITS.md"
            ).read_text(encoding="utf-8")
            self.assertIn("## Capability Keys", knowledge_text)
            self.assertNotIn("## Current Limits", knowledge_text)
            self.assertIn("## Seed Backlog", limits_text)

    def test_resolve_repo_root_uses_pyproject_marker(self) -> None:
        self.assertEqual(resolve_repo_root(), REPO_ROOT)

    def test_plan_project_returns_recommendations(self) -> None:
        brief = ProjectBrief(
            project_name="smoke-plan",
            dataset_path=str(DATASET),
            target_column="churned",
            task_type="classification",
            problem_description="Predict churn from simple customer behavior features.",
        )
        _, profile, recommendations = plan_project(brief, repo_root=REPO_ROOT)
        self.assertEqual(profile.target_name, "churned")
        self.assertEqual(set(profile.categorical_cardinality), set(profile.categorical_features))
        self.assertEqual(len(recommendations), 9)
        self.assertNotIn("04_feature_selection", {item.step_id for item in recommendations})
        self.assertIn("09_final_validation", {item.step_id for item in recommendations})
        for recommendation in recommendations:
            self.assertNotIn("Memory signal not available yet for this step.", recommendation.reasoning)

    def test_execute_workflow_exports_only_a_notebook(self) -> None:
        brief = ProjectBrief(
            project_name="smoke-run",
            dataset_path=str(DATASET),
            target_column="churned",
            task_type="classification",
            problem_description="Predict churn from simple customer behavior features.",
            competition_enabled=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = execute_workflow(
                brief,
                repo_root=REPO_ROOT,
                output_root=tmpdir,
            )
            self.assertFalse((Path(tmpdir) / "memory").exists())
            self.assertFalse(any(path.name == "memory" for path in Path(tmpdir).rglob("memory")))
            self.assertIn(artifacts.selected_metric, {"balanced_accuracy", "f1_macro"})
            self.assertIsNotNone(artifacts.output_notebook_path)
            self.assertTrue(Path(artifacts.output_notebook_path).exists())
            self.assertEqual(Path(artifacts.output_notebook_path).suffix, ".ipynb")
            self.assertEqual(Path(artifacts.output_notebook_path).parent.resolve(), Path(tmpdir).resolve())
            exported_files = sorted(path.resolve() for path in Path(tmpdir).iterdir() if path.is_file())
            self.assertEqual(exported_files, [Path(artifacts.output_notebook_path).resolve()])
            notebook = json.loads(Path(artifacts.output_notebook_path).read_text(encoding="utf-8"))
            notebook_source = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook.get("cells", [])
            )
            self.assertIn("def preprocessing", notebook_source)
            self.assertIn("def split_data", notebook_source)
            self.assertIn("def train_model", notebook_source)
            self.assertIn("def hyperparameter_optimization", notebook_source)
            self.assertIn("def validate", notebook_source)
            self.assertIn("competition_dashboard", notebook_source)
            self.assertIn("prediction_frame", notebook_source)
            self.assertIn("transformed_dataset = preprocessing(", notebook_source)
            self.assertIn('training_state = train_model(split_state["train"])', notebook_source)
            self.assertIn('display(pd.DataFrame([optimization_state["summary"]]))', notebook_source)
            self.assertIn('display(optimization_state["competition_dashboard"].head(12))', notebook_source)
            self.assertIn('validation_state = validate(split_state["test"], winning_state)', notebook_source)
            self.assertIn("## 1. Preprocessing", notebook_source)
            self.assertIn("## 2. Data Splitting", notebook_source)
            self.assertIn("## 3. Model Training", notebook_source)
            self.assertIn("## 4. Hyperparameter Optimization", notebook_source)
            self.assertIn("## 5. Validation", notebook_source)
            self.assertIn("Relevant input information:", notebook_source)
            self.assertIn("Selected policies:", notebook_source)
            self.assertIn("- Training: `standard_training`", notebook_source)
            self.assertNotIn("Problem description:", notebook_source)
            self.assertNotIn(f'"dataset_path": "{DATASET.resolve()}"', notebook_source)
            self.assertNotIn("def build_preprocessor", notebook_source)
            self.assertNotIn("class DatasetPreparer", notebook_source)
            self.assertNotIn("def run_exported_workflow", notebook_source)
            self.assertNotIn("def apply_numeric_transform", notebook_source)
            self.assertNotIn("def apply_imputation", notebook_source)
            self.assertNotIn("raw_numeric_features", notebook_source)
            self.assertNotIn("one_hot_features", notebook_source)
            self.assertNotIn("ordinal_features", notebook_source)
            self.assertNotIn('if BASELINE_STRATEGY ==', notebook_source)
            self.assertNotIn("from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score", notebook_source)
            self.assertNotIn("from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score", notebook_source)
            self.assertNotIn("return pd.read_parquet(path)", notebook_source)
            self.assertNotIn("Unsupported dataset format", notebook_source)
            self.assertNotIn('if method == "minmax"', notebook_source)
            self.assertNotIn('if method == "robust"', notebook_source)
            self.assertNotIn("from agentic_automl import", notebook_source)
            self.assertNotIn("from agentic_automl.", notebook_source)
            self.assertNotIn("\nimport agentic_automl\n", notebook_source)
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                code = "".join(cell.get("source", []))
                compile(code, f"notebook_cell_{index}", "exec")
            self.assertIn("winner_test_metric", artifacts.metrics_summary)

    def test_execute_workflow_honors_preprocessing_feature_role_overrides(self) -> None:
        frame = pd.DataFrame(
            {
                "support_tickets": [0, 1, 2, 0, 1, 2, 3, 1, 0, 2, 1, 0, 2, 3, 1, 0, 2, 1, 0, 3],
                "tenure_months": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39],
                "region": [
                    "north", "south", "east", "west", "north",
                    "south", "east", "west", "north", "south",
                    "east", "west", "north", "south", "east",
                    "west", "north", "south", "east", "west",
                ],
                "churned": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "feature_role_override.csv"
            frame.to_csv(dataset_path, index=False)
            brief = ProjectBrief(
                project_name="override-run",
                dataset_path=str(dataset_path),
                target_column="churned",
                task_type="classification",
                problem_description="Predict churn from simple customer behavior features.",
            )
            context = prepare_execution_context(
                brief,
                selected_options={"01_preprocessing": "custom"},
                step_feedback={
                    "01_preprocessing": {
                        "agreement": "different",
                        "custom_note": "Treat support_tickets as categorical.",
                        "policy_summary": "Use custom for preprocessing with support_tickets as categorical.",
                        "policy_metadata": {
                            "preprocessing_overrides": {
                                "force_categorical_features": ["support_tickets"]
                            }
                        },
                    }
                },
                repo_root=REPO_ROOT,
            )
            training_state = train_model_phase(context)
            pipeline = training_state["pipeline"]
            plan = pipeline.named_steps["preprocessor"].named_steps["prepare"].plan

            self.assertIn("support_tickets", plan.raw_categorical_features)
            self.assertNotIn("support_tickets", plan.raw_numeric_features)


if __name__ == "__main__":
    unittest.main()
