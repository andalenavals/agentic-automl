from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_automl.orchestrator import execute_workflow, plan_project
from agentic_automl.schemas import ProjectBrief


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "examples" / "customer_churn_demo.csv"


class AgenticAutomlSmokeTest(unittest.TestCase):
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
        self.assertGreaterEqual(len(recommendations), 8)

    def test_execute_workflow_exports_a_bundle(self) -> None:
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
            self.assertIn(artifacts.selected_metric, {"balanced_accuracy", "f1_macro"})
            self.assertTrue(Path(artifacts.bundle_dir).exists())
            self.assertIn("winner_test_metric", artifacts.metrics_summary)


if __name__ == "__main__":
    unittest.main()
