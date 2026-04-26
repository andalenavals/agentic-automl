from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .paths import resolve_repo_root


@dataclass(frozen=True, slots=True)
class StepDefinition:
    step_id: str
    step_number: int
    title: str


STEP_DEFINITIONS: list[StepDefinition] = [
    StepDefinition("00_intake", 0, "Intake"),
    StepDefinition("01_preprocessing", 1, "Preprocessing"),
    StepDefinition("02_data_splitting", 2, "Data splitting"),
    StepDefinition("03_model_selection", 3, "Model selection"),
    StepDefinition("04_feature_selection", 4, "Feature selection"),
    StepDefinition("05_metric_selection", 5, "Metric selection"),
    StepDefinition("06_training_configuration", 6, "Training configuration"),
    StepDefinition("07_hyperparameter_optimization", 7, "Hyperparameter optimization"),
    StepDefinition("08_validation_and_baseline", 8, "Validation and baseline comparison"),
]


def get_workflow_markdown(repo_root: str | Path | None = None) -> str:
    root = resolve_repo_root(repo_root)
    workflow_path = root / "workflow" / "automl_workflow.md"
    if workflow_path.exists():
        return workflow_path.read_text(encoding="utf-8")

    packaged = resources.files("agentic_automl.assets").joinpath("automl_workflow.md")
    return packaged.read_text(encoding="utf-8")
