from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path


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
    StepDefinition("05_metric_selection", 4, "Metric selection"),
    StepDefinition("06_training_configuration", 5, "Training"),
    StepDefinition("08_validation_and_baseline", 6, "Validation"),
    StepDefinition("07_hyperparameter_optimization", 7, "Hyperparameter optimization"),
    StepDefinition("09_final_validation", 9, "Final validation"),
]


def get_workflow_markdown(repo_root: str | Path | None = None) -> str:
    packaged = resources.files("agentic_automl.assets").joinpath("automl_workflow.md")
    return packaged.read_text(encoding="utf-8")
