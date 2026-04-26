from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TaskType = Literal["classification", "regression"]


@dataclass(slots=True)
class ProjectBrief:
    project_name: str
    dataset_path: str
    target_column: str
    problem_description: str
    task_type: TaskType
    date_column: str | None = None
    baseline_metric: str | None = None
    competition_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DataProfile:
    rows: int
    columns: int
    numeric_features: list[str]
    categorical_features: list[str]
    missing_fraction: float
    target_cardinality: int
    target_name: str
    target_skew: float | None = None
    class_imbalance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowRecommendation:
    step_id: str
    step_number: int
    title: str
    recommendation: str
    reasoning: list[str]
    options: list[str] = field(default_factory=list)
    selected_option: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateResult:
    model_name: str
    role: str
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]
    params: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunArtifacts:
    project_dir: str
    bundle_dir: str
    selected_metric: str
    higher_is_better: bool
    winner: str
    baseline: CandidateResult
    leaderboard: list[CandidateResult]
    report_markdown: str
    metrics_summary: dict[str, float]
    predictions_preview: list[dict[str, Any]]
    workflow_decisions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_dir": self.project_dir,
            "bundle_dir": self.bundle_dir,
            "selected_metric": self.selected_metric,
            "higher_is_better": self.higher_is_better,
            "winner": self.winner,
            "baseline": self.baseline.to_dict(),
            "leaderboard": [entry.to_dict() for entry in self.leaderboard],
            "report_markdown": self.report_markdown,
            "metrics_summary": self.metrics_summary,
            "predictions_preview": self.predictions_preview,
            "workflow_decisions": self.workflow_decisions,
        }
