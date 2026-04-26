from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .advisor import build_recommendations
from .data import format_profile_summary, load_dataset, profile_dataset
from .memory import append_memory_entry, ensure_memory_layout, load_all_memories, slugify, write_project_memory
from .modeling import (
    baseline_estimator,
    build_pipeline,
    cross_validation_strategy,
    export_bundle,
    leaderboard_row,
    metric_configuration,
    model_catalog,
    pick_best_result,
    run_competition,
    selected_model_names,
    split_data,
)
from .schemas import CandidateResult, ProjectBrief, RunArtifacts, WorkflowRecommendation


def plan_project(brief: ProjectBrief, repo_root: str | Path | None = None):
    root = ensure_memory_layout(repo_root)
    frame = load_dataset(brief.dataset_path)
    profile = profile_dataset(frame, brief)
    recommendations = build_recommendations(brief, profile, load_all_memories(root))
    return frame, profile, recommendations


def execute_workflow(
    brief: ProjectBrief,
    selected_options: dict[str, str] | None = None,
    repo_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> RunArtifacts:
    root = ensure_memory_layout(repo_root)
    frame, profile, recommendations = plan_project(brief, root)
    selected_options = selected_options or {item.step_id: item.selected_option or "" for item in recommendations}
    for recommendation in recommendations:
        recommendation.selected_option = selected_options.get(recommendation.step_id, recommendation.selected_option)

    split_option = selected_options.get("02_data_splitting", "random_holdout")
    feature_option = selected_options.get("04_feature_selection", "skip")
    model_option = selected_options.get("03_model_selection", "balanced_candidate_set")
    selected_metric = selected_options.get("05_metric_selection", "f1_macro" if brief.task_type == "classification" else "rmse")
    training_option = selected_options.get("06_training_configuration", "standard_cv")
    hpo_option = selected_options.get("07_hyperparameter_optimization", "skip")

    X_train, X_test, y_train, y_test = split_data(frame, brief, split_option)
    scoring, higher_is_better = metric_configuration(brief.task_type, selected_metric)
    cv = cross_validation_strategy(brief.task_type, training_option)
    catalog = model_catalog(brief.task_type, profile.class_imbalance)

    baseline_pipeline = build_pipeline(profile, brief.task_type, "skip", baseline_estimator(brief.task_type))
    baseline_result = leaderboard_row(
        baseline_pipeline,
        "baseline_dummy",
        "baseline",
        scoring,
        brief.task_type,
        X_train,
        y_train,
        X_test,
        y_test,
        cv,
    )

    leaderboard: list[CandidateResult] = [baseline_result]
    trained_pipelines = {}
    for model_name in selected_model_names(model_option):
        pipeline = build_pipeline(profile, brief.task_type, feature_option, catalog[model_name])
        leaderboard.append(
            leaderboard_row(
                pipeline,
                model_name,
                "candidate",
                scoring,
                brief.task_type,
                X_train,
                y_train,
                X_test,
                y_test,
                cv,
            )
        )
        trained_pipelines[model_name] = pipeline.fit(X_train, y_train)

    if hpo_option != "skip":
        ranked = sorted(
            [row for row in leaderboard if row.role == "candidate"],
            key=lambda row: row.cv_metrics[selected_metric],
            reverse=higher_is_better,
        )
        leaderboard.extend(
            run_competition(
                shortlisted_names=[row.model_name for row in ranked[:2]],
                task_type=brief.task_type,
                profile=profile,
                feature_option=feature_option,
                scoring=scoring,
                selected_metric=selected_metric,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv=cv,
                imbalance_ratio=profile.class_imbalance,
            )
        )

    winner = pick_best_result([row for row in leaderboard if row.role != "baseline"], selected_metric, higher_is_better)
    trained_name = winner.model_name.replace("_competition", "")
    winning_pipeline = trained_pipelines.get(trained_name)
    if winner.role == "competition" or winning_pipeline is None:
        winning_pipeline = build_pipeline(profile, brief.task_type, feature_option, catalog[trained_name])
        if winner.role == "competition":
            winning_pipeline.set_params(**winner.params)
        winning_pipeline.fit(X_train, y_train)

    project_slug = slugify(brief.project_name)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    project_dir = Path(output_root or root / "projects" / f"{project_slug}-{now}").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    preview = pd.DataFrame({"actual": y_test.reset_index(drop=True), "prediction": pd.Series(winning_pipeline.predict(X_test)).reset_index(drop=True)}).head(25)
    decision_map = {item.step_id: selected_options.get(item.step_id, item.selected_option or "") for item in recommendations}
    metrics_summary = {
        "winner_cv_metric": winner.cv_metrics.get(selected_metric, float("nan")),
        "winner_test_metric": winner.test_metrics.get(selected_metric, float("nan")),
        "baseline_test_metric": baseline_result.test_metrics.get(selected_metric, float("nan")),
    }
    report_markdown = render_report(brief, profile, recommendations, leaderboard, winner, baseline_result, selected_metric)
    bundle_payload = {
        "brief": brief.to_dict(),
        "profile": profile.to_dict(),
        "workflow_decisions": decision_map,
        "leaderboard": [entry.to_dict() for entry in leaderboard],
        "generated_at": now,
    }
    bundle_dir, model_path = export_bundle(project_dir, winning_pipeline, bundle_payload)

    (project_dir / "report.md").write_text(report_markdown, encoding="utf-8")
    (project_dir / "workflow_decisions.json").write_text(json.dumps(decision_map, indent=2), encoding="utf-8")
    (project_dir / "leaderboard.json").write_text(json.dumps([entry.to_dict() for entry in leaderboard], indent=2), encoding="utf-8")
    preview.to_csv(project_dir / "predictions_preview.csv", index=False)
    (project_dir / "README.md").write_text(
        (
            f"# {brief.project_name}\n\n"
            f"- Dataset: `{brief.dataset_path}`\n"
            f"- Target: `{brief.target_column}`\n"
            f"- Selected metric: `{selected_metric}`\n"
            f"- Winning model: `{winner.model_name}`\n"
            f"- Bundle: `{bundle_dir}`\n"
            f"- Model artifact: `{model_path}`\n"
        ),
        encoding="utf-8",
    )

    for recommendation in recommendations:
        selected_option = selected_options.get(recommendation.step_id, recommendation.selected_option or "")
        append_memory_entry(
            recommendation.step_id,
            brief.project_name,
            recommendation.recommendation,
            selected_option,
            recommendation.reasoning,
            root,
        )
        write_project_memory(
            project_dir.name,
            recommendation.step_id,
            recommendation.title,
            recommendation.recommendation,
            selected_option,
            recommendation.reasoning,
            root,
        )

    return RunArtifacts(
        project_dir=str(project_dir),
        bundle_dir=bundle_dir,
        selected_metric=selected_metric,
        higher_is_better=higher_is_better,
        winner=winner.model_name,
        baseline=baseline_result,
        leaderboard=leaderboard,
        report_markdown=report_markdown,
        metrics_summary=metrics_summary,
        predictions_preview=preview.to_dict(orient="records"),
        workflow_decisions=decision_map,
    )


def render_report(
    brief: ProjectBrief,
    profile,
    recommendations: list[WorkflowRecommendation],
    leaderboard: list[CandidateResult],
    winner: CandidateResult,
    baseline: CandidateResult,
    selected_metric: str,
) -> str:
    lines = [
        f"# Agentic AutoML Report: {brief.project_name}",
        "",
        "## Project brief",
        f"- Dataset: `{brief.dataset_path}`",
        f"- Problem: {brief.problem_description}",
        f"- Task type: `{brief.task_type}`",
        f"- Target column: `{brief.target_column}`",
        "",
        "## Data profile",
        f"- {format_profile_summary(profile)}",
        "",
        "## Workflow decisions",
    ]
    for recommendation in recommendations:
        lines.append(f"- Step {recommendation.step_number}: {recommendation.title} -> `{recommendation.selected_option}` | {recommendation.recommendation}")

    lines.extend(["", "## Leaderboard"])
    for result in leaderboard:
        metric_value = result.test_metrics.get(selected_metric, result.cv_metrics.get(selected_metric, float('nan')))
        lines.append(f"- {result.model_name} ({result.role}) -> {selected_metric}: {metric_value:.4f}")

    winner_score = winner.test_metrics.get(selected_metric, winner.cv_metrics.get(selected_metric, float("nan")))
    baseline_score = baseline.test_metrics.get(selected_metric, baseline.cv_metrics.get(selected_metric, float("nan")))
    lines.extend(
        [
            "",
            "## Outcome",
            f"- Winning model: `{winner.model_name}`",
            f"- Winner {selected_metric}: {winner_score:.4f}",
            f"- Baseline {selected_metric}: {baseline_score:.4f}",
            "",
            "## Export bundle",
            "- The bundle includes the winning model, workflow decisions, leaderboard, and preview predictions.",
        ]
    )
    return "\n".join(lines) + "\n"
