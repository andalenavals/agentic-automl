from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .orchestrator import execute_workflow, plan_project
from .schemas import ProjectBrief
from .workflow import get_workflow_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic AutoML CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("workflow", help="Print the fixed automl_workflow.md")

    plan_parser = subparsers.add_parser("plan", help="Profile data and print workflow recommendations")
    add_project_arguments(plan_parser)

    run_parser = subparsers.add_parser("run", help="Execute the AutoML workflow and export a single rerunnable workflow notebook")
    add_project_arguments(run_parser)
    run_parser.add_argument("--output-dir", help="Optional explicit output directory or `.ipynb` path for the generated workflow notebook.")

    subparsers.add_parser("ui", help="Launch the Streamlit chat UI")
    return parser


def add_project_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task-type", required=True, choices=["classification", "regression"])
    parser.add_argument("--problem", required=True)
    parser.add_argument("--date-column")
    parser.add_argument("--baseline-metric")
    parser.add_argument("--competition", action="store_true")


def build_brief(args: argparse.Namespace) -> ProjectBrief:
    return ProjectBrief(
        project_name=args.project_name,
        dataset_path=args.dataset,
        target_column=args.target,
        task_type=args.task_type,
        problem_description=args.problem,
        date_column=args.date_column,
        baseline_metric=args.baseline_metric,
        competition_enabled=args.competition,
    )


def launch_ui() -> int:
    script_path = Path(__file__).resolve().with_name("ui_app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(script_path)])


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "workflow":
        print(get_workflow_markdown())
        return 0

    if args.command == "ui":
        return launch_ui()

    brief = build_brief(args)
    if args.command == "plan":
        _, profile, recommendations = plan_project(brief)
        print(f"Profile: {profile.to_dict()}")
        for recommendation in recommendations:
            print(f"{recommendation.step_number}. {recommendation.title}: {recommendation.selected_option}")
            print(f"   {recommendation.recommendation}")
        return 0

    artifacts = execute_workflow(brief, output_root=args.output_dir)
    if artifacts.output_notebook_path:
        print(f"Notebook ready at: {artifacts.output_notebook_path}")
    print(f"Winning model: {artifacts.winner}")
    print(f"Selected metric: {artifacts.selected_metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
