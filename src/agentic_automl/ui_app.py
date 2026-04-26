from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.express as px
import streamlit as st

from .memory import load_all_memories
from .orchestrator import execute_workflow, plan_project
from .paths import resolve_repo_root
from .schemas import ProjectBrief
from .workflow import STEP_DEFINITIONS, get_workflow_markdown


STARTER_MESSAGE = """Explain the project in a short structured brief:

project_name: customer-churn-mvp
dataset_path: /absolute/path/to/data.csv
target_column: churned
task_type: classification
problem_description: predict which customers are likely to churn next month
competition_enabled: yes
date_column:
"""


def parse_brief(text: str):
    payload: dict[str, str] = {}
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        payload[key.strip()] = value.strip()

    required = ["project_name", "dataset_path", "target_column", "task_type", "problem_description"]
    missing = [field for field in required if not payload.get(field)]
    if missing:
        return None, missing

    task_type = payload["task_type"].lower()
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be classification or regression.")

    return ProjectBrief(
        project_name=payload["project_name"],
        dataset_path=payload["dataset_path"],
        target_column=payload["target_column"],
        task_type=task_type,
        problem_description=payload["problem_description"],
        date_column=payload.get("date_column") or None,
        baseline_metric=payload.get("baseline_metric") or None,
        competition_enabled=payload.get("competition_enabled", "no").lower() in {"yes", "true", "1"},
    ), []


def initialize_state() -> None:
    st.session_state.setdefault("messages", [{"role": "assistant", "content": STARTER_MESSAGE}])
    st.session_state.setdefault("brief", None)
    st.session_state.setdefault("profile", None)
    st.session_state.setdefault("recommendations", [])
    st.session_state.setdefault("selected_options", {})
    st.session_state.setdefault("artifacts", None)


def render_sidebar(repo_root) -> None:
    st.sidebar.header("Workflow status")
    for step in STEP_DEFINITIONS:
        selected = st.session_state.get("selected_options", {}).get(step.step_id)
        st.sidebar.write(f"{step.step_number}. {step.title} -> {selected or 'pending'}")

    st.sidebar.divider()
    st.sidebar.caption(f"Repository root: `{repo_root}`")
    if st.session_state.get("artifacts"):
        st.sidebar.success(f"Bundle ready: `{st.session_state['artifacts']['bundle_dir']}`")


def render_chat() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("Paste the project brief or update the workflow choices.")
    if not user_message:
        return

    st.session_state["messages"].append({"role": "user", "content": user_message})
    try:
        brief, missing = parse_brief(user_message)
    except ValueError as error:
        st.session_state["messages"].append({"role": "assistant", "content": str(error)})
        st.rerun()
        return

    if brief is None:
        st.session_state["messages"].append({"role": "assistant", "content": "I still need: " + ", ".join(missing)})
        st.rerun()
        return

    _, profile, recommendations = plan_project(brief)
    st.session_state["brief"] = brief
    st.session_state["profile"] = asdict(profile)
    st.session_state["recommendations"] = [item.to_dict() for item in recommendations]
    st.session_state["selected_options"] = {item.step_id: item.selected_option for item in recommendations if item.selected_option}
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": (
                f"Project loaded for `{brief.project_name}`.\n\n"
                f"- Dataset: `{brief.dataset_path}`\n"
                f"- Target: `{brief.target_column}`\n"
                f"- Task type: `{brief.task_type}`\n\n"
                "I generated recommendations for every workflow step. Review them in the Workflow tab, then run the bundle build."
            ),
        }
    )
    st.rerun()


def render_workflow_tab(repo_root) -> None:
    brief = st.session_state.get("brief")
    recommendations = st.session_state.get("recommendations") or []
    if not brief or not recommendations:
        st.info("Load a project brief in the chat first.")
        return

    profile = st.session_state.get("profile") or {}
    st.subheader("Dataset profile")
    st.dataframe(
        pd.DataFrame(
            [
                {"metric": "rows", "value": profile.get("rows")},
                {"metric": "columns", "value": profile.get("columns")},
                {"metric": "numeric_features", "value": len(profile.get("numeric_features", []))},
                {"metric": "categorical_features", "value": len(profile.get("categorical_features", []))},
                {"metric": "missing_fraction", "value": profile.get("missing_fraction")},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    memories = load_all_memories(repo_root)
    for recommendation in recommendations:
        step_id = recommendation["step_id"]
        with st.expander(f"{recommendation['step_number']}. {recommendation['title']}", expanded=step_id == "01_preprocessing"):
            st.markdown(recommendation["recommendation"])
            st.write("Reasoning")
            for line in recommendation["reasoning"]:
                st.write(f"- {line}")
            options = recommendation.get("options", [])
            selected = recommendation.get("selected_option")
            if options:
                index = options.index(selected) if selected in options else 0
                st.session_state["selected_options"][step_id] = st.selectbox(
                    f"Choose the policy for {recommendation['title'].lower()}",
                    options=options,
                    index=index,
                    key=f"select_{step_id}",
                )
            st.caption("Memory")
            st.code(memories.get(step_id, "No memory found."), language="markdown")


def render_results_tab() -> None:
    artifacts = st.session_state.get("artifacts")
    if not artifacts:
        st.info("Run the workflow to generate leaderboard, bundle artifacts, and visual summaries.")
        return

    summary = artifacts["metrics_summary"]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Winner", artifacts["winner"])
    col_b.metric(f"Winner {artifacts['selected_metric']}", f"{summary['winner_test_metric']:.4f}")
    col_c.metric(f"Baseline {artifacts['selected_metric']}", f"{summary['baseline_test_metric']:.4f}")

    leaderboard = pd.DataFrame(
        [
            {
                "model_name": row["model_name"],
                "role": row["role"],
                "cv_metric": row["cv_metrics"].get(artifacts["selected_metric"]),
                "test_metric": row["test_metrics"].get(artifacts["selected_metric"]),
            }
            for row in artifacts["leaderboard"]
        ]
    )
    st.plotly_chart(
        px.bar(leaderboard, x="model_name", y="test_metric", color="role", title=f"Test-set {artifacts['selected_metric']} by model"),
        use_container_width=True,
    )
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    st.subheader("Prediction preview")
    st.dataframe(pd.DataFrame(artifacts["predictions_preview"]), use_container_width=True, hide_index=True)
    st.subheader("Report")
    st.markdown(artifacts["report_markdown"])


def run_app() -> None:
    st.set_page_config(page_title="Agentic AutoML", layout="wide")
    initialize_state()
    repo_root = resolve_repo_root()

    st.title("Agentic AutoML")
    st.caption("A chat-led AutoML workspace for Codex, Claude Code, OpenClaw, and similar AI operators.")
    render_sidebar(repo_root)

    chat_tab, workflow_tab, results_tab, markdown_tab = st.tabs(["Chat", "Workflow", "Results", "automl_workflow.md"])
    with chat_tab:
        render_chat()
    with workflow_tab:
        render_workflow_tab(repo_root)
        if st.session_state.get("brief") and st.button("Run workflow and build bundle", type="primary"):
            with st.spinner("Training candidates, comparing the baseline, and exporting the bundle..."):
                st.session_state["artifacts"] = execute_workflow(
                    brief=st.session_state["brief"],
                    selected_options=st.session_state["selected_options"],
                ).to_dict()
            st.rerun()
    with results_tab:
        render_results_tab()
    with markdown_tab:
        st.code(get_workflow_markdown(repo_root), language="markdown")


if __name__ == "__main__":
    run_app()
