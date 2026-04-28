from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit runs this file as a script, so we need the package root on sys.path
# when the app is launched via `streamlit run src/agentic_automl/ui_app.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_automl.orchestrator import (
    materialize_workflow_artifacts,
    plan_project,
    prepare_execution_context,
    hyperparameter_optimization_phase,
    train_model_phase,
    validation_phase,
)
from agentic_automl.paths import resolve_repo_root
from agentic_automl.preprocessing import (
    describe_preprocessing_execution_steps,
    extract_preprocessing_overrides_from_feedback,
)
from agentic_automl.schemas import DataProfile
from agentic_automl.ui_logic import (
    ACTION_HISTORY_KEY,
    AGREEMENT_LABELS,
    DISCUSSION_HISTORY_KEY,
    MAIN_CHAT_HEIGHT,
    REQUIRED_FIELDS,
    STARTER_MESSAGE,
    STEP_CHAT_HEIGHT,
    answer_contextual_step_question,
    answer_general_discussion_question,
    append_chat_history_entry,
    build_brief_from_fields,
    build_codex_discussion_activity,
    build_discussion_context,
    build_initial_step_feedback,
    build_policy_summary,
    build_step_discussion_context_block,
    build_workflow_step_feedback_payload,
    extract_brief_fields,
    find_step_for_custom_chat,
    format_missing_fields_message,
    infer_execution_option,
    is_question,
    maybe_call_local_codex_discussion,
    normalize_step_feedback_entry,
    parse_codex_exec_json_events,
    respond_to_discussion_message,
    respond_to_step_discussion,
    summarize_step_feedback,
    validate_step_feedback,
)
from agentic_automl.workflow import STEP_DEFINITIONS, get_workflow_markdown

__all__ = [
    "ACTION_HISTORY_KEY",
    "AGREEMENT_LABELS",
    "DISCUSSION_HISTORY_KEY",
    "MAIN_CHAT_HEIGHT",
    "REQUIRED_FIELDS",
    "STARTER_MESSAGE",
    "STEP_CHAT_HEIGHT",
    "answer_contextual_step_question",
    "answer_general_discussion_question",
    "append_chat_history_entry",
    "build_brief_from_fields",
    "build_codex_discussion_activity",
    "build_discussion_context",
    "build_initial_step_feedback",
    "build_policy_summary",
    "build_preprocessing_policy_details",
    "build_step_discussion_context_block",
    "build_workflow_step_feedback_payload",
    "extract_brief_fields",
    "find_step_for_custom_chat",
    "format_missing_fields_message",
    "infer_execution_option",
    "is_question",
    "maybe_call_local_codex_discussion",
    "normalize_step_feedback_entry",
    "parse_codex_exec_json_events",
    "respond_to_discussion_message",
    "respond_to_step_discussion",
    "summarize_step_feedback",
    "validate_step_feedback",
]


def build_preprocessing_policy_details(
    recommendation: dict[str, Any],
    profile_dict: dict[str, Any] | None,
    selected_option: str | None = None,
    feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    option = selected_option or recommendation.get("selected_option") or "auto_tabular_preprocessing"
    if not profile_dict:
        return {
            "selected_option": option,
            "steps": [],
        }

    profile = DataProfile(**profile_dict)
    preprocessing_overrides = extract_preprocessing_overrides_from_feedback(feedback)
    return {
        "selected_option": option,
        "steps": describe_preprocessing_execution_steps(profile, option, overrides=preprocessing_overrides),
    }


def build_model_policy_details(
    recommendation: dict[str, Any],
    selected_option: str | None = None,
    feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy_metadata = (feedback or {}).get("policy_metadata", {})
    recommendation_metadata = recommendation.get("metadata", {})
    return {
        "selected_option": selected_option or recommendation.get("selected_option") or "",
        "model_parameters": policy_metadata.get("model_parameters", recommendation_metadata.get("model_parameters", {})),
    }


def build_training_policy_details(
    recommendation: dict[str, Any],
    selected_option: str | None = None,
    feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy_metadata = (feedback or {}).get("policy_metadata", {})
    recommendation_metadata = recommendation.get("metadata", {})
    return {
        "selected_option": selected_option or recommendation.get("selected_option") or "",
        "training_config": policy_metadata.get(
            "training_config",
            recommendation_metadata.get("training_config", {}),
        ),
    }


def build_hpo_policy_details(
    recommendation: dict[str, Any],
    selected_option: str | None = None,
    feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy_metadata = (feedback or {}).get("policy_metadata", {})
    recommendation_metadata = recommendation.get("metadata", {})
    hpo_config = policy_metadata.get("hpo_config", {})
    return {
        "selected_option": selected_option or recommendation.get("selected_option") or "",
        "search_parameters": hpo_config.get(
            "search_parameters",
            recommendation_metadata.get("search_parameters", []),
        ),
        "search_space": hpo_config.get(
            "search_space",
            recommendation_metadata.get("search_space", {}),
        ),
    }


def format_preprocessing_steps_for_chat(details: dict[str, Any]) -> str:
    lines = ["Generated execution steps:"]
    for index, entry in enumerate(details.get("steps", []), start=1):
        lines.append(f"{index}. {entry['step']}")
        lines.append(f"   Why: {entry['why']}")
    return "\n".join(lines)


def format_key_value_lines(mapping: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in mapping.items():
        if value is None or value == "":
            continue
        lines.append(f"- `{key}`: `{value}`")
    return lines


def render_step_mode_selector(step_id: str) -> str:
    mode_key = f"step_mode_{step_id}"
    st.session_state.setdefault(mode_key, "Discussion")
    selector_options = ["Discussion", "Action"]
    if hasattr(st, "segmented_control"):
        selection = st.segmented_control(
            "Step chat mode",
            selector_options,
            key=mode_key,
            label_visibility="collapsed",
        )
        return selection or st.session_state.get(mode_key, "Discussion")

    return st.radio(
        "Step chat mode",
        selector_options,
        key=mode_key,
        horizontal=True,
        label_visibility="collapsed",
    )


def render_dataset_profile(profile: dict[str, Any]) -> None:
    rows_column, value_column = st.columns([2, 3])
    with rows_column:
        st.write("Rows")
    with value_column:
        st.write(profile.get("rows"))

    columns_column, columns_value = st.columns([2, 3])
    with columns_column:
        st.write("Columns")
    with columns_value:
        st.write(profile.get("columns"))

    numeric_features = profile.get("numeric_features", [])
    numeric_column, numeric_value, numeric_details = st.columns([2, 1, 3])
    with numeric_column:
        st.write("Numeric features")
    with numeric_value:
        st.write(len(numeric_features))
    with numeric_details:
        with st.expander("View numeric feature names", expanded=False):
            if numeric_features:
                for feature in numeric_features:
                    st.write(f"- `{feature}`")
            else:
                st.caption("No numeric features detected.")

    categorical_features = profile.get("categorical_features", [])
    categorical_column, categorical_value, categorical_details = st.columns([2, 1, 3])
    with categorical_column:
        st.write("Categorical features")
    with categorical_value:
        st.write(len(categorical_features))
    with categorical_details:
        with st.expander("View categorical feature names", expanded=False):
            if categorical_features:
                for feature in categorical_features:
                    st.write(f"- `{feature}`")
            else:
                st.caption("No categorical features detected.")

    missing_column, missing_value = st.columns([2, 3])
    with missing_column:
        st.write("Missing fraction")
    with missing_value:
        missing_fraction = profile.get("missing_fraction")
        st.write(f"{missing_fraction:.1%}" if isinstance(missing_fraction, (int, float)) else missing_fraction)


def apply_step_chat_message(step_id: str, user_message: str, mode: str) -> str:
    recommendation = next(item for item in st.session_state.get("recommendations", []) if item["step_id"] == step_id)
    feedback = normalize_step_feedback_entry(
        st.session_state["step_feedback"].setdefault(step_id, build_initial_step_feedback([recommendation])[step_id]),
        recommendation,
    )
    current_option = st.session_state["selected_options"].get(step_id, recommendation.get("selected_option") or "")
    if mode == "discussion":
        append_chat_history_entry(feedback, DISCUSSION_HISTORY_KEY, "user", user_message)
        codex_result = maybe_call_local_codex_discussion(
            user_message,
            feedback.get(DISCUSSION_HISTORY_KEY, [])[:-1],
            brief=st.session_state.get("brief"),
            recommendation=recommendation,
            recommendations=st.session_state.get("recommendations", []),
            step_feedback=st.session_state.get("step_feedback", {}),
            selected_options=st.session_state.get("selected_options", {}),
        )
        if codex_result:
            for item in codex_result.get("activity", []):
                append_chat_history_entry(feedback, DISCUSSION_HISTORY_KEY, item["role"], item["content"])
            append_chat_history_entry(feedback, DISCUSSION_HISTORY_KEY, "assistant", codex_result["reply"])
            return codex_result["reply"]

        reply = respond_to_discussion_message(
            recommendation,
            user_message,
            brief=st.session_state.get("brief"),
            profile=st.session_state.get("profile"),
            history=feedback.get(DISCUSSION_HISTORY_KEY, []),
            recommendations=st.session_state.get("recommendations", []),
            step_feedback=st.session_state.get("step_feedback", {}),
            selected_options=st.session_state.get("selected_options", {}),
            allow_local_codex=False,
        )
        append_chat_history_entry(feedback, DISCUSSION_HISTORY_KEY, "assistant", reply)
        return reply

    response = respond_to_step_discussion(
        recommendation,
        user_message,
        current_option=current_option,
        profile=st.session_state.get("profile"),
        selected_options=st.session_state.get("selected_options", {}),
        current_policy_metadata=feedback.get("policy_metadata", {}),
    )
    assistant_reply = response["reply"]
    if response["updated_policy"]:
        st.session_state["selected_options"][step_id] = response["selected_option"]
        feedback["custom_note"] = response["custom_note"]
        feedback["policy_summary"] = response["policy_summary"]
        feedback["policy_metadata"] = response.get("policy_metadata", {})
        feedback["policy_confirmed"] = False
        if step_id == "01_preprocessing":
            details = build_preprocessing_policy_details(
                recommendation,
                st.session_state.get("profile"),
                selected_option=response["selected_option"],
                feedback=feedback,
            )
            assistant_reply = response["reply"] + "\n\n" + format_preprocessing_steps_for_chat(details)
    append_chat_history_entry(feedback, ACTION_HISTORY_KEY, "user", user_message)
    append_chat_history_entry(feedback, ACTION_HISTORY_KEY, "assistant", assistant_reply)
    return assistant_reply


def answer_step_follow_up_in_chat(user_message: str) -> str | None:
    recommendations = st.session_state.get("recommendations", [])
    if not recommendations:
        return None

    step_id = find_step_for_custom_chat(
        user_message,
        recommendations,
        st.session_state.get("step_feedback", {}),
    )
    if not step_id:
        return None

    feedback = st.session_state.get("step_feedback", {}).get(step_id, {})
    recommendation = next(item for item in recommendations if item["step_id"] == step_id)
    if feedback.get("agreement") != "different":
        return (
            f"I can help with {recommendation['title'].lower()}.\n\n"
            "Switch that step to `I want something different` in `Workflow Builder`, and then ask your question or describe the alternative. "
            "I will turn it into a working execution policy."
        )
    return apply_step_chat_message(step_id, user_message, mode="discussion")


def answer_follow_up(user_message: str) -> str:
    lowered = user_message.lower()
    selected_hpo = st.session_state.get("selected_options", {}).get("07_hyperparameter_optimization", "skip")
    training_ready = bool(st.session_state.get("training_state"))
    validation_ready = bool(st.session_state.get("validation_state"))
    optimization_ready = bool(st.session_state.get("optimization_state"))
    final_validation_ready = bool(st.session_state.get("final_validation_state"))
    artifacts_ready = bool(st.session_state.get("artifacts"))

    if artifacts_ready and any(token in lowered for token in ["what do i do now", "what next", "next", "now what"]):
        return (
            "The output notebook is already built.\n\n"
            "Next you can:\n"
            "1. Review the Results tab.\n"
            "2. Inspect the exported notebook path in the sidebar.\n"
            "3. Start a new project or update this brief if you want another run."
        )

    if any(token in lowered for token in ["what do i do now", "what next", "next step", "now what"]):
        if not training_ready:
            return (
                "You are ready to start execution.\n\n"
                "1. Finish reviewing the workflow steps in `Workflow Builder`.\n"
                "2. Click `Start training`.\n"
                "3. Then run validation, and optionally build the current notebook output or continue to tuning."
            )
        if training_ready and not validation_ready:
            return (
                "Training is complete.\n\n"
                "The next step is to click `Run validation` so we can compare the trained model against the no-model baseline."
            )
        if validation_ready and selected_hpo == "skip" and not final_validation_ready:
            return (
                "Validation is ready.\n\n"
                "You can now either:\n"
                "1. Build the notebook output from the current validated model.\n"
                "2. Run final validation to prepare the final dashboard story."
            )
        if validation_ready and selected_hpo != "skip" and not optimization_ready:
            return (
                "Initial validation is ready.\n\n"
                "You can now either:\n"
                "1. Build the notebook output from the current validated model.\n"
                "2. Run hyperparameter optimization before the final validation step."
            )
        if optimization_ready and not final_validation_ready:
            return (
                "Hyperparameter optimization is complete.\n\n"
                "The next step is `Run final validation` so we can compare the tuned model against the baseline and prepare the final export story."
            )
        if final_validation_ready and not artifacts_ready:
            return (
                "Final validation is ready.\n\n"
                "If you like the result, click `Build notebook output from final validated model`."
            )

    if "workflow tab" in lowered or "where is the workflow" in lowered:
        return (
            "The step-by-step flow is in the `Workflow Builder` tab.\n\n"
            "That is where you review defaults, customize steps, and run training, validation, tuning, and export."
        )

    if "run the bundle" in lowered or "bundle build" in lowered or "how do i run" in lowered:
        return (
            "The output build happens after validation.\n\n"
            "Use the buttons in `Workflow Builder`:\n"
            "1. `Start training`\n"
            "2. `Run validation`\n"
            "3. Then choose `Build notebook output from current validated model` or continue to tuning and final validation first."
        )

    return (
        "I kept your current project brief.\n\n"
        "If you want to change it, send fields like `target_column: ...` or `dataset_path: ...`.\n"
        "If not, review the workflow steps and move into training, validation, and optional tuning from `Workflow Builder`."
    )


def parse_or_update_brief(text: str):
    extracted = extract_brief_fields(text)
    current_fields = dict(st.session_state.get("brief_fields", {}))
    if extracted:
        current_fields.update(extracted)
        st.session_state["brief_fields"] = current_fields
        return build_brief_from_fields(current_fields), current_fields, True

    missing = [field for field in REQUIRED_FIELDS if not current_fields.get(field)]
    return (None, missing), current_fields, False


def initialize_state() -> None:
    st.session_state.setdefault("messages", [{"role": "assistant", "content": STARTER_MESSAGE}])
    st.session_state.setdefault("brief", None)
    st.session_state.setdefault("brief_fields", {})
    st.session_state.setdefault("profile", None)
    st.session_state.setdefault("recommendations", [])
    st.session_state.setdefault("selected_options", {})
    st.session_state.setdefault("step_feedback", {})
    st.session_state.setdefault("execution_context", None)
    st.session_state.setdefault("execution_signature", None)
    st.session_state.setdefault("execution_notice", None)
    st.session_state.setdefault("training_state", None)
    st.session_state.setdefault("validation_state", None)
    st.session_state.setdefault("optimization_state", None)
    st.session_state.setdefault("final_validation_state", None)
    st.session_state.setdefault("artifacts", None)


def clear_execution_state(*, preserve_notice: bool = False) -> None:
    st.session_state["execution_context"] = None
    st.session_state["execution_signature"] = None
    st.session_state["training_state"] = None
    st.session_state["validation_state"] = None
    st.session_state["optimization_state"] = None
    st.session_state["final_validation_state"] = None
    st.session_state["artifacts"] = None
    if not preserve_notice:
        st.session_state["execution_notice"] = None


def build_workflow_signature() -> str:
    brief = st.session_state.get("brief")
    payload = build_workflow_step_feedback_payload(
        st.session_state.get("recommendations", []),
        st.session_state.get("step_feedback", {}),
        st.session_state.get("selected_options", {}),
    )
    return json.dumps(
        {
            "brief": brief.to_dict() if brief else None,
            "selected_options": st.session_state.get("selected_options", {}),
            "step_feedback": payload,
        },
        sort_keys=True,
        default=str,
    )


def sync_execution_state_with_workflow() -> None:
    if not st.session_state.get("brief"):
        return
    current_signature = build_workflow_signature()
    executed_signature = st.session_state.get("execution_signature")
    if executed_signature and executed_signature != current_signature:
        clear_execution_state(preserve_notice=True)
        st.session_state["execution_notice"] = (
            "Workflow changes cleared the previous training, validation, tuning, and export results."
        )


def render_chat_window(messages: list[dict[str, str]], height: int) -> None:
    with st.container(height=height, border=True):
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def inject_step_chat_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stFormSubmitButton"] {
            display: none;
        }
        div[data-testid="InputInstructions"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_step_chat_input(
    *,
    form_key: str,
    input_key: str,
    placeholder: str,
) -> str | None:
    with st.form(form_key, clear_on_submit=True):
        user_message = st.text_input(
            "",
            key=input_key,
            placeholder=placeholder,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send")
    if not submitted:
        return None
    return user_message.strip() or None


def render_sidebar(repo_root: Path) -> None:
    st.sidebar.header("Workflow status")
    for step in STEP_DEFINITIONS:
        selected = st.session_state.get("selected_options", {}).get(step.step_id)
        status = summarize_step_feedback(st.session_state.get("step_feedback", {}), step.step_id)
        st.sidebar.write(f"{step.step_number}. {step.title} -> {selected or 'pending'} ({status})")

    st.sidebar.divider()
    st.sidebar.write(f"Training: {'done' if st.session_state.get('training_state') else 'pending'}")
    st.sidebar.write(f"Validation: {'done' if st.session_state.get('validation_state') else 'pending'}")
    hpo_option = st.session_state.get("selected_options", {}).get("07_hyperparameter_optimization", "skip")
    if hpo_option == "skip":
        st.sidebar.write("Hyperparameter optimization: skipped")
    else:
        st.sidebar.write(
            f"Hyperparameter optimization: {'done' if st.session_state.get('optimization_state') else 'pending'}"
        )
    st.sidebar.write(
        f"Final validation: {'done' if st.session_state.get('final_validation_state') else 'pending'}"
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Repository root: `{repo_root}`")
    artifacts = st.session_state.get("artifacts")
    if artifacts and artifacts.get("output_notebook_path"):
        st.sidebar.success(f"Workflow notebook: `{artifacts['output_notebook_path']}`")


def render_chat() -> None:
    st.caption("You can provide the brief gradually. The chat will keep what you already told me.")
    render_chat_window(st.session_state["messages"], MAIN_CHAT_HEIGHT)

    user_message = st.chat_input("Paste the project brief or update the workflow choices.")
    if not user_message:
        return

    st.session_state["messages"].append({"role": "user", "content": user_message})
    try:
        (brief, missing), current_fields, had_brief_fields = parse_or_update_brief(user_message)
    except ValueError as error:
        st.session_state["messages"].append({"role": "assistant", "content": str(error)})
        st.rerun()
        return

    if brief is None and st.session_state.get("brief") is not None:
        step_reply = answer_step_follow_up_in_chat(user_message)
        st.session_state["messages"].append(
            {"role": "assistant", "content": step_reply or answer_follow_up(user_message)}
        )
        st.rerun()
        return

    if brief is None:
        if not had_brief_fields and is_question(user_message):
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "You can give me the project brief gradually.\n\n"
                        "To start the workflow I need:\n"
                        "- dataset_path\n"
                        "- target_column\n"
                        "- task_type\n"
                        "- problem_description\n\n"
                        "Example:\n"
                        "`dataset_path: /path/to/data.csv`"
                    ),
                }
            )
            st.rerun()
            return
        st.session_state["messages"].append(
            {"role": "assistant", "content": format_missing_fields_message(current_fields, missing)}
        )
        st.rerun()
        return

    _, profile, recommendations = plan_project(brief)
    st.session_state["brief"] = brief
    st.session_state["brief_fields"] = {
        "project_name": brief.project_name,
        "dataset_path": brief.dataset_path,
        "target_column": brief.target_column,
        "task_type": brief.task_type,
        "problem_description": brief.problem_description,
        "date_column": brief.date_column or "",
        "baseline_metric": brief.baseline_metric or "",
        "competition_enabled": "yes" if brief.competition_enabled else "no",
    }
    st.session_state["profile"] = asdict(profile)
    st.session_state["recommendations"] = [item.to_dict() for item in recommendations]
    st.session_state["selected_options"] = {
        item.step_id: item.selected_option
        for item in recommendations
        if item.selected_option
    }
    st.session_state["step_feedback"] = build_initial_step_feedback(st.session_state["recommendations"])
    clear_execution_state()
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": (
                f"Project loaded for `{brief.project_name}`.\n\n"
                f"- Dataset: `{brief.dataset_path}`\n"
                f"- Target: `{brief.target_column}`\n"
                f"- Task type: `{brief.task_type}`\n\n"
                "I generated a step-by-step workflow.\n\n"
                "Next:\n"
                "1. Open `Workflow Builder`.\n"
                "2. Review each workflow step and either agree or customize it.\n"
                "3. Click `Start training`.\n"
                "4. Then run validation, optional hyperparameter optimization, final validation, and notebook output build."
            ),
        }
    )
    st.rerun()


def build_current_feedback_payload() -> dict[str, dict[str, Any]]:
    return build_workflow_step_feedback_payload(
        st.session_state.get("recommendations", []),
        st.session_state.get("step_feedback", {}),
        st.session_state.get("selected_options", {}),
    )


def prepare_ui_execution_context() -> dict[str, Any]:
    context = prepare_execution_context(
        st.session_state["brief"],
        selected_options=st.session_state.get("selected_options", {}),
        step_feedback=build_current_feedback_payload(),
    )
    st.session_state["execution_context"] = context
    return context


def run_training_from_ui() -> bool:
    issues = validate_step_feedback(
        st.session_state.get("recommendations", []),
        st.session_state.get("step_feedback", {}),
        st.session_state.get("selected_options", {}),
    )
    if issues:
        st.warning("Please review every workflow step before starting training:")
        for issue in issues:
            st.write(f"- {issue}")
        return False

    progress = st.progress(0, text="Preparing the execution context...")
    context = prepare_ui_execution_context()
    progress.progress(30, text="Training the selected model...")
    training_state = train_model_phase(context)
    progress.progress(100, text="Training complete.")

    st.session_state["execution_signature"] = build_workflow_signature()
    st.session_state["training_state"] = training_state
    st.session_state["validation_state"] = None
    st.session_state["optimization_state"] = None
    st.session_state["final_validation_state"] = None
    st.session_state["artifacts"] = None
    return True


def run_validation_from_ui() -> bool:
    training_state = st.session_state.get("training_state")
    if not training_state:
        st.warning("Start training before running validation.")
        return False

    progress = st.progress(0, text="Loading the trained model state...")
    context = st.session_state.get("execution_context") or prepare_ui_execution_context()
    progress.progress(45, text="Comparing the trained model against the no-model baseline...")
    validation_state = validation_phase(context, training_state, role="candidate", tuned=False)
    progress.progress(100, text="Validation complete.")

    st.session_state["execution_signature"] = build_workflow_signature()
    st.session_state["validation_state"] = validation_state
    st.session_state["optimization_state"] = None
    st.session_state["final_validation_state"] = None
    st.session_state["artifacts"] = None
    return True


def run_hpo_from_ui() -> bool:
    training_state = st.session_state.get("training_state")
    if not training_state:
        st.warning("Start training before hyperparameter optimization.")
        return False

    context = st.session_state.get("execution_context") or prepare_ui_execution_context()
    if context["config"]["hpo_option"] == "skip":
        st.info("The current workflow is configured to skip hyperparameter optimization.")
        return False

    progress = st.progress(0, text="Preparing the tuning competition...")
    progress.progress(35, text="Running the hyperparameter search...")
    optimization_state = hyperparameter_optimization_phase(context, training_state)
    progress.progress(100, text="Hyperparameter optimization complete.")

    st.session_state["execution_signature"] = build_workflow_signature()
    st.session_state["optimization_state"] = optimization_state
    st.session_state["final_validation_state"] = None
    st.session_state["artifacts"] = None
    return optimization_state is not None


def run_final_validation_from_ui() -> bool:
    training_state = st.session_state.get("training_state")
    validation_state = st.session_state.get("validation_state")
    if not training_state or not validation_state:
        st.warning("Run training and validation before the final validation step.")
        return False

    context = st.session_state.get("execution_context") or prepare_ui_execution_context()
    optimization_state = st.session_state.get("optimization_state")
    source_state = optimization_state or training_state
    tuned = optimization_state is not None
    role = "competition" if tuned else "candidate"

    progress = st.progress(0, text="Preparing the final validation view...")
    progress.progress(40, text="Revalidating the final model against the baseline...")
    final_validation_state = validation_phase(context, source_state, role=role, tuned=tuned)
    progress.progress(100, text="Final validation complete.")

    st.session_state["execution_signature"] = build_workflow_signature()
    st.session_state["final_validation_state"] = final_validation_state
    st.session_state["artifacts"] = None
    return True


def build_output_notebook_from_ui(*, use_final_model: bool) -> bool:
    training_state = st.session_state.get("training_state")
    validation_state = st.session_state.get("validation_state")
    if not training_state or not validation_state:
        st.warning("Run training and validation before building the notebook output.")
        return False

    final_state = st.session_state.get("final_validation_state") if use_final_model else validation_state
    if use_final_model and not final_state:
        st.warning("Run final validation before building the final notebook output.")
        return False

    progress = st.progress(0, text="Preparing the workflow notebook export...")
    context = st.session_state.get("execution_context") or prepare_ui_execution_context()
    optimization_state = st.session_state.get("optimization_state") if use_final_model else None
    progress.progress(50, text="Writing the single-file workflow notebook...")
    artifacts = materialize_workflow_artifacts(
        context,
        training_state,
        validation_state,
        optimization_state=optimization_state,
        final_state=final_state,
        build_output_notebook=True,
    )
    progress.progress(100, text="Workflow notebook built.")

    st.session_state["execution_signature"] = build_workflow_signature()
    st.session_state["artifacts"] = artifacts.to_dict()
    return True


def render_mapping_block(title: str, mapping: dict[str, Any]) -> None:
    lines = format_key_value_lines(mapping)
    st.write(title)
    if not lines:
        st.caption("No explicit values for this section yet.")
        return
    for line in lines:
        st.write(line)


def render_step_policy_snapshot(
    recommendation: dict[str, Any],
    feedback: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    step_id = recommendation["step_id"]
    current_option = st.session_state.get("selected_options", {}).get(step_id, recommendation.get("selected_option") or "")

    if step_id == "01_preprocessing":
        details = build_preprocessing_policy_details(
            recommendation,
            profile,
            selected_option=current_option,
            feedback=feedback,
        )
        st.write(f"Recommended execution policy: `{details['selected_option']}`")
        st.write("Recommended execution policy steps")
        for entry in details["steps"]:
            st.write(f"- {entry['step']}")
            st.caption(f"Why: {entry['why']}")
        return

    if step_id == "03_model_selection":
        details = build_model_policy_details(recommendation, current_option, feedback)
        st.write(f"Recommended model: `{details['selected_option']}`")
        render_mapping_block("Recommended initial model parameters", details["model_parameters"])
        return

    if step_id == "05_metric_selection":
        st.write(f"Recommended primary metric: `{current_option}`")
        for line in recommendation.get("reasoning", []):
            st.write(f"- {line}")
        return

    if step_id == "06_training_configuration":
        details = build_training_policy_details(recommendation, current_option, feedback)
        st.write(f"Recommended training policy: `{details['selected_option']}`")
        render_mapping_block("Recommended training parameters", details["training_config"])
        return

    if step_id == "07_hyperparameter_optimization":
        details = build_hpo_policy_details(recommendation, current_option, feedback)
        st.write(f"Recommended HPO policy: `{details['selected_option']}`")
        if details["search_parameters"]:
            st.write("Hyperparameters to optimize")
            for item in details["search_parameters"]:
                st.write(f"- `{item}`")
        else:
            st.caption("The packaged search scope will follow the selected model if this step is enabled.")
        for line in recommendation.get("reasoning", []):
            st.write(f"- {line}")
        return

    if step_id == "08_validation_and_baseline":
        st.write(f"Recommended validation flow: `{current_option}`")
        baseline_strategy = recommendation.get("metadata", {}).get("baseline_strategy")
        if baseline_strategy:
            st.write(f"- Baseline strategy: `{baseline_strategy}`")
        for line in recommendation.get("reasoning", []):
            st.write(f"- {line}")
        return

    if step_id == "09_final_validation":
        st.write(f"Recommended final validation flow: `{current_option}`")
        for line in recommendation.get("reasoning", []):
            st.write(f"- {line}")
        return

    st.markdown(recommendation["recommendation"])
    st.write("Reasoning")
    for line in recommendation["reasoning"]:
        st.write(f"- {line}")


def render_action_policy_summary(
    recommendation: dict[str, Any],
    feedback: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    step_id = recommendation["step_id"]
    current_option = st.session_state["selected_options"].get(step_id, recommendation.get("selected_option") or "")

    if step_id == "01_preprocessing":
        details = build_preprocessing_policy_details(
            recommendation,
            profile,
            selected_option=current_option,
            feedback=feedback,
        )
        st.write("Final recommended execution steps")
        for entry in details["steps"]:
            st.write(f"- {entry['step']}")
            st.caption(f"Why: {entry['why']}")
        return

    if step_id == "03_model_selection":
        details = build_model_policy_details(recommendation, current_option, feedback)
        st.write(f"- Selected model: `{details['selected_option']}`")
        for line in format_key_value_lines(details["model_parameters"]):
            st.write(line)
        return

    if step_id == "06_training_configuration":
        details = build_training_policy_details(recommendation, current_option, feedback)
        st.write(f"- Training policy: `{details['selected_option']}`")
        for line in format_key_value_lines(details["training_config"]):
            st.write(line)
        return

    if step_id == "07_hyperparameter_optimization":
        details = build_hpo_policy_details(recommendation, current_option, feedback)
        st.write(f"- HPO policy: `{details['selected_option']}`")
        if details["search_parameters"]:
            st.write("- Search scope: " + ", ".join(f"`{item}`" for item in details["search_parameters"]))
        else:
            st.write("- Search scope: packaged default search for the selected model")
        return

    if step_id == "05_metric_selection":
        st.write(f"- Primary metric: `{current_option}`")
        return

    st.write(f"- Working execution policy: `{current_option}`")


def render_step_review_controls(recommendation: dict[str, Any], profile: dict[str, Any]) -> None:
    step_id = recommendation["step_id"]
    default_option = recommendation.get("selected_option") or ""
    feedback = normalize_step_feedback_entry(
        st.session_state["step_feedback"].setdefault(
            step_id,
            build_initial_step_feedback([recommendation])[step_id],
        ),
        recommendation,
    )
    agreement_options = list(AGREEMENT_LABELS.keys())
    current_agreement = feedback.get("agreement", "agree")
    if current_agreement not in agreement_options:
        current_agreement = "agree"

    feedback["agreement"] = st.selectbox(
        "Do you agree with the default procedure for this step?",
        options=agreement_options,
        format_func=lambda key: AGREEMENT_LABELS[key],
        index=agreement_options.index(current_agreement),
        key=f"agreement_{step_id}",
    )

    if feedback["agreement"] == "agree":
        st.session_state["selected_options"][step_id] = default_option
        feedback["custom_note"] = ""
        feedback["policy_summary"] = ""
        feedback["policy_metadata"] = {}
        feedback["policy_confirmed"] = True
        st.caption(f"Default execution policy: `{default_option}`")
        return

    if feedback["agreement"] == "different":
        if not feedback.get("custom_note") and not feedback.get("policy_summary"):
            feedback["policy_confirmed"] = False
        active_mode = render_step_mode_selector(step_id)

        if active_mode == "Discussion":
            st.caption("Open chat. This mode can use the intake and current workflow context, but it does not change the policy.")
            render_chat_window(feedback.setdefault(DISCUSSION_HISTORY_KEY, []), STEP_CHAT_HEIGHT)
            user_message = render_step_chat_input(
                form_key=f"step_discussion_form_{step_id}",
                input_key=f"step_discussion_input_{step_id}",
                placeholder="Chat with me here...",
            )
            if user_message:
                apply_step_chat_message(step_id, user_message, mode="discussion")
                st.rerun()
        else:
            st.caption("Describe a change and I will build the execution policy.")
            render_chat_window(feedback.setdefault(ACTION_HISTORY_KEY, []), STEP_CHAT_HEIGHT)
            user_message = render_step_chat_input(
                form_key=f"step_action_form_{step_id}",
                input_key=f"step_action_input_{step_id}",
                placeholder="Describe the new idea or policy change...",
            )
            if user_message:
                apply_step_chat_message(step_id, user_message, mode="action")
                st.rerun()

            if feedback.get("policy_summary"):
                st.success("Working policy ready")
                render_action_policy_summary(recommendation, feedback, profile)
                confirmation_label = (
                    "Policy confirmed" if feedback.get("policy_confirmed") else "Confirm this working policy"
                )
                if st.button(
                    confirmation_label,
                    key=f"confirm_policy_{step_id}",
                    disabled=bool(feedback.get("policy_confirmed")),
                    use_container_width=True,
                ):
                    feedback["policy_confirmed"] = True
                    st.rerun()

                if feedback.get("policy_confirmed"):
                    st.success("This custom policy is confirmed and ready to run.")
                else:
                    st.warning("Review the final policy and confirm it before running the workflow.")
            else:
                st.info("No custom working policy has been built for this step yet.")
        return

    st.session_state["selected_options"][step_id] = default_option


def render_validation_summary(title: str, state: dict[str, Any], task_type: str) -> None:
    summary = state["summary"]
    model_metrics = summary["model_metrics"]
    baseline_metrics = summary["baseline_metrics"]
    selected_metric = summary["selected_metric"]
    prediction_frame = state["prediction_frame"]

    st.subheader(title)
    card_a, card_b, card_c = st.columns(3)
    card_a.metric("Selected metric", selected_metric)
    card_b.metric("Model", f"{model_metrics[selected_metric]:.4f}")
    card_c.metric("Baseline", f"{baseline_metrics[selected_metric]:.4f}")
    st.caption(
        f"Baseline strategy: `{summary['baseline_strategy']}`. "
        f"Delta vs baseline on `{selected_metric}`: `{summary['metric_delta_vs_baseline']:.4f}`."
    )

    metric_rows: list[dict[str, Any]] = []
    for metric_name, metric_value in model_metrics.items():
        metric_rows.append({"metric": metric_name, "series": "Model", "value": metric_value})
    for metric_name, metric_value in baseline_metrics.items():
        metric_rows.append({"metric": metric_name, "series": "Baseline", "value": metric_value})
    metric_frame = pd.DataFrame(metric_rows)
    st.plotly_chart(
        px.bar(
            metric_frame,
            x="metric",
            y="value",
            color="series",
            barmode="group",
            title=f"{title} metrics",
        ),
        use_container_width=True,
    )

    if task_type == "classification":
        confusion = pd.crosstab(
            prediction_frame["actual"],
            prediction_frame["prediction"],
            rownames=["Actual"],
            colnames=["Predicted"],
        )
        st.plotly_chart(
            px.imshow(
                confusion,
                text_auto=True,
                aspect="auto",
                title=f"{title} confusion matrix",
            ),
            use_container_width=True,
        )
    else:
        st.plotly_chart(
            px.scatter(
                prediction_frame,
                x="actual",
                y="prediction",
                title=f"{title} actual vs prediction",
            ),
            use_container_width=True,
        )

    st.write("Prediction preview")
    st.dataframe(state["predictions_preview"], use_container_width=True, hide_index=True)


def render_execution_controls() -> None:
    issues = validate_step_feedback(
        st.session_state.get("recommendations", []),
        st.session_state.get("step_feedback", {}),
        st.session_state.get("selected_options", {}),
    )
    st.subheader("Execution")
    if issues:
        st.warning("Each workflow step needs an explicit review before execution can start.")
        return

    training_state = st.session_state.get("training_state")
    validation_state = st.session_state.get("validation_state")
    optimization_state = st.session_state.get("optimization_state")
    final_validation_state = st.session_state.get("final_validation_state")
    selected_hpo = st.session_state.get("selected_options", {}).get("07_hyperparameter_optimization", "skip")

    st.success("All workflow steps have been reviewed.")

    if not training_state:
        if st.button("Start training", type="primary", use_container_width=True):
            if run_training_from_ui():
                st.rerun()
        return

    st.success("Training is complete.")
    if not validation_state:
        if st.button("Run validation", type="primary", use_container_width=True):
            if run_validation_from_ui():
                st.rerun()
        return

    validation_col, build_current_col = st.columns(2)
    with validation_col:
        if selected_hpo != "skip" and not optimization_state:
            if st.button("Run hyperparameter optimization", use_container_width=True):
                if run_hpo_from_ui():
                    st.rerun()
        elif selected_hpo == "skip" and not final_validation_state:
            if st.button("Run final validation", use_container_width=True):
                if run_final_validation_from_ui():
                    st.rerun()
        elif optimization_state and not final_validation_state:
            if st.button("Run final validation", type="primary", use_container_width=True):
                if run_final_validation_from_ui():
                    st.rerun()
        else:
            st.caption("No additional execution step is pending here.")
    with build_current_col:
        if st.button("Build notebook output from current validated model", use_container_width=True):
            if build_output_notebook_from_ui(use_final_model=False):
                st.rerun()

    if final_validation_state:
        st.success("Final validation is complete.")
        if st.button(
            "Build notebook output from final validated model",
            type="primary",
            use_container_width=True,
        ):
            if build_output_notebook_from_ui(use_final_model=True):
                st.rerun()
    elif selected_hpo == "skip":
        st.caption("Final validation can still be run even when hyperparameter optimization is skipped.")


def render_workflow_tab() -> None:
    brief = st.session_state.get("brief")
    recommendations = st.session_state.get("recommendations") or []
    if not brief or not recommendations:
        st.info("Load a project brief in the chat first.")
        return

    sync_execution_state_with_workflow()
    notice = st.session_state.get("execution_notice")
    if notice:
        st.info(notice)
        st.session_state["execution_notice"] = None

    profile = st.session_state.get("profile") or {}
    st.subheader("Dataset profile")
    render_dataset_profile(profile)

    for recommendation in recommendations:
        step_id = recommendation["step_id"]
        current_feedback = st.session_state.get("step_feedback", {}).get(step_id, {})
        with st.expander(
            f"{recommendation['step_number']}. {recommendation['title']}",
            expanded=step_id == "01_preprocessing",
        ):
            render_step_policy_snapshot(recommendation, current_feedback, profile)
            render_step_review_controls(recommendation, profile)

    render_execution_controls()


def render_results_tab() -> None:
    brief = st.session_state.get("brief")
    if not brief:
        st.info("Load a project brief first.")
        return

    training_state = st.session_state.get("training_state")
    validation_state = st.session_state.get("validation_state")
    optimization_state = st.session_state.get("optimization_state")
    final_validation_state = st.session_state.get("final_validation_state")
    artifacts = st.session_state.get("artifacts")

    if not any([training_state, validation_state, optimization_state, final_validation_state, artifacts]):
        st.info("Use `Workflow Builder` to start training and move through validation, tuning, and export.")
        return

    if training_state:
        st.subheader("Training")
        summary = training_state["training_summary"]
        card_a, card_b, card_c = st.columns(3)
        card_a.metric("Selected model", summary["selected_model_label"])
        card_b.metric("Train rows", summary["train_rows"])
        card_c.metric("Holdout rows", summary["holdout_rows"])
        render_mapping_block("Model parameters", summary["model_parameters"])
        render_mapping_block("Training configuration", summary["training_configuration"])
        render_mapping_block("Cross-validation metrics", summary["cv_metrics"])

    if validation_state:
        render_validation_summary("Validation", validation_state, brief.task_type)

    if optimization_state:
        st.subheader("Hyperparameter optimization")
        summary = optimization_state["summary"]
        st.write(f"- Selected model: `{summary['selected_model']}`")
        st.write(f"- Best CV metric: `{summary['best_cv_metric']}`")
        if summary.get("best_params"):
            render_mapping_block("Best hyperparameters", summary["best_params"])
        search_rows = []
        for key, values in summary.get("search_space", {}).items():
            search_rows.append({"hyperparameter": key, "values": ", ".join(values)})
        if search_rows:
            st.write("Search space")
            st.dataframe(pd.DataFrame(search_rows), use_container_width=True, hide_index=True)

    if final_validation_state:
        render_validation_summary("Final validation", final_validation_state, brief.task_type)

    if artifacts:
        st.subheader("Output")
        output_a, output_b = st.columns(2)
        output_a.write(f"- Output directory: `{artifacts['project_dir']}`")
        if artifacts.get("output_notebook_path"):
            output_b.write(f"- Workflow notebook: `{artifacts['output_notebook_path']}`")
        output_b.write(f"- Winner: `{artifacts['winner']}`")
        st.markdown(artifacts["report_markdown"])


def run_app() -> None:
    st.set_page_config(page_title="Agentic AutoML", layout="wide")
    initialize_state()
    inject_step_chat_styles()
    repo_root = resolve_repo_root()

    st.title("Agentic AutoML")
    st.caption("A chat-led AutoML workspace for Codex, Claude Code, OpenClaw, and similar AI operators.")
    render_sidebar(repo_root)

    chat_tab, workflow_tab, results_tab, markdown_tab = st.tabs(
        ["Chat", "Workflow Builder", "Results", "Workflow File"]
    )
    with chat_tab:
        render_chat()
    with workflow_tab:
        render_workflow_tab()
    with results_tab:
        render_results_tab()
    with markdown_tab:
        st.code(get_workflow_markdown(repo_root), language="markdown")


if __name__ == "__main__":
    run_app()
