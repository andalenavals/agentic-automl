from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from .paths import resolve_repo_root


STEP_SKILL_DIRECTORIES = {
    "00_intake": "00-intake",
    "01_preprocessing": "01-preprocessing",
    "02_data_splitting": "02-data-splitting",
    "03_model_selection": "03-model-selection",
    "05_metric_selection": "05-metric-selection",
    "06_training_configuration": "06-training-configuration",
    "07_hyperparameter_optimization": "07-hyperparameter-optimization",
    "08_validation_and_baseline": "08-validation-and-baseline",
    "09_final_validation": "09-final-validation",
}


def skills_root(repo_root: str | Path | None = None) -> Path:
    root = resolve_repo_root(repo_root)
    return root / "src" / "agentic_automl" / "assets" / "skills"


def skill_directory(step_id: str, repo_root: str | Path | None = None) -> Path:
    slug = STEP_SKILL_DIRECTORIES[step_id]
    return skills_root(repo_root) / slug


def knowledge_file(step_id: str, repo_root: str | Path | None = None) -> Path:
    return skill_directory(step_id, repo_root) / "KNOWLEDGE.md"


def limits_file(step_id: str, repo_root: str | Path | None = None) -> Path:
    return skill_directory(step_id, repo_root) / "LIMITS.md"


def ensure_limits_file(step_id: str, repo_root: str | Path | None = None) -> Path:
    path = limits_file(step_id, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "\n".join(
                [
                    f"# {STEP_SKILL_DIRECTORIES[step_id]} LIMITS",
                    "",
                    "This file stores unsupported step requests and seed backlog items.",
                    "",
                    "## Seed Backlog",
                    "",
                    "## Pending Requests",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return path


def append_limit_request(
    step_id: str,
    request: str,
    repo_root: str | Path | None = None,
) -> Path:
    path = ensure_limits_file(step_id, repo_root)
    cleaned_request = " ".join(request.strip().split())
    current = path.read_text(encoding="utf-8")
    if cleaned_request in current:
        return path

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    entry = "\n".join(
        [
            f"- [ ] {timestamp}",
            f"  Request: {cleaned_request}",
            "",
        ]
    )
    path.write_text(current + entry, encoding="utf-8")
    return path


CAPABILITY_HEADER = "## Capability Keys"
CAPABILITY_PATTERN = re.compile(r"^\s*-\s*`?([a-z0-9_.-]+)`?\s*:", re.IGNORECASE)
HEADING_PATTERN = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
LIST_ITEM_PATTERN = re.compile(r"^\s*-\s+(.*)$")


def load_capability_keys(
    step_id: str,
    repo_root: str | Path | None = None,
) -> set[str]:
    path = knowledge_file(step_id, repo_root)
    if not path.exists():
        return set()

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    in_capability_section = False
    capabilities: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if stripped == CAPABILITY_HEADER:
            in_capability_section = True
            continue
        if in_capability_section and stripped.startswith("## "):
            break
        if not in_capability_section:
            continue
        match = CAPABILITY_PATTERN.match(stripped)
        if match:
            capabilities.add(match.group(1))
    return capabilities


def capability_is_supported(
    step_id: str,
    capability_key: str,
    repo_root: str | Path | None = None,
) -> bool:
    return capability_key in load_capability_keys(step_id, repo_root)


def load_knowledge_sections(
    step_id: str,
    repo_root: str | Path | None = None,
) -> dict[str, list[str]]:
    path = knowledge_file(step_id, repo_root)
    if not path.exists():
        return {}

    sections: dict[str, list[str]] = {}
    current_heading: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            current_heading = heading_match.group(2).strip()
            sections.setdefault(current_heading, [])
            continue
        if current_heading is not None:
            sections[current_heading].append(line.rstrip())
    return sections


def section_items(
    step_id: str,
    heading: str,
    repo_root: str | Path | None = None,
) -> list[str]:
    sections = load_knowledge_sections(step_id, repo_root)
    lines = sections.get(heading, [])
    items: list[str] = []
    for line in lines:
        match = LIST_ITEM_PATTERN.match(line)
        if match:
            items.append(match.group(1).strip())
    return items


# Backward-compatible aliases for older local sessions.
todo_file = limits_file
ensure_todo_file = ensure_limits_file
append_todo_request = append_limit_request
