from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from .paths import resolve_repo_root
from .workflow import STEP_DEFINITIONS


def ensure_memory_layout(repo_root: str | Path | None = None) -> Path:
    root = resolve_repo_root(repo_root)
    (root / "projects").mkdir(parents=True, exist_ok=True)
    for step in STEP_DEFINITIONS:
        directory = root / "memory" / "global" / step.step_id
        directory.mkdir(parents=True, exist_ok=True)
        memory_file = directory / "step_memory.md"
        if not memory_file.exists():
            memory_file.write_text(
                f"# {step.title} Memory\n\nNo preferences recorded yet.\n",
                encoding="utf-8",
            )
    return root


def load_all_memories(repo_root: str | Path | None = None) -> dict[str, str]:
    root = ensure_memory_layout(repo_root)
    return {
        step.step_id: (root / "memory" / "global" / step.step_id / "step_memory.md").read_text(encoding="utf-8")
        for step in STEP_DEFINITIONS
    }


def slugify(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    compact = "-".join(part for part in normalized.split("-") if part)
    return compact or "project"


def append_memory_entry(
    step_id: str,
    project_name: str,
    recommendation: str,
    selected_option: str,
    reasoning: Iterable[str],
    repo_root: str | Path | None = None,
) -> None:
    root = ensure_memory_layout(repo_root)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    memory_file = root / "memory" / "global" / step_id / "step_memory.md"
    entry = (
        f"\n## {timestamp} | {project_name}\n"
        f"- Recommendation: {recommendation}\n"
        f"- Selected option: {selected_option}\n"
        f"- Reasoning: {' | '.join(reasoning)}\n"
    )
    with memory_file.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def write_project_memory(
    project_slug: str,
    step_id: str,
    title: str,
    recommendation: str,
    selected_option: str,
    reasoning: Iterable[str],
    repo_root: str | Path | None = None,
) -> None:
    root = ensure_memory_layout(repo_root)
    directory = root / "projects" / project_slug / "memory"
    directory.mkdir(parents=True, exist_ok=True)
    memory_file = directory / f"{step_id}.md"
    content = (
        f"# {title}\n\n"
        f"- Recommendation: {recommendation}\n"
        f"- Selected option: {selected_option}\n"
        + "\n".join(f"- {line}" for line in reasoning)
        + "\n"
    )
    memory_file.write_text(content, encoding="utf-8")
