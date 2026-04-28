from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT_ENV = "AGENTIC_AUTOML_HOME"


def resolve_repo_root(explicit_root: str | Path | None = None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve()

    env_root = os.getenv(REPO_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "agentic_automl").exists():
            return parent
    return Path.cwd().resolve()
