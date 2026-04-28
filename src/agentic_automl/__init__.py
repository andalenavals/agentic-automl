"""Agentic AutoML package.

Keep the package import lightweight so tooling can inspect the package
without immediately importing the full orchestration stack.
"""

from __future__ import annotations

from typing import Any


def plan_project(*args: Any, **kwargs: Any):
    from .orchestrator import plan_project as _plan_project

    return _plan_project(*args, **kwargs)


def execute_workflow(*args: Any, **kwargs: Any):
    from .orchestrator import execute_workflow as _execute_workflow

    return _execute_workflow(*args, **kwargs)


__all__ = ["execute_workflow", "plan_project"]
