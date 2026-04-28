# Agentic AutoML

`agentic-automl` is an AutoML package built for AI operators rather than human click-through usage.

The intended user is a coding agent such as Codex, Claude Code, or OpenClaw. The human should only need to explain:

- where the dataset is
- what the target is
- what kind of problem should be solved
- what success means

The package then follows a fixed workflow, recommends each step, runs the workflow, and exports a single workflow notebook.

Feature pruning and feature-subset decisions are handled inside preprocessing, so there is no separate feature-selection step in the workflow.

Each workflow step now ships with:

- `SKILLS.md` for the reasoning/operating guide
- `KNOWLEDGE.md` for the currently supported executable actions and capability keys
- `LIMITS.md` for unsupported Action-mode requests and seed backlog items that should be implemented later

All Action-mode step agents follow the same philosophy:

- natural follow-up conversation
- awareness of previous context and policy selections
- awareness of limits and knowledge
- clear distinction between consultation and execution
- suggestions based on available policies and current customizations
- flexibility to revise policy multiple times in one conversation

## MVP Scope

- tabular CSV and Parquet datasets
- classification and regression
- fixed canonical `automl_workflow.md` under `src/agentic_automl/assets/`
- chat-style UI with workflow recommendations and visualizations
- optional compact hyperparameter competition
- final exported workflow notebook

## Structure

```text
agentic-automl/
├── docs/
├── examples/
├── projects/
├── src/agentic_automl/
│   └── assets/automl_workflow.md
│   └── assets/skills/
├── tests/
└── README.md
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
agentic-automl ui
```

## Documentation

This repo now ships with a small Sphinx documentation site under `docs/`.

Published docs:

- [https://andalenavals.github.io/agentic-automl/](https://andalenavals.github.io/agentic-automl/)

The site is deployed by the GitHub Pages workflow in `.github/workflows/github-pages.yml` and uses the Read the Docs Sphinx theme.

Build it locally with:

```bash
source .venv/bin/activate
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The main pages are:

- `docs/workflow.rst`
- `docs/action_agents.rst`
- `docs/output_notebook.rst`

Plan a project from the CLI:

```bash
agentic-automl plan \
  --project-name customer-churn-mvp \
  --dataset examples/customer_churn_demo.csv \
  --target churned \
  --task-type classification \
  --problem "Predict which customers are likely to churn next month."
```

Run a project end to end:

```bash
agentic-automl run \
  --project-name customer-churn-mvp \
  --dataset examples/customer_churn_demo.csv \
  --target churned \
  --task-type classification \
  --problem "Predict which customers are likely to churn next month." \
  --competition
```

## What Gets Exported

Each completed run exports one notebook file that includes:

- the workflow decisions
- metric summaries
- explanations of the selected choices
- plotting cells for the selected run
- an embedded standalone runtime copied into the notebook itself
- code cells that can rerun the full workflow from a dataset path resolved relative to the notebook location, without importing `agentic_automl`
- the final holdout prediction table regenerated in memory when the notebook is executed

The notebook is intentionally workflow-specific. It contains only the code
needed for the selected path and does not import `agentic_automl` when rerun.

## Why This Design

- project-specific data work can vary widely
- the AutoML workflow can stay standardized
- the UI and CLI both use the same workflow engine
