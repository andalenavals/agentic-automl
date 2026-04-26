# Agentic AutoML

`agentic-automl` is an AutoML package built for AI operators rather than human click-through usage.

The intended user is a coding agent such as Codex, Claude Code, or OpenClaw. The human should only need to explain:

- where the dataset is
- what the target is
- what kind of problem should be solved
- what success means

The package then follows a fixed workflow, recommends each step, stores reusable memory, runs the workflow, and exports a winning model bundle with explainable artifacts.

## MVP Scope

- tabular CSV and Parquet datasets
- classification and regression
- fixed `automl_workflow.md`
- per-step memory under `memory/global/<step>/step_memory.md`
- chat-style UI with workflow recommendations and visualizations
- optional compact hyperparameter competition
- final exported winning-model bundle

## Structure

```text
agentic-automl/
├── examples/
├── memory/
│   └── global/
├── projects/
├── src/agentic_automl/
├── tests/
└── workflow/automl_workflow.md
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
agentic-automl ui
```

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

Each completed project bundle includes:

- the winning model artifact
- workflow decisions
- leaderboard metrics
- a human-readable report
- a prediction preview
- per-step project memory

## Why This Design

- project-specific data work can vary widely
- the AutoML workflow can stay standardized
- the UI and CLI both use the same workflow engine
- every delivered project can improve future recommendations through stored memory
