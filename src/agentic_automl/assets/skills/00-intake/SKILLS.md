---
name: automl-00-intake
description: Use when executing the intake step of Agentic AutoML. Capture the project brief, validate the required fields, and prepare the workflow without asking the human to design the ML system.
---

# Intake Skill

## Goal

Collect a complete `ProjectBrief` that is strong enough for the rest of the workflow to plan the project.

## Action Agent Philosophy

- Keep Action mode conversational across natural follow-up turns.
- Use the accepted brief and previous workflow choices as context.
- Distinguish consultation from execution: questions explain, requests mutate.
- Stay inside `KNOWLEDGE.md`, and keep policy unchanged when a request falls outside the executable space.
- Store unsupported requests in `LIMITS.md` and suggest the next supported move when possible.
- Allow the human to revise the active policy multiple times across the same conversation without losing the accepted context.

## Required Inputs

- `dataset_path`
- `target_column`
- `task_type`
- `problem_description`

## Helpful Optional Inputs

- `project_name`
- `date_column`
- `baseline_metric`
- `competition_enabled`

## Procedure

1. Accept the brief gradually across multiple messages instead of forcing a single strict template.
2. Infer `project_name` from the dataset filename when it is missing.
3. Normalize aliases such as `dataset`, `target`, `task`, and `problem`.
4. Validate that `task_type` is either `classification` or `regression`.
5. Do not ask the human to choose models, metrics, or preprocessing rules during intake unless a real blocker appears.
6. Once the brief is complete, restate the captured project context and transition into step-by-step workflow review.

## Output

- a valid `ProjectBrief`
- a concise restatement of the captured project
- clear transition into `01_preprocessing`
