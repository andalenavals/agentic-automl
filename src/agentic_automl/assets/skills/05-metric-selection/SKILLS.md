---
name: automl-05-metric-selection
description: Use when executing the metric-selection step of Agentic AutoML. Choose the primary optimization metric and ensure the same metric is used for model ranking and baseline comparison.
---

# Metric Selection Skill

## Goal

Pick one primary metric that matches the real business objective and governs winner selection consistently.

## Action Agent Philosophy

- Keep Action mode conversational across natural follow-up turns.
- Use the accepted brief and previous workflow choices as context.
- Distinguish consultation from execution: questions explain, requests mutate.
- Stay inside `KNOWLEDGE.md`, and keep policy unchanged when a request falls outside the executable space.
- Store unsupported requests in `LIMITS.md` and suggest the next supported move when possible.
- Allow the human to revise the active policy multiple times across the same conversation without losing the accepted context.

## Inputs

- `ProjectBrief`
- `DataProfile`
- target imbalance or regression skew signals

## Procedure

1. Honor `baseline_metric` if the brief already gives a valid metric override.
2. For classification, favor `balanced_accuracy` when imbalance matters and `f1_macro` when balanced precision/recall is the main concern.
3. For regression, prefer `mae` when skew/outlier robustness matters and `rmse` otherwise, with `r2` only when explained variance is the main objective.
4. Keep the primary metric consistent across cross-validation ranking, competition ranking, and final baseline comparison.
5. Mention supporting metrics, but keep one clear winner metric.

## Output

- selected primary metric
- short objective-driven rationale
- note that the same metric governs ranking and baseline comparison
