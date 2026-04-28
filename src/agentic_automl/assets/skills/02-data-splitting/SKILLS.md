---
name: automl-02-data-splitting
description: Use when executing the data-splitting step of Agentic AutoML. Choose the holdout strategy that preserves evaluation integrity and avoids leakage.
---

# Data Splitting Skill

## Goal

Create a trustworthy final holdout strategy before any model training starts.

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
- whether a usable `date_column` exists

## Procedure

1. Use `time_ordered_holdout` when chronology matters or a valid `date_column` is available.
2. Use `stratified_holdout` for classification when class balance should be preserved.
3. Use `random_holdout` for standard regression when no temporal constraint exists.
4. Keep cross-validation entirely inside the training partition.
5. Call out leakage risks explicitly when the split choice is not obvious.

## Output

- selected split policy
- short leakage-aware rationale
- confirmation question for the human
