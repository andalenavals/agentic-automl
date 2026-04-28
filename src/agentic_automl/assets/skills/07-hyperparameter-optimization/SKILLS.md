---
name: automl-07-hyperparameter-optimization
description: Use when executing the optional hyperparameter-optimization step of Agentic AutoML. Decide whether to skip tuning or run a focused competition on the already selected model.
---

# Hyperparameter Optimization Skill

## Goal

Control tuning spend so the competition stage improves the selected model without turning the workflow into blind search.

## Action Agent Philosophy

- Keep Action mode conversational across natural follow-up turns.
- Use the accepted brief and previous workflow choices as context.
- Distinguish consultation from execution: questions explain, requests mutate.
- Stay inside `KNOWLEDGE.md`, and keep policy unchanged when a request falls outside the executable space.
- Store unsupported requests in `LIMITS.md` and suggest the next supported move when possible.
- Allow the human to revise the active policy multiple times across the same conversation without losing the accepted context.

## Inputs

- `ProjectBrief`
- current selected model and its parameters
- accepted training budget

## Procedure

1. Use `skip` when `competition_enabled` is false or when tuning budget is intentionally off.
2. Use `small_competition` as the normal tuning mode when the user wants optimization.
3. Use `expanded_competition` only when broader tuning budget is explicitly justified.
4. Restrict tuning to the already selected model instead of reopening model search.
5. In Action mode, treat a named hyperparameter list as an executable search-scope request, not as free-form guidance.
6. Merge or replace the active HPO search scope according to the user request:
   - replace when the user gives a fresh explicit list
   - extend when the user says add or include
   - subtract when the user says remove, drop, or exclude
   - rank-select when the user asks for the top or most important `N` hyperparameters
   - keep follow-up scope refinements executable even when the user only names supported hyperparameters directly
7. Only accept hyperparameters that are listed in `KNOWLEDGE.md` for the current selected model.
8. If the request is outside the executable knowledge space, keep the current HPO policy unchanged and store the request in `LIMITS.md`.
9. Keep the tuning story explainable and budget-aware.

## Output

- selected HPO option
- selected search scope for the current model
- brief explanation of scope and budget
