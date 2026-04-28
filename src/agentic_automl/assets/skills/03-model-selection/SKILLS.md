---
name: automl-03-model-selection
description: Use when executing the model-selection step of Agentic AutoML. Choose one concrete starting model and its initial parameters based on task type, dataset scale, and feature mix.
---

# Model Selection Skill

## Goal

Select one concrete starting model that balances problem fit, runtime cost, and parameter transparency.

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
- already accepted workflow decisions

## Procedure

1. Start from the problem type, dataset scale, and feature mix.
2. Choose one explicit model name instead of a multi-model strategy.
3. Recommend initial model parameters that are sensible before any tuning.
4. Explain why this one model is the best starting point for the current project.
5. Keep the model choice easy to override later from Action mode.

## Output

- selected concrete model
- initial model parameters
- rationale tied to scale and feature mix
