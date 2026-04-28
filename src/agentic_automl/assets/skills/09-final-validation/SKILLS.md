---
name: automl-09-final-validation
description: Use when executing the final validation step of Agentic AutoML. Revalidate the tuned model, compare it against the baseline again, and prepare the final dashboard plus notebook export.
---

# Final Validation Skill

## Goal

Finish the workflow with the final tuned-versus-baseline comparison and the export-ready dashboard story.

## Action Agent Philosophy

- Keep Action mode conversational across natural follow-up turns.
- Use the accepted brief and previous workflow choices as context.
- Distinguish consultation from execution: questions explain, requests mutate.
- Stay inside `KNOWLEDGE.md`, and keep policy unchanged when a request falls outside the executable space.
- Store unsupported requests in `LIMITS.md` and suggest the next supported move when possible.
- Allow the human to revise the active policy multiple times across the same conversation without losing the accepted context.

## Inputs

- tuned model when hyperparameter optimization was run
- untuned model when tuning was skipped
- untouched test partition
- selected primary metric
- accepted workflow decisions
- optimization summary when available

## Procedure

1. Evaluate the final model on the untouched test split.
2. Reuse the same baseline logic from the earlier validation step so comparisons stay fair.
3. Make the untuned-versus-tuned story explicit when optimization was run.
4. Show the most relevant final plots for the task type.
5. Include the hyperparameter optimization summary in the final dashboard when tuning was run.
6. Export exactly one workflow notebook file when the operator chooses to build it, and embed the minimal standalone workflow runtime directly inside that notebook so it can rerun without importing `agentic_automl`.

## Output

- final validated model
- explicit baseline comparison
- tuned-versus-untuned comparison when available
- final dashboard-ready summary
- export-ready notebook context
