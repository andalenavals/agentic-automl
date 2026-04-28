---
name: automl-08-validation-and-baseline
description: Use when executing the validation step of Agentic AutoML. Validate the current trained model on the untouched holdout and compare it against a no-model baseline.
---

# Validation and Baseline Skill

## Goal

Produce a trustworthy untuned validation result before any optional hyperparameter optimization starts.

## Action Agent Philosophy

- Keep Action mode conversational across natural follow-up turns.
- Use the accepted brief and previous workflow choices as context.
- Distinguish consultation from execution: questions explain, requests mutate.
- Stay inside `KNOWLEDGE.md`, and keep policy unchanged when a request falls outside the executable space.
- Store unsupported requests in `LIMITS.md` and suggest the next supported move when possible.
- Allow the human to revise the active policy multiple times across the same conversation without losing the accepted context.

## Inputs

- current trained model
- untouched test partition
- selected primary metric
- accepted workflow decisions

## Procedure

1. Evaluate the current trained model on the untouched holdout split.
2. Compare it against a no-model baseline using the same primary metric.
3. Choose the strongest simple baseline that uses no input features and relies only on the training target distribution.
4. For classification, use a class-prior baseline; for hard-label metrics, prefer `stratified_random`, and optionally report `most_frequent` or `uniform_random` for context.
5. For regression, use a target-only constant baseline; prefer `mean_value` for squared-error metrics such as RMSE/MSE/R2 and `median_value` for absolute-error metrics such as MAE.
6. Show both the metric outcome and the validation plots that are relevant for the task type.
7. Allow the operator to decide whether to build the output notebook from the current untuned model.
8. When exporting, keep the output to a single notebook file that can rerun the workflow and regenerate the holdout predictions in memory.

## Output

- validated current model
- explicit baseline comparison
- baseline rationale
