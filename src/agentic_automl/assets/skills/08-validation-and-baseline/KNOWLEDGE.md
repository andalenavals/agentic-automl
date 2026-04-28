# Validation And Baseline Knowledge

## Current Capability

This step can validate the current model on the untouched test set, compare it against the strongest simple no-feature baseline, and prepare the optional output notebook.

## Supported Base Policies

- `test_set_with_baseline`

## Supported Action-Mode Changes

- keep the untouched holdout comparison
- compare the current trained model against a no-model baseline
- explain which baseline strategy is being used
- use a class-prior baseline from the training target distribution for classification
- use a target-only constant baseline from the training target distribution for regression
- explain the output notebook contents
- explain that the notebook can regenerate the holdout predictions in memory

## Capability Keys

- `validation_test_set_with_baseline`: keep the untouched holdout comparison
- `validation_compare_no_model_baseline`: compare the current trained model against a no-model baseline
- `validation_explain_baseline_strategy`: explain the baseline strategy
- `validation_class_prior_baseline`: use a class-prior baseline for classification
- `validation_target_only_constant_baseline`: use a target-only constant baseline for regression
- `validation_explain_output_notebook`: explain the output notebook contents
- `validation_explain_predictions_in_notebook`: explain the in-notebook prediction regeneration
