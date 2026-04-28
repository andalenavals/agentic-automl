# Final Validation Knowledge

## Current Capability

This step can revalidate the final model on the untouched test split, compare it against the baseline again, and prepare the final dashboard/export story.

## Supported Base Policies

- `final_validation_dashboard`

## Supported Action-Mode Changes

- keep the final dashboard flow
- explain the tuned-versus-untuned comparison
- explain which optimization results are included
- explain the final output notebook contents
- explain that the final notebook can regenerate the holdout predictions in memory

## Capability Keys

- `final_validation_dashboard`: keep the final dashboard flow
- `final_validation_explain_tuned_vs_untuned`: explain the tuned-versus-untuned comparison
- `final_validation_explain_optimization_summary`: explain which optimization results are included
- `final_validation_explain_output_notebook`: explain the final output notebook contents
- `final_validation_explain_predictions_in_notebook`: explain the in-notebook prediction regeneration
