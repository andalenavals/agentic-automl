# Training Configuration Knowledge

## Current Capability

This step can select one packaged training policy and override the main training parameters exposed by the chosen model.
Not every selected model supports every training override, so Action mode should reject no-op requests instead of storing them as if they changed the runtime.

## Supported Base Policies

- `fast_training`
- `standard_training`
- `thorough_training`

## Supported Action-Mode Changes

- favor speed over confidence
- use the standard reproducible setup
- spend more validation budget for a stronger estimate
- override `cv_folds`
- override `random_seed`
- override `optimizer` when the chosen model supports it
- override `learning_rate` when the chosen model supports it
- override `epochs`
- override `mini_batch` when the chosen model supports it
- override `early_stopping`

## Capability Keys

- `training_fast`: favor speed over confidence
- `training_standard`: use the standard reproducible setup
- `training_thorough`: spend more validation budget for a stronger estimate
- `training_override_cv_folds`: override `cv_folds`
- `training_override_random_seed`: override `random_seed`
- `training_override_optimizer`: override `optimizer`
- `training_override_learning_rate`: override `learning_rate`
- `training_override_epochs`: override `epochs`
- `training_override_mini_batch`: override `mini_batch`
- `training_override_early_stopping`: override `early_stopping`
