# Model Selection Knowledge

## Current Capability

This step can choose one packaged starting model and declare its initial parameter configuration.
Only the parameters explicitly exposed by the selected model are executable here.

## Supported Base Policies

### Classification

- `logistic_regression`
- `random_forest_classifier`
- `hist_gradient_boosting_classifier`
- `mlp_classifier`

### Regression

- `ridge_regression`
- `random_forest_regressor`
- `hist_gradient_boosting_regressor`
- `mlp_regressor`

## Supported Action-Mode Changes

- switch to another supported packaged model
- override initial model parameters that the packaged model already exposes
- ask which models are currently supported

## Capability Keys

- `model_logistic_regression`: use `logistic_regression`
- `model_random_forest_classifier`: use `random_forest_classifier`
- `model_hist_gradient_boosting_classifier`: use `hist_gradient_boosting_classifier`
- `model_mlp_classifier`: use `mlp_classifier`
- `model_ridge_regression`: use `ridge_regression`
- `model_random_forest_regressor`: use `random_forest_regressor`
- `model_hist_gradient_boosting_regressor`: use `hist_gradient_boosting_regressor`
- `model_mlp_regressor`: use `mlp_regressor`
- `model_override_c`: override `C` when the chosen model supports it
- `model_override_class_weight`: override `class_weight` when the chosen model supports it
- `model_override_alpha`: override `alpha` when the chosen model supports it
- `model_override_n_estimators`: override `n_estimators` when the chosen model supports it
- `model_override_max_depth`: override `max_depth` when the chosen model supports it
- `model_override_min_samples_leaf`: override `min_samples_leaf` when the chosen model supports it
- `model_override_max_leaf_nodes`: override `max_leaf_nodes` when the chosen model supports it
- `model_override_l2_regularization`: override `l2_regularization` when the chosen model supports it
- `model_override_hidden_layer_sizes`: override `hidden_layer_sizes` when the chosen model supports it
