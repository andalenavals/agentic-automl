# Metric Selection Knowledge

## Current Capability

This step can choose the packaged winner metric that governs ranking and baseline comparison.

## Supported Base Policies

### Classification

- `balanced_accuracy`
- `f1_macro`
- `accuracy`

### Regression

- `rmse`
- `mae`
- `r2`

## Supported Action-Mode Changes

- switch the primary winner metric among the packaged choices
- explain tradeoffs among class-balance, precision-recall, absolute error, squared error, and explained variance

## Capability Keys

- `metric_balanced_accuracy`: use `balanced_accuracy` as the primary metric
- `metric_f1_macro`: use `f1_macro` as the primary metric
- `metric_accuracy`: use `accuracy` as the primary metric
- `metric_rmse`: use `rmse` as the primary metric
- `metric_mae`: use `mae` as the primary metric
- `metric_r2`: use `r2` as the primary metric
- `metric_explain_tradeoffs`: explain metric tradeoffs for the current problem
