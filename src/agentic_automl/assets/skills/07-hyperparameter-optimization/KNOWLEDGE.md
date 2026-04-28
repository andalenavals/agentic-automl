# Hyperparameter Optimization Knowledge

## Current Capability

This step can keep tuning disabled, run a compact competition, or run a broader competition on the already selected model.

Action mode can also edit the HPO search scope by naming the exact hyperparameters to optimize, as long as those hyperparameters are supported for the current selected model. It can also select the top recommended hyperparameters by count, using the current model's built-in priority order. Follow-up Action messages can refine the active HPO scope by naming supported hyperparameters directly, without repeating the word `hyperparameter`.

## Supported Base Policies

- `skip`
- `small_competition`
- `expanded_competition`

## Supported Action-Mode Changes

- skip tuning
- run a compact tuning round
- run a broader tuning round
- replace the HPO search scope with an explicit list of supported hyperparameters
- add supported hyperparameters to the current HPO search scope
- remove supported hyperparameters from the current HPO search scope
- refine the active HPO scope across turns by naming supported hyperparameters directly
- use only the top recommended `N` hyperparameters for the current selected model
- ask which hyperparameters are currently supported for the selected model
- ask which hyperparameters are recommended for the selected model

## Logistic Regression

- `C`: inverse regularization strength
- `class_weight`: class imbalance handling
- `optimizer`: solver choice

## Ridge Regression

- `alpha`: regularization strength

## Random Forest Classifier

- `max_depth`: tree depth control
- `min_samples_leaf`: minimum samples per leaf
- `min_samples_split`: minimum samples before splitting
- `max_features`: feature subsampling per split
- `n_estimators`: number of trees
- `class_weight`: class imbalance handling

## Random Forest Regressor

- `max_depth`: tree depth control
- `min_samples_leaf`: minimum samples per leaf
- `min_samples_split`: minimum samples before splitting
- `max_features`: feature subsampling per split
- `n_estimators`: number of trees

## Hist Gradient Boosting Classifier

- `learning_rate`: shrinkage per boosting round
- `max_depth`: tree depth control
- `max_leaf_nodes`: leaf complexity budget
- `l2_regularization`: leaf-value regularization
- `epochs`: number of boosting iterations

## Hist Gradient Boosting Regressor

- `learning_rate`: shrinkage per boosting round
- `max_depth`: tree depth control
- `max_leaf_nodes`: leaf complexity budget
- `l2_regularization`: leaf-value regularization
- `epochs`: number of boosting iterations

## MLP Classifier

- `alpha`: weight decay
- `learning_rate`: optimizer step size
- `hidden_layer_sizes`: network width and depth
- `optimizer`: solver choice
- `epochs`: maximum iterations
- `mini_batch`: batch size

## MLP Regressor

- `alpha`: weight decay
- `learning_rate`: optimizer step size
- `hidden_layer_sizes`: network width and depth
- `optimizer`: solver choice
- `epochs`: maximum iterations
- `mini_batch`: batch size

## Capability Keys

- `hpo_skip`: skip tuning
- `hpo_small_competition`: run a compact tuning round
- `hpo_expanded_competition`: run a broader tuning round
- `hpo_select_search_parameters`: replace, add, remove, or top-`N` select supported hyperparameters in the active HPO search scope
- `hpo_tune_c`: tune `C`
- `hpo_tune_alpha`: tune `alpha`
- `hpo_tune_class_weight`: tune `class_weight`
- `hpo_tune_optimizer`: tune `optimizer`
- `hpo_tune_learning_rate`: tune `learning_rate`
- `hpo_tune_n_estimators`: tune `n_estimators`
- `hpo_tune_max_depth`: tune `max_depth`
- `hpo_tune_min_samples_leaf`: tune `min_samples_leaf`
- `hpo_tune_min_samples_split`: tune `min_samples_split`
- `hpo_tune_max_features`: tune `max_features`
- `hpo_tune_max_leaf_nodes`: tune `max_leaf_nodes`
- `hpo_tune_l2_regularization`: tune `l2_regularization`
- `hpo_tune_epochs`: tune `epochs`
- `hpo_tune_hidden_layer_sizes`: tune `hidden_layer_sizes`
- `hpo_tune_mini_batch`: tune `mini_batch`
