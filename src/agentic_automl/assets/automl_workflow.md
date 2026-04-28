# Agentic AutoML Workflow

This workflow is fixed and is meant to be followed by an AI coding agent.

The human should only need to explain the data and the problem. The AI agent should then guide the project step by step, make recommendations, explain tradeoffs, run the AutoML process, and export a single rerunnable workflow notebook.

## Step Skills

When executing a workflow step, first load the linked step skill and its companion `KNOWLEDGE.md`.

- `SKILLS.md` describes how the step should reason and operate.
- `KNOWLEDGE.md` describes which executable actions the step currently supports and declares stable capability keys for runtime validation.
- `LIMITS.md` stores unsupported Action-mode requests and seed backlog items for future implementation.

- Intake: [skills/00-intake/SKILLS.md](skills/00-intake/SKILLS.md), [skills/00-intake/KNOWLEDGE.md](skills/00-intake/KNOWLEDGE.md), [skills/00-intake/LIMITS.md](skills/00-intake/LIMITS.md)
- Preprocessing: [skills/01-preprocessing/SKILLS.md](skills/01-preprocessing/SKILLS.md), [skills/01-preprocessing/KNOWLEDGE.md](skills/01-preprocessing/KNOWLEDGE.md), [skills/01-preprocessing/LIMITS.md](skills/01-preprocessing/LIMITS.md)
- Data splitting: [skills/02-data-splitting/SKILLS.md](skills/02-data-splitting/SKILLS.md), [skills/02-data-splitting/KNOWLEDGE.md](skills/02-data-splitting/KNOWLEDGE.md), [skills/02-data-splitting/LIMITS.md](skills/02-data-splitting/LIMITS.md)
- Model selection: [skills/03-model-selection/SKILLS.md](skills/03-model-selection/SKILLS.md), [skills/03-model-selection/KNOWLEDGE.md](skills/03-model-selection/KNOWLEDGE.md), [skills/03-model-selection/LIMITS.md](skills/03-model-selection/LIMITS.md)
- Metric selection: [skills/05-metric-selection/SKILLS.md](skills/05-metric-selection/SKILLS.md), [skills/05-metric-selection/KNOWLEDGE.md](skills/05-metric-selection/KNOWLEDGE.md), [skills/05-metric-selection/LIMITS.md](skills/05-metric-selection/LIMITS.md)
- Training configuration: [skills/06-training-configuration/SKILLS.md](skills/06-training-configuration/SKILLS.md), [skills/06-training-configuration/KNOWLEDGE.md](skills/06-training-configuration/KNOWLEDGE.md), [skills/06-training-configuration/LIMITS.md](skills/06-training-configuration/LIMITS.md)
- Validation and baseline comparison: [skills/08-validation-and-baseline/SKILLS.md](skills/08-validation-and-baseline/SKILLS.md), [skills/08-validation-and-baseline/KNOWLEDGE.md](skills/08-validation-and-baseline/KNOWLEDGE.md), [skills/08-validation-and-baseline/LIMITS.md](skills/08-validation-and-baseline/LIMITS.md)
- Hyperparameter optimization: [skills/07-hyperparameter-optimization/SKILLS.md](skills/07-hyperparameter-optimization/SKILLS.md), [skills/07-hyperparameter-optimization/KNOWLEDGE.md](skills/07-hyperparameter-optimization/KNOWLEDGE.md), [skills/07-hyperparameter-optimization/LIMITS.md](skills/07-hyperparameter-optimization/LIMITS.md)
- Final validation: [skills/09-final-validation/SKILLS.md](skills/09-final-validation/SKILLS.md), [skills/09-final-validation/KNOWLEDGE.md](skills/09-final-validation/KNOWLEDGE.md), [skills/09-final-validation/LIMITS.md](skills/09-final-validation/LIMITS.md)

## Action Mode Execution Contract

When the human uses Action mode for a workflow step, the agent must follow this loop:

1. Parse the message into candidate executable intents for that step.
2. Map each intent to one or more capability keys.
3. Check those capability keys against the current step `KNOWLEDGE.md`.
4. If every requested capability is supported, merge the resulting policy changes into the current working policy.
5. If any requested capability is unsupported, do not mutate the working policy, explain that the request is outside the current executable space, and append it to that step `LIMITS.md`.

This contract is especially important for multi-turn customization: unsupported follow-up requests must never erase or overwrite a previously valid custom policy.

## Intake

Step skill: [skills/00-intake/SKILLS.md](skills/00-intake/SKILLS.md)

Ask the human for a structured brief with:

- `project_name`
- `dataset_path`
- `target_column`
- `task_type` as `classification` or `regression`
- `problem_description`
- optional `date_column`
- optional `baseline_metric`
- optional `competition_enabled`

The AI should not ask the human to design the ML system manually unless a real blocker appears.

After each workflow recommendation, the AI must ask the human whether they agree with the default procedure or want something different. If they want something different, the AI should open a short discussion, answer questions, turn the discussion into a working execution policy, and store that policy summary with the final notebook export.

## Step 1. Preprocessing

Step skill: [skills/01-preprocessing/SKILLS.md](skills/01-preprocessing/SKILLS.md)

Review the dataset shape, feature types, missingness, obvious leakage risks, and target definition.

Recommend:

- how to clean missing data
- how to encode categorical data
- how to scale or normalize numeric features
- how to prune, subset, or re-role features when needed
- whether to keep a simple automatic preprocessing path or a custom path

## Step 2. Data Splitting

Step skill: [skills/02-data-splitting/SKILLS.md](skills/02-data-splitting/SKILLS.md)

Recommend the split strategy based on the problem:

- stratified holdout for classification when appropriate
- random holdout for standard regression
- time-ordered holdout when a time column exists or leakage risk suggests it

Cross-validation must happen only inside the training partition.

## Step 3. Model Selection

Step skill: [skills/03-model-selection/SKILLS.md](skills/03-model-selection/SKILLS.md)

Recommend one specific initial model based on:

- problem type
- dataset size
- feature mix

The recommendation must include:

- the exact selected model
- the initial model parameters
- the reason this model is the best starting point before any tuning

## Step 4. Metric Selection

Step skill: [skills/05-metric-selection/SKILLS.md](skills/05-metric-selection/SKILLS.md)

Recommend one primary metric and a few supporting metrics.

The recommendation should reflect:

- business objective
- target imbalance
- problem type
- robustness to outliers or class imbalance

The same primary metric must be used for model ranking and baseline comparison.

## Step 5. Training

Step skill: [skills/06-training-configuration/SKILLS.md](skills/06-training-configuration/SKILLS.md)

Recommend:

- the training parameters relevant to the selected model
- optimizer when applicable
- learning rate when applicable
- epochs or maximum iterations when applicable
- mini-batch size when applicable
- validation or cross-validation budget
- random seed strategy

The training step should be executable independently and should expose a clear button to start training.

## Step 6. Validation

Step skill: [skills/08-validation-and-baseline/SKILLS.md](skills/08-validation-and-baseline/SKILLS.md)

Validate the trained model on the untouched test set and compare it against a no-model baseline.

Baseline choice should be problem-aware:

- classification should use a class-prior baseline drawn from the training target distribution; for hard-label metrics, prefer a stratified baseline, and optionally report `most_frequent` or `uniform_random` for context
- regression should use a target-only constant baseline that ignores input features; prefer the mean for squared-error metrics such as RMSE/MSE/R2 and the median for absolute-error metrics such as MAE

This step should also allow the operator to decide whether to build the workflow notebook immediately from the current untuned model.

## Step 7. Hyperparameter Optimization (Gym Competition, Optional)

Step skill: [skills/07-hyperparameter-optimization/SKILLS.md](skills/07-hyperparameter-optimization/SKILLS.md)

If enabled, optimize the selected model by tuning a focused set of hyperparameters.

The competition should:

- stay budget-aware
- remain explainable
- optimize the current selected model rather than reopening model search
- expose which hyperparameters are being tuned

If disabled, record that choice and proceed.

## Step 9. Final Validation

Step skill: [skills/09-final-validation/SKILLS.md](skills/09-final-validation/SKILLS.md)

Repeat validation with the tuned model from Step 7 and compare it again against the same baseline logic.

The final dashboard should make it easy to see:

- how the untuned and tuned models compare
- how much each version improves over the baseline
- which workflow decisions and hyperparameter optimization steps led to the final exported notebook

## Output Requirements

The final project output must include:

- exactly one workflow notebook file
- the workflow decisions
- metric summaries
- hyperparameter optimization details when tuning was run
- the reasoning and plots for the selected run
- an embedded standalone runtime copied into the notebook itself
- code cells that can rerun the full workflow from a dataset path resolved relative to the notebook location, without importing `agentic_automl`
- a final holdout prediction table regenerated in memory when the notebook is executed
