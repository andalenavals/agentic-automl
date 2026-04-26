# Agentic AutoML Workflow

This workflow is fixed and is meant to be followed by an AI coding agent.

The human should only need to explain the data and the problem. The AI agent should then guide the project step by step, make recommendations, explain tradeoffs, store memory, run the AutoML process, and export the winning model bundle.

## Intake

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

## Step 1. Preprocessing

Review the dataset shape, feature types, missingness, obvious leakage risks, and target definition.

Recommend:

- how to clean missing data
- how to encode categorical data
- how to scale or normalize numeric features
- whether to keep a simple automatic preprocessing path or a custom path

## Step 2. Data Splitting

Recommend the split strategy based on the problem:

- stratified holdout for classification when appropriate
- random holdout for standard regression
- time-ordered holdout when a time column exists or leakage risk suggests it

Cross-validation must happen only inside the training partition.

## Step 3. Model Selection

Recommend a candidate set of models based on:

- problem type
- dataset size
- feature mix
- previous accepted choices stored in memory

The candidate set should usually include:

- one simple model
- one tree ensemble
- one stronger nonlinear candidate

## Step 4. Feature Selection

Recommend whether to:

- skip feature selection
- use automatic selection
- use a lightweight pruning strategy

Explain the reason in terms of dimensionality, noise, and interpretability.

## Step 5. Metric Selection

Recommend one primary metric and a few supporting metrics.

The recommendation should reflect:

- business objective
- target imbalance
- problem type
- robustness to outliers or class imbalance

The same primary metric must be used for model ranking and baseline comparison.

## Step 6. Training Configuration

Recommend:

- cross-validation depth
- random seed strategy
- iteration budget
- whether speed or thoroughness should be prioritized

The initial MVP should default to reproducible settings.

## Step 7. Hyperparameter Optimization (Gym Competition, Optional)

If enabled, run a compact competition on the best candidate families.

The competition should:

- stay budget-aware
- remain explainable
- improve the best baseline candidate rather than replacing the whole workflow with blind search

If disabled, record that choice and proceed.

## Step 8. Validation In Test Set And Baseline Comparison

Validate the winner on the untouched test set.

Always compare against a simple baseline.

The output should make it easy to see:

- why the winner won
- how much it improved over the baseline
- which workflow decisions led to the final bundle

## Output Requirements

The final project output must include:

- the winning model artifact
- workflow decisions
- metric summaries
- a human-readable report
- a preview of predictions
- per-step project memory

## Memory Rules

Each step has a matching `memory/global/<step>/step_memory.md` file.

When a project finishes:

- write the accepted decision for each step into the project memory
- append a compact summary to the global step memory
- use those memories as recommendation signals in future projects
