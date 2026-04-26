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

## Step 1. Preprocessing

Review the dataset shape, feature types, missingness, obvious leakage risks, and target definition.

## Step 2. Data Splitting

Recommend a split strategy that preserves evaluation integrity and avoids leakage.

## Step 3. Model Selection

Recommend a candidate set grounded in problem type, feature mix, data scale, and remembered preferences.

## Step 4. Feature Selection

Recommend whether to skip, automate, or lightly prune the transformed feature space.

## Step 5. Metric Selection

Recommend one primary optimization metric and supporting diagnostics.

## Step 6. Training Configuration

Recommend a reproducible training setup, balancing speed and thoroughness.

## Step 7. Hyperparameter Optimization (Gym Competition, Optional)

If enabled, run a compact competition on the strongest candidate families.

## Step 8. Validation In Test Set And Baseline Comparison

Validate the winner on the untouched test set and compare against a simple baseline.
