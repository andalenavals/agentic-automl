Workflow
========

The workflow is fixed and agent-centered. The human provides the brief, reviews
the recommendation for each step, and can revise the active policy through
Action mode at any point.

Canonical workflow
------------------

The canonical workflow lives in:

* ``src/agentic_automl/assets/automl_workflow.md``

The steps are:

1. Intake
2. Preprocessing
3. Data splitting
4. Model selection
5. Metric selection
6. Training configuration
7. Validation and baseline
8. Hyperparameter optimization
9. Final validation

Per-step assets
---------------

Each workflow step ships with three companion files under
``src/agentic_automl/assets/skills/<step>/``:

* ``SKILLS.md`` describes how the step reasons and operates.
* ``KNOWLEDGE.md`` declares the currently supported executable actions.
* ``LIMITS.md`` stores unsupported requests and seed backlog items.

Step intent
-----------

* Intake captures the minimum project brief.
* Preprocessing owns data cleaning, feature pruning, feature-role changes, and
  executable preprocessing overrides.
* Data splitting owns the final holdout strategy only.
* Model selection chooses one specific starting model and its initial
  parameters.
* Metric selection chooses one winner metric that also governs baseline
  comparison.
* Training configuration controls the executable training parameters relevant to
  the selected model.
* Validation and baseline compare the current model against the strongest simple
  no-feature baseline.
* Hyperparameter optimization optionally runs a focused competition on the
  selected model.
* Final validation summarizes the tuned-versus-untuned outcome and prepares the
  final notebook story.
