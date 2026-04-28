Output Notebook
===============

Export contract
---------------

The workflow exports exactly one file: a self-contained Jupyter notebook.

The notebook is intentionally simpler than the package runtime. It contains only
the code required for the agreed workflow path and does not import
``agentic_automl`` when executed.

Notebook structure
------------------

The exported notebook is organized around five explicit workflow functions:

1. ``preprocessing(dataset_path, target, features, transformation_kwargs)``
2. ``split_data(transformed_data)``
3. ``train_model(train_split)``
4. ``hyperparameter_optimization(train_split)``
5. ``validate(test_split, trained_model)``

Each section includes only the code relevant to the selected workflow path. The
exporter prunes unused helpers so the notebook reflects the actual agreed
workflow rather than a generic package runtime.

What the notebook includes
--------------------------

* relevant input information
* selected policies
* metric summaries
* model and baseline comparisons
* hyperparameter competition output when tuning was run
* plots relevant to the chosen task
* a rerunnable minimal runtime embedded directly in the notebook

Path handling
-------------

The notebook resolves the dataset path relative to the notebook location, so the
export stays portable within the project tree.

Validation story
----------------

Validation compares the trained model against the strongest simple no-feature
baseline:

* classification uses a class-prior baseline from the training target
  distribution
* regression uses a target-only constant baseline from the training target
  distribution

Final validation can also include the tuned-versus-untuned comparison and the
hyperparameter competition dashboard.
