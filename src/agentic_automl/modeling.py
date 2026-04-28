from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor
from .schemas import CandidateResult, DataProfile, ProjectBrief


CLASSIFICATION_MODELS = [
    "logistic_regression",
    "random_forest_classifier",
    "hist_gradient_boosting_classifier",
    "mlp_classifier",
]
REGRESSION_MODELS = [
    "ridge_regression",
    "random_forest_regressor",
    "hist_gradient_boosting_regressor",
    "mlp_regressor",
]


HPO_SEARCH_LIBRARY: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "compact_default": ["C", "class_weight"],
        "expanded_default": ["C", "class_weight", "optimizer"],
        "parameters": {
            "C": {
                "grid_key": "model__C",
                "compact": [0.1, 1.0, 3.0],
                "expanded": [0.01, 0.1, 1.0, 3.0, 10.0],
                "aliases": ["parameter c", "inverse regularization", "regularization strength"],
                "reason": "controls how strongly the model regularizes the decision boundary",
            },
            "class_weight": {
                "grid_key": "model__class_weight",
                "compact": [None, "balanced"],
                "expanded": [None, "balanced"],
                "aliases": ["class weight", "class_weight"],
                "reason": "helps when the positive and negative classes carry different effective weight",
            },
            "optimizer": {
                "grid_key": "model__solver",
                "compact": ["lbfgs", "liblinear"],
                "expanded": ["lbfgs", "liblinear", "saga"],
                "aliases": ["optimizer", "solver"],
                "reason": "changes the optimization behavior and can affect convergence on tabular problems",
            },
        },
    },
    "ridge_regression": {
        "compact_default": ["alpha"],
        "expanded_default": ["alpha"],
        "parameters": {
            "alpha": {
                "grid_key": "model__alpha",
                "compact": [0.1, 1.0, 10.0],
                "expanded": [0.01, 0.1, 1.0, 10.0, 100.0],
                "aliases": ["alpha", "regularization"],
                "reason": "sets how strongly the regression coefficients are shrunk",
            },
        },
    },
    "random_forest_classifier": {
        "compact_default": ["max_depth", "min_samples_leaf", "max_features", "class_weight"],
        "expanded_default": [
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "class_weight",
            "min_samples_split",
            "n_estimators",
        ],
        "parameters": {
            "max_depth": {
                "grid_key": "model__max_depth",
                "compact": [None, 4, 8, 12],
                "expanded": [None, 4, 8, 12, 16, 24],
                "aliases": ["max depth", "max_depth", "tree depth"],
                "reason": "is usually the strongest control on tree complexity and overfitting",
            },
            "min_samples_leaf": {
                "grid_key": "model__min_samples_leaf",
                "compact": [1, 2, 4],
                "expanded": [1, 2, 4, 8, 12],
                "aliases": ["min samples leaf", "min_samples_leaf", "leaf size"],
                "reason": "smooths noisy leaves and is often high leverage on tabular classification data",
            },
            "min_samples_split": {
                "grid_key": "model__min_samples_split",
                "compact": [2, 5, 10],
                "expanded": [2, 5, 10, 20],
                "aliases": ["min samples split", "min_samples_split", "split size"],
                "reason": "works with leaf size to prevent overly specific branches",
            },
            "max_features": {
                "grid_key": "model__max_features",
                "compact": ["sqrt", 0.5],
                "expanded": ["sqrt", "log2", 0.3, 0.5, 0.8],
                "aliases": ["max features", "max_features", "feature subsampling"],
                "reason": "changes how diverse the trees are and can improve generalization",
            },
            "n_estimators": {
                "grid_key": "model__n_estimators",
                "compact": [200, 400, 600],
                "expanded": [200, 400, 600, 800],
                "aliases": ["n_estimators", "estimators", "number of trees"],
                "reason": "mostly trades extra compute for a more stable ensemble estimate",
            },
            "class_weight": {
                "grid_key": "model__class_weight",
                "compact": [None, "balanced"],
                "expanded": [None, "balanced", "balanced_subsample"],
                "aliases": ["class weight", "class_weight"],
                "reason": "is worth testing when class imbalance matters for the chosen classification metric",
            },
        },
    },
    "random_forest_regressor": {
        "compact_default": ["max_depth", "min_samples_leaf", "max_features"],
        "expanded_default": ["max_depth", "min_samples_leaf", "max_features", "min_samples_split", "n_estimators"],
        "parameters": {
            "max_depth": {
                "grid_key": "model__max_depth",
                "compact": [None, 4, 8, 12],
                "expanded": [None, 4, 8, 12, 16, 24],
                "aliases": ["max depth", "max_depth", "tree depth"],
                "reason": "is usually the strongest control on tree complexity and overfitting",
            },
            "min_samples_leaf": {
                "grid_key": "model__min_samples_leaf",
                "compact": [1, 2, 4],
                "expanded": [1, 2, 4, 8, 12],
                "aliases": ["min samples leaf", "min_samples_leaf", "leaf size"],
                "reason": "smooths noisy leaves and can stabilize regression trees",
            },
            "min_samples_split": {
                "grid_key": "model__min_samples_split",
                "compact": [2, 5, 10],
                "expanded": [2, 5, 10, 20],
                "aliases": ["min samples split", "min_samples_split", "split size"],
                "reason": "prevents the trees from splitting on tiny, unstable sample groups",
            },
            "max_features": {
                "grid_key": "model__max_features",
                "compact": ["sqrt", 0.5],
                "expanded": ["sqrt", "log2", 0.3, 0.5, 0.8],
                "aliases": ["max features", "max_features", "feature subsampling"],
                "reason": "changes tree diversity and can improve out-of-sample stability",
            },
            "n_estimators": {
                "grid_key": "model__n_estimators",
                "compact": [200, 400, 600],
                "expanded": [200, 400, 600, 800],
                "aliases": ["n_estimators", "estimators", "number of trees"],
                "reason": "mostly trades extra compute for a more stable ensemble estimate",
            },
        },
    },
    "hist_gradient_boosting_classifier": {
        "compact_default": ["learning_rate", "max_depth", "max_leaf_nodes"],
        "expanded_default": ["learning_rate", "max_depth", "max_leaf_nodes", "l2_regularization", "epochs"],
        "parameters": {
            "learning_rate": {
                "grid_key": "model__learning_rate",
                "compact": [0.05, 0.1],
                "expanded": [0.03, 0.05, 0.1, 0.2],
                "aliases": ["learning rate", "learning_rate"],
                "reason": "controls how aggressively each boosting step updates the ensemble",
            },
            "max_depth": {
                "grid_key": "model__max_depth",
                "compact": [None, 4, 8],
                "expanded": [None, 4, 8, 12],
                "aliases": ["max depth", "max_depth", "tree depth"],
                "reason": "controls the interaction depth available to each boosting tree",
            },
            "max_leaf_nodes": {
                "grid_key": "model__max_leaf_nodes",
                "compact": [15, 31, 63],
                "expanded": [15, 31, 63, 127],
                "aliases": ["max leaf nodes", "max_leaf_nodes", "leaf nodes"],
                "reason": "sets the complexity budget of each tree in the boosting stage",
            },
            "l2_regularization": {
                "grid_key": "model__l2_regularization",
                "compact": [0.0, 0.1],
                "expanded": [0.0, 0.1, 1.0],
                "aliases": ["l2 regularization", "l2_regularization"],
                "reason": "can dampen unstable leaf values when the dataset is noisy",
            },
            "epochs": {
                "grid_key": "model__max_iter",
                "compact": [100, 200],
                "expanded": [100, 200, 300],
                "aliases": ["epochs", "iterations", "max_iter"],
                "reason": "changes how many boosting rounds the model can fit",
            },
        },
    },
    "hist_gradient_boosting_regressor": {
        "compact_default": ["learning_rate", "max_depth", "max_leaf_nodes"],
        "expanded_default": ["learning_rate", "max_depth", "max_leaf_nodes", "l2_regularization", "epochs"],
        "parameters": {
            "learning_rate": {
                "grid_key": "model__learning_rate",
                "compact": [0.05, 0.1],
                "expanded": [0.03, 0.05, 0.1, 0.2],
                "aliases": ["learning rate", "learning_rate"],
                "reason": "controls how aggressively each boosting step updates the ensemble",
            },
            "max_depth": {
                "grid_key": "model__max_depth",
                "compact": [None, 4, 8],
                "expanded": [None, 4, 8, 12],
                "aliases": ["max depth", "max_depth", "tree depth"],
                "reason": "controls the interaction depth available to each boosting tree",
            },
            "max_leaf_nodes": {
                "grid_key": "model__max_leaf_nodes",
                "compact": [15, 31, 63],
                "expanded": [15, 31, 63, 127],
                "aliases": ["max leaf nodes", "max_leaf_nodes", "leaf nodes"],
                "reason": "sets the complexity budget of each tree in the boosting stage",
            },
            "l2_regularization": {
                "grid_key": "model__l2_regularization",
                "compact": [0.0, 0.1],
                "expanded": [0.0, 0.1, 1.0],
                "aliases": ["l2 regularization", "l2_regularization"],
                "reason": "can dampen unstable leaf values when the target is noisy",
            },
            "epochs": {
                "grid_key": "model__max_iter",
                "compact": [100, 200],
                "expanded": [100, 200, 300],
                "aliases": ["epochs", "iterations", "max_iter"],
                "reason": "changes how many boosting rounds the model can fit",
            },
        },
    },
    "mlp_classifier": {
        "compact_default": ["alpha", "learning_rate", "hidden_layer_sizes"],
        "expanded_default": ["alpha", "learning_rate", "hidden_layer_sizes", "optimizer", "epochs", "mini_batch"],
        "parameters": {
            "alpha": {
                "grid_key": "model__alpha",
                "compact": [0.0001, 0.001],
                "expanded": [0.0001, 0.001, 0.01],
                "aliases": ["alpha", "regularization"],
                "reason": "controls weight decay and is often a high-leverage stabilizer for MLPs",
            },
            "learning_rate": {
                "grid_key": "model__learning_rate_init",
                "compact": [0.0005, 0.001],
                "expanded": [0.0003, 0.0005, 0.001, 0.003],
                "aliases": ["learning rate", "learning_rate"],
                "reason": "strongly affects convergence behavior in neural network training",
            },
            "hidden_layer_sizes": {
                "grid_key": "model__hidden_layer_sizes",
                "compact": [(64, 32), (128, 64)],
                "expanded": [(64, 32), (128, 64), (128, 128)],
                "aliases": ["hidden layers", "hidden_layer_sizes", "layers", "network width"],
                "reason": "changes the model capacity and the types of interactions the network can represent",
            },
            "optimizer": {
                "grid_key": "model__solver",
                "compact": ["adam", "sgd"],
                "expanded": ["adam", "sgd", "lbfgs"],
                "aliases": ["optimizer", "solver"],
                "reason": "changes the training dynamics and can matter a lot on smaller tabular datasets",
            },
            "epochs": {
                "grid_key": "model__max_iter",
                "compact": [200, 300],
                "expanded": [150, 250, 350],
                "aliases": ["epochs", "iterations", "max_iter"],
                "reason": "controls how long the optimizer can keep improving the network",
            },
            "mini_batch": {
                "grid_key": "model__batch_size",
                "compact": [16, 32],
                "expanded": [16, 32, 64],
                "aliases": ["mini batch", "mini_batch", "batch size", "batch_size"],
                "reason": "changes the gradient noise level and the effective training dynamics",
            },
        },
    },
    "mlp_regressor": {
        "compact_default": ["alpha", "learning_rate", "hidden_layer_sizes"],
        "expanded_default": ["alpha", "learning_rate", "hidden_layer_sizes", "optimizer", "epochs", "mini_batch"],
        "parameters": {
            "alpha": {
                "grid_key": "model__alpha",
                "compact": [0.0001, 0.001],
                "expanded": [0.0001, 0.001, 0.01],
                "aliases": ["alpha", "regularization"],
                "reason": "controls weight decay and is often a high-leverage stabilizer for MLPs",
            },
            "learning_rate": {
                "grid_key": "model__learning_rate_init",
                "compact": [0.0005, 0.001],
                "expanded": [0.0003, 0.0005, 0.001, 0.003],
                "aliases": ["learning rate", "learning_rate"],
                "reason": "strongly affects convergence behavior in neural network training",
            },
            "hidden_layer_sizes": {
                "grid_key": "model__hidden_layer_sizes",
                "compact": [(64, 32), (128, 64)],
                "expanded": [(64, 32), (128, 64), (128, 128)],
                "aliases": ["hidden layers", "hidden_layer_sizes", "layers", "network width"],
                "reason": "changes the model capacity and the types of interactions the network can represent",
            },
            "optimizer": {
                "grid_key": "model__solver",
                "compact": ["adam", "sgd"],
                "expanded": ["adam", "sgd", "lbfgs"],
                "aliases": ["optimizer", "solver"],
                "reason": "changes the training dynamics and can matter a lot on smaller tabular datasets",
            },
            "epochs": {
                "grid_key": "model__max_iter",
                "compact": [200, 300],
                "expanded": [150, 250, 350],
                "aliases": ["epochs", "iterations", "max_iter"],
                "reason": "controls how long the optimizer can keep improving the network",
            },
            "mini_batch": {
                "grid_key": "model__batch_size",
                "compact": [16, 32],
                "expanded": [16, 32, 64],
                "aliases": ["mini batch", "mini_batch", "batch size", "batch_size"],
                "reason": "changes the gradient noise level and the effective training dynamics",
            },
        },
    },
}


def hpo_model_spec(model_name: str) -> dict[str, Any]:
    return HPO_SEARCH_LIBRARY.get(model_name, {})


def hpo_parameter_specs(model_name: str) -> dict[str, dict[str, Any]]:
    return dict(hpo_model_spec(model_name).get("parameters", {}))


def hpo_supported_parameters(model_name: str) -> list[str]:
    return list(hpo_parameter_specs(model_name))


def hpo_parameter_reason_map(model_name: str) -> dict[str, str]:
    return {
        parameter_name: str(spec.get("reason", ""))
        for parameter_name, spec in hpo_parameter_specs(model_name).items()
    }


def hpo_parameter_alias_lookup(model_name: str) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for parameter_name, spec in hpo_parameter_specs(model_name).items():
        aliases = [parameter_name, parameter_name.replace("_", " ")]
        aliases.extend(spec.get("aliases", []))
        for alias in aliases:
            lookup[alias.lower()] = parameter_name
    return lookup


def default_hpo_search_parameters(model_name: str, hpo_option: str) -> list[str]:
    spec = hpo_model_spec(model_name)
    fallback = hpo_supported_parameters(model_name)
    if not fallback:
        return []
    if hpo_option == "expanded_competition":
        return [item for item in spec.get("expanded_default", fallback) if item in fallback] or fallback
    return [item for item in spec.get("compact_default", fallback) if item in fallback] or fallback


def normalize_hpo_search_parameters(
    model_name: str,
    hpo_option: str,
    requested_parameters: list[str] | None = None,
) -> list[str]:
    supported = set(hpo_supported_parameters(model_name))
    if requested_parameters:
        normalized = [item for item in requested_parameters if item in supported]
        if normalized:
            return normalized
    return default_hpo_search_parameters(model_name, hpo_option)


def hpo_search_space_config(
    model_name: str,
    hpo_option: str,
    requested_parameters: list[str] | None = None,
) -> dict[str, Any]:
    width = "expanded" if hpo_option == "expanded_competition" else "compact"
    selected_parameters = normalize_hpo_search_parameters(model_name, hpo_option, requested_parameters)
    specs = hpo_parameter_specs(model_name)
    estimator_grid: dict[str, list[Any]] = {}
    display_grid: dict[str, list[Any]] = {}
    for parameter_name in selected_parameters:
        spec = specs.get(parameter_name, {})
        values = list(spec.get(width, []))
        grid_key = str(spec.get("grid_key", ""))
        if not grid_key or not values:
            continue
        estimator_grid[grid_key] = values
        display_grid[parameter_name] = values
    return {
        "selected_parameters": selected_parameters,
        "estimator_grid": estimator_grid,
        "display_grid": display_grid,
    }


def split_data(frame: pd.DataFrame, brief: ProjectBrief, split_option: str, test_size: float = 0.2):
    features = frame.drop(columns=[brief.target_column])
    target = frame[brief.target_column]

    if split_option == "time_ordered_holdout":
        if not brief.date_column or brief.date_column not in frame.columns:
            raise ValueError("Time-ordered holdout requires a valid date_column in the project brief.")
        ordered = frame.sort_values(by=brief.date_column)
        split_index = int(len(ordered) * (1 - test_size))
        train_frame = ordered.iloc[:split_index]
        test_frame = ordered.iloc[split_index:]
        return (
            train_frame.drop(columns=[brief.target_column]),
            test_frame.drop(columns=[brief.target_column]),
            train_frame[brief.target_column],
            test_frame[brief.target_column],
        )

    stratify = None
    if split_option == "stratified_holdout" and brief.task_type == "classification":
        if target.nunique(dropna=False) > 1 and target.value_counts().min() > 1:
            stratify = target

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )


def model_options_for_task(task_type: str) -> list[str]:
    return list(CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS)


def default_model_option(task_type: str, profile: DataProfile) -> str:
    if task_type == "classification":
        if profile.rows >= 50_000:
            return "hist_gradient_boosting_classifier"
        if not profile.categorical_features and profile.rows <= 10_000:
            return "logistic_regression"
        if profile.high_cardinality_categorical_features or profile.rows >= 5_000:
            return "hist_gradient_boosting_classifier"
        return "random_forest_classifier"

    if profile.rows >= 50_000:
        return "hist_gradient_boosting_regressor"
    if not profile.categorical_features and profile.rows <= 10_000:
        return "ridge_regression"
    if profile.target_skew and abs(profile.target_skew) >= 1.0:
        return "hist_gradient_boosting_regressor"
    return "random_forest_regressor"


def humanize_model_name(model_name: str) -> str:
    return model_name.replace("_", " ").capitalize()


def default_model_parameters(
    task_type: str,
    model_name: str,
    profile: DataProfile,
) -> dict[str, Any]:
    if model_name == "logistic_regression":
        return {
            "C": 1.0,
            "class_weight": "balanced" if profile.class_imbalance and profile.class_imbalance >= 1.5 else None,
        }
    if model_name == "ridge_regression":
        return {"alpha": 1.0}
    if model_name in {"random_forest_classifier", "random_forest_regressor"}:
        return {
            "n_estimators": 300 if profile.rows <= 10_000 else 500,
            "max_depth": None if profile.rows <= 20_000 else 16,
            "min_samples_leaf": 1 if profile.rows <= 5_000 else 2,
        }
    if model_name in {"hist_gradient_boosting_classifier", "hist_gradient_boosting_regressor"}:
        return {
            "max_depth": None,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.0,
        }
    if model_name in {"mlp_classifier", "mlp_regressor"}:
        return {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.0001,
        }
    return {}


def default_training_configuration(
    task_type: str,
    model_name: str,
    profile: DataProfile,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "cv_folds": 5 if profile.rows > 2_000 else 3,
        "random_seed": 42,
        "optimizer": None,
        "learning_rate": None,
        "epochs": None,
        "mini_batch": None,
        "early_stopping": False,
        "validation_fraction": 0.1,
    }
    if model_name in {"logistic_regression", "ridge_regression"}:
        config["epochs"] = 300
        return config
    if model_name in {"random_forest_classifier", "random_forest_regressor"}:
        return config
    if model_name in {"hist_gradient_boosting_classifier", "hist_gradient_boosting_regressor"}:
        config["learning_rate"] = 0.1
        config["epochs"] = 200
        config["early_stopping"] = True
        return config
    if model_name in {"mlp_classifier", "mlp_regressor"}:
        config["optimizer"] = "adam"
        config["learning_rate"] = 0.001
        config["epochs"] = 250
        config["mini_batch"] = 32
        config["early_stopping"] = True
        return config
    return config


def normalize_model_parameters(
    task_type: str,
    model_name: str,
    profile: DataProfile,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = default_model_parameters(task_type, model_name, profile)
    if not overrides:
        return params
    params.update({key: value for key, value in overrides.items() if value is not None})
    return params


def normalize_training_configuration(
    task_type: str,
    model_name: str,
    profile: DataProfile,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = default_training_configuration(task_type, model_name, profile)
    if not overrides:
        return config
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def metric_configuration(task_type: str, selected_metric: str) -> tuple[dict[str, str], bool]:
    if task_type == "classification":
        return {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
        }, True

    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }, (selected_metric == "r2")


def cross_validation_strategy(task_type: str, option_or_config: str | dict[str, Any]):
    if isinstance(option_or_config, dict):
        folds = int(option_or_config.get("cv_folds", 5))
    else:
        option = option_or_config
        folds = 5 if option in {"standard_cv", "thorough_cv"} else 3
    if task_type == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def build_estimator(
    task_type: str,
    model_name: str,
    model_params: dict[str, Any],
    training_config: dict[str, Any],
) -> Any:
    random_seed = int(training_config.get("random_seed", 42))
    epochs = training_config.get("epochs")
    learning_rate = training_config.get("learning_rate")
    optimizer = training_config.get("optimizer")
    mini_batch = training_config.get("mini_batch")
    early_stopping = bool(training_config.get("early_stopping", False))
    validation_fraction = float(training_config.get("validation_fraction", 0.1))

    if model_name == "logistic_regression":
        solver = optimizer if optimizer in {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"} else "lbfgs"
        return LogisticRegression(
            C=float(model_params.get("C", 1.0)),
            class_weight=model_params.get("class_weight"),
            max_iter=int(epochs or 300),
            solver=solver,
            random_state=random_seed,
        )
    if model_name == "ridge_regression":
        return Ridge(
            alpha=float(model_params.get("alpha", 1.0)),
            random_state=random_seed,
        )
    if model_name == "random_forest_classifier":
        return RandomForestClassifier(
            n_estimators=int(model_params.get("n_estimators", 300)),
            max_depth=model_params.get("max_depth"),
            min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
            class_weight=model_params.get("class_weight"),
            random_state=random_seed,
            n_jobs=1,
        )
    if model_name == "random_forest_regressor":
        return RandomForestRegressor(
            n_estimators=int(model_params.get("n_estimators", 300)),
            max_depth=model_params.get("max_depth"),
            min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
            random_state=random_seed,
            n_jobs=1,
        )
    if model_name == "hist_gradient_boosting_classifier":
        return HistGradientBoostingClassifier(
            learning_rate=float(learning_rate or 0.1),
            max_iter=int(epochs or 200),
            max_depth=model_params.get("max_depth"),
            max_leaf_nodes=model_params.get("max_leaf_nodes"),
            l2_regularization=float(model_params.get("l2_regularization", 0.0)),
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_seed,
        )
    if model_name == "hist_gradient_boosting_regressor":
        return HistGradientBoostingRegressor(
            learning_rate=float(learning_rate or 0.1),
            max_iter=int(epochs or 200),
            max_depth=model_params.get("max_depth"),
            max_leaf_nodes=model_params.get("max_leaf_nodes"),
            l2_regularization=float(model_params.get("l2_regularization", 0.0)),
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_seed,
        )
    if model_name == "mlp_classifier":
        return MLPClassifier(
            hidden_layer_sizes=tuple(model_params.get("hidden_layer_sizes", (64, 32))),
            alpha=float(model_params.get("alpha", 0.0001)),
            solver=optimizer or "adam",
            learning_rate_init=float(learning_rate or 0.001),
            max_iter=int(epochs or 250),
            batch_size=int(mini_batch) if mini_batch else "auto",
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_seed,
        )
    if model_name == "mlp_regressor":
        return MLPRegressor(
            hidden_layer_sizes=tuple(model_params.get("hidden_layer_sizes", (64, 32))),
            alpha=float(model_params.get("alpha", 0.0001)),
            solver=optimizer or "adam",
            learning_rate_init=float(learning_rate or 0.001),
            max_iter=int(epochs or 250),
            batch_size=int(mini_batch) if mini_batch else "auto",
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_seed,
        )
    raise ValueError(f"Unsupported model option: {model_name}")


def build_pipeline(
    profile: DataProfile,
    task_type: str,
    preprocessing_option: str,
    model_name: str,
    model_params: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    preprocessing_overrides: dict[str, Any] | None = None,
) -> Pipeline:
    params = model_params or {}
    config = training_config or {}
    estimator = build_estimator(task_type, model_name, params, config)
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(profile, preprocessing_option, overrides=preprocessing_overrides)),
            ("model", estimator),
        ]
    )


def convert_metric(name: str, value: float) -> float:
    return abs(float(value)) if name in {"rmse", "mae"} else float(value)


def evaluate_predictions(task_type: str, y_true: pd.Series, predictions) -> dict[str, float]:
    if task_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "f1_macro": float(f1_score(y_true, predictions, average="macro")),
        }
    return {
        "rmse": float(mean_squared_error(y_true, predictions, squared=False)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    model_name: str,
    role: str,
    scoring: dict[str, str],
    task_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv,
    notes: list[str] | None = None,
) -> CandidateResult:
    cv_results = cross_validate(clone(pipeline), X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
    fitted_pipeline = clone(pipeline).fit(X_train, y_train)
    predictions = fitted_pipeline.predict(X_test)
    return CandidateResult(
        model_name=model_name,
        role=role,
        cv_metrics={metric: convert_metric(metric, cv_results[f"test_{metric}"].mean()) for metric in scoring},
        test_metrics=evaluate_predictions(task_type, y_test, predictions),
        params={
            "model_type": fitted_pipeline.named_steps["model"].__class__.__name__,
            "pipeline_steps": [name for name, _ in fitted_pipeline.steps],
        },
        notes=notes or [],
    )


def choose_baseline_strategy(
    brief: ProjectBrief,
    profile: DataProfile,
    selected_metric: str,
) -> tuple[str, str]:
    if brief.task_type == "classification":
        return (
            "stratified_random",
            "Classification baselines should come from the training target distribution. "
            f"For the hard-label metric `{selected_metric}`, a stratified class-prior baseline is the strongest simple reference that uses no input features. "
            "`most_frequent` and `uniform_random` can still be reported as extra context when needed.",
        )

    if selected_metric in {"mae"}:
        return (
            "median_value",
            "Regression baselines should ignore the input features and use only the training target distribution. "
            f"For the absolute-error metric `{selected_metric}`, the median target value is the strongest simple no-feature baseline.",
        )
    return (
        "mean_value",
        "Regression baselines should ignore the input features and use only the training target distribution. "
        f"For the squared-error style metric `{selected_metric}`, the mean target value is the strongest simple no-feature baseline.",
    )


def baseline_predictions(
    task_type: str,
    strategy: str,
    y_train: pd.Series,
    y_test: pd.Series,
    random_seed: int = 42,
):
    if task_type == "classification":
        classes = y_train.value_counts(normalize=True).sort_index()
        labels = classes.index.to_list()
        probabilities = classes.to_numpy(dtype=float)
        rng = np.random.default_rng(random_seed)
        if strategy == "most_frequent":
            return np.repeat(y_train.mode(dropna=False).iloc[0], len(y_test))
        if strategy == "uniform_random":
            return rng.choice(labels, size=len(y_test))
        if strategy == "stratified_random":
            return rng.choice(labels, size=len(y_test), p=probabilities)
        raise ValueError(f"Unsupported classification baseline strategy: {strategy}")

    if strategy == "median_value":
        return np.repeat(float(y_train.median()), len(y_test))
    if strategy == "mean_value":
        return np.repeat(float(y_train.mean()), len(y_test))
    raise ValueError(f"Unsupported regression baseline strategy: {strategy}")


def evaluate_baseline(
    brief: ProjectBrief,
    profile: DataProfile,
    selected_metric: str,
    y_train: pd.Series,
    y_test: pd.Series,
) -> CandidateResult:
    strategy, reason = choose_baseline_strategy(brief, profile, selected_metric)
    predictions = baseline_predictions(brief.task_type, strategy, y_train, y_test)
    return CandidateResult(
        model_name=strategy,
        role="baseline",
        cv_metrics={},
        test_metrics=evaluate_predictions(brief.task_type, y_test, predictions),
        params={"baseline_strategy": strategy},
        notes=[reason],
    )


def hpo_parameter_grid(
    task_type: str,
    model_name: str,
    model_params: dict[str, Any],
    training_config: dict[str, Any],
    hpo_option: str,
    search_parameters: list[str] | None = None,
) -> dict[str, list[Any]]:
    del task_type, model_params, training_config
    search_config = hpo_search_space_config(model_name, hpo_option, requested_parameters=search_parameters)
    return dict(search_config["estimator_grid"])


def optimize_pipeline(
    pipeline: Pipeline,
    task_type: str,
    model_name: str,
    selected_metric: str,
    hpo_option: str,
    model_params: dict[str, Any],
    training_config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    hpo_config: dict[str, Any] | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    requested_search_parameters: list[str] | None = None
    if isinstance(hpo_config, dict):
        requested_search_parameters = list(hpo_config.get("search_parameters", []) or [])
    search_config = hpo_search_space_config(
        model_name,
        hpo_option,
        requested_parameters=requested_search_parameters,
    )
    grid = hpo_parameter_grid(
        task_type,
        model_name,
        model_params,
        training_config,
        hpo_option,
        search_parameters=requested_search_parameters,
    )
    scoring_name = metric_configuration(task_type, selected_metric)[0][selected_metric]
    search = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=grid,
        scoring=scoring_name,
        cv=cv,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    optimization_summary = {
        "selected_model": model_name,
        "search_parameters": list(search_config["selected_parameters"]),
        "search_space": {
            key: [str(item) for item in values]
            for key, values in search_config["display_grid"].items()
        },
        "best_params": search.best_params_,
        "best_cv_metric": convert_metric(selected_metric, search.best_score_),
    }
    return search.best_estimator_, optimization_summary


def selected_model_names(option: str) -> list[str]:
    return [option]


def model_catalog(task_type: str, imbalance_ratio: float | None) -> dict[str, Any]:
    dummy_profile = DataProfile(
        rows=1,
        columns=1,
        numeric_features=[],
        categorical_features=[],
        categorical_cardinality={},
        missing_by_feature={},
        constant_features=[],
        likely_identifier_features=[],
        date_like_features=[],
        date_parse_failure_by_feature={},
        high_cardinality_categorical_features={},
        missing_fraction=0.0,
        target_cardinality=2,
        target_name="target",
        class_imbalance=imbalance_ratio,
    )
    catalog: dict[str, Any] = {}
    for model_name in model_options_for_task(task_type):
        params = default_model_parameters(task_type, model_name, dummy_profile)
        training = default_training_configuration(task_type, model_name, dummy_profile)
        catalog[model_name] = build_estimator(task_type, model_name, params, training)
    return catalog


def leaderboard_row(
    pipeline: Pipeline,
    model_name: str,
    role: str,
    scoring: dict[str, str],
    task_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv,
) -> CandidateResult:
    return evaluate_pipeline(pipeline, model_name, role, scoring, task_type, X_train, y_train, X_test, y_test, cv)


def run_competition(
    shortlisted_names: list[str],
    task_type: str,
    profile: DataProfile,
    preprocessing_option: str,
    preprocessing_overrides: dict[str, Any] | None,
    scoring: dict[str, str],
    selected_metric: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    imbalance_ratio: float | None,
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for model_name in shortlisted_names:
        model_params = default_model_parameters(task_type, model_name, profile)
        training_config = default_training_configuration(task_type, model_name, profile)
        pipeline = build_pipeline(
            profile,
            task_type,
            preprocessing_option,
            model_name,
            model_params=model_params,
            training_config=training_config,
            preprocessing_overrides=preprocessing_overrides,
        )
        tuned_pipeline, optimization_summary = optimize_pipeline(
            pipeline,
            task_type,
            model_name,
            selected_metric,
            "small_competition",
            model_params,
            training_config,
            X_train,
            y_train,
            cv,
        )
        results.append(
            CandidateResult(
                model_name=f"{model_name}_tuned",
                role="competition",
                cv_metrics={selected_metric: float(optimization_summary["best_cv_metric"])},
                test_metrics={},
                params=optimization_summary["best_params"],
                notes=["Best result from the configured hyperparameter optimization stage."],
            )
        )
    return results


def pick_best_result(results: list[CandidateResult], selected_metric: str, higher_is_better: bool) -> CandidateResult:
    key = lambda result: result.cv_metrics.get(selected_metric, result.test_metrics.get(selected_metric, float("-inf")))
    return max(results, key=key) if higher_is_better else min(results, key=key)
