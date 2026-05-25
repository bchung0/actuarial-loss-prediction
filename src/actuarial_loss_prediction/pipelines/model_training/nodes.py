"""Nodes for the model training pipeline."""

import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


def build_model_pipeline(
    feature_engineering_pipeline: Pipeline, model_params: dict = {}
) -> Pipeline:
    """Build the model pipeline.

    Args:
        feature_engineering_pipeline (Pipeline): The feature engineering pipeline.
        model_params (dict): The parameters for the HistGradientBoostingRegressor model.

    Returns:
        Pipeline: The complete model pipeline.
    """
    return Pipeline(
        [
            ("feature_engineering", feature_engineering_pipeline),
            ("model", HistGradientBoostingRegressor(**model_params)),
        ]
    )


def objective(
    trial,
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_space: dict,
    cv_random_state: int,
):
    """Objective function for Optuna optimization.

    Args:
        trial: The Optuna trial object.
        model_pipeline (Pipeline): The model pipeline.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        search_space (dict): The search space for hyperparameter tuning.
        cv_random_state (int): The random state for cross-validation.

    Returns:
        The mean cross-validation score for the given hyperparameters.
    """
    # feature engs
    # set model params
    params = dict()
    for param_name, info in search_space.items():
        params[param_name] = getattr(trial, f"suggest_{info['type']}")(
            param_name, **info["params"]
        )
    model_pipeline["model"].set_params(**params)

    # fit model pipeline

    # model_pipeline["model"].fit(X_train, y_train)

    score = cross_val_score(
        estimator=model_pipeline["model"],
        X=X_train,
        y=y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=cv_random_state),
        scoring="neg_root_mean_squared_error",
    )
    mean_score = -score.mean()

    return mean_score


def model_tuning(
    df_train: pd.DataFrame,
    model_pipeline: Pipeline,
    target_name: int,
    search_space: dict,
    cv_random_state: int,
    n_trials: int,
):
    """Tune the model using Optuna.

    Args:
        df_train (pd.DataFrame): The training data.
        model_pipeline (Pipeline): The model pipeline.
        target_name (str): The name of the target variable.
        search_space (dict): The search space for hyperparameter tuning.
        cv_random_state (int): The random state for cross-validation.
        n_trials (int): The number of trials for Optuna optimization.

    Returns:
        The best model after hyperparameter tuning.
    """
    X_train = df_train.drop(columns=[target_name])
    y_train = df_train[target_name]

    X_train_fe = model_pipeline["feature_engineering"].fit_transform(X_train, y_train)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial, model_pipeline, X_train_fe, y_train, search_space, cv_random_state
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_model = build_model_pipeline(
        model_pipeline["feature_engineering"], study.best_params
    )
    best_model.fit(X_train, y_train)

    return best_model
