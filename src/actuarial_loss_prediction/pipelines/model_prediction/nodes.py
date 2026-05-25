"""Nodes for the model prediction pipeline."""

import pandas as pd


def best_model_prediction(
    best_model,
    df_test: pd.DataFrame,
):
    """Make predictions using the best model.

    Args:
        best_model: The best model after hyperparameter tuning.
        df_test (pd.DataFrame): The test data.

    Returns:
            pd.DataFrame: The test data with the claim number and predictions.
    """
    predictions = best_model.predict(df_test)

    df_test["UltimateIncurredClaimCost"] = predictions

    return df_test[["ClaimNumber", "UltimateIncurredClaimCost"]]
