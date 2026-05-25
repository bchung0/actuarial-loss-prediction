from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import best_model_prediction


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model prediction pipeline."""
    return Pipeline(
        [
            Node(
                best_model_prediction,
                inputs=[
                    "best_model",
                    "df_test",
                ],
                outputs="predictions",
                name="Model_Prediction",
            )
        ],
        namespace="model_prediction",
        inputs=["best_model", "df_test"],
        outputs="predictions",
    )
