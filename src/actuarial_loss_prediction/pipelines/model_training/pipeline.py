from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import build_model_pipeline, model_tuning


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model training pipeline."""
    return Pipeline(
        [
            Node(
                build_model_pipeline,
                inputs="feature_engineering_pipeline",
                outputs="model_pipeline",
                name="Build_Model_Pipeline",
            ),
            Node(
                model_tuning,
                inputs=[
                    "df_train",
                    "model_pipeline",
                    "params:target_name",
                    "params:search_space",
                    "params:cv_random_state",
                    "params:n_trials",
                ],
                outputs="best_model",
                name="Model_Tuning",
            ),
        ],
        namespace="model_training",
        inputs=["feature_engineering_pipeline", "df_train"],
        parameters=[
            "params:target_name",
            "params:search_space",
            "params:cv_random_state",
            "params:n_trials",
        ],
        outputs="best_model",
    )
