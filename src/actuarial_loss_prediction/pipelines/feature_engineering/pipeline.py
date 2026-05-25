"""Pipeline definition for the feature engineering pipeline."""

from kedro.pipeline import Node, Pipeline, pipeline

from .nodes import setup_feature_engineering_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline."""
    return pipeline(
        [
            Node(
                setup_feature_engineering_pipeline,
                inputs="params:feature_engineering_parameters",
                outputs="feature_engineering_pipeline",
                name="Setup_Feature_Engineering",
            )
        ],
        namespace="feature_engineering",
        inputs=None,
        parameters="params:feature_engineering_parameters",
        outputs="feature_engineering_pipeline",
    )
