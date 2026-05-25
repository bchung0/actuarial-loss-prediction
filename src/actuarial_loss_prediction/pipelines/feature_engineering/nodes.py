"""Nodes for the feature engineering pipeline."""

from sklearn import set_config
from sklearn.pipeline import FeatureUnion, Pipeline

from .transformers import (
    AccidentTime,
    ClaimDescriptionWordCount,
    ColumnDropper,
    ColumnSelector,
    ConstantImputer,
    EncodeCatFeatures,
    PCATransformer,
    ReportDelayHours,
    ReportTime,
    SentenceEmbedding,
)


def setup_feature_engineering_pipeline(feature_engineering_params: dict) -> Pipeline:
    """Set up the feature engineering pipeline.

    Args:
        feature_engineering_params (dict): The parameters for the feature engineering pipeline.

    Returns:
        Pipeline: The feature engineering pipeline.
    """
    set_config(transform_output="pandas")  # to get dataframe as Transformer output

    date_feature_eng = FeatureUnion(
        [
            ("add_report_delay_days", ReportDelayHours()),
            ("add_accident_time_features", AccidentTime()),
            ("add_report_time_features", ReportTime()),
        ]
    )
    desc_embedding_pca = Pipeline(
        [
            (
                "select_desc_column",
                ColumnSelector(feature_engineering_params["selected_columns"]),
            ),
            (
                "embed_desc",
                SentenceEmbedding(
                    feature_engineering_params["emb_model_name"],
                    feature_engineering_params["batch_size"],
                    feature_engineering_params["load_embeddings_path"],
                    feature_engineering_params["save_embeddings_path"],
                ),
            ),
            (
                "pca_embed_desc",
                PCATransformer(feature_engineering_params["n_components"]),
            ),
        ]
    )

    claim_desc_feature_eng = FeatureUnion(
        [
            ("add_desc_word_count", ClaimDescriptionWordCount()),
            ("add_desc_embedding_pca", desc_embedding_pca),
        ]
    )

    cat_feature_eng = Pipeline(
        [
            (
                "impute_MaritalStatus",
                ConstantImputer(
                    feature_engineering_params["impute_column"],
                    feature_engineering_params["impute_fill_value"],
                ),
            ),
            (
                "encode_cat_features",
                EncodeCatFeatures(feature_engineering_params["cat_features"]),
            ),
        ]
    )

    feature_engineering_pipeline = Pipeline(
        [
            ("cat_feature_eng", cat_feature_eng),
            (
                "add_new_features",
                FeatureUnion(
                    [
                        ("date_fe", date_feature_eng),
                        ("claim_desc_fe", claim_desc_feature_eng),
                        (
                            "drop_cols",
                            ColumnDropper(feature_engineering_params["to_drop"]),
                        ),
                    ]
                ),
            ),
        ]
    )

    #  feature_engineering_pipeline.set_params(**feature_engineering_params)

    return feature_engineering_pipeline
