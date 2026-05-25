"""Custom transformers for feature engineering in the actuarial loss prediction pipeline."""

import logging
import os
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def ColumnDropper(to_drop=None):
    """Transformer to drop specified columns from the dataset."""

    class _ColumnDropper(BaseEstimator, TransformerMixin):
        def __init__(self, to_drop):
            self.to_drop = to_drop

        def fit(self, X, y=None):
            self.fitted_ = True
            return self

        def transform(self, X):
            return X.drop(self.to_drop, axis=1).reset_index(drop=True)

    return _ColumnDropper(to_drop)


def ConstantImputer(impute_columns, impute_fill_value):
    """Transformer to impute missing values in specified columns with a constant value."""
    return ColumnTransformer(
        transformers=[
            (
                "impute_MaritalStatus",
                SimpleImputer(strategy="constant", fill_value=impute_fill_value),
                impute_columns,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def EncodeCatFeatures(cat_features):
    """Transformer to encode categorical features using OneHotEncoder."""
    return ColumnTransformer(
        transformers=[
            (
                "encode_cat_features",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


# date features


def ReportDelayHours():
    """Transformer to compute the report delay in hours between DateTimeOfAccident and DateReported."""

    class _ReportDelayHours(BaseEstimator, TransformerMixin):
        def __init__(self):
            return

        def fit(self, X, y=None):
            self.fitted_ = True
            return self

        def transform(self, X):
            X = X.copy()
            X["report_delay"] = X["DateReported"] - X["DateTimeOfAccident"]
            X["report_delay_hours"] = X["report_delay"].dt.total_seconds() / 3600

            # reset index to concatenate correctly with FeatureUnion to pca embeddings (our row key is ClaimNumber)
            X.reset_index(inplace=True)

            return X[["report_delay_hours"]]

    return _ReportDelayHours()


def AccidentTime():
    """Transformer to extract accident time features (hour of day, day of week) from the DateTimeOfAccident column."""

    class _AccidentTime(BaseEstimator, TransformerMixin):
        def __init__(self):
            return

        def fit(self, X, y=None):
            self.fitted_ = True
            return self

        def transform(self, X):
            X = X.copy()
            X["accident_year"] = X["DateTimeOfAccident"].dt.year
            X["accident_month"] = X["DateTimeOfAccident"].dt.month
            X["accident_day"] = X["DateTimeOfAccident"].dt.day
            X["accident_hour"] = X["DateTimeOfAccident"].dt.hour

            # reset index to concatenate correctly with FeatureUnion to pca embeddings (our row key is ClaimNumber)
            X.reset_index(inplace=True)

            return X[
                ["accident_year", "accident_month", "accident_day", "accident_hour"]
            ]

    return _AccidentTime()


def ReportTime():
    """Transformer to extract report time features from the DateReported column."""

    class _ReportTime(BaseEstimator, TransformerMixin):
        def __init__(self):
            return

        def fit(self, X, y=None):
            self.fitted_ = True
            return self

        def transform(self, X):
            X = X.copy()
            X["reported_year"] = X["DateReported"].dt.year
            X["reported_month"] = X["DateReported"].dt.month
            X["reported_day"] = X["DateReported"].dt.day

            # reset index to concatenate correctly with FeatureUnion to pca embeddings (our row key is ClaimNumber)
            X.reset_index(inplace=True)

            return X[["reported_year", "reported_month", "reported_day"]]

    return _ReportTime()


# Claim description word count


def ClaimDescriptionWordCount():
    """Transformer to compute the word count of the claim description."""

    class _ClaimDescriptionWordCount(BaseEstimator, TransformerMixin):
        def __init__(self):
            return

        def fit(self, X, y=None):
            self.fitted_ = True
            return self

        def transform(self, X):
            X = X.copy()
            X["ClaimDescription_word_count"] = X["ClaimDescription"].apply(
                lambda x: len(str(x).split())
            )

            # reset index to concatenate correctly with FeatureUnion to pca embeddings (our row key is ClaimNumber)
            X.reset_index(inplace=True)

            return X[["ClaimDescription_word_count"]]

    return _ClaimDescriptionWordCount()


def ColumnSelector(selected_columns):
    """Transformer to select specified columns from the dataset.

    Args:
        selected_columns (list): List of column names to select from the dataset.
    """

    class _ColumnSelector(BaseEstimator, TransformerMixin):
        def __init__(self, selected_columns):
            self.columns = selected_columns

        def fit(self, X, y=None):
            self.fitted_ = True
            return self

        def transform(self, X):
            return X[self.columns]

    return _ColumnSelector(selected_columns)


def SentenceEmbedding(
    emb_model_name, batch_size, load_embeddings_path, save_embeddings_path, verbose=True
):
    """Sentence embedding transformer using SentenceTransformer with options to load/save embeddings and enriched logging.

    Args:
        emb_model_name (str): The name of the SentenceTransformer model to use.
        batch_size (int): The batch size for embedding computation.
        load_embeddings_path (str): Path to load precomputed embeddings from (if exists).
        save_embeddings_path (str): Directory path to save computed embeddings with versioning.
        verbose (bool): Whether to log detailed information about the embedding process.
    """

    class _SentenceEmbedding(BaseEstimator, TransformerMixin):
        def __init__(
            self,
            emb_model_name,
            batch_size,
            load_embeddings_path,
            save_embeddings_path,
            verbose,
        ):
            self.model_name = emb_model_name
            self.batch_size = batch_size
            self.load_embeddings_path = load_embeddings_path
            self.save_embeddings_path = save_embeddings_path
            self.verbose = verbose

        def _model_tag(self):
            return self.model_name.replace("/", "_").replace("\\", "_")

        def _log(self, msg):
            if self.verbose:
                logger.info(f"[SentenceEmbedding] {msg}")

        def _version_path(self):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_tag = self._model_tag()
            path = os.path.join(
                self.save_embeddings_path, f"desc_embeddings_{model_tag}_{ts}.db"
            )
            return path

        def fit(self, X, y=None):
            # Skip fitting if loading precomputed embeddings
            if self.load_embeddings_path is not None and os.path.exists(
                self.load_embeddings_path
            ):
                self._log(
                    f"Skipping fit (loading embeddings from {self.load_embeddings_path})"
                )
                return self

            self._log(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.fitted_ = True
            return self

        def transform(self, X):
            # Option 1: load existing embeddings
            if self.load_embeddings_path is not None and os.path.exists(
                self.load_embeddings_path
            ):
                self._log(f"Loading embeddings from {self.load_embeddings_path} ...")
                conn = sqlite3.connect(self.load_embeddings_path)
                emb_df = pd.read_sql("SELECT * FROM desc_embeddings", conn)
                conn.close()

                self._log(f"Loaded embeddings shape: {emb_df.shape}")

                # filter by ClaimNumber in dataset
                emb_df_filtered = X[["ClaimNumber"]].merge(
                    emb_df, on="ClaimNumber", how="left"
                )
                emb = emb_df_filtered.drop(["ClaimNumber"], axis=1)

                # replace NaNs and infs
                embeddings = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

                return embeddings

            # Option 2: compute embeddings
            self._log("Computing embeddings...")

            # input format to sentence transformer is list of strings
            X_list = pd.Series(X).fillna("").astype(str).tolist()

            start = time.time()
            emb = self.model.encode(
                X_list, batch_size=self.batch_size, show_progress_bar=True
            )
            self._log(f"Embedding done in {time.time() - start:.2f}s")

            embeddings = emb.to_numpy(dtype=np.float32)

            # Save embeddings dataframe as a SQL table to versionned path
            self._log("Saving embeddings...")
            df_embeddings = X[["ClaimNumber"]].join(pd.DataFrame(embeddings))
            version_path = self._version_path()

            conn = sqlite3.connect(version_path)

            df_embeddings.to_sql(
                name="desc_embeddings", con=conn, if_exists="fail", index=False
            )

            conn.close()

            self._log(f"Saved embeddings to {version_path}")

            return embeddings

    return _SentenceEmbedding(
        emb_model_name, batch_size, load_embeddings_path, save_embeddings_path, verbose
    )


def PCATransformer(n_components, verbose=True):
    """PCA transformer for dimensionality reduction of embeddings with enriched logging.

    Args:
        n_components (int): The number of principal components to keep.
        verbose (bool): Whether to log the time taken for fitting and transforming.
    """

    class _PCATransformer(BaseEstimator, TransformerMixin):
        def __init__(self, n_components, verbose):
            self.n_components = n_components
            self.verbose = verbose

        def _log(self, msg):
            if self.verbose:
                logger.info(f"[PCA] {msg}")

        def _model_tag(self):
            return f"pca_{self.n_components}"

        def fit(self, X, y=None):
            self._log(f"Fitting PCA (n_components={self.n_components})...")
            start = time.time()

            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(X)

            self._log(f"PCA fitted in {time.time() - start:.2f}s")
            self.fitted_ = True

            return self

        def transform(self, X):
            # Compute PCA
            self._log("Transforming embeddings with PCA...")

            start = time.time()
            X_pca = self.pca.transform(X)

            self._log(f"PCA done in {time.time() - start:.2f}s")
            self._log(f"PCA output shape: {X_pca.shape}")

            return X_pca

    return _PCATransformer(n_components, verbose)
