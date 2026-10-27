from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cyber_threat_ml.features import URLFeatureExtractor, add_malware_features, add_network_features


def split_features_labels(df: pd.DataFrame, label_col: str = "label", test_size: float = 0.25, random_state: int = 42):
    """Create a stratified train/test split."""
    features = df.drop(columns=[label_col])
    labels = df[label_col]
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


def _structured_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [col for col in features.columns if col not in numeric_columns]

    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def build_malware_pipeline(train_features: pd.DataFrame) -> Pipeline:
    """Build a malware classifier pipeline."""
    engineered_features = add_malware_features(train_features)
    preprocessor = _structured_preprocessor(engineered_features)

    return Pipeline(
        [
            ("features", FunctionTransformer(add_malware_features, validate=False)),
            ("preprocessor", preprocessor),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=160,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_network_pipeline(train_features: pd.DataFrame) -> Pipeline:
    """Build a network intrusion classifier pipeline."""
    engineered_features = add_network_features(train_features)
    preprocessor = _structured_preprocessor(engineered_features)

    return Pipeline(
        [
            ("features", FunctionTransformer(add_network_features, validate=False)),
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=140,
                    learning_rate=0.06,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )


def build_phishing_pipeline() -> Pipeline:
    """Build a phishing URL classifier using raw URL text plus numeric URL stats."""
    feature_union = FeatureUnion(
        [
            (
                "url_stats",
                Pipeline(
                    [
                        ("stats", URLFeatureExtractor()),
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=15000,
                    lowercase=True,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", feature_union),
            (
                "classifier",
                LogisticRegression(
                    C=3.0,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def prediction_frame(model: Pipeline, features: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Return a standard probability-scoring dataframe."""
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame({"index": features.index, "prob_label_1": probabilities})
