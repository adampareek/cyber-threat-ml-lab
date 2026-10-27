from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(labels: pd.Series, probabilities, threshold: float = 0.5) -> dict[str, float]:
    """Calculate common binary classification metrics."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "precision": round(precision_score(labels, predictions, zero_division=0), 4),
        "recall": round(recall_score(labels, predictions, zero_division=0), 4),
        "f1": round(f1_score(labels, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(labels, probabilities), 4),
    }


def evaluate_model(model, features: pd.DataFrame | pd.Series, labels: pd.Series) -> dict[str, float]:
    """Score a fitted model on a held-out dataset."""
    probabilities = model.predict_proba(features)[:, 1]
    return classification_metrics(labels, probabilities)
