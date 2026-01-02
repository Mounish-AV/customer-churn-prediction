"""
Model evaluation functions for the customer churn prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.helpers import calculate_business_value, save_json

logger = setup_logger(__name__)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X: Features
        y: True labels
        threshold: Classification threshold

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model...")

    # Get predictions
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba)
    }

    # Log metrics
    logger.info("Model Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Get confusion matrix and its components.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Tuple of (confusion matrix, components dict)
    """
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    components = {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }

    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    return cm, components


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> str:
    """
    Get detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Classification report string
    """
    report = classification_report(y_true, y_pred)
    logger.info(f"Classification Report:\n{report}")

    return report


def evaluate_business_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clv: float = 1000,
    retention_cost: float = 100,
    acquisition_cost: float = 500
) -> Dict[str, float]:
    """
    Evaluate business impact of the model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        clv: Customer lifetime value
        retention_cost: Cost to retain a customer
        acquisition_cost: Cost to acquire a new customer

    Returns:
        Dictionary with business metrics
    """
    logger.info("Evaluating business impact...")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    business_metrics = calculate_business_value(
        tp, tn, fp, fn,
        clv, retention_cost, acquisition_cost
    )

    logger.info("Business Impact:")
    for metric, value in business_metrics.items():
        logger.info(f"  {metric}: ${value:,.2f}")

    return business_metrics


def compare_models(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series
) -> pd.DataFrame:
    """
    Compare multiple models.

    Args:
        models: Dictionary of model name to model instance
        X: Features
        y: True labels

    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Comparing {len(models)} models...")

    results = []

    for name, model in models.items():
        metrics = evaluate_model(model, X, y)
        metrics['model'] = name
        results.append(metrics)

    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('model')

    logger.info(f"Model Comparison:\n{df_results}")

    return df_results


def select_best_model(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    metric: str = 'roc_auc',
    minimize: bool = False
) -> Tuple[str, Any]:
    """
    Select best model based on a metric.

    Args:
        models: Dictionary of model name to model instance
        X: Features
        y: True labels
        metric: Metric to use for selection
        minimize: Whether to minimize the metric

    Returns:
        Tuple of (best model name, best model)
    """
    logger.info(f"Selecting best model based on {metric}...")

    df_results = compare_models(models, X, y)

    if minimize:
        best_model_name = df_results[metric].idxmin()
    else:
        best_model_name = df_results[metric].idxmax()

    best_model = models[best_model_name]
    best_score = df_results.loc[best_model_name, metric]

    logger.info(f"Best model: {best_model_name} with {metric}={best_score:.4f}")

    return best_model_name, best_model


def save_evaluation_results(
    metrics: Dict[str, Any],
    output_path: str = 'reports/evaluation_results.json'
) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to save results
    """
    save_json(metrics, output_path)
    logger.info(f"Evaluation results saved to: {output_path}")
