"""
Visualization functions for the customer churn prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.helpers import create_directory

logger = setup_logger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
    """
    logger.info("Plotting confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        create_directory(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save figure
    """
    logger.info("Plotting ROC curve...")

    fpr, tpr, _ = roc_curve(y_true, y_proba)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        create_directory(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to: {save_path}")

    plt.close()


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    logger.info("Plotting feature importance...")

    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        create_directory(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to: {save_path}")

    plt.close()


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    save_path: Optional[str] = None
) -> None:
    """
    Plot target variable distribution.

    Args:
        df: DataFrame with target column
        target_col: Name of target column
        save_path: Path to save figure
    """
    logger.info("Plotting target distribution...")

    plt.figure(figsize=(8, 6))
    df[target_col].value_counts().plot(kind='bar')
    plt.title(f'{target_col} Distribution')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_path:
        create_directory(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Target distribution plot saved to: {save_path}")

    plt.close()


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot correlation matrix for numerical features.

    Args:
        df: DataFrame
        save_path: Path to save figure
    """
    logger.info("Plotting correlation matrix...")

    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(12, 10))
    sns.heatmap(numerical_df.corr(), annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()

    if save_path:
        create_directory(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix saved to: {save_path}")

    plt.close()


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = 'roc_auc',
    save_path: Optional[str] = None
) -> None:
    """
    Plot model comparison.

    Args:
        results_df: DataFrame with model comparison results
        metric: Metric to plot
        save_path: Path to save figure
    """
    logger.info(f"Plotting model comparison for {metric}...")

    plt.figure(figsize=(10, 6))
    results_df[metric].plot(kind='bar')
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xlabel('Model')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        create_directory(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to: {save_path}")

    plt.close()
