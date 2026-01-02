"""
Model prediction functions for the customer churn prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Optional, Union
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.helpers import load_pickle, load_config

logger = setup_logger(__name__)


def load_production_model(
    model_name: str = 'best_model',
    config_path: str = 'config.yaml'
) -> Any:
    """
    Load production model from file.

    Args:
        model_name: Name of the model file
        config_path: Path to configuration file

    Returns:
        Loaded model
    """
    config = load_config(config_path)
    model_dir = Path(config['persistence']['production_dir'])
    model_path = model_dir / f"{model_name}.pkl"

    logger.info(f"Loading model from: {model_path}")
    model = load_pickle(str(model_path))

    return model


def predict(
    model: Any,
    X: pd.DataFrame,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Make predictions using trained model.

    Args:
        model: Trained model
        X: Features to predict on
        threshold: Classification threshold

    Returns:
        Predictions array
    """
    logger.info(f"Making predictions on {X.shape[0]} samples...")

    # Get probability predictions
    y_proba = model.predict_proba(X)[:, 1]

    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)

    logger.info(f"Predictions completed. Positive class: {y_pred.sum()}")

    return y_pred


def predict_proba(
    model: Any,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Get probability predictions.

    Args:
        model: Trained model
        X: Features to predict on

    Returns:
        Probability predictions array
    """
    logger.info(f"Getting probability predictions on {X.shape[0]} samples...")

    y_proba = model.predict_proba(X)[:, 1]

    return y_proba


def batch_predict(
    model_path: str,
    data_path: str,
    output_path: str,
    threshold: float = 0.5
) -> None:
    """
    Make batch predictions on a dataset.

    Args:
        model_path: Path to trained model
        data_path: Path to input data
        output_path: Path to save predictions
        threshold: Classification threshold
    """
    logger.info("Running batch prediction...")

    # Load model
    model = load_pickle(model_path)

    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {df.shape}")

    # Make predictions
    y_proba = predict_proba(model, df)
    y_pred = (y_proba >= threshold).astype(int)

    # Add predictions to dataframe
    df['churn_probability'] = y_proba
    df['churn_prediction'] = y_pred

    # Save results
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to: {output_path}")


def predict_single(
    model: Any,
    features: dict,
    feature_names: list
) -> dict:
    """
    Make prediction for a single customer.

    Args:
        model: Trained model
        features: Dictionary of feature values
        feature_names: List of feature names in correct order

    Returns:
        Dictionary with prediction and probability
    """
    # Create dataframe from features
    df = pd.DataFrame([features])

    # Ensure correct column order
    df = df[feature_names]

    # Make prediction
    proba = model.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)

    result = {
        'prediction': pred,
        'probability': float(proba),
        'churn_risk': 'High' if proba >= 0.7 else 'Medium' if proba >= 0.3 else 'Low'
    }

    logger.info(f"Single prediction: {result}")

    return result
