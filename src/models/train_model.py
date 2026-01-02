"""
Model training functions for the customer churn prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.helpers import load_config, save_pickle, create_directory

logger = setup_logger(__name__)


def get_model(model_name: str, params: Dict[str, Any]) -> Any:
    """
    Get model instance based on name and parameters.

    Args:
        model_name: Name of the model
        params: Model parameters

    Returns:
        Model instance
    """
    models = {
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'XGBClassifier': xgb.XGBClassifier
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name](**params)


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config_path: str = 'config.yaml'
) -> Any:
    """
    Train baseline logistic regression model.

    Args:
        X_train: Training features
        y_train: Training target
        config_path: Path to configuration file

    Returns:
        Trained model
    """
    logger.info("Training baseline model...")

    config = load_config(config_path)
    baseline_config = config['models']['baseline']

    model = get_model(baseline_config['name'], baseline_config['params'])
    model.fit(X_train, y_train)

    # Save baseline model
    model_dir = Path(config['persistence']['baseline_dir'])
    create_directory(model_dir)
    model_path = model_dir / 'logistic_regression.pkl'
    save_pickle(model, str(model_path))

    logger.info(f"Baseline model trained and saved to: {model_path}")

    return model


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    config_path: str = 'config.yaml'
) -> Any:
    """
    Train a specific model.

    Args:
        X_train: Training features
        y_train: Training target
        model_name: Name of model to train
        config_path: Path to configuration file

    Returns:
        Trained model
    """
    logger.info(f"Training {model_name}...")

    config = load_config(config_path)

    if model_name not in config['models']:
        raise ValueError(f"Model {model_name} not found in config")

    model_config = config['models'][model_name]
    model = get_model(model_config['name'], model_config['params'])

    # Train model
    model.fit(X_train, y_train)

    logger.info(f"{model_name} training completed")

    return model


def train_with_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: Any,
    cv_folds: int = 5,
    scoring: str = 'roc_auc'
) -> Tuple[Any, np.ndarray]:
    """
    Train model with cross-validation.

    Args:
        X_train: Training features
        y_train: Training target
        model: Model instance
        cv_folds: Number of CV folds
        scoring: Scoring metric

    Returns:
        Tuple of (trained model, cv scores)
    """
    logger.info(f"Training with {cv_folds}-fold cross-validation...")

    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1
    )

    logger.info(f"CV {scoring} scores: {cv_scores}")
    logger.info(f"Mean CV {scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full training set
    model.fit(X_train, y_train)

    return model, cv_scores


def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    config_path: str = 'config.yaml'
) -> Any:
    """
    Perform hyperparameter tuning using GridSearchCV.

    Args:
        X_train: Training features
        y_train: Training target
        model_name: Name of model to tune
        config_path: Path to configuration file

    Returns:
        Best model
    """
    logger.info(f"Performing hyperparameter tuning for {model_name}...")

    config = load_config(config_path)

    # Get base model
    model_config = config['models'][model_name]
    base_model = get_model(model_config['name'], {'random_state': 42})

    # Get parameter grid
    param_grid = config['tuning']['param_grids'].get(model_name, {})

    if not param_grid:
        logger.warning(f"No parameter grid found for {model_name}, using default params")
        return train_model(X_train, y_train, model_name, config_path)

    # Perform grid search
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=config['tuning']['cv_folds'],
        scoring=config['tuning']['scoring'],
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config_path: str = 'config.yaml',
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Train all models defined in config.

    Args:
        X_train: Training features
        y_train: Training target
        config_path: Path to configuration file
        save_models: Whether to save trained models

    Returns:
        Dictionary of trained models
    """
    logger.info("Training all models...")

    config = load_config(config_path)
    models = {}

    # Train each model
    for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
        if model_name in config['models']:
            try:
                model = train_model(X_train, y_train, model_name, config_path)
                models[model_name] = model

                # Save model
                if save_models:
                    model_dir = Path(config['persistence']['experiments_dir'])
                    create_directory(model_dir)
                    model_path = model_dir / f"{model_name}_v1.pkl"
                    save_pickle(model, str(model_path))
                    logger.info(f"Model saved to: {model_path}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")

    logger.info(f"Trained {len(models)} models successfully")

    return models


def save_production_model(
    model: Any,
    model_name: str = 'best_model',
    config_path: str = 'config.yaml'
) -> None:
    """
    Save model to production directory.

    Args:
        model: Trained model
        model_name: Name for the model file
        config_path: Path to configuration file
    """
    config = load_config(config_path)
    model_dir = Path(config['persistence']['production_dir'])
    create_directory(model_dir)

    model_path = model_dir / f"{model_name}.pkl"
    save_pickle(model, str(model_path))

    logger.info(f"Production model saved to: {model_path}")
